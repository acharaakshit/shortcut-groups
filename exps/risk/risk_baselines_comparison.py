from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from exps.interventions.common import load_audited_model, load_test_dataset

from exps.risk.common import (
    balanced_split,
    dataset_half_fit_size,
    kmeans_risk_scores,
    shortcut_group_error_rates,
    load_test_results,
    nmf_risk_scores,
)
from shortcut_groups.datasets import DATASET_LABELS, dataset_names
from shortcut_groups.experiment_utils import (
    artifact_prefix,
    completed_csv_keys,
    merge_csv_rows,
    require_exact_keys,
    errors_found_at_coverage,
)
from shortcut_groups.io import load_model_attribution_triplet
from shortcut_groups.interventions.evaluation import BATCH_SIZE, DTYPE, batch_from_dataset
from shortcut_groups.methods import build_shortcut_group_features
from shortcut_groups.partial_correlations import build_shortcut_task_maps

from domino._slice.mixture import DominoSlicer



# ablation on the risk scores derived using various shortcut-task representations and raw attributions

METHODS = (
    "joint_m_sc_m_task",
    "confidence",
    "rho_sc",
    "rho_task",
    "m_sc",
    "m_task",
    "raw_attr",
    "representation_kmeans",
    "domino",
    "combined_quadrants",
)
KS = (2, 4, 8, 16, 32)
SCALARS = {"confidence", "rho_sc", "rho_task"}
OUT = Path("results/risk_baselines/k_sweep")
REPRESENTATION_DIR = Path("results/risk_baselines/representations")
KEYS = ["dataset", "model", "seed", "split_seed", "method", "k"]


def softmax(logits):
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def penultimate_representations(model, images, model_name):
    if model_name == "resnet":
        return model.forward_head(model.forward_features(images), pre_logits=True)
    tokens = model._process_input(images)
    cls = model.class_token.expand(len(images), -1, -1)
    return model.encoder(torch.cat((cls, tokens), dim=1))[:, 0]


@torch.no_grad()
def load_representations(root, dataset_name, model_name, seed):
    path = REPRESENTATION_DIR / f"{dataset_name}_{model_name}_seed{seed}_penultimate.npy"
    if path.exists():
        return np.load(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_test_dataset(root, dataset_name)
    model = load_audited_model(root, dataset_name, model_name, seed, device)
    batches = []
    for start in tqdm(range(0, len(dataset), BATCH_SIZE), unit="batch", leave=False,
                      desc=f"{dataset_name} {model_name} seed={seed} representations"):
        images, _, _ = batch_from_dataset(dataset, range(start, min(start+BATCH_SIZE, len(dataset))), device)
        batches.append(penultimate_representations(model, images.to(DTYPE), model_name).cpu().numpy())
    representations = np.concatenate(batches)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, representations)
    temporary.replace(path)
    return representations

# just a secondary ranking to avoid selecting randomly ordered images when there are ties
def domino_secondary_ranking(embeddings, memberships, component_indices, means, variances):
    assigned = component_indices[memberships.argmax(axis=1)]
    variance = variances[assigned]
    # negative squared Mahalanobis distance to the assigned DOMINO group's mean embedding
    return -np.sum((embeddings - means[assigned]) ** 2 / variance, axis=1)


def domino_risk_scores(representations, probabilities, targets, errors, fit_idx, heldout_idx, seed, k=8):
    fit = np.asarray(representations[fit_idx], dtype=np.float64)
    heldout = np.asarray(representations[heldout_idx], dtype=np.float64)
    # fit DOMINO with same representation, number of slices, mixtures are  no of shortcut groups
    slicer = DominoSlicer(n_slices=k, n_mixture_components=k, n_pca_components=None,
                         init_params="kmeans", random_state=seed, max_iter=200, pbar=False)
    # fit using embeddings, target labels and probabilities
    slicer.fit(data=None, embeddings=fit, targets=targets[fit_idx],
               pred_probs=np.asarray(probabilities[fit_idx], dtype=np.float64))
    # Assign memberships using embeddings only, including on held-out images.
    memberships = slicer.predict_proba(data=None, embeddings=fit, targets=None, pred_probs=None)
    heldout_memberships = slicer.predict_proba(data=None, embeddings=heldout, targets=None, pred_probs=None)
    rates = shortcut_group_error_rates(memberships, errors[fit_idx])
    gaussian = next(variable for variable in slicer.mm.variables if variable.name == "embeddings")
    # break risk ties using proximity to the assigned DOMINO group's mean embedding
    proximity = domino_secondary_ranking(heldout, heldout_memberships, slicer.slice_cluster_indices,
                                               gaussian.means_, gaussian.covariances_)
    return heldout_memberships @ rates, proximity


def load_ablation_inputs(root, dataset, model, seed):
    pa, ba, bi = load_model_attribution_triplet(
        root,
        dataset,
        model,
        "LRP",
        3136,
        seed,
    )
    alignment_outputs = build_shortcut_task_maps(bi, ba, pa)
    test, errors = load_test_results(root, dataset, model, seed)
    probabilities = softmax(test["pred_logits"])
    return {
        "test": test,
        "errors": errors,
        "probabilities": probabilities,
        "confidence": -probabilities.max(axis=1),
        "combined_quadrants": build_shortcut_group_features(alignment_outputs, "signed-four-channel"),
        "joint_m_sc_m_task": np.concatenate(
            [
                alignment_outputs["shortcut_evidence"],
                alignment_outputs["task_evidence"],
            ],
            axis=1,
        ),
        "m_sc": alignment_outputs["shortcut_evidence"],
        "m_task": alignment_outputs["task_evidence"],
        "raw_attr": np.maximum(bi, 0.0),
        "rho_sc": alignment_outputs["shortcut_scores"],
        "rho_task": alignment_outputs["task_scores"],
    }


def evaluate_ablation_method(
    dataset, model, seed, split_seed, method, inputs, *, k=8
):
    targets = inputs["test"]["targets"]
    attrs = inputs["test"]["attrs"]
    fit_idx, heldout_idx = balanced_split(
        targets, attrs, dataset_half_fit_size(dataset), split_seed
    )
    secondary_ranking = None
    if method in {"confidence", "rho_sc"}:
        scores = inputs[method][heldout_idx]
    elif method == "rho_task":
        scores = -inputs[method][heldout_idx]
    elif method == "representation_kmeans":
        scores, secondary_ranking = kmeans_risk_scores(
            inputs["representations"], inputs["errors"], fit_idx, heldout_idx, k
        )
    elif method == "domino":
        scores, secondary_ranking = domino_risk_scores(
            inputs["representations"], inputs["probabilities"], targets,
            inputs["errors"], fit_idx, heldout_idx, split_seed, k=k,
        )
    else:
        scores = nmf_risk_scores(
            inputs[method],
            inputs["errors"],
            fit_idx,
            heldout_idx,
            k,
        )
    # compute errors found for each of the methods
    capture = errors_found_at_coverage(
        inputs["errors"][heldout_idx], scores, 0.2, secondary_ranking=secondary_ranking
    )
    return {
        "dataset": dataset,
        "dataset_label": DATASET_LABELS[dataset],
        "model": model,
        "seed": seed,
        "split_seed": split_seed,
        "attribution_method": "LRP",
        "regions": 3136,
        "num_shortcut_groups": k,
        "method": method,
        "coverage": 0.2,
        "baseline_err": inputs["errors"][heldout_idx].mean(),
        "capture": capture,
        "errors_found_pct": 100.0 * capture,
    }

def main():
    root = artifact_prefix()
    paths = []
    for dataset in dataset_names():
        for model in ("vit", "resnet"):
            for seed in range(4): # four independently trained models
                path = OUT / f"{dataset}_{model}_seed{seed}.csv"
                paths.append(path)
                completed = completed_csv_keys(path, KEYS)
                pending = [
                    (split_seed, method, k)
                    for split_seed in range(5) # five held-out sets
                    for k in KS
                    for method in METHODS
                    if (dataset, model, seed, split_seed, method, k)
                    not in completed
                ]
                if not pending:
                    continue
                inputs, scalar_rows, rows = None, {}, []
                for split_seed, method, k in pending:
                    row = scalar_rows.get((split_seed, method))
                    if row is None:
                        if inputs is None:
                            inputs = load_ablation_inputs(root, dataset, model, seed)
                        if method in {"representation_kmeans", "domino"} and "representations" not in inputs:
                            inputs["representations"] = load_representations(root, dataset, model, seed)
                        row = evaluate_ablation_method(
                            dataset, model, seed, split_seed, method, inputs, k=k
                        )
                    if method in SCALARS:
                        scalar_rows[(split_seed, method)] = row
                    rows.append(row | {"k": k, "num_shortcut_groups": k})
                merge_csv_rows(path, rows, KEYS)
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    raw = raw[raw.method.isin(METHODS)]
    require_exact_keys(
        raw, KEYS, product(dataset_names(), ("vit", "resnet"), range(4), range(5), METHODS, KS),
        "baseline K sweep",
    )
    raw.to_csv(OUT / "all.csv", index=False)
    print(f"wrote {OUT / 'all.csv'}")


if __name__ == "__main__":
    main()
