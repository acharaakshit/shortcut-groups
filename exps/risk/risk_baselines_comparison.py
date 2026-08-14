from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

from exps.risk.common import (
    balanced_split,
    dataset_half_fit_size,
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
from shortcut_groups.partial_correlations import build_shortcut_task_maps


# ablation on the risk scores derived using various shortcut-task representations and raw attributions

METHODS = (
    "joint_m_sc_m_task",
    "rho_sc",
    "rho_task",
    "m_sc",
    "m_task",
    "raw_attr",
)
RAW_OUT = Path(
    "results/risk_baselines/heldout_risk_ablation_lrp_56x56_k8.csv"
)
SUMMARY_OUT = Path(
    "results/risk_baselines/heldout_risk_ablation_lrp_56x56_k8_summary.csv"
)
RESULT_KEYS = ["dataset", "model", "seed", "split_seed", "method"]


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
    if len(errors) != bi.shape[0]:
        raise ValueError(
            f"{dataset} {model} seed={seed} attribution/test row mismatch: "
            f"attributions={bi.shape[0]}, test={len(errors)}"
        )
    return {
        "test": test,
        "errors": errors,
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
    dataset, model, seed, split_seed, method, inputs
):
    targets = inputs["test"]["targets"]
    attrs = inputs["test"]["attrs"]
    fit_idx, heldout_idx = balanced_split(
        targets, attrs, dataset_half_fit_size(dataset), split_seed
    )
    if method == "rho_sc":
        scores = inputs[method][heldout_idx]
    elif method == "rho_task":
        scores = -inputs[method][heldout_idx]
    else:
        scores = nmf_risk_scores(
            inputs[method],
            inputs["errors"],
            fit_idx,
            heldout_idx,
            8,
        )
    # compute errors found for each of the methods
    capture = errors_found_at_coverage(
        inputs["errors"][heldout_idx], scores, 0.2
    )
    return {
        "dataset": dataset,
        "dataset_label": DATASET_LABELS[dataset],
        "model": model,
        "seed": seed,
        "split_seed": split_seed,
        "attribution_method": "LRP",
        "regions": 3136,
        "num_shortcut_groups": 8,
        "method": method,
        "coverage": 0.2,
        "baseline_err": inputs["errors"][heldout_idx].mean(),
        "capture": capture,
        "errors_found_pct": 100.0 * capture,
    }

# for each model seed, average across five held-out sets and then show deviation across model seeds
def summarise(raw):
    expected = product(
        dataset_names(), ("vit", "resnet"), range(4), range(5), METHODS
    )
    require_exact_keys(
        raw,
        ["dataset", "model", "seed", "split_seed", "method"],
        expected,
        "risk-ablation results",
    )
    # Average across the five held-out splits within each training seed.
    per_seed = (
        raw.groupby(
            [
                "dataset",
                "dataset_label",
                "model",
                "seed",
                "method",
            ],
            as_index=False,
        )
        .agg(errors_found_pct=("errors_found_pct", "mean"))
    )
    # Summarise across the four training seeds.
    return (
        per_seed.groupby(
            ["dataset", "dataset_label", "model", "method"],
            as_index=False,
        )
        .agg(
            mean=("errors_found_pct", "mean"),
            std=("errors_found_pct", lambda x: x.std(ddof=0)),
            n_seeds=("seed", "size"),
        )
    )


def main():
    root = artifact_prefix()
    completed = completed_csv_keys(
        RAW_OUT,
        RESULT_KEYS,
    )
    for dataset in dataset_names():
        for model in ("vit", "resnet"):
            for seed in range(4): # four independently trained models
                pending = [
                    (split_seed, method)
                    for split_seed in range(5) # five held-out sets
                    for method in METHODS
                    if (dataset, model, seed, split_seed, method)
                    not in completed
                ]
                if not pending:
                    continue
                inputs = load_ablation_inputs(root, dataset, model, seed)
                rows = [
                    evaluate_ablation_method(
                        dataset, model, seed, split_seed, method, inputs
                    )
                    for split_seed, method in pending
                ]
                merge_csv_rows(
                    RAW_OUT,
                    rows,
                    RESULT_KEYS,
                )
                completed.update(
                    (dataset, model, seed, split_seed, method)
                    for split_seed, method in pending
                )
    raw = pd.read_csv(RAW_OUT)
    summary = summarise(raw)
    summary.to_csv(SUMMARY_OUT, index=False)
    print(f"wrote {RAW_OUT}")
    print(f"wrote {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
