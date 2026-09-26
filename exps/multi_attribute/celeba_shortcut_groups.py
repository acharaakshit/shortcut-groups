from itertools import product
from pathlib import Path
import pickle

import numpy as np
import torch
from datasets2d import get_biased_multiattribute_celeba_splits
from exps.interventions.common import load_audited_model

from exps.risk.common import nmf_risk_scores
from shortcut_groups.experiment_utils import artifact_prefix, errors_found_at_coverage
from shortcut_groups.methods import fit_nmf_mixture
from shortcut_groups.partial_correlations import residualise
from shortcut_groups.interventions.evaluation import predict_batches
from shortcut_groups.prototypes import membership_weighted_group_prototypes

NUM_GROUPS = 8
RISK_SPLITS = 5
RISK_COVERAGE = .2
OUT = Path("results/figures/multi_attribute")

INTERSECTION_CELLS = tuple(product((0, 1), repeat=3))


def load_rank_arrays(results_dir):
    # saved attribution ranks
    roles = {"ts": "ts", "ba": "ba", "sa_male": "sa_multilabel_male", "sa_smiling": "sa_multilabel_smiling"}
    arrays = {}
    for role, filename_role in roles.items():
        path = results_dir / (f"celeba_gender_multiattr_vit_MULTIATTR_{filename_role}_LRP__"
                              "partition_grid_regions_3136_seed_0.pkl")
        with path.open("rb") as handle:
            arrays[role] = np.asarray(pickle.load(handle), dtype=float)
    return arrays


def correlation_contributions(z1, z2):
    z1 = (z1 - z1.mean()) / (z1.std(ddof=1) + 1e-12)
    z2 = (z2 - z2.mean()) / (z2.std(ddof=1) + 1e-12)
    return z1 * z2


def partial_contribution_maps(first, second, *controls):
    maps = np.empty_like(first)
    for i in range(len(first)):
        # residualise with respect to all controls
        control = np.column_stack([values[i] for values in controls])
        maps[i] = correlation_contributions(*residualise(first[i], second[i], control))
    return np.maximum(maps, 0)

# contributions for partial multiple correlation
def combined_shortcut_maps(ts, ba, gender_sa, smiling_sa):
    maps = np.empty_like(ts)
    for i in range(len(ts)):
        residual, sa = residualise(ts[i], np.column_stack([gender_sa[i], smiling_sa[i]]), ba[i])
        # jointly predict task residuals from both sensitive attribute residuals
        coefficients = np.linalg.lstsq(sa, residual, rcond=None)[0]
        maps[i] = correlation_contributions(residual, sa @ coefficients)
    return np.maximum(maps, 0)


def fit_groups(arrays, num_groups):
    ts, ba, male, smiling = (arrays[key] for key in ("ts", "ba", "sa_male", "sa_smiling"))
    # contribution maps
    gender = partial_contribution_maps(ts, male, ba, smiling)
    smile = partial_contribution_maps(ts, smiling, ba, male)
    combined = combined_shortcut_maps(ts, ba, male, smiling)
    task = partial_contribution_maps(ts, ba, male, smiling)
    # representation for grouping
    features = np.concatenate([combined, task], axis=1)
    # fit NMF
    memberships, _, _ = fit_nmf_mixture(features, num_groups)
    # group level membership weighted contribution maps
    combined_prototypes, task_prototypes = np.split(membership_weighted_group_prototypes(memberships, features), 2, axis=1)
    # order groups by membership mass
    order = np.argsort(memberships.sum(axis=0))[::-1]
    return dict(features=features, memberships=memberships[:, order],
                gender_shortcut_prototypes=membership_weighted_group_prototypes(memberships, gender)[order],
                smiling_shortcut_prototypes=membership_weighted_group_prototypes(memberships, smile)[order],
                combined_shortcut_prototypes=combined_prototypes[order], task_prototypes=task_prototypes[order])


def save_task_predictions(prefix, output_path):
    dataset = get_biased_multiattribute_celeba_splits(
        root=str(prefix), balanced=False, attr_labs=False
    )[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = prefix / "celeba_gender_multiattr/saved_data/models/MODEL_vit_3_SEED2D_0_MULTIATTR_ts_F1.ckpt"
    model = load_audited_model(prefix, "celeba_gender_multiattr", "vit", 0,
                               device, checkpoint=checkpoint)
    # task predictions on the test set
    dataset.attributes = dataset.attributes.tolist()
    predictions, targets, _ = predict_batches(model, dataset, range(len(dataset)), device)
    np.savez_compressed(output_path, predictions=predictions, targets=targets,
                        male_attributes=np.asarray(dataset.male_attributes),
                        smiling_attributes=np.asarray(dataset.smiling_attributes), dataset_indices=np.asarray(dataset.indices))


def load_test_results(path, expected_count):
    with np.load(path) as data:
        values = [data[key].reshape(-1) for key in ("targets", "male_attributes", "smiling_attributes")]
        predictions = data["predictions"] if "predictions" in data else data["pred_logits"].argmax(axis=1)
    values.append(predictions)
    if any(len(value) != expected_count for value in values):
        raise ValueError(f"Test-result length mismatch; expected {expected_count}")
    return tuple(values)


def intersection_balanced_split(targets, male, smiling, seed):
    rng = np.random.default_rng(seed)
    fit = []
    # sample half of each intersection for fitting
    for target, gender, smile in INTERSECTION_CELLS:
        cell = np.flatnonzero((targets == target) & (male == gender) & (smiling == smile))
        fit.append(rng.choice(cell, len(cell) // 2, replace=False))
    fit = np.sort(np.concatenate(fit))
    heldout = np.ones(len(targets), dtype=bool)
    heldout[fit] = False
    return fit, np.flatnonzero(heldout)


def heldout_error_capture(groups, targets, male, smiling, errors):
    captures = []
    for seed in range(RISK_SPLITS):
        fit, heldout = intersection_balanced_split(targets, male, smiling, seed)
        # refit NMF and estimate group risk using only the fitting half
        scores = nmf_risk_scores(groups["features"], errors, fit, heldout,
                                 groups["memberships"].shape[1])
        # errors found in the highest-risk heldout images
        capture = errors_found_at_coverage(errors[heldout], scores, RISK_COVERAGE)
        captures.append(capture)
    return np.mean(captures)


def main():
    prefix = artifact_prefix()
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = load_rank_arrays(prefix / "celeba_gender_multiattr/results")
    groups = fit_groups(arrays, NUM_GROUPS)
    stem = OUT / f"celeba_vit_lrp_multiattribute_multilabel_k{NUM_GROUPS}"
    saved = {key: value for key, value in groups.items() if key != "features"}
    test_path = OUT / "celeba_vit_multiattribute_test_results_seed_0.npz"
    # reuse saved predictions
    if not test_path.exists():
        save_task_predictions(prefix, test_path)
    targets, male, smiling, predictions = load_test_results(test_path, len(groups["memberships"]))
    errors = predictions != targets
    capture = heldout_error_capture(groups, targets, male, smiling, errors)
    print(f"held-out error capture at {100*RISK_COVERAGE:.0f}% coverage: {100*capture:.2f}%")
    # save group maps for plotting
    path = stem.with_name(stem.name + "_groups.npz")
    np.savez_compressed(path, **saved)


if __name__ == "__main__":
    main()
