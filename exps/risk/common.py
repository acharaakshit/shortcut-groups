import numpy as np

from shortcut_groups.datasets import dataset_test_count
from shortcut_groups.io import load_shortcut_task_maps, load_test_split
from shortcut_groups.methods import (
    fit_kmeans,
    fit_nmf_mixture,
    one_hot,
    row_l2_normalize,
    transform_nmf_mixture,
)


ATTRIBUTION_METHODS = {
    "vit": ("LRP", "CX", "TiS"),
    "resnet": ("LRP", "GradCAM"),
}

# fitting and heldout sets are both equal halves of the entire test set
def dataset_half_fit_size(dataset):
    return dataset_test_count(dataset) // 2

# this is to generate a balanced fitting and heldout set for the risk experiments
def balanced_split(targets, attrs, fit_size, seed):
    rng = np.random.default_rng(seed)
    fit = []
    # we need balanced subgroups for both fitting and heldout sets
    for target in (0, 1):
        for attr in (0, 1):
            group = np.flatnonzero((targets == target) & (attrs == attr))
            fit.append(rng.choice(group, size=fit_size // 4, replace=False))
    fit_idx = np.sort(np.concatenate(fit))
    heldout = np.ones(len(targets), dtype=bool)
    heldout[fit_idx] = False
    return fit_idx, np.flatnonzero(heldout)

# shortcut group error rates
def shortcut_group_error_rates(memberships, errors):
    weights = memberships.sum(axis=0)
    rates = memberships.T @ errors
    return np.divide(rates, weights, out=np.zeros_like(rates), where=weights > 1e-12)

# from our saved npz files consisting of model logits, labels and attributes, read the errors
def load_test_results(artifact_root, dataset, model, seed):
    test = load_test_split(artifact_root, dataset, model, seed)
    logits = test["pred_logits"]
    errors = logits.argmax(axis=1) != test["targets"]
    return test, errors

# load contribution maps, preds
def load_risk_inputs(
    artifact_root,
    dataset,
    model,
    seed,
    attribution_method="LRP",
    regions=3136,
):
    alignment_outputs = load_shortcut_task_maps(
        artifact_root,
        dataset,
        model,
        attribution_method,
        regions,
        seed,
    )
    test, errors = load_test_results(artifact_root, dataset, model, seed)
    n = len(test["targets"])
    if len(alignment_outputs["shortcut_evidence"]) != n:
        raise ValueError(
            f"{dataset} {model} seed={seed} map/test row mismatch"
        )
    features = np.concatenate([
        alignment_outputs["shortcut_evidence"],
        alignment_outputs["task_evidence"],
    ], axis=1)
    return features, test, errors

# compute the risk for the heldout set
def nmf_risk_scores(
    features,
    errors,
    fit_idx,
    heldout_idx,
    num_shortcut_groups,
):
    # fit nmf on the fitting set
    fit_memberships, _, model = fit_nmf_mixture(
        features[fit_idx],
        num_shortcut_groups,
    )
    # apply nmf transform to the heldout set
    heldout_memberships = transform_nmf_mixture(
        model,
        features[heldout_idx],
    )
    # compute risk on the heldout set based on the error rates from the fitting set
    group_error_rates = shortcut_group_error_rates(
        fit_memberships,
        errors[fit_idx],
    )
    return heldout_memberships @ group_error_rates

# risk scores for k-means
def kmeans_risk_scores(
    features,
    errors,
    fit_idx,
    heldout_idx,
    num_shortcut_groups,
):
    # get non-zero, zero rows
    normalized_features, zero_rows = row_l2_normalize(features)
    # fit
    fit_informative = ~zero_rows[fit_idx]
    # heldout
    heldout_informative = ~zero_rows[heldout_idx]
    fit_memberships = np.zeros((len(fit_idx), num_shortcut_groups), dtype=float)
    heldout_memberships = np.zeros((len(heldout_idx), num_shortcut_groups), dtype=float)
    if not fit_informative.any() or not heldout_informative.any():
        raise ValueError("K-means risk scoring requires informative fit and held-out rows")
    # again, fit and transform to heldout
    labels, model = fit_kmeans(
        normalized_features[fit_idx][fit_informative],
        num_shortcut_groups,
    )
    # covert to one-hot but one image only belongs to one group
    fit_memberships[fit_informative] = one_hot(labels, num_shortcut_groups)
    # assign shortcut groups to the held-out images
    heldout_features = normalized_features[heldout_idx][heldout_informative]
    heldout_labels = np.asarray(model.predict(heldout_features)).reshape(-1)
    heldout_memberships[heldout_informative] = one_hot(
        heldout_labels,
        num_shortcut_groups,
    )
    # error rate computation for shortcut groups (basically mean)
    group_error_rates = shortcut_group_error_rates(
        fit_memberships,
        errors[fit_idx],
    )
    # risk score (one-hot memberships in this case)
    scores = heldout_memberships @ group_error_rates
    # some images might have to be selected from the same shortcut group so secondary risk score based on centroid proximity is used
    centroid_proximity = np.full(len(heldout_idx), -np.inf)
    assigned_centroids = np.asarray(model.cluster_centers_)[heldout_labels]
    centroid_proximity[heldout_informative] = -np.linalg.norm(
        heldout_features - assigned_centroids,
        axis=1,
    )
    return scores, centroid_proximity
