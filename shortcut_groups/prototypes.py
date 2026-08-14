import numpy as np
# this loads the attribution stats files from OSCAR which are ranked per image grid vectors
from .io import load_shortcut_task_maps
from .methods import (
    fit_kmeans,
    fit_nmf_mixture,
    # row level l2 normalisation where each row contains a shortcut and task map
    row_l2_normalize,
    # row level normalisation only for non-zero rows
    row_simplex_normalize,
)

# our three intervention sources
DATASET_SOURCE = "Dataset"
SHORTCUT_GROUP_SOURCE = "Shortcut Group"
IMAGE_SOURCE = "Image"
CONTRIBUTION_SOURCES = (DATASET_SOURCE, SHORTCUT_GROUP_SOURCE, IMAGE_SOURCE)


def membership_weighted_group_prototypes(memberships, image_maps):
    """Average image maps for each group, weighted by soft memberships."""
    # we normalise the non-zero maps
    image_maps, zero_maps = row_simplex_normalize(image_maps)
    memberships = memberships.copy()
    # assign zero membership to the zero maps indices in the membership matrix
    memberships[zero_maps] = 0.0
    # Total soft-membership mass assigned to each shortcut group.
    group_membership_mass = memberships.sum(axis=0)
    # normalisation of group memberships
    normalized_membership_weights = np.zeros_like(memberships)
    nonempty_groups = group_membership_mass > 1e-12
    normalized_membership_weights[:, nonempty_groups] = (
        memberships[:, nonempty_groups]
        / group_membership_mass[nonempty_groups]
    )
    # finally we just assign weight to every image level contribution map based on it's group membership
    return normalized_membership_weights.T @ image_maps


def build_nmf_shortcut_groups(
    image_shortcut_maps,
    image_task_maps,
    num_shortcut_groups=8,
):
    """Build soft groups, their average maps, and per-image group maps."""
    # form the joint representation first
    joint_image_maps = np.concatenate(
        [image_shortcut_maps, image_task_maps],
        axis=1,
    )
    # obtain memberships from NMF
    memberships, _, _ = fit_nmf_mixture(
        joint_image_maps,
        num_shortcut_groups,
    )
    # use the memberships to obtain group level prototypes from image level maps (note that membership matrix should have K cols where K is no of scut groups)
    group_prototypes = membership_weighted_group_prototypes(
        memberships,
        joint_image_maps,
    )
    # image group maps are just like 
    image_group_maps = memberships @ group_prototypes
    num_regions = image_shortcut_maps.shape[1]
    return {
        "memberships": memberships,
        "shortcut_group_prototypes": group_prototypes[:, :num_regions],
        "task_group_prototypes": group_prototypes[:, num_regions:],
        "image_shortcut_group_maps": image_group_maps[:, :num_regions],
        "image_task_group_maps": image_group_maps[:, num_regions:],
    }


def contribution_sources(
    maps,
    selected=CONTRIBUTION_SOURCES,
):
    sources = {}
    if IMAGE_SOURCE in selected:
        sources[IMAGE_SOURCE] = (
            maps["shortcut_evidence"],
            maps["task_evidence"],
        )
    if DATASET_SOURCE in selected:
        sources[DATASET_SOURCE] = (
            maps["dataset_shortcut_evidence"],
            maps["dataset_task_evidence"],
        )
    if SHORTCUT_GROUP_SOURCE in selected:
        shortcut_groups = build_nmf_shortcut_groups(
            maps["shortcut_evidence"],
            maps["task_evidence"],
            8,
        )
        sources[SHORTCUT_GROUP_SOURCE] = (
            shortcut_groups["image_shortcut_group_maps"],
            shortcut_groups["image_task_group_maps"],
        )
    return sources


def hard_cluster_average_prototypes(labels, image_maps, num_shortcut_groups):
    row_mass = image_maps.sum(axis=1)
    informative = row_mass > 1e-12
    normalized_image_maps = row_simplex_normalize(image_maps)[0]
    group_prototypes = np.zeros(
        (num_shortcut_groups, image_maps.shape[1]),
        dtype=float,
    )
    for group in range(num_shortcut_groups):
        selected = (labels == group) & informative
        if np.any(selected):
            group_prototypes[group] = normalized_image_maps[selected].mean(axis=0)
    return group_prototypes


def compute_nmf_group_prototypes(
    artifact_root,
    dataset,
    model,
    attribution_method="LRP",
    num_shortcut_groups=8,
):
    maps = load_shortcut_task_maps(
        artifact_root,
        dataset,
        model,
        attribution_method,
        3136,
        0,
    )
    shortcut_groups = build_nmf_shortcut_groups(
        maps["shortcut_evidence"],
        maps["task_evidence"],
        num_shortcut_groups,
    )
    return {
        "memberships": shortcut_groups["memberships"],
        "shortcut_group_prototypes": shortcut_groups["shortcut_group_prototypes"],
        "task_group_prototypes": shortcut_groups["task_group_prototypes"],
    }


def compute_kmeans_group_prototypes(
    artifact_root,
    dataset,
    model,
    attribution_method="LRP",
    num_shortcut_groups=8,
):
    maps = load_shortcut_task_maps(
        artifact_root,
        dataset,
        model,
        attribution_method,
        3136,
        0,
    )
    joint_image_maps = np.concatenate(
        [maps["shortcut_evidence"], maps["task_evidence"]],
        axis=1,
    )
    # L2 normalisation for K-means, L1 for NMF earlier
    normalized_joint_image_maps, zero_rows = row_l2_normalize(joint_image_maps)
    informative = ~zero_rows
    labels = np.full(len(joint_image_maps), -1, dtype=int)
    labels[informative] = fit_kmeans(
        normalized_joint_image_maps[informative],
        num_shortcut_groups,
    )[0]
    num_regions = maps["shortcut_evidence"].shape[1]
    group_prototypes = hard_cluster_average_prototypes(
        labels,
        joint_image_maps,
        num_shortcut_groups,
    )
    return {
        "shortcut_group_prototypes": group_prototypes[:, :num_regions],
        "task_group_prototypes": group_prototypes[:, num_regions:],
    }
