import pickle
from pathlib import Path

import numpy as np

from .datasets import dataset_counts, dataset_test_count
from .partial_correlations import build_shortcut_task_maps

def load_model_attribution_triplet(
    artifact_root,
    dataset,
    model,
    attribution_method,
    regions,
    seed,
    *,
    pa_seed=None,
    ba_seed=None,
):
    """Load protected-attribute, baseline, and biased-model attributions."""
    # model training seeds of sensitive attr and baseline models are used for auxiliary model experiments
    pa_seed = seed if pa_seed is None else pa_seed # sensitive attr model seed
    ba_seed = seed if ba_seed is None else ba_seed # baseline model seed

    # location of attribution rank files
    results = Path(artifact_root) / dataset / "results"
    # attribution rank file suffix
    attribution_suffix = (
        f"_{attribution_method}__partition_grid_regions_{regions}_seed_"
    )
    # dataset train and val counts needed to load the biased model attribution rank files
    train_count, val_count = dataset_counts(dataset)

    # define paths and load the pickle files
    paths = (
        results / f"{dataset}_{model}_True_True{attribution_suffix}{pa_seed}.pkl",
        results / f"{dataset}_{model}_True_None{attribution_suffix}{ba_seed}.pkl",
        results
        / (
            f"{dataset}_{model}_None_None{attribution_suffix}{seed}_"
            f"{train_count}_{val_count}.pkl"
        ),
    )

    arrays = []
    for path in paths:
        with open(path, "rb") as handle:
            arrays.append(np.asarray(pickle.load(handle), dtype=float))
    pa, ba, bi = arrays

    if pa.shape != ba.shape or pa.shape != bi.shape:
        raise ValueError(
            "Attribution triplet shape mismatch: "
            f"PA={pa.shape}, BA={ba.shape}, BI={bi.shape}."
        )
    expected_shape = (dataset_test_count(dataset), regions)
    if pa.shape != expected_shape:
        raise ValueError(
            f"{dataset} must contain attribution arrays with shape "
            f"{expected_shape}; got {pa.shape}"
        )
    return pa, ba, bi

# contribuion maps built from attribution rank files
def load_shortcut_task_maps(
    artifact_root,
    dataset,
    model,
    attribution_method,
    regions,
    seed,
):
    pa, ba, bi = load_model_attribution_triplet(
        artifact_root,
        dataset,
        model,
        attribution_method,
        regions,
        seed,
    )
    return build_shortcut_task_maps(bi, ba, pa)


# latent files obtained from probing_2d.py in OSCAR, this is for computing errors, etc
def load_test_split(artifact_root, dataset, model, seed):
    train_count, _ = dataset_counts(dataset)
    path = (
        Path(artifact_root)
        / dataset
        / "latents"
        / f"{dataset}_{model}_None_{train_count}_seed_{seed}.npz"
    )
    with np.load(path, allow_pickle=True) as archive:
        stored = archive["test"].item()
    return {
        key: np.asarray(stored[key])
        for key in ("pred_logits", "targets", "attrs")
    }
