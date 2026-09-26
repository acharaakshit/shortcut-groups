from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from shortcut_groups.datasets import dataset_names
from shortcut_groups.experiment_utils import artifact_prefix
from shortcut_groups.io import load_shortcut_task_maps
from shortcut_groups.methods import fit_nmf_mixture
from shortcut_groups.prototypes import membership_weighted_group_prototypes

KS = (2, 4, 8, 16, 32)
REFERENCE_K = 8
OUT = Path("results/prototype_k_sensitivity")


def compare_prototypes(root, dataset, model, seed):
    # contribution maps
    maps = load_shortcut_task_maps(root, dataset, model, "LRP", 3136, seed)
    # representation for grouping
    joint = np.concatenate([maps["shortcut_evidence"], maps["task_evidence"]], axis=1)
    prototypes = {}
    for k in KS:
        # fit NMF
        memberships, _, _ = fit_nmf_mixture(joint, k)
        # shortcut group level per-image membership weighted contribution maps
        prototypes[k] = membership_weighted_group_prototypes(memberships, joint)
    prototype_rows = []
    for k in KS:
        # skip if same
        if k == REFERENCE_K:
            continue
        distances = cdist(prototypes[k], prototypes[REFERENCE_K], metric="cosine")
        forward, backward = distances.min(axis=1), distances.min(axis=0)
        comparison = dict(dataset=dataset, model=model, seed=seed, method="nmf", reference_k=REFERENCE_K, k=k)
        for direction, values, matches in (
            ("k_to_reference", forward, distances.argmin(axis=1)),
            ("reference_to_k", backward, distances.argmin(axis=0)),
        ):
            prototype_rows.extend(dict(**comparison, direction=direction, prototype=index,
                                       matched_prototype=int(match), distance=float(value), weight=.5 / len(values))
                                  for index, (value, match) in enumerate(zip(values, matches)))
    return prototype_rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    root = artifact_prefix()
    prototype_rows = []
    for dataset, model, seed in product(dataset_names(), ("resnet", "vit"), range(4)):
        prototype_rows.extend(compare_prototypes(root, dataset, model, seed))
    pd.DataFrame(prototype_rows).to_csv(OUT / "reference_prototype_distances.csv", index=False)


if __name__ == "__main__":
    main()
