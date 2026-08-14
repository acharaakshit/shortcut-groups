import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shortcut_groups.methods import fit_nmf_mixture, row_simplex_normalize
from shortcut_groups.partial_correlations import build_shortcut_task_maps

DATASETS = (
    ("celeba_gender", "CelebA"),
    ("chexpert_pleuraleffusiongender", "CheXpert"),
    ("waterbirds", "Waterbirds"),
    ("camelyon17", "Camelyon17"),
    ("isic_source", "ISIC2019"),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("vit", "resnet"), default="vit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    fig, axes = plt.subplots(2 * len(DATASETS), 8, figsize=(12, 15))
    for dataset_index, (dataset, label) in enumerate(DATASETS):
        path = Path(__file__).parent / "artifacts" / f"{dataset}.npz"
        with np.load(path) as data:
            maps = build_shortcut_task_maps(
                data[f"{args.model}_ts_ranks"], data[f"{args.model}_ba_ranks"],
                data[f"{args.model}_sa_ranks"],
            )
        joint = np.concatenate(
            [maps["shortcut_evidence"], maps["task_evidence"]], axis=1
        )
        memberships, _, _ = fit_nmf_mixture(joint, 8)
        joint, zero = row_simplex_normalize(joint)
        memberships[zero] = 0
        weights = memberships / np.maximum(memberships.sum(axis=0), 1e-12)
        shortcut, task = np.split(weights.T @ joint, 2, axis=1)
        for group in range(8):
            axes[2 * dataset_index, group].imshow(
                shortcut[group].reshape(56, 56), cmap="Oranges",
                vmin=0, vmax=shortcut.max(),
            )
            axes[2 * dataset_index + 1, group].imshow(
                task[group].reshape(56, 56), cmap="Blues",
                vmin=0, vmax=task.max(),
            )
            if dataset_index == 0:
                axes[0, group].set_title(f"Group {group + 1}")
        axes[2 * dataset_index, 0].set_ylabel(f"{label}\nShortcut")
        axes[2 * dataset_index + 1, 0].set_ylabel(f"{label}\nTask")
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.tight_layout()
    if args.output:
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
