import argparse
from pathlib import Path

import numpy as np

from exps.risk.common import balanced_split, nmf_risk_scores
from shortcut_groups.experiment_utils import errors_found_at_coverage
from shortcut_groups.partial_correlations import build_shortcut_task_maps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--model", choices=("vit", "resnet"), default="vit")
    args = parser.parse_args()

    path = Path(__file__).parent / "artifacts" / f"{args.dataset}.npz"
    with np.load(path) as data:
        maps = build_shortcut_task_maps(
            data[f"{args.model}_ts_ranks"],
            data[f"{args.model}_ba_ranks"],
            data[f"{args.model}_sa_ranks"],
        )
        features = np.concatenate(
            [maps["shortcut_evidence"], maps["task_evidence"]], axis=1
        )
        targets = data["targets"]
        attributes = data["attributes"]
        errors = data[f"{args.model}_predictions"] != targets

    captures = []
    for seed in range(5):
        fit, heldout = balanced_split(
            targets, attributes, len(targets) // 2, seed,
        )
        scores = nmf_risk_scores(features, errors, fit, heldout, 8)
        capture = errors_found_at_coverage(
            errors[heldout], scores, 0.2,
        )
        captures.append(capture)
        print(
            f"split {seed}: errors found={100 * capture:.1f}%"
        )
    print(f"mean: {100 * np.mean(captures):.1f}%")


if __name__ == "__main__":
    main()
