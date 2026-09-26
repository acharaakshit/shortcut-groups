import argparse

import numpy as np
import torch

from datasets2d import get_biased_multiattribute_celeba_splits
from exps.interventions.common import evaluate_intervention, load_audited_model
from exps.multi_attribute.celeba_shortcut_groups import fit_groups, load_rank_arrays

from shortcut_groups.experiment_utils import artifact_prefix
from shortcut_groups.interventions.common import (
    COMBINED_INTERVENTION,
    SHORTCUT_SUPPRESSION,
    TASK_AMPLIFICATION,
    TASK_SUPPRESSION,
)
from shortcut_groups.interventions.evaluation import predict_batches



BASELINE = "No intervention"
INTERVENTIONS = (
    SHORTCUT_SUPPRESSION,
    TASK_AMPLIFICATION,
    COMBINED_INTERVENTION,
    TASK_SUPPRESSION,
)


def metrics(predictions, targets, attributes):
    correct = predictions == targets
    row = {"accuracy": float(correct.mean())}
    intersection_accuracies = []
    for target in (0, 1):
        for gender in (0, 1):
            for smiling in (0, 1):
                selected = (
                    (targets == target)
                    & (attributes[:, 0] == gender)
                    & (attributes[:, 1] == smiling)
                )
                accuracy = float(correct[selected].mean())
                intersection_accuracies.append(accuracy)
    # largest accuracy gap across the eight subgroups
    row["lpg"] = max(intersection_accuracies) - min(intersection_accuracies)

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    prefix = artifact_prefix()

    device = torch.device(args.device)
    checkpoint = (prefix / "celeba_gender_multiattr/saved_data/models"
                  / "MODEL_vit_3_SEED2D_0_MULTIATTR_ts_F1.ckpt")
    model = load_audited_model(prefix, "celeba_gender_multiattr", "vit", 0,
                               device, checkpoint=checkpoint)
    dataset = get_biased_multiattribute_celeba_splits(
        root=str(prefix), balanced=False, attr_labs=False,
    )[2]
    dataset.attributes = dataset.attributes.tolist()
    indices = list(range(len(dataset)))
    # fit groups from the saved attribution ranks
    arrays = load_rank_arrays(prefix / "celeba_gender_multiattr/results")
    groups = fit_groups(arrays, 8)
    memberships = groups["memberships"].astype(np.float32)
    shortcut = memberships @ groups["combined_shortcut_prototypes"].astype(np.float32)
    task = memberships @ groups["task_prototypes"].astype(np.float32)

    baseline_predictions, targets, attributes = predict_batches(
        model,
        dataset,
        indices,
        device,
        desc=BASELINE,
    )
    rows = [{"intervention": BASELINE, **metrics(
        baseline_predictions, targets, attributes
    )}]
    for intervention in INTERVENTIONS:
        predictions, _, _ = (
            evaluate_intervention(
                model,
                dataset,
                indices,
                shortcut, # suppress
                task, # amplify
                intervention,
                "vit",
                device,
                intervention,
                seed=0,
                vit_layer_mode="all",
                vit_targets=("value",),
                return_predictions=True,
            )
        )
        rows.append({"intervention": intervention, **metrics(
            predictions, targets, attributes
        )})

    for row in rows[1:]:
        print(
            f"{row['intervention']}: "
            f"delta acc={100 * (row['accuracy'] - rows[0]['accuracy']):+.1f}, "
            f"delta LPG_red={100 * (rows[0]['lpg'] - row['lpg']):+.1f}"
        )


if __name__ == "__main__":
    main()
