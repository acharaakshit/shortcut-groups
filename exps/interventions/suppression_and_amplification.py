import argparse
from pathlib import Path

import torch

from exps.interventions.common import (
    evaluate_intervention,
    load_audited_model,
    load_intervention_maps,
    load_test_dataset,
)
from shortcut_groups.datasets import dataset_names
from shortcut_groups.experiment_utils import (
    artifact_prefix,
    completed_csv_keys,
    merge_csv_rows,
)
# suppression and amplification approaches
from shortcut_groups.interventions.common import (
    COMBINED_INTERVENTION,
    SHUFFLED_SUPPRESSION,
    SHORTCUT_SUPPRESSION,
    TASK_AMPLIFICATION,
    TASK_SUPPRESSION,
)
from shortcut_groups.interventions.evaluation import load_baseline_metrics
from shortcut_groups.prototypes import (
    CONTRIBUTION_SOURCES,
    IMAGE_SOURCE,
    contribution_sources,
)


MODELS = ("resnet", "vit")
INTERVENTIONS = (
    SHORTCUT_SUPPRESSION,
    SHUFFLED_SUPPRESSION,
    TASK_SUPPRESSION,
    TASK_AMPLIFICATION,
    COMBINED_INTERVENTION,
)
OUT_CSV = Path("results/fixed_interventions/intervention_seeds.csv")
SHUFFLE_SEEDS = range(5)
NO_SHUFFLE_SEED = -1
RESULT_KEY_COLUMNS = [
    "dataset",
    "model",
    "seed",
    "source",
    "intervention",
    "shuffle_seed",
]


def intervention_conditions(selected_sources, selected_interventions):
    """Return source, intervention, and shuffled-trial."""
    conditions = []
    for intervention in selected_interventions:
        if intervention == SHUFFLED_SUPPRESSION:
            if IMAGE_SOURCE in selected_sources:
                conditions.extend(
                    (IMAGE_SOURCE, intervention, shuffle_seed)
                    for shuffle_seed in SHUFFLE_SEEDS
                )
        elif intervention == TASK_SUPPRESSION:
            if IMAGE_SOURCE in selected_sources:
                conditions.append(
                    (IMAGE_SOURCE, intervention, NO_SHUFFLE_SEED)
                )
        else:
            conditions.extend(
                (source, intervention, NO_SHUFFLE_SEED)
                for source in selected_sources
            )
    return conditions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Different internal interventions (suppression of task, shortcut, shuffled shortcut suppression as ablation, combined approach as main, task amplification as well)."
    )
    parser.add_argument("--dataset", action="append", choices=dataset_names())
    parser.add_argument("--model", action="append", choices=MODELS)
    parser.add_argument("--seed", action="append", type=int, choices=range(4))
    parser.add_argument(
        "--source",
        action="append",
        choices=CONTRIBUTION_SOURCES,
    )
    parser.add_argument(
        "--intervention",
        action="append",
        choices=INTERVENTIONS,
    )
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    return parser.parse_args()


def main():
    args = parse_args()
    root = artifact_prefix()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_datasets = args.dataset or dataset_names()
    selected_models = args.model or MODELS
    selected_seeds = args.seed or range(4)
    selected_sources = args.source or CONTRIBUTION_SOURCES
    selected_interventions = (
        INTERVENTIONS
        if args.intervention is None
        else [
            intervention
            for intervention in INTERVENTIONS
            if intervention in args.intervention
        ]
    )
    conditions = intervention_conditions(
        selected_sources,
        selected_interventions,
    )
    completed = completed_csv_keys(
        args.out_csv,
        RESULT_KEY_COLUMNS,
    )
    for dataset_name in selected_datasets:
        dataset = load_test_dataset(root, dataset_name)
        indices = range(len(dataset))
        for model_name in selected_models:
            for seed in selected_seeds:
                expected = {
                    (
                        dataset_name,
                        model_name,
                        seed,
                        source,
                        intervention,
                        shuffle_seed,
                    )
                    for source, intervention, shuffle_seed in conditions
                }
                if expected <= completed:
                    continue
                result_rows = []
                model = load_audited_model(
                    root, dataset_name, model_name, seed, device
                )
                alignment_outputs = load_intervention_maps(
                    root, dataset_name, model_name, seed
                )
                source_maps = contribution_sources(alignment_outputs)
                baseline = load_baseline_metrics(
                    root,
                    dataset_name,
                    model_name,
                    seed,
                )
                for source, intervention, shuffle_seed in conditions:
                    shortcut, task = source_maps[source]
                    result_key = (
                        dataset_name,
                        model_name,
                        seed,
                        source,
                        intervention,
                        shuffle_seed,
                    )
                    if result_key in completed:
                        continue
                    evaluation_seed = (
                        shuffle_seed
                        if intervention == SHUFFLED_SUPPRESSION
                        else seed
                    )
                    description = (
                        f"{dataset_name} {model_name} seed{seed} {source} "
                        f"{intervention}"
                    )
                    if intervention == SHUFFLED_SUPPRESSION:
                        description += f" shuffle{shuffle_seed}"
                    metrics = evaluate_intervention(
                        model,
                        dataset,
                        indices,
                        shortcut,
                        task,
                        intervention,
                        model_name,
                        device,
                        description,
                        seed=evaluation_seed,
                    )
                    result_rows.append({
                        "dataset": dataset_name,
                        "model": model_name,
                        "seed": seed,
                        "source": source,
                        "intervention": intervention,
                        "shuffle_seed": shuffle_seed,
                        **metrics,
                        "delta_accuracy": metrics["accuracy"] - baseline["accuracy"],
                    })
                merge_csv_rows(
                    args.out_csv,
                    result_rows,
                    RESULT_KEY_COLUMNS,
                )
                completed.update(
                    (
                        dataset_name,
                        model_name,
                        seed,
                        row["source"],
                        row["intervention"],
                        row["shuffle_seed"],
                    )
                    for row in result_rows
                )
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
