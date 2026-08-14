from pathlib import Path

from exps.risk.common import (
    ATTRIBUTION_METHODS,
    balanced_split,
    dataset_half_fit_size,
    kmeans_risk_scores,
    load_risk_inputs,
    nmf_risk_scores,
)
from shortcut_groups.datasets import DATASET_LABELS, dataset_names
from shortcut_groups.experiment_utils import (
    artifact_prefix,
    completed_csv_keys,
    merge_csv_rows,
    errors_found_at_coverage,
)

# main risk experiment

MODELS = ("vit", "resnet")
METHODS = ("kmeans_joint", "nmf_joint")
GRID_REGIONS = (64, 256, 1024, 3136)
# for the main experiment with four model seeds and all attribution methods for both resnet and vit and 5 heldout sets
OUT_CSV = Path("results/stability/heldout_seed_stability_k8_all_xai_56x56.csv")
# for various grid resolutions, vit models (four seeds) and all vit based attribution methods
GRID_OUT_CSV = Path("results/stability/vit_nmf_grid_partitions_heldout.csv")
RISK_RESULT_KEYS = [
    "dataset",
    "model",
    "attribution_method",
    "seed",
    "split_seed",
    "method",
]
GRID_RESULT_KEYS = [
    "dataset",
    "attribution_method",
    "regions",
    "seed",
    "split_seed",
]

def evaluate_heldout_split(
    features, test, errors, dataset, model, seed, split_seed, attribution, regions=3136, methods=METHODS
):
    # fitting and heldout sets
    fit, heldout = balanced_split(
        test["targets"],
        test["attrs"],
        dataset_half_fit_size(dataset),
        split_seed,
    )
    risk_rows = []
    for method in methods:
        if method == "nmf_joint":
            scores = nmf_risk_scores(
                features,
                errors,
                fit,
                heldout,
                8,
            )
            secondary_ranking = None
        else:
            scores, secondary_ranking = kmeans_risk_scores(
                features,
                errors,
                fit,
                heldout,
                8,
            )
        # store the relevant details
        capture = errors_found_at_coverage(
            errors[heldout],
            scores,
            0.2,
            secondary_ranking=secondary_ranking,
        )
        risk_rows.append({
            "dataset": dataset,
            "dataset_label": DATASET_LABELS[dataset],
            "model": model,
            "seed": seed,
            "split_seed": split_seed,
            "attribution_method": attribution,
            "partition": "grid",
            "regions": regions,
            "num_shortcut_groups": 8,
            "method": method,
            "fit_size": len(fit),
            "heldout_size": len(heldout),
            "coverage": 0.2, # we fix 20% here for these experiments, other coverage ablations are in other scripts
            "baseline_err": errors[heldout].mean(),
            "capture": capture,
        })
    return risk_rows

# mutliple grid rows for the vit model
def grid_capture_row(
    dataset,
    attribution,
    regions,
    seed,
    split_seed,
    risk_result,
):
    return dict(
        dataset=dataset, model="vit", attribution_method=attribution, regions=regions,
        seed=seed, split_seed=split_seed, num_shortcut_groups=8,
        capture=risk_result["capture"],
    )


def main():
    root = artifact_prefix()
    # check if the results already exist first, recomputation is costly but remove this to run anyways
    completed = completed_csv_keys(
        OUT_CSV,
        RISK_RESULT_KEYS,
    )
    completed_grid = completed_csv_keys(
        GRID_OUT_CSV,
        GRID_RESULT_KEYS,
    )
    for model in MODELS:
        for attribution in ATTRIBUTION_METHODS[model]:
            for seed in range(4):
                for dataset in dataset_names():
                    pending = [
                        split_seed
                        for split_seed in range(5)
                        if any(
                            (
                                dataset,
                                model,
                                attribution,
                                seed,
                                split_seed,
                                method,
                            )
                            not in completed
                            for method in METHODS
                        )
                        or (
                            model == "vit"
                            and (
                                dataset,
                                attribution,
                                3136,
                                seed,
                                split_seed,
                            )
                            not in completed_grid
                        )
                    ]
                    # if nothing is pending, skip everything
                    if not pending:
                        continue
                    # load the test set, errors
                    features, test, errors = load_risk_inputs(
                        root, dataset, model, seed, attribution
                    )
                    for split_seed in pending:
                        missing_methods = tuple(
                            method
                            for method in METHODS
                            if (
                                dataset,
                                model,
                                attribution,
                                seed,
                                split_seed,
                                method,
                            )
                            not in completed
                        )
                        grid_key = (dataset, attribution, 3136, seed, split_seed)
                        needs_grid = (
                            model == "vit" and grid_key not in completed_grid
                        )
                        methods_to_evaluate = tuple(
                            method
                            for method in METHODS
                            if method in missing_methods
                            or (needs_grid and method == "nmf_joint")
                        )
                        print(f"{dataset} {model} {attribution} seed={seed} split={split_seed}", flush=True)
                        risk_rows = evaluate_heldout_split(
                            features,
                            test,
                            errors,
                            dataset,
                            model,
                            seed,
                            split_seed,
                            attribution,
                            methods=methods_to_evaluate,
                        )
                        if missing_methods:
                            new_risk_rows = [
                                row
                                for row in risk_rows
                                if row["method"] in missing_methods
                            ]
                            merge_csv_rows(
                                OUT_CSV,
                                new_risk_rows,
                                RISK_RESULT_KEYS,
                            )
                            completed.update(
                                (
                                    dataset,
                                    model,
                                    attribution,
                                    seed,
                                    split_seed,
                                    method,
                                )
                                for method in missing_methods
                            )
                        if needs_grid:
                            nmf = next(
                                row
                                for row in risk_rows
                                if row["method"] == "nmf_joint"
                            )
                            merge_csv_rows(
                                GRID_OUT_CSV,
                                [
                                    grid_capture_row(
                                        dataset,
                                        attribution,
                                        3136,
                                        seed,
                                        split_seed,
                                        nmf,
                                    )
                                ],
                                GRID_RESULT_KEYS,
                            )
                            completed_grid.add(grid_key)

    for attribution in ATTRIBUTION_METHODS["vit"]:
        for regions in GRID_REGIONS[:-1]:
            for seed in range(4):
                for dataset in dataset_names():
                    pending = [
                        split_seed
                        for split_seed in range(5)
                        if (
                            dataset,
                            attribution,
                            regions,
                            seed,
                            split_seed,
                        )
                        not in completed_grid
                    ]
                    if not pending:
                        continue
                    features, test, errors = load_risk_inputs(
                        root, dataset, "vit", seed, attribution, regions
                    )
                    for split_seed in pending:
                        grid_key = (
                            dataset,
                            attribution,
                            regions,
                            seed,
                            split_seed,
                        )
                        if grid_key in completed_grid:
                            continue
                        print(f"{dataset} vit {attribution} regions={regions} seed={seed} "
                              f"split={split_seed}", flush=True)
                        risk_rows = evaluate_heldout_split(
                            features, test, errors, dataset, "vit", seed, split_seed, attribution,
                            regions, (METHODS[1],),
                        )
                        merge_csv_rows(
                            GRID_OUT_CSV,
                            [
                                grid_capture_row(
                                    dataset,
                                    attribution,
                                    regions,
                                    seed,
                                    split_seed,
                                    risk_rows[0],
                                )
                            ],
                            GRID_RESULT_KEYS,
                        )
                        completed_grid.add(grid_key)
    print(f"wrote {OUT_CSV}", flush=True)
    print(f"wrote {GRID_OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
