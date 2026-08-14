from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from exps.interventions.common import load_audited_model, load_test_dataset
from shortcut_groups.datasets import dataset_names
from shortcut_groups.experiment_utils import (
    artifact_prefix,
    completed_csv_keys,
    merge_csv_rows,
)
from shortcut_groups.interventions.common import square_side
from shortcut_groups.interventions.evaluation import (
    BATCH_SIZE,
    DTYPE,
    batch_from_dataset,
    metrics_from_predictions,
)
from shortcut_groups.io import load_shortcut_task_maps
from shortcut_groups.prototypes import CONTRIBUTION_SOURCES, contribution_sources


MODELS = ("vit", "resnet")
SEEDS = range(4)
# main experiment is for 256 regions and appendix for 3136, 3136 shows a little more distribution shift (It is based on the poor performance of random occlusions and we have not measured the shift explicitly)
REGION_COUNTS = (256, 3136)
OUT_CSV = Path("results/occlusion.csv")


def select_nonopposing_mask_regions(
    masking_contribution, opposite_contribution, mask_count
):
    regions = np.arange(len(masking_contribution))
    # Identify the opposite contribution map’s top 12.5%
    opposite_top = np.lexsort(
        (regions, -opposite_contribution)
    )[:mask_count]
    # Exclude these regions
    available = np.setdiff1d(regions, opposite_top)
    # Select the strongest masking contributions from the remaining regions
    return available[
        np.lexsort(
            (available, -masking_contribution[available])
        )[:mask_count]
    ]

# evaluation of masked images
@torch.no_grad()
def evaluate_masks(
    model,
    dataset,
    indices,
    occlusion_specs,
    regions,
    device,
    desc,
):
    indices = list(indices)
    side = square_side(regions)
    image_batch_size = max(1, BATCH_SIZE // len(occlusion_specs))
    predictions = []
    labels = []
    attributes = []

    for start in tqdm(
        range(0, len(indices), image_batch_size),
        desc=desc,
    ):
        batch_ids = indices[start:start + image_batch_size]
        x, y, attrs = batch_from_dataset(dataset, batch_ids, device)
        x = x.to(DTYPE)

        batch_selections = []
        for _metadata, selected_regions in occlusion_specs:
            selected_regions = (
                selected_regions
                if len(selected_regions) == 1
                else selected_regions[batch_ids]
            )
            selected_regions = torch.as_tensor(
                selected_regions,
                device=device,
            )
            # make the shape compatible
            if len(selected_regions) == 1:
                selected_regions = selected_regions.expand(
                    len(batch_ids),
                    -1,
                )
            batch_selections.append(selected_regions)

        region_mask = torch.zeros(
            (len(occlusion_specs), len(batch_ids), regions),
            device=device,
            dtype=torch.bool,
        )
        region_mask.scatter_(2, torch.stack(batch_selections), True)
        pixel_mask = torch.nn.functional.interpolate(
            region_mask.reshape(-1, 1, side, side).float(),
            size=x.shape[-2:],
            mode="nearest",
        ).bool()
        masked = x.unsqueeze(0).expand(
            len(occlusion_specs),
            *x.shape,
        ).reshape(
            -1, *x.shape[1:]
        ).masked_fill(pixel_mask, 0)

        pred = model(masked).argmax(dim=1)
        predictions.append(
            pred.reshape(len(occlusion_specs), len(batch_ids)).cpu()
        )
        labels.append(y.cpu())
        attributes.append(attrs.cpu())

    predictions = torch.cat(predictions, dim=1)
    labels = torch.cat(labels)
    attributes = torch.cat(attributes)
    return [
        metrics_from_predictions(mask_predictions, labels, attributes)
        for mask_predictions in predictions
    ]


def evaluate_occlusion_resolution(
    root,
    dataset_name,
    model_name,
    seed,
    regions,
    device,
    dataset,
    model,
    indices,
):
    alignment_outputs = load_shortcut_task_maps(
        root, dataset_name, model_name, "LRP", regions, seed
    )
    source_maps = contribution_sources(alignment_outputs)
    mask_count = regions // 8
    occlusion_rows = []
    occlusion_specs = []

    for source in CONTRIBUTION_SOURCES:
        shortcut, task = source_maps[source]
        shared = shortcut.ndim == 1
        if shared:
            shortcut, task = shortcut[None], task[None]
        # shortcut and task regions to select from
        shortcut_regions = [
            select_nonopposing_mask_regions(s, t, mask_count)
            for s, t in zip(shortcut, task)
        ]
        task_regions = [
            select_nonopposing_mask_regions(t, s, mask_count)
            for s, t in zip(shortcut, task)
        ]
        for strategy, selected_regions in (
            ("m_sc_top", shortcut_regions),
            ("m_task_top", task_regions),
        ):
            occlusion_specs.append((
                {"source": source, "strategy": strategy, "trial": 0},
                np.stack(selected_regions),
            ))

        for trial in range(5):
            rng = np.random.default_rng(trial)
            random_regions = [
                rng.choice(
                    np.setdiff1d(np.arange(regions), selected_regions),
                    mask_count,
                    replace=False,
                )
                for selected_regions in shortcut_regions
            ]
            occlusion_specs.append((
                {
                    "source": source,
                    "strategy": "random_outside_m_sc_top",
                    "trial": trial,
                },
                np.stack(random_regions),
            ))

    metrics_by_occlusion = evaluate_masks(
        model,
        dataset,
        indices,
        occlusion_specs,
        regions,
        device,
        f"{dataset_name} {model_name} {square_side(regions)}x{square_side(regions)}",
    )
    for (metadata, _selected_regions), occlusion_metrics in zip(
        occlusion_specs,
        metrics_by_occlusion,
    ):
        occlusion_rows.append({
            "dataset": dataset_name,
            "model": model_name,
            "seed": seed,
            "regions": regions,
            **metadata,
            **occlusion_metrics,
        })
    return occlusion_rows


def main():
    root = artifact_prefix()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    completed = completed_csv_keys(
        OUT_CSV,
        ["dataset", "model", "seed", "regions"],
    )
    for dataset_name in dataset_names():
        for model_name in MODELS:
            dataset = load_test_dataset(root, dataset_name)
            indices = np.arange(len(dataset))
            for seed in SEEDS:
                pending = [
                    regions
                    for regions in REGION_COUNTS
                    if (dataset_name, model_name, seed, regions) not in completed
                ]
                if not pending:
                    continue
                # intervention inference always on f_TS 
                model = load_audited_model(
                    root, dataset_name, model_name, seed, device
                )
                # only run experiments that are not already completed
                for regions in pending:
                    merge_csv_rows(
                        OUT_CSV,
                        evaluate_occlusion_resolution(
                            root,
                            dataset_name,
                            model_name,
                            seed,
                            regions,
                            device,
                            dataset,
                            model,
                            indices,
                        ),
                        [
                            "dataset",
                            "model",
                            "seed",
                            "regions",
                            "source",
                            "strategy",
                            "trial",
                        ],
                    )
                    completed.add((dataset_name, model_name, seed, regions))
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
