import argparse
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from explain_2d.partition import square_atlas_grid
from scipy.ndimage import mean as labelled_mean
from scipy.stats import rankdata
from tqdm import tqdm
from zennit.attribution import Gradient
from zennit.composites import EpsilonPlusFlat
from zennit.torchvision import ResNetCanonizer

from exps.interventions.common import (
    default_intervention_target,
    load_audited_model,
    load_test_dataset,
)
from shortcut_groups.datasets import dataset_names
from shortcut_groups.experiment_utils import (
    artifact_prefix,
    completed_csv_keys,
    merge_csv_rows,
)
from shortcut_groups.interventions.common import (
    COMBINED_INTERVENTION,
    SHORTCUT_SUPPRESSION,
    TASK_AMPLIFICATION,
)
from shortcut_groups.interventions.evaluation import (
    batch_from_dataset,
    evaluate_batches,
    load_baseline_metrics,
)
from shortcut_groups.interventions.resnet import (
    RESNET_TARGET_STAGES,
    RESNET_STAGES,
    resnet_intervention_scales,
)
from shortcut_groups.interventions.vit import (
    attention_lrp_patched,
    patch_vit,
    vit_intervention_token_scale,
)
from shortcut_groups.io import load_model_attribution_triplet
from shortcut_groups.partial_correlations import build_shortcut_task_maps
from shortcut_groups.prototypes import (
    CONTRIBUTION_SOURCES,
    IMAGE_SOURCE,
    SHORTCUT_GROUP_SOURCE,
    contribution_sources,
    build_nmf_shortcut_groups,
)

# both architectures in our exps
MODELS = ("vit", "resnet")
# we have four independently trained models
SEEDS = range(4)
COMBINED_THEN_INTERVENTION = (
    "Combined first, then shortcut suppression"
)
# number of iterations in our exps are always fixed to 5
ITERATIONS = 5
INTERVENTIONS = (
    SHORTCUT_SUPPRESSION,
    TASK_AMPLIFICATION,
    COMBINED_INTERVENTION,
    COMBINED_THEN_INTERVENTION,
)
DEFAULT_SOURCES = (SHORTCUT_GROUP_SOURCE, IMAGE_SOURCE)
DEFAULT_INTERVENTIONS = (
    COMBINED_THEN_INTERVENTION,
)
# this is used for plotting and mainly used to avoid recomputing iterative maps for the chexpert figures, can be safely ignored here
ITERATIVE_MAP_TRACE_DATASET = "chexpert_pleuraleffusiongender"
ITERATIVE_MAP_TRACE_MODEL = "vit"
ITERATIVE_MAP_TRACE_SEED = 0
OUT_CSV = Path("results/fixed_interventions/iterative_interventions.csv")
ITERATIVE_MAP_TRACE_DIR = Path(
    "results/fixed_interventions/iterative_map_traces"
)

# this is for recomputing the attribution ranks after the interventions are run
def attribution_ranks(relevance_map, atlas, region_ids):
    # computation similar to oscar
    scores = labelled_mean(relevance_map, labels=atlas, index=region_ids)
    return rankdata(-scores, method="average")

# AttnLRP code for ViT
def _recompute_lrp_rankings(
    model,
    dataset,
    indices,
    scales,
    atlas,
    region_ids,
    dataset_name,
    device,
):
    rank_vectors = []
    indices = list(indices)
    batch_size = 16
    for start in tqdm(
        range(0, len(indices), batch_size),
        desc=f"{dataset_name} recompute LRP",
        unit="batch",
        leave=False,
    ):
        batch_ids = indices[start:start + batch_size]
        x, _labels, _attrs = batch_from_dataset(dataset, batch_ids, device)
        x = x.detach().requires_grad_(True)
        patch_activations = {}

        def save_patch_embeddings(_module, _inputs, patch_embeddings):
            patch_embeddings.retain_grad()
            patch_activations["value"] = patch_embeddings

        handle = model.conv_proj.register_forward_hook(save_patch_embeddings)
        try:
            # batched inference
            token_scale = torch.as_tensor(
                scales[batch_ids],
                device=device,
                dtype=x.dtype,
            )
            # apply patching to update Value interventions
            with patch_vit(
                model,
                token_scale,
                layer_mode=default_intervention_target("vit"),
            ):
                # once monkey patch and intervention applied to relevant blocks, do model inference
                logits = model(x)
                # predictions
                predicted_logit_sum = logits.gather(
                    1,
                    logits.detach().argmax(dim=1, keepdim=True),
                ).sum()
            model.zero_grad(set_to_none=True)
            predicted_logit_sum.backward()
            patch_embeddings = patch_activations["value"]
            # relevance computation using gradient times activations
            relevance = torch.relu(
                (patch_embeddings.grad * patch_embeddings).sum(
                    dim=1,
                    keepdim=True,
                )
            )
            # resize to the image size
            relevance = F.interpolate(
                relevance,
                size=atlas.shape,
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            # compute the ranks
            rank_vectors.extend(
                attribution_ranks(relevance_map, atlas, region_ids)
                for relevance_map in relevance.detach().cpu().numpy()
            )
        finally:
            handle.remove()
            model.zero_grad(set_to_none=True)
    return np.asarray(rank_vectors, dtype=float)

# AttnLRP ViT attribution computation
def recompute_lrp_rankings(
    model,
    dataset,
    indices,
    scales,
    atlas,
    region_ids,
    dataset_name,
    device,
):
    # apply monkey patching using lxt, zennit
    with attention_lrp_patched(model):
        return _recompute_lrp_rankings(
            model,
            dataset,
            indices,
            scales,
            atlas,
            region_ids,
            dataset_name,
            device,
        )

# apply intervention on ResNet and perform inference
def forward_resnet_cumulative(model, x, batch_ids, scales):
    features = model.maxpool(model.act1(model.bn1(model.conv1(x))))
    for stage_name in RESNET_STAGES:
        features = getattr(model, stage_name)(features)
        stage_scales = scales[stage_name]
        if stage_scales is not None:
            scale = torch.as_tensor(
                stage_scales[batch_ids],
                device=features.device,
                dtype=features.dtype,
            )
            features = features * scale[:, None]
    return model.forward_head(features)

# ResNet attribution ranking computation
def recompute_resnet_rankings(
    model,
    dataset,
    indices,
    scales,
    atlas,
    region_ids,
    dataset_name,
    device,
):
    rank_vectors = []
    indices = list(indices)
    batch_size = 64
    for start in tqdm(
        range(0, len(indices), batch_size),
        desc=f"{dataset_name} recompute ResNet",
        unit="batch",
        leave=False,
    ):
        batch_ids = indices[start:start + batch_size]
        x, _labels, _attrs = batch_from_dataset(dataset, batch_ids, device)
        x = x.detach().requires_grad_(True)
        original_forward = model.forward

        def intervened_forward(self, inputs):
            return forward_resnet_cumulative(self, inputs, batch_ids, scales)

        model.forward = MethodType(intervened_forward, model)
        try:
            with torch.no_grad():
                predicted_classes = model(x).argmax(dim=1)
            one_hot = torch.zeros((len(batch_ids), 2), device=device, dtype=x.dtype)
            one_hot.scatter_(1, predicted_classes[:, None], 1.0)
            composite = EpsilonPlusFlat(canonizers=[ResNetCanonizer()])
            with Gradient(model, composite) as attribution:
                _, relevance = attribution(x, one_hot)
        finally:
            model.forward = original_forward
            model.zero_grad(set_to_none=True)

        relevance = torch.relu(relevance.sum(dim=1, keepdim=True))
        relevance = F.interpolate(
            relevance,
            size=atlas.shape,
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        rank_vectors.extend(
            attribution_ranks(relevance_map, atlas, region_ids)
            for relevance_map in relevance.detach().cpu().numpy()
        )
    return np.asarray(rank_vectors, dtype=float)


def update_resnet_cumulative_scales(
    scales,
    shortcut,
    task,
    intervention,
    target=None,
):
    shortcut = torch.as_tensor(shortcut, dtype=torch.float32)
    task = torch.as_tensor(task, dtype=torch.float32)
    target = target or default_intervention_target("resnet")
    for stage, size in zip(RESNET_STAGES, (56, 28, 14, 7)):
        if stage not in RESNET_TARGET_STAGES[target]:
            continue
        step = resnet_intervention_scales(
            shortcut,
            task,
            size,
            size,
            intervention,
        ).numpy()
        scales[stage] = step if scales[stage] is None else scales[stage] * step


def iterative_map_trace_path(dataset, model, intervention, seed=0):
    intervention_slug = intervention.lower().replace(" ", "_")
    seed_suffix = "" if seed == 0 else f"_seed{seed}"
    return (
        ITERATIVE_MAP_TRACE_DIR
        / f"{dataset}_{model}{seed_suffix}_image_{intervention_slug}.npz"
    )


def iterative_map_trace_is_complete(dataset, model, intervention, seed=0):
    path = iterative_map_trace_path(dataset, model, intervention, seed)
    if not path.exists():
        return False
    try:
        with np.load(path) as trace:
            return (
                trace["shortcut"].shape == (ITERATIONS + 1, 8, 3136)
                and trace["task"].shape == trace["shortcut"].shape
            )
    except (KeyError, OSError, ValueError):
        return False


def records_iterative_map_trace(dataset, model, seed, source):
    return (
        dataset == ITERATIVE_MAP_TRACE_DATASET
        and model == ITERATIVE_MAP_TRACE_MODEL
        and seed == ITERATIVE_MAP_TRACE_SEED
        and source == IMAGE_SOURCE
    )


def group_prototypes_for_trace(alignment_outputs):
    groups = build_nmf_shortcut_groups(
        alignment_outputs["shortcut_evidence"],
        alignment_outputs["task_evidence"],
        8,
    )
    return (
        groups["shortcut_group_prototypes"],
        groups["task_group_prototypes"],
    )

# irrelvant as such for this experiment, just used to save cache for plotting
def save_iterative_map_trace(
    dataset,
    model,
    intervention,
    prototype_trace,
    seed=0,
):
    path = iterative_map_trace_path(dataset, model, intervention, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        shortcut=np.stack(
            [shortcut for shortcut, _task in prototype_trace]
        ),
        task=np.stack([task for _shortcut, task in prototype_trace]),
    )
    temporary.replace(path)


def evaluate_cumulative_intervention(
    model,
    dataset,
    indices,
    scales,
    model_name,
    device,
    description,
):
    def intervened_logits(x, batch_ids):
        if model_name == "resnet":
            return forward_resnet_cumulative(model, x, batch_ids, scales)
        token_scale = torch.as_tensor(
            scales[batch_ids],
            device=x.device,
            dtype=x.dtype,
        )
        with patch_vit(
            model,
            token_scale,
            layer_mode=default_intervention_target("vit"),
        ):
            return model(x)

    return evaluate_batches(
        model,
        dataset,
        indices,
        device,
        logits_fn=intervened_logits,
        desc=description,
    )

# take the contribution maps compute the cumulative scales begining from the no intervention
def update_cumulative_scales(
    scales,
    shortcut,
    task,
    model_name,
    intervention,
    num_images,
    resnet_target=None,
):
    if shortcut.ndim == 1:
        shortcut = np.repeat(shortcut[None, :], num_images, axis=0)
        task = np.repeat(task[None, :], num_images, axis=0)
    if model_name == "resnet":
        update_resnet_cumulative_scales(
            scales,
            shortcut,
            task,
            intervention,
            resnet_target,
        )
        return
    shortcut = torch.as_tensor(shortcut, dtype=torch.float32)
    task = torch.as_tensor(task, dtype=torch.float32)
    step = np.stack([
        vit_intervention_token_scale(
            shortcut[index],
            task[index],
            intervention,
        ).numpy()
        for index in range(len(shortcut))
    ])
    scales *= step


def recompute_attribution_rankings(
    model,
    dataset,
    indices,
    scales,
    atlas,
    model_name,
    dataset_name,
    device,
):
    region_ids = np.unique(atlas)
    if model_name == "resnet":
        return recompute_resnet_rankings(model, dataset, indices, scales, atlas, region_ids, dataset_name, device)
    return recompute_lrp_rankings(
        model,
        dataset,
        indices,
        scales,
        atlas,
        region_ids,
        dataset_name,
        device,
    )


def iterative_intervention_row(
    dataset,
    model,
    seed,
    source,
    intervention,
    iteration,
    alignment_outputs,
    metrics,
    baseline,
    baseline_shortcut_score_mean,
    baseline_task_score_mean,
):
    shortcut_scores = alignment_outputs["shortcut_scores"]
    task_scores = alignment_outputs["task_scores"]
    shortcut_score_mean = float(np.mean(shortcut_scores))
    task_score_mean = float(np.mean(task_scores))
    return {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "source": source,
        "intervention": intervention,
        "iteration": iteration,
        "shortcut_score_mean": shortcut_score_mean,
        "task_score_mean": task_score_mean,
        "delta_shortcut_score_mean": (
            shortcut_score_mean - baseline_shortcut_score_mean
        ),
        "delta_task_score_mean": task_score_mean - baseline_task_score_mean,
        **metrics,
        "delta_accuracy": metrics["accuracy"] - baseline["accuracy"],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="iterative internal interventions"
    )
    parser.add_argument("--dataset", action="append", choices=dataset_names())
    parser.add_argument("--model", action="append", choices=MODELS)
    parser.add_argument("--seed", action="append", type=int, choices=SEEDS)
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
    selected_seeds = args.seed or SEEDS
    selected_sources = [
        source
        for source in CONTRIBUTION_SOURCES
        if source in (args.source or DEFAULT_SOURCES)
    ]
    selected_interventions = [
        intervention
        for intervention in (
            INTERVENTIONS if args.intervention else DEFAULT_INTERVENTIONS
        )
        if args.intervention is None or intervention in args.intervention
    ]
    completed = completed_csv_keys(
        args.out_csv,
        ["dataset", "model", "seed", "source", "intervention", "iteration"],
    )
    for dataset_name in selected_datasets:
        for model_name in selected_models:
            dataset = None
            for seed in selected_seeds:
                if all(
                    all(
                        (
                            dataset_name,
                            model_name,
                            seed,
                            source,
                            intervention,
                            iteration,
                        ) in completed
                        for iteration in range(1, ITERATIONS + 1)
                    )
                    and (
                        not records_iterative_map_trace(
                            dataset_name,
                            model_name,
                            seed,
                            source,
                        )
                        or iterative_map_trace_is_complete(
                            dataset_name,
                            model_name,
                            intervention,
                            seed,
                        )
                    )
                    for intervention in selected_interventions
                    for source in selected_sources
                ):
                    continue
                # load attribution ranked maps per image for AttnLRP+ViT, LRP+ResNet, 4 model seeds
                pa, ba, initial_bi = load_model_attribution_triplet(
                    root,
                    dataset_name,
                    model_name,
                    "LRP",
                    3136,
                    seed,
                )
                if dataset is None:
                    dataset = load_test_dataset(root, dataset_name)
                model = load_audited_model(
                    root, dataset_name, model_name, seed, device
                )
                indices = range(len(dataset))
                atlas = square_atlas_grid(
                    size=(224, 224),
                    grid=(56, 56),
                )
                # no intervention performance
                baseline = load_baseline_metrics(
                    root, dataset_name, model_name, seed
                )
                # shortcut and task alignment score -> shortcut and task contribution maps
                initial_alignment = build_shortcut_task_maps(initial_bi, ba, pa)
                # we track both partial correlations/alignment scores across the five interventions
                baseline_shortcut_score_mean = float(
                    np.mean(initial_alignment["shortcut_scores"])
                )
                baseline_task_score_mean = float(
                    np.mean(initial_alignment["task_scores"])
                )
                # intervention types
                for intervention in selected_interventions:
                    # image and shortcut group level
                    for source in selected_sources:
                        intervention_key = (
                            dataset_name,
                            model_name,
                            seed,
                            source,
                            intervention,
                        )
                        # if the csv already contains the result row, the experiment will not run for these again
                        intervention_complete = all(
                            (*intervention_key, iteration) in completed
                            for iteration in range(1, ITERATIONS + 1)
                        )
                        # again, not relevant
                        record_trace = records_iterative_map_trace(
                            dataset_name,
                            model_name,
                            seed,
                            source,
                        )
                        if intervention_complete and (
                            not record_trace
                            or iterative_map_trace_is_complete(
                                dataset_name,
                                model_name,
                                intervention,
                                seed,
                            )
                        ):
                            continue
                        iteration_rows = []
                        current_alignment = initial_alignment
                        prototype_trace = (
                            [group_prototypes_for_trace(current_alignment)]
                            if record_trace
                            else None
                        )
                        # resnet target scale differs for last two residual stages but for vit it is 14by14 (plus CLS which is unchanged)
                        scales = (
                            {stage: None for stage in RESNET_STAGES}
                            if model_name == "resnet"
                            else np.ones((len(dataset), 197), dtype=np.float32)
                        )
                        # actual experiment begins
                        for iteration in range(1, ITERATIONS + 1):
                            step_intervention = intervention
                            # main intervention is first combined and then shortcut suppression
                            if intervention == COMBINED_THEN_INTERVENTION:
                                step_intervention = (
                                    COMBINED_INTERVENTION
                                    if iteration == 1
                                    else SHORTCUT_SUPPRESSION
                                )
                            # maps for the current iteration
                            shortcut, task = contribution_sources(
                                current_alignment,
                                (source,),
                            )[source]
                            # compute cumulative scale for the latest intervention
                            update_cumulative_scales(
                                scales,
                                shortcut,
                                task,
                                model_name,
                                step_intervention,
                                len(dataset),
                            )
                            # apply the latest intervention
                            metrics = evaluate_cumulative_intervention(
                                model, dataset, indices, scales, model_name, device,
                                f"{dataset_name} {model_name} seed {seed} "
                                f"{source} {intervention} {iteration}",
                            )
                            # biased model rankings have to be recomputed after the intervention
                            bi = recompute_attribution_rankings(
                                model,
                                dataset,
                                indices,
                                scales,
                                atlas,
                                model_name,
                                dataset_name,
                                device,
                            )
                            # recompute shortcut, task alignment
                            current_alignment = build_shortcut_task_maps(
                                bi,
                                ba,
                                pa,
                            )
                            if prototype_trace is not None:
                                prototype_trace.append(
                                    group_prototypes_for_trace(
                                        current_alignment
                                    )
                                )

                            # result writing, not important
                            iteration_rows.append(iterative_intervention_row(
                                dataset_name,
                                model_name,
                                seed,
                                source,
                                intervention,
                                iteration,
                                current_alignment,
                                metrics,
                                baseline,
                                baseline_shortcut_score_mean,
                                baseline_task_score_mean,
                            ))
                            merge_csv_rows(
                                args.out_csv,
                                iteration_rows,
                                [
                                    "dataset",
                                    "model",
                                    "seed",
                                    "source",
                                    "intervention",
                                    "iteration",
                                ],
                            )
                        if prototype_trace is not None:
                            save_iterative_map_trace(
                                dataset_name,
                                model_name,
                                intervention,
                                prototype_trace,
                                seed,
                            )
                        completed.update(
                            (*intervention_key, iteration)
                            for iteration in range(1, ITERATIONS + 1)
                        )
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
