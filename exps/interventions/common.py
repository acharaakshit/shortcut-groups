from pathlib import Path

import timm
import torch
from models import Classifier2D

from shortcut_groups.datasets import dataset_counts, get_dataset
from shortcut_groups.interventions.evaluation import evaluate_batches
from shortcut_groups.interventions.resnet import forward_resnet_intervention
from shortcut_groups.interventions.vit import (
    patch_vit,
    vit_intervention_token_scale,
)
from shortcut_groups.io import load_shortcut_task_maps

# Default targets selected using the multi-seed shortcut suppression sweep.
def default_intervention_target(model):
    return "last2" if model == "resnet" else "all"

# load the shortcut and task contribution maps
def load_intervention_maps(artifact_root, dataset, model, seed=0):
    return load_shortcut_task_maps(
        artifact_root, dataset, model, "LRP", 3136, seed,
    )

# load the datasets
def load_test_dataset(artifact_root, dataset):
    return get_dataset(dataset, artifact_root)

# load the model using OSCAR
def load_audited_model(artifact_root, dataset, model, seed, device):
    train_count, val_count = dataset_counts(dataset)
    ckpt = Path(artifact_root) / dataset / "saved_data/models" / (
        f"MODEL_{model}_3_SEED2D_{seed}_BASELINE_None_None_"
        f"{train_count}_{val_count}_F1.ckpt"
    )
    checkpoint_kwargs = {
        "checkpoint_path": ckpt,
        "model_alias": "resnet50.am_in1k" if model == "resnet" else "vit",
        "num_classes": 2,
        "in_channels": 3,
        "lr": 1e-5 if "chexpert" in dataset else 1e-4,
        "img_size": 224,
        "strict": False,
    }
    if model == "resnet":
        original_create_model = timm.create_model

        def create_model_without_pretrained(*args, **kwargs):
            kwargs["pretrained"] = False
            return original_create_model(*args, **kwargs)

        timm.create_model = create_model_without_pretrained
        try:
            wrapped = Classifier2D.load_from_checkpoint(**checkpoint_kwargs)
        finally:
            timm.create_model = original_create_model
    else:
        wrapped = Classifier2D.load_from_checkpoint(**checkpoint_kwargs)
    return wrapped.model.to(device).eval()


def evaluate_intervention(
    model,
    dataset,
    indices,
    shortcut,
    task,
    intervention,
    model_name,
    device,
    desc,
    *,
    seed=0,
    resnet_target=None,
    vit_targets=("value",),
    vit_layer_mode=None,
):
    per_image = shortcut.ndim == 2
    fixed_token_scale = None
    if resnet_target is None:
        resnet_target = default_intervention_target("resnet")
    if vit_layer_mode is None:
        vit_layer_mode = default_intervention_target("vit")
    if model_name == "vit":
        shortcut = torch.as_tensor(shortcut, dtype=torch.float32)
        task = torch.as_tensor(task, dtype=torch.float32)
    if model_name == "vit" and not per_image:
        fixed_token_scale = vit_intervention_token_scale(
            shortcut,
            task,
            intervention,
            seed=seed,
        ).to(device)

    def intervened_logits(x, batch_ids):
        if model_name == "resnet" and per_image:
            return forward_resnet_intervention(
                model,
                x,
                shortcut[batch_ids],
                task[batch_ids],
                target=resnet_target,
                intervention=intervention,
                seed=[seed + index for index in batch_ids],
            )
        if model_name == "resnet":
            return forward_resnet_intervention(
                model,
                x,
                shortcut,
                task,
                target=resnet_target,
                intervention=intervention,
                seed=seed,
            )
        if per_image:
            token_scales = torch.stack([
                vit_intervention_token_scale(
                    shortcut[idx],
                    task[idx],
                    intervention,
                    seed=seed + idx,
                )
                for idx in batch_ids
            ]).to(device=x.device, dtype=x.dtype)
            with patch_vit(
                model,
                token_scales,
                targets=vit_targets,
                layer_mode=vit_layer_mode,
            ):
                return model(x)
        with patch_vit(
            model,
            fixed_token_scale,
            targets=vit_targets,
            layer_mode=vit_layer_mode,
        ):
            return model(x)

    return evaluate_batches(
        model,
        dataset,
        indices,
        device,
        logits_fn=intervened_logits,
        desc=desc,
    )
