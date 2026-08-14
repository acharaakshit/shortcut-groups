import torch

from .common import (
    region_map_to_grid,
    SHORTCUT_SUPPRESSION,
    spatial_intervention_scale,
)

# layers that we consider in our target ablation experiments
RESNET_STAGES = ("layer1", "layer2", "layer3", "layer4")
# we either intervene after outputs at layer4, combo of 3 and 4, combo of 2,3 and 4 and finally all residual layers
RESNET_TARGET_STAGES = {
    "layer4": ("layer4",),
    "last2": ("layer3", "layer4"),
    "last3": ("layer2", "layer3", "layer4"),
    "all": RESNET_STAGES,
}


def feature_weights(region_scale):
    if region_scale.ndim == 2:
        return region_scale[None, None]
    return region_scale[:, None]

# obtain the intervention scales based on the feature map resolutions
def resnet_intervention_scale(
    shortcut,
    task,
    feature_height,
    feature_width,
    intervention,
    seed=0,
):
    feature_shortcut = region_map_to_grid(shortcut, feature_height, feature_width)
    feature_task = region_map_to_grid(task, feature_height, feature_width)
    return spatial_intervention_scale(
        feature_shortcut,
        feature_task,
        intervention,
        seed=seed,
    )


def resnet_intervention_scales(
    shortcut,
    task,
    feature_height,
    feature_width,
    intervention,
    seed=0,
):
    if shortcut.ndim == 1:
        return resnet_intervention_scale(
            shortcut,
            task,
            feature_height,
            feature_width,
            intervention,
            seed=int(seed),
        )
    seeds = [int(seed)] * len(shortcut) if isinstance(seed, int) else seed
    return torch.stack(
        [
            resnet_intervention_scale(
                shortcut[index],
                task[index],
                feature_height,
                feature_width,
                intervention,
                seed=int(seeds[index]),
            )
            for index in range(len(shortcut))
        ]
    )

# we apply suppression after getting the output of the residual stages
def forward_resnet_intervention(
    model,
    x,
    shortcut,
    task,
    target="layer4",
    intervention=SHORTCUT_SUPPRESSION,
    seed=0,
):
    out = model.conv1(x)
    out = model.bn1(out)
    out = model.act1(out)
    out = model.maxpool(out)
    shortcut = torch.as_tensor(shortcut, device=out.device, dtype=torch.float32)
    task = torch.as_tensor(task, device=out.device, dtype=torch.float32)
    selected_stages = RESNET_TARGET_STAGES[target]

    for name in RESNET_STAGES:
        out = getattr(model, name)(out)
        if name in selected_stages:
            scale = resnet_intervention_scales(
                shortcut,
                task,
                out.shape[-2],
                out.shape[-1],
                intervention,
                seed=seed,
            )
            out = out * feature_weights(scale).to(out.dtype)

    return model.forward_head(out)

# the typical inference would flow like this
# out = model.conv1(x)
# out = model.bn1(out)
# out = model.act1(out)
# out = model.maxpool(out)
# out = model.layer1(out)
# out = model.layer2(out)
# out = model.layer3(out)
# out = model.layer4(out)
# out = model.forward_head(out)
