import math

import torch
import torch.nn.functional as F

# our suppression approach that uses cosine sim with task and then suppresses residual
SHORTCUT_SUPPRESSION = "Shortcut suppression"
# suppress shortcut-exclusive residual evidence while amplifying task-exclusive
# residual evidence.
COMBINED_INTERVENTION = (
    "Combined shortcut suppression and task amplification"
)
# amplify task-exclusive residual evidence without suppressing shortcut evidence.
TASK_AMPLIFICATION = "Task amplification"

# residual suppression on task evidence/contribution
TASK_SUPPRESSION = "Task suppression"
# random baseline suppression outside of shortcut suppression to serve as a random baseline
SHUFFLED_SUPPRESSION = (
    "Outside-shortcut-support shuffled suppression"
)

# just the square root of regions to obtain side
def square_side(regions):
    return math.isqrt(regions)

# map the regions to a grid -> image size, feature map size etc
def region_map_to_grid(region_map, height, width):
    # get side
    side = square_side(region_map.numel())
    # create grid suitable to be overlayed
    grid = region_map.reshape(1, 1, side, side)
    # interpolate to image/feature size
    return F.interpolate(grid, size=(height, width), mode="area")[0, 0]

# shortcut residual suppression
def shortcut_residual_map(shortcut, task):
    # normalise the nonnegative shortcut and task maps
    shortcut = shortcut / shortcut.norm().clamp_min(1e-8)
    task = task / task.norm().clamp_min(1e-8)
    # remove the projection of shortcut along task and remove from shortcut
    residual = torch.clamp(
        shortcut - torch.sum(shortcut * task) * task,
        min=0.0,
    )
    # now, we will normalise by dividing with max value
    peak = residual.max()
    normalised = residual / peak.clamp_min(1e-8)
    # if peak is very very small then essentially don't apply suppression
    return normalised * (peak > 1e-8).to(residual.dtype)

# this is the application of actuall residual suppression
def residual_suppression_scale(shortcut, task):
    # ensure that the residual is not negative or else it might be counter/anti suppression
    shortcut_residual = shortcut_residual_map(shortcut, task)
    return torch.clamp(1.0 - shortcut_residual, min=0.0)


def task_residual_amplification_scale(shortcut, task):
    task_residual = shortcut_residual_map(task, shortcut)
    return 1.0 + task_residual


def combined_residual_intervention_scale(shortcut, task):
    shortcut_residual = shortcut_residual_map(shortcut, task)
    suppression = torch.clamp(
        1.0 - shortcut_residual,
        min=0.0,
    )
    task_amplification = task_residual_amplification_scale(shortcut, task)
    return suppression * task_amplification

# take the residual and shuffle
def shuffled_residual_suppression_scale(
    shortcut,
    task,
    seed=0,
):
    residual = shortcut_residual_map(shortcut, task).flatten()
    support = residual > 1e-8
    values = residual[support]
    if not len(values):
        return torch.ones_like(shortcut)

    # take regions outside of shortcut support first
    outside = torch.nonzero(~support, as_tuple=False).flatten()
    # take remaining regions from the lowest support regions
    if len(outside) < len(values):
        inside = torch.nonzero(support, as_tuple=False).flatten()
        inside = inside[torch.argsort(residual[inside], stable=True)]
        targets = torch.cat((outside, inside[: len(values) - len(outside)]))
    else:
        targets = outside

    # shuffle and apply now
    generator = torch.Generator(device=residual.device).manual_seed(int(seed))
    target_order = torch.randperm(
        len(targets),
        generator=generator,
        device=residual.device,
    )[: len(values)]
    value_order = torch.randperm(
        len(values),
        generator=generator,
        device=residual.device,
    )
    shuffled_residual = torch.zeros_like(residual)
    shuffled_residual[targets[target_order]] = values[value_order]
    return torch.clamp(
        1.0
        - shuffled_residual.reshape_as(shortcut),
        min=0.0,
    )

# main intervention application function
def spatial_intervention_scale(
    shortcut,
    task,
    intervention,
    seed=0,
):
    if intervention == SHUFFLED_SUPPRESSION:
        return shuffled_residual_suppression_scale(
            shortcut,
            task,
            seed=seed,
        )
    if intervention == SHORTCUT_SUPPRESSION:
        return residual_suppression_scale(shortcut, task)
    if intervention == COMBINED_INTERVENTION:
        return combined_residual_intervention_scale(shortcut, task)
    if intervention == TASK_AMPLIFICATION:
        return task_residual_amplification_scale(shortcut, task)
    if intervention == TASK_SUPPRESSION:
        return residual_suppression_scale(task, shortcut)
    raise ValueError(f"Unknown spatial intervention: {intervention}")
