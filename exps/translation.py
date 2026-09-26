import argparse
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from exps.interventions.common import load_audited_model
from exps.risk.common import balanced_split, nmf_risk_scores
from shortcut_groups.datasets import get_dataset
from shortcut_groups.experiment_utils import artifact_prefix, errors_found_at_coverage
from shortcut_groups.interventions.vit import attention_lrp_patched
from shortcut_groups.io import load_model_attribution_triplet, load_test_split
from shortcut_groups.methods import fit_nmf_mixture, transform_nmf_mixture
from shortcut_groups.prototypes import membership_weighted_group_prototypes
from shortcut_groups.partial_correlations import build_shortcut_task_maps

# our model triplet
MODEL_ROLES = ("ts", "ba", "sa")
TRANSLATIONS = (0, 10, 20, 30, 40)  # Include the unshifted baseline.
COVERAGE = 0.2
DIRECTION_SEED = 0
NUM_GROUPS = 8
RISK_SPLITS = 5
OUT = Path("results/alignment")
# directions of translation
CARDINAL_DIRECTIONS = np.asarray([(1, 0), (-1, 0), (0, 1), (0, -1)], dtype=int)


def balanced_directions(targets, attributes, seed):
    rng = np.random.default_rng(seed)
    directions = np.empty((len(targets), 2), dtype=int)
    # balance directions within each subgroup
    for cell_index, (target, attribute) in enumerate(product((0, 1), repeat=2)):
        positions = np.flatnonzero((targets == target) & (attributes == attribute))
        repeated = np.resize(np.roll(CARDINAL_DIRECTIONS, -cell_index, axis=0), (len(positions), 2))
        directions[positions] = repeated[rng.permutation(len(positions))]
    return directions


def translate_batch(images, directions, displacement):
    pad = int(displacement)
    # reflection padding for the translated crop
    padded = functional.pad(images, (pad, pad, pad, pad), mode="reflect")
    height, width = images.shape[-2:]
    translated = []
    for image, (unit_x, unit_y) in zip(padded, directions):
        x, y = pad - int(unit_x) * pad, pad - int(unit_y) * pad
        translated.append(image[:, y:y+height, x:x+width])
    return torch.stack(translated)


def relevance_ranks(model, images, predictions):
    patch_activations = None

    def save_patch_activations(_module, _inputs, output):
        nonlocal patch_activations
        patch_activations = output
        patch_activations.retain_grad()

    # patch activations and their gradients
    inputs = images.detach().requires_grad_(True)
    handle = model.conv_proj.register_forward_hook(save_patch_activations)
    try:
        logits = model(inputs)
    finally:
        handle.remove()
    # explain the classes predicted before patching
    selected_logits = logits.gather(1, predictions[:, None]).sum()
    model.zero_grad(set_to_none=True)
    selected_logits.backward()

    # positive patch relevance
    relevance = torch.relu(
        (patch_activations * patch_activations.grad).sum(1, keepdim=True)
    )
    relevance = functional.interpolate(
        relevance, size=(224, 224), mode="bilinear", align_corners=False
    )
    # mean relevance in each 4 x 4 region, matching OSCAR
    regional_scores = relevance.reshape(
        len(relevance), 1, 56, 4, 56, 4
    ).mean(dim=(3, 5))[:, 0]
    # rank regions per image, averaging ties
    ranks = rankdata(
        -regional_scores.detach().cpu().numpy().reshape(len(relevance), -1),
        method="average",
        axis=1,
    )
    return ranks


def cache_path(args, role, displacement):
    count = len(args.targets)
    return args.cache_dir / (
        f"{args.dataset_name}_vit_lrp_{role}_translation_{displacement}px_"
        f"n{count}_seed0_unpatched_logits.npz"
    )


def compute_translated_ranks(args, directions, test_dataset):
    # compute missing translations
    missing = {role: [d for d in TRANSLATIONS if d > 0 and not cache_path(args, role, d).exists()]
               for role in MODEL_ROLES}
    if not any(missing.values()):
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(test_dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for role, pending in missing.items():
        if not pending:
            continue
        model = load_audited_model(args.prefix, args.dataset, "vit", 0, device, role=role)
        for displacement in pending:
            all_ranks, all_logits, offset = [], [], 0
            for images, _ in tqdm(loader, desc=f"{role}: translation={displacement}px", leave=True):
                shifted = translate_batch(images.to(device), directions[offset:offset+len(images)], displacement)
                # predictions from the original model
                with torch.no_grad():
                    logits = model(shifted)
                # temporary patch for LRP
                with attention_lrp_patched(model):
                    ranks = relevance_ranks(model, shifted, logits.argmax(dim=1))
                all_ranks.append(ranks)
                all_logits.append(logits.float().cpu().numpy())
                offset += len(images)
            np.savez_compressed(cache_path(args, role, displacement), ranks=np.concatenate(all_ranks),
                                logits=np.concatenate(all_logits))
        model.to("cpu")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def load_condition(args, displacement):
    ranks, task_logits = {}, None
    for role in MODEL_ROLES:
        path = cache_path(args, role, displacement)
        with np.load(path) as archive:
            ranks[role] = np.asarray(archive["ranks"], dtype=float)
            if role == "ts":
                task_logits = archive["logits"]
    return ranks, task_logits


def heldout_error_capture(features, errors, targets, attributes, num_groups, splits):
    captures = []
    for seed in range(splits):
        # balanced fit and heldout halves
        fit, heldout = balanced_split(targets, attributes, len(targets) // 2, seed)
        # fit NMF and group risk on the fit half
        scores = nmf_risk_scores(features, errors, fit, heldout, num_groups)
        # error capture in the highest-risk 20% of heldout images
        captures.append(errors_found_at_coverage(errors[heldout], scores, COVERAGE))
    return np.mean(captures)


def translation_metrics(args, original_ranks, targets, attributes, original_logits):
    # contribution maps
    original_maps = build_shortcut_task_maps(original_ranks["ts"], original_ranks["ba"], original_ranks["sa"])
    # representation for grouping
    original_features = np.concatenate([original_maps["shortcut_evidence"], original_maps["task_evidence"]], axis=1)
    # fit NMF on the original images for comparing group maps
    original_memberships, _, fixed_model = fit_nmf_mixture(original_features, NUM_GROUPS)
    order = np.argsort(original_memberships.sum(axis=0))[::-1]
    shortcut_maps, task_maps = [], []
    for displacement in TRANSLATIONS:
        if displacement == 0:
            features, memberships, logits = original_features, original_memberships, original_logits
        else:
            ranks, logits = load_condition(args, displacement)
            maps = build_shortcut_task_maps(ranks["ts"], ranks["ba"], ranks["sa"])
            features = np.concatenate([maps["shortcut_evidence"], maps["task_evidence"]], axis=1)
            # NMF memberships
            memberships = transform_nmf_mixture(fixed_model, features)
        errors = logits.argmax(axis=1) != targets
        # group level membership weighted contribution maps
        shortcut, task = np.split(membership_weighted_group_prototypes(memberships, features), 2, axis=1)
        shortcut_maps.append(shortcut[order])
        task_maps.append(task[order])
        # refit NMF for each risk split for this translation
        capture = heldout_error_capture(features, errors, targets, attributes,
                                        NUM_GROUPS, RISK_SPLITS)
        print(f"{args.dataset_name} {displacement:>2}px: error capture={100*capture:.1f}%")
    maps = dict(translations=np.asarray(TRANSLATIONS), shortcut_prototypes=np.asarray(shortcut_maps),
                task_prototypes=np.asarray(task_maps))
    return maps



def evaluate_translations(args):
    dataset = get_dataset(args.dataset, str(args.prefix))
    # original ranks and task predictions
    sa, ba, ts = load_model_attribution_triplet(args.prefix, args.dataset, "vit", "LRP", 3136, 0)
    ranks = {"ts": ts, "ba": ba, "sa": sa}
    test = load_test_split(args.prefix, args.dataset, "vit", 0)
    targets, attributes, logits = test["targets"], test["attrs"], test["pred_logits"]
    # keep each image direction fixed across translation distances
    directions = balanced_directions(targets, attributes, DIRECTION_SEED)
    args.targets = targets
    compute_translated_ranks(args, directions, dataset)
    maps = translation_metrics(args, ranks, targets, attributes, logits)
    # save group maps
    stem = OUT / f"{args.dataset_name}_vit_lrp_translation_n{len(dataset)}_seed0"
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(stem.with_name(stem.name + f"_k{NUM_GROUPS}_maps.npz"), **maps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    args.prefix = artifact_prefix()
    for args.dataset_name, args.dataset in (("waterbirds", "waterbirds"), ("celeba", "celeba_gender")):
        args.cache_dir = OUT / f"{args.dataset_name}_translation_cache"
        evaluate_translations(args)


if __name__ == "__main__":
    main()
