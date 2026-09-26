import argparse
from copy import copy
from itertools import product
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from transformers import CLIPImageProcessor, CLIPModel

from exps.interventions.common import load_audited_model
from shortcut_groups.datasets import dataset_names, get_dataset
from shortcut_groups.experiment_utils import artifact_prefix
from shortcut_groups.io import load_shortcut_task_maps
from shortcut_groups.prototypes import build_nmf_shortcut_groups

CLIP_MODEL = "openai/clip-vit-base-patch16"
NUM_NEAREST = 5

# just z_score the chest xray images
def normalise(images, dataset):
    images = images.float() / 255
    # CheXpert normalises inside __getitem__ so we need explicit normalisation here
    if not hasattr(dataset, "transform"):
        mean = images.mean((1, 2, 3), keepdim=True)
        std = images.std((1, 2, 3), keepdim=True, correction=0).clamp_min(1e-8)
        return ((images - mean) / std).expand(-1, 3, -1, -1)
    transform = dataset.transform
    return (transform.transforms[-1] if isinstance(transform, T.Compose) else transform)(images)


def get_images(dataset, name):
    raw_dataset = copy(dataset)
    if hasattr(dataset, "transform"):
        raw_dataset.transform = (T.Compose(dataset.transform.transforms[:-1])
                                 if isinstance(dataset.transform, T.Compose) else torch.nn.Identity())
    available_images = torch.empty((len(dataset), 3, 224, 224), dtype=torch.uint8)
    for i in range(len(dataset)):
        if name == "chexpert_pleuraleffusiongender":
            with Image.open(Path(dataset.root) / dataset.df.iloc[i]["Path"]) as image:
                raw = dataset.pre_transform(image.convert("L")).repeat(3, 1, 1)
        else:
            raw, _ = raw_dataset[i]
        available_images[i] = (raw * 255).round().to(torch.uint8)
    return available_images


def nearest_images(features, labels, attrs):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / norms
    similarity = features @ features.T
    nearest_images = {}
    same_label = labels[:, None] == labels[None, :]
    same_attr = attrs[:, None] == attrs[None, :]
    for image_type in ("same", "opposite"):
        eligible = same_label & (same_attr if image_type == "same" else ~same_attr)
        # exclude the same image from its nearest neighbours
        np.fill_diagonal(eligible, False)
        nearest_images[image_type] = np.argsort(-np.where(eligible, similarity, -np.inf), axis=1,
                                  kind="stable")[:, :NUM_NEAREST]
    return nearest_images


@torch.inference_mode()
def clip_selected_images(available_images, labels, attrs, args):
    processor = CLIPImageProcessor.from_pretrained(
        CLIP_MODEL, local_files_only=True, cache_dir=args.clip_cache_dir)
    model = CLIPModel.from_pretrained(
        CLIP_MODEL, local_files_only=True, cache_dir=args.clip_cache_dir).eval().to(args.device)
    features = []
    for batch in available_images.split(args.batch_size):
        images = [Image.fromarray(x.permute(1, 2, 0).numpy()) for x in batch]
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(args.device)
        # feature embedding
        features.append(model.get_image_features(pixel_values=pixels).float().cpu().numpy())
    return nearest_images(np.concatenate(features), labels, attrs)


def shortcut_mask(scores, fraction):
    # top-20% or 30% shortcut contributions per image
    regions = np.broadcast_to(np.arange(scores.shape[1]), scores.shape)
    order = np.lexsort((regions, -scores), axis=1)[:, :ceil(fraction * scores.shape[1])]
    mask = np.zeros_like(scores, dtype=bool)
    np.put_along_axis(mask, order, True, axis=1)
    return mask


@torch.inference_mode()
def predict(model, available_images, dataset, args, nearest_images=None, masks=None):
    predictions = []
    for start in range(0, len(available_images), args.batch_size):
        rows = slice(start, start + args.batch_size)
        images = available_images[rows]
        if nearest_images is not None:
            # highest shortcut regions
            mask = torch.as_tensor(masks[rows]).reshape(-1, 1, 56, 56)
            # interpolate to image size
            mask = F.interpolate(mask.float(), size=images.shape[-2:], mode="nearest").bool()
            # replace the regions with the nearest images
            images = torch.where(mask, available_images[nearest_images[rows]], images)
        # perform inference
        logits = model(normalise(images.to(args.device), dataset))
        # binary classification
        predictions.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", choices=dataset_names(), default=dataset_names())
    parser.add_argument("--model", nargs="+", choices=("resnet", "vit"), default=("resnet", "vit"))
    parser.add_argument("--seed", nargs="+", type=int, choices=range(4), default=list(range(4)))
    parser.add_argument("--fraction", nargs="+", type=float, choices=(.2, .3), default=(.2, .3))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--clip-cache-dir", type=Path, help="Location of existing CLIP model")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    root = artifact_prefix()
    results = []
    for name in args.dataset:
        print(f"Loading {name}")
        # test set
        dataset = get_dataset(name, root)
        # get all images from the test set
        available_images = get_images(dataset, name)
        # task labels and sensitive attributes
        labels, attrs = np.asarray(dataset.labels), np.asarray(dataset.attributes)
        # nearest images for each image based on CLIP
        nearest_images = clip_selected_images(available_images, labels, attrs, args)
        for model_name, seed in product(args.model, args.seed):
            # contribution maps
            maps = load_shortcut_task_maps(root, name, model_name, "LRP", 3136, seed)
            # grouping
            groups = build_nmf_shortcut_groups(maps["shortcut_evidence"], maps["task_evidence"], 8)
            scores = groups["image_shortcut_group_maps"]
            model = load_audited_model(root, name, model_name, seed, args.device)
            # baseline predictions for computing changes later
            baseline = predict(model, available_images, dataset, args)
            for fraction in args.fraction:
                mask = shortcut_mask(scores, fraction)
                rates = {kind: [] for kind in nearest_images}
                for rank in range(NUM_NEAREST):
                    pair = {kind: indices[:, rank] for kind, indices in nearest_images.items()}
                    for kind in nearest_images:
                        after = predict(model, available_images, dataset, args, pair[kind], mask)
                        rates[kind].append(100 * np.mean(after != baseline))
                row = dict(dataset=name, model=model_name, seed=seed, fraction=fraction,
                           same=np.mean(rates["same"]), opposite=np.mean(rates["opposite"]))
                results.append(row)
                print(f"{name} {model_name} seed={seed} {fraction:.0%}: "
                      f"same={row['same']:.2f}% opposite={row['opposite']:.2f}%")
            del model
    summary = pd.DataFrame(results).groupby(["dataset", "model", "fraction"])[["same", "opposite"]].mean()
    print("\nFlip rates (%), averaged over five nearest_images and the model seeds:")
    print(summary.to_string(float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
