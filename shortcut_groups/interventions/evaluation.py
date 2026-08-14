import torch
from tqdm import tqdm

from shortcut_groups.io import load_test_split


BATCH_SIZE = 128
# Match the saved FP32 logits used for no-intervention baselines.
DTYPE = torch.float32


def batch_from_dataset(dataset, indices, device):
    images, labels, attrs = [], [], []
    for idx in indices:
        image, label = dataset[int(idx)]
        images.append(image)
        labels.append(label)
        attrs.append(dataset.attributes[int(idx)])
    return (
        torch.stack(images).to(device),
        torch.as_tensor(labels, device=device),
        torch.as_tensor(attrs, device=device),
    )


def update_group_counts(pred, y, attrs, correct, total):
    for label in (0, 1):
        for attr in (0, 1):
            group = (y == label) & (attrs == attr)
            key = f"group_{label}{attr}"
            total[key] += int(group.sum().item())
            correct[key] += int((pred[group] == y[group]).sum().item())


def group_summary(correct, total):
    metrics = {}
    accuracies = []
    for label in (0, 1):
        for attr in (0, 1):
            key = f"group_{label}{attr}"
            acc_key = f"{key}_accuracy"
            count_key = f"{key}_count"
            metrics[count_key] = total[key]
            metrics[acc_key] = correct[key] / total[key]
            accuracies.append(metrics[acc_key])
    metrics["worst_group_accuracy"] = min(accuracies)
    return metrics

def metrics_from_predictions(pred, y, attrs):
    pred = torch.as_tensor(pred)
    y = torch.as_tensor(y)
    attrs = torch.as_tensor(attrs)
    group_correct = {
        f"group_{label}{attr}": 0
        for label in (0, 1)
        for attr in (0, 1)
    }
    group_total = group_correct.copy()
    update_group_counts(pred, y, attrs, group_correct, group_total)
    metrics = group_summary(group_correct, group_total)
    metrics["accuracy"] = (pred == y).float().mean().item()
    return metrics


@torch.no_grad()
def predict_batches(model, dataset, indices, device, logits_fn=None, desc=None):
    predictions, labels, attributes = [], [], []
    indices = list(indices)

    for start in tqdm(
        range(0, len(indices), BATCH_SIZE),
        desc=desc,
        unit="batch",
        leave=False,
    ):
        batch_ids = indices[start:start + BATCH_SIZE]
        x, y, attrs = batch_from_dataset(dataset, batch_ids, device)
        x = x.to(DTYPE)
        logits = model(x) if logits_fn is None else logits_fn(x, batch_ids)
        predictions.append(logits.argmax(dim=1).cpu())
        labels.append(y.cpu())
        attributes.append(attrs.cpu())

    return tuple(
        torch.cat(values).numpy()
        for values in (predictions, labels, attributes)
    )


def evaluate_batches(model, dataset, indices, device, logits_fn=None, desc=None):
    return metrics_from_predictions(*predict_batches(
        model,
        dataset,
        indices,
        device,
        logits_fn,
        desc=desc,
    ))


def load_baseline_metrics(artifact_root, dataset, model, seed):
    test = load_test_split(artifact_root, dataset, model, seed)
    return metrics_from_predictions(
        test["pred_logits"].argmax(axis=1),
        test["targets"],
        test["attrs"],
    )
