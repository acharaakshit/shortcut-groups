import os

from datasets2d import (
    get_biased_celeba_splits, # CelebA
    get_biased_chexpert_splits, # CheXpert
    get_biased_waterbirds_splits, # Waterbirds
    get_camelyon_splits, # Camelyon
    get_isic_biased_splits, # ISIC
)


def _select_split(datasets, split):
    return datasets[("train", "val", "test").index(split)]

# get biased datasets (but essentially the balanced test set from these biased datasets which is same for the triplet models/datasets)

def get_celeba(prefix, split="test"):
    datasets = get_biased_celeba_splits(
        root=prefix,
        label="Blond_Hair",
        attribute="Male",
        balanced=False,
        train_samples=10000,
        test_samples=1000,
        val_samples=500,
        bias_samples_train=500,
        bias_samples_val=40,
        attr_labs=False,
    )
    return _select_split(datasets, split)


def get_chexpert(prefix, split="test"):
    datasets = get_biased_chexpert_splits(
        root=os.path.join(prefix, "chexpert/data/chexpertchestxrays-u20210408/"),
        label="Pleural Effusion",
        attribute="Sex",
        balanced=False,
        train_samples=10000,
        test_samples=1000,
        val_samples=500,
        bias_samples_train=500,
        bias_samples_val=40,
        attr_labs=False,
    )
    return _select_split(datasets, split)


def get_waterbirds(prefix, split="test"):
    datasets = get_biased_waterbirds_splits(
        root=os.path.join(
            prefix,
            "CUB",
            "waterbirds",
            "waterbird_stratified_pool_forest2water2",
        ),
        label="y",
        attribute="place",
        balanced=False,
        train_samples=10000,
        test_samples=1000,
        val_samples=500,
        bias_samples_train=100,
        bias_samples_val=40,
        attr_labs=False,
    )
    return _select_split(datasets, split)


def get_isic(prefix, split="test"):
    datasets = get_isic_biased_splits(
        root=os.path.join(prefix, "isic2019"),
        label="malignant",
        attribute="source",
        balanced=False,
        train_samples=5000,
        test_samples=1000,
        val_samples=250,
        bias_samples_train=500,
        bias_samples_val=40,
        attr_labs=False,
        train_augment=True,
    )
    return _select_split(datasets, split)


def get_camelyon(prefix, split="test"):
    datasets = get_camelyon_splits(
        root=os.path.join(prefix, "camelyon17_v1.0"),
        label="tumor",
        attribute="center",
        balanced=False,
        train_samples=30000,
        test_samples=2000,
        val_samples=1000,
        bias_samples_train=500,
        bias_samples_val=80,
        attr_labs=False,
    )
    return _select_split(datasets, split)


DATASETS = {
    "celeba_gender": (get_celeba, 500, 40),
    "chexpert_pleuraleffusiongender": (get_chexpert, 500, 40),
    "waterbirds": (get_waterbirds, 100, 40),
    "camelyon17": (get_camelyon, 500, 80),
    "isic_source": (get_isic, 500, 40),
}

TEST_COUNTS = {
    "celeba_gender": 1000,
    "chexpert_pleuraleffusiongender": 1000,
    "waterbirds": 1000,
    "camelyon17": 2000,
    "isic_source": 1000,
}

DATASET_LABELS = {
    "celeba_gender": "CelebA",
    "chexpert_pleuraleffusiongender": "CheXpert",
    "waterbirds": "Waterbirds",
    "camelyon17": "Camelyon17",
    "isic_source": "ISIC2019",
}


def dataset_names():
    return tuple(DATASETS)


def dataset_counts(name):
    return DATASETS[name][1:]


def dataset_test_count(name):
    return TEST_COUNTS[name]


def get_dataset(name, prefix):
    """Return the requested test dataset using its canonical construction."""
    return DATASETS[name][0](prefix)
