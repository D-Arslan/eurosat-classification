"""EuroSAT dataset loading and preprocessing."""

import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]

# ImageNet normalization (used with pretrained ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool = True):
    """Get data transforms with augmentation for training."""
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def load_eurosat(data_dir: str = "data", batch_size: int = 64, seed: int = 42):
    """Load EuroSAT dataset with train/val/test split (70/15/15).

    Downloads the dataset automatically via torchvision if not present.
    Split strategy: random split with fixed seed for reproducibility.
    """
    # Download EuroSAT via torchvision
    full_dataset = datasets.EuroSAT(
        root=data_dir,
        download=True,
        transform=None  # Apply transforms after splitting
    )

    # Split: 70% train, 15% val, 15% test
    n_total = len(full_dataset)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Wrap with transforms
    train_set = TransformSubset(train_set, get_transforms(train=True))
    val_set = TransformSubset(val_set, get_transforms(train=False))
    test_set = TransformSubset(test_set, get_transforms(train=False))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    print(f"Dataset split: train={n_train}, val={n_val}, test={n_test}")
    return train_loader, val_loader, test_loader


class TransformSubset(torch.utils.data.Dataset):
    """Apply transforms to a Subset."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)
