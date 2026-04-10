"""
dataset.py
----------
Dataset loading, preprocessing, and augmentation pipeline.
Handles: resizing, normalization, augmentation (as described in proposal).
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import (
    DATA_TRAIN_DIR, DATA_VAL_DIR, DATA_TEST_DIR,
    IMG_SIZE, MEAN, STD, BATCH_SIZE
)


# ── Transforms ────────────────────────────────────────────────────────────────

def get_train_transforms():
    """
    Training transforms: resizing + normalization + augmentation.
    Augmentation improves robustness as described in the proposal.
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_val_test_transforms():
    """
    Validation/Test transforms: only resize + normalize (no augmentation).
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_inference_transform():
    """
    Transform for a single PIL image during real-time inference.
    Returns a tensor ready for the model (no batch dimension yet).
    """
    return get_val_test_transforms()


# ── Loaders ───────────────────────────────────────────────────────────────────

def get_dataloaders(batch_size=BATCH_SIZE, num_workers=0):
    """
    Returns (train_loader, val_loader, test_loader, class_names).
    num_workers=0 is safer on Windows; increase on Linux/Mac.
    """
    train_dataset = datasets.ImageFolder(
        root=DATA_TRAIN_DIR,
        transform=get_train_transforms()
    )
    val_dataset = datasets.ImageFolder(
        root=DATA_VAL_DIR,
        transform=get_val_test_transforms()
    )
    test_dataset = datasets.ImageFolder(
        root=DATA_TEST_DIR,
        transform=get_val_test_transforms()
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=False
    )

    class_names = train_dataset.classes
    print(f"[Dataset] Train: {len(train_dataset)} | "
          f"Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"[Dataset] Classes: {class_names}")

    return train_loader, val_loader, test_loader, class_names


def get_class_counts(data_dir):
    """Print the number of images per class in a given split directory."""
    print(f"\n[Class Counts] {data_dir}")
    for cls in sorted(os.listdir(data_dir)):
        cls_path = os.path.join(data_dir, cls)
        if os.path.isdir(cls_path):
            count = len(os.listdir(cls_path))
            print(f"  {cls:12s}: {count} images")