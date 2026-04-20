import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import DATA_TRAIN_DIR, DATA_VAL_DIR, DATA_TEST_DIR, IMG_SIZE, MEAN, STD, BATCH_SIZE


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15),
        transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def get_val_test_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_inference_transform():
    return get_val_test_transforms()


def get_dataloaders(batch_size=BATCH_SIZE, num_workers=0):
    train_dataset = datasets.ImageFolder(root=DATA_TRAIN_DIR, transform=get_train_transforms())
    val_dataset   = datasets.ImageFolder(root=DATA_VAL_DIR,   transform=get_val_test_transforms())
    test_dataset  = datasets.ImageFolder(root=DATA_TEST_DIR,  transform=get_val_test_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=False)

    class_names = train_dataset.classes
    counts = np.bincount(train_dataset.targets)
    print(f"[Dataset] Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"[Dataset] Classes: {class_names}")
    print(f"[Dataset] Train counts: " + " | ".join(f"{class_names[i]}={counts[i]}" for i in range(len(class_names))))
    return train_loader, val_loader, test_loader, class_names


def get_class_counts(data_dir):
    print(f"\n[Class Counts] {data_dir}")
    for cls in sorted(os.listdir(data_dir)):
        cls_path = os.path.join(data_dir, cls)
        if os.path.isdir(cls_path):
            print(f"  {cls:12s}: {len(os.listdir(cls_path))} images")