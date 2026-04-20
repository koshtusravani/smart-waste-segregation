import os
import sys
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_TRAIN_DIR, DATA_VAL_DIR, DATA_TEST_DIR,
    MODEL_DIR, RESULTS_DIR, WEIGHT_DECAY
)
from dataset import get_train_transforms, get_val_test_transforms
from model import build_model, count_parameters

FOUR_CLASS_MODEL_PATH = os.path.join(MODEL_DIR, "best_model_4class.pth")
FOUR_CLASS_PLOTS_DIR  = os.path.join(RESULTS_DIR, "plots_4class")

FOUR_CLASS_NAMES = ["metal", "organic", "paper", "plastic"]

def get_four_class_dataloaders(batch_size=32, num_workers=0):

    train_full = datasets.ImageFolder(root=DATA_TRAIN_DIR, transform=get_train_transforms())
    val_full   = datasets.ImageFolder(root=DATA_VAL_DIR,   transform=get_val_test_transforms())
    test_full  = datasets.ImageFolder(root=DATA_TEST_DIR,  transform=get_val_test_transforms())

    glass_idx  = train_full.class_to_idx["glass"]
    old_classes = train_full.classes                          
    new_classes = [c for c in old_classes if c != "glass"]   
    old_to_new  = {old_classes.index(c): new_classes.index(c) for c in new_classes}

    def filter_indices(dataset):
        return [i for i, (_, lbl) in enumerate(dataset.samples) if lbl != glass_idx]

    def collate_remap(batch):
        imgs, labels = zip(*batch)
        new_labels = torch.tensor([old_to_new[int(l)] for l in labels])
        return torch.stack(imgs), new_labels

    def make_loader(dataset, shuffle):
        idx = filter_indices(dataset)
        return DataLoader(
            Subset(dataset, idx),
            batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=False,
            collate_fn=collate_remap,
        )

    train_loader = make_loader(train_full, shuffle=True)
    val_loader   = make_loader(val_full,   shuffle=False)
    test_loader  = make_loader(test_full,  shuffle=False)

    n_train = len(filter_indices(train_full))
    n_val   = len(filter_indices(val_full))
    n_test  = len(filter_indices(test_full))

    print(f"[4-Class] Train: {n_train} | Val: {n_val} | Test: {n_test}")
    print(f"[4-Class] Classes: {new_classes}")
    return train_loader, val_loader, test_loader, new_classes

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total


def save_curves(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"],   label="Val Loss")
    plt.title("4-Class Model — Loss Curve")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve_4class.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc")
    plt.plot(epochs, [a * 100 for a in history["val_acc"]],   label="Val Acc")
    plt.title("4-Class Model — Accuracy Curve")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve_4class.png"), dpi=150)
    plt.close()

    print(f"[Train] Curves saved to {save_dir}")


def _print_header():
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>8} | {'Time':>6}")
    print("-" * 65)

def train_4class():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")
    print(f"[Train] 4-CLASS MODE (glass excluded)")
    print(f"[Train] Saves to: {FOUR_CLASS_MODEL_PATH}\n")

    train_loader, val_loader, _, class_names = get_four_class_dataloaders(
        batch_size=32, num_workers=0
    )

    model = build_model(num_classes=len(class_names), freeze_backbone=True)
    model.to(device)
    count_parameters(model)

    criterion    = nn.CrossEntropyLoss()
    history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n[Train] Phase 1: Head only (30 epochs, backbone frozen)\n")
    _print_header()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=WEIGHT_DECAY,
    )
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(1, 31):
        t0 = time.time()
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = validate(model, val_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)
        print(f"{epoch:>6} | {tl:>10.4f} | {ta*100:>8.2f}% | {vl:>8.4f} | {va*100:>7.2f}% | {time.time()-t0:>5.1f}s")
        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), FOUR_CLASS_MODEL_PATH)
            print(f"            Best model saved (val acc: {best_val_acc*100:.2f}%)")

    print("\n[Train] Phase 2: Full fine-tune (15 epochs, lr=1e-4)\n")
    _print_header()

    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(31, 46):
        t0 = time.time()
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = validate(model, val_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)
        print(f"{epoch:>6} | {tl:>10.4f} | {ta*100:>8.2f}% | {vl:>8.4f} | {va*100:>7.2f}% | {time.time()-t0:>5.1f}s")
        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), FOUR_CLASS_MODEL_PATH)
            print(f"           Best model saved (val acc: {best_val_acc*100:.2f}%)")

    save_curves(history, FOUR_CLASS_PLOTS_DIR)

    print(f"\n[Train]  Done. Best Val Acc (4-class): {best_val_acc*100:.2f}%")
    print(f"[Train] Model saved to: {FOUR_CLASS_MODEL_PATH}")


if __name__ == "__main__":
    train_4class()