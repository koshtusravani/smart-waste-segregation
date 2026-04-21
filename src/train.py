# train.py
# Hema Sravani Koshtu
# April 20, 2026
# purpose: trains the 5-class waste classification model using a two-phase strategy

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BEST_MODEL_PATH, MODEL_DIR, PLOTS_DIR, WEIGHT_DECAY
from dataset import get_dataloaders
from model import build_model, count_parameters, load_model


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


def save_curves(history, save_dir, start_epoch=1):
    #saves loss and accuracy plots to disk with correct x-axis epoch offset
    os.makedirs(save_dir, exist_ok=True)
    n = len(history["train_loss"])
    epochs = range(start_epoch, start_epoch + n)

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"],   label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc")
    plt.plot(epochs, [a * 100 for a in history["val_acc"]],   label="Val Acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=150)
    plt.close()

    print(f"[Train] Curves saved to {save_dir}")


def _print_header():
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>8} | {'Time':>6}")
    print("-" * 65)


def _run_epoch(epoch, model, train_loader, val_loader, criterion, optimizer,
               scheduler, history, device, best_val_acc):
    #runs one train and validation epoch, saves checkpoint if validation accuracy improves
    t0 = time.time()
    tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
    vl, va = validate(model, val_loader, criterion, device)
    scheduler.step()

    history["train_loss"].append(tl)
    history["val_loss"].append(vl)
    history["train_acc"].append(ta)
    history["val_acc"].append(va)

    elapsed = time.time() - t0
    print(f"{epoch:>6} | {tl:>10.4f} | {ta * 100:>8.2f}% | {vl:>8.4f} | {va * 100:>7.2f}% | {elapsed:>5.1f}s")

    if va > best_val_acc:
        best_val_acc = va
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"            Best model saved (val acc: {best_val_acc * 100:.2f}%)")

    return best_val_acc


def train_from_scratch(train_loader, val_loader, class_names, device):
    #phase 1: trains head only with backbone frozen, phase 2: full fine-tune
    model = build_model(num_classes=len(class_names), freeze_backbone=True)
    model.to(device)
    count_parameters(model)

    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
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
        best_val_acc = _run_epoch(
            epoch, model, train_loader, val_loader,
            criterion, optimizer, scheduler, history, device, best_val_acc,
        )

    print("\n[Train] Phase 2: Full fine-tune (15 epochs, lr=1e-4, StepLR)\n")
    _print_header()

    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(31, 46):
        best_val_acc = _run_epoch(
            epoch, model, train_loader, val_loader,
            criterion, optimizer, scheduler, history, device, best_val_acc,
        )

    save_curves(history, PLOTS_DIR, start_epoch=1)
    return best_val_acc


def continue_training(train_loader, val_loader, class_names, device):
    #resumes training from the saved checkpoint using cosine annealing lr
    print(f"[Train] Loading checkpoint: {BEST_MODEL_PATH}")
    model = load_model(BEST_MODEL_PATH, num_classes=len(class_names), device=device)
    count_parameters(model)

    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()

    #measure baseline accuracy from checkpoint before continuing
    print("[Train] Measuring checkpoint baseline...")
    _, best_val_acc = validate(model, val_loader, criterion, device)
    print(f"[Train] Checkpoint baseline val acc: {best_val_acc * 100:.2f}%\n")

    EPOCHS = 25
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    os.makedirs(MODEL_DIR, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    print(f"[Train] Continuing fine-tune for {EPOCHS} more epochs at lr=5e-5 (cosine decay)...\n")
    _print_header()

    for epoch in range(1, EPOCHS + 1):
        best_val_acc = _run_epoch(
            epoch, model, train_loader, val_loader,
            criterion, optimizer, scheduler, history, device, best_val_acc,
        )

    #start_epoch=46 so x-axis continues from where initial training left off
    save_curves(history, PLOTS_DIR, start_epoch=46)
    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train or continue training the model.")
    parser.add_argument(
        "--continue", dest="resume", action="store_true",
        help="Resume fine-tuning from existing best_model.pth",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    train_loader, val_loader, _, class_names = get_dataloaders(batch_size=32, num_workers=0)

    if args.resume:
        best_val_acc = continue_training(train_loader, val_loader, class_names, device)
    else:
        best_val_acc = train_from_scratch(train_loader, val_loader, class_names, device)

    print(f"\n[Train]  Done. Best Val Acc: {best_val_acc * 100:.2f}%")
    print(f"[Train] Model saved to: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()