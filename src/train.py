"""
train.py
--------
Full training loop for the Smart Waste Segregation CNN.

Features:
  - Train + validation loop with loss and accuracy tracking
  - Best model checkpoint saved automatically
  - Learning rate scheduling (StepLR)
  - Training curves saved to results/plots/
  - All hyperparameters pulled from config.py
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

# Add src/ to path when running from project root
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    BEST_MODEL_PATH, MODEL_DIR, PLOTS_DIR,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LR_STEP_SIZE, LR_GAMMA, BATCH_SIZE
)
from dataset import get_dataloaders
from model import build_model, count_parameters


# ── Helpers ───────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted  = outputs.max(1)
        correct       += predicted.eq(labels).sum().item()
        total         += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            loss           = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted  = outputs.max(1)
            correct       += predicted.eq(labels).sum().item()
            total         += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


def save_training_curves(history, save_dir):
    """Save loss and accuracy plots to results/plots/."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    plt.plot(epochs, history["val_loss"],   label="Val Loss",   marker="o")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150)
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc", marker="o")
    plt.plot(epochs, [a * 100 for a in history["val_acc"]],   label="Val Acc",   marker="o")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=150)
    plt.close()

    print(f"[Train] Curves saved to {save_dir}")


# ── Main Training Loop ────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # Data
    train_loader, val_loader, _, class_names = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=0
    )

    # Model
    model = build_model(num_classes=len(class_names), freeze_backbone=True)
    model.to(device)
    count_parameters(model)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    # Tracking
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc  = 0.0
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"\n[Train] Starting training for {NUM_EPOCHS} epochs...\n")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>8} | {'Time':>6}")
    print("-" * 65)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc*100:>8.2f}% | "
              f"{val_loss:>8.4f} | {val_acc*100:>7.2f}% | {elapsed:>5.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"           ✅ Best model saved (val acc: {best_val_acc*100:.2f}%)")

    print(f"\n[Train] Training complete. Best Val Acc: {best_val_acc*100:.2f}%")

    # After frozen training, fine-tune by unfreezing backbone for a few epochs
    print("\n[Train] Phase 2: Fine-tuning full model for 10 more epochs...")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(1, 11):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"{NUM_EPOCHS+epoch:>6} | {train_loss:>10.4f} | {train_acc*100:>8.2f}% | "
              f"{val_loss:>8.4f} | {val_acc*100:>7.2f}% | {elapsed:>5.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"           ✅ Best model saved (val acc: {best_val_acc*100:.2f}%)")

    save_training_curves(history, PLOTS_DIR)
    print(f"\n[Train] Final Best Val Acc: {best_val_acc*100:.2f}%")
    print(f"[Train] Model saved to: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    train()