"""
evaluate.py
-----------
Evaluation module for the Smart Waste Segregation system.

Computes (as specified in the proposal):
  - Accuracy, Precision, Recall, F1-Score (per class + macro average)
  - Confusion Matrix (saved as image)
  - Classification report printed to console
  - Highlights visually similar class pairs: glass/plastic, paper/organic
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    BEST_MODEL_PATH, CM_DIR, CLASS_NAMES, NUM_CLASSES
)
from dataset import get_dataloaders
from model import load_model


# ── Core Evaluation ───────────────────────────────────────────────────────────

def get_predictions(model, loader, device):
    """Run inference over a dataloader and return all labels + predictions."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            preds   = probs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def print_metrics(y_true, y_pred, class_names):
    """Print accuracy + per-class precision, recall, F1."""
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*55}")
    print(f"  Overall Accuracy: {acc*100:.2f}%")
    print(f"{'='*55}")
    print("\nPer-Class Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Highlight commonly confused pairs (as noted in proposal)
    print("⚠️  Watch for confusion between:")
    print("   - glass vs plastic (visually similar)")
    print("   - paper vs organic (visually similar)")


def save_confusion_matrix(y_true, y_pred, class_names, save_dir):
    """Save a styled confusion matrix image."""
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)  # normalize

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicted Label"); axes[0].set_ylabel("True Label")

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Predicted Label"); axes[1].set_ylabel("True Label")

    plt.suptitle("Smart Waste Segregation — Test Set Evaluation", fontsize=15, y=1.02)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] Confusion matrix saved to {save_path}")


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluate] Using device: {device}")

    # Load data
    _, _, test_loader, class_names = get_dataloaders(num_workers=0)

    # Load model
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"[Evaluate] ERROR: No model found at {BEST_MODEL_PATH}")
        print("  → Run python src/train.py first.")
        return

    model = load_model(BEST_MODEL_PATH, num_classes=len(class_names), device=device)

    # Predictions
    print("[Evaluate] Running inference on test set...")
    y_true, y_pred, y_probs = get_predictions(model, test_loader, device)

    # Metrics
    print_metrics(y_true, y_pred, class_names)

    # Confusion matrix
    save_confusion_matrix(y_true, y_pred, class_names, CM_DIR)

    print("\n[Evaluate] ✅ Evaluation complete.")


if __name__ == "__main__":
    evaluate()