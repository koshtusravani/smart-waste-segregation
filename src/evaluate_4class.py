# evaluate_4class.py
# Hema Sravani Koshtu
# April 20, 2026
# purpose: evaluates the 4-class model on the test set and saves the confusion matrix

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_DIR, RESULTS_DIR
from model import build_model
from train_4class import get_four_class_dataloaders

FOUR_CLASS_MODEL_PATH = os.path.join(MODEL_DIR, "best_model_4class.pth")
FOUR_CLASS_CM_DIR     = os.path.join(RESULTS_DIR, "confusion_matrix_4class")


def get_predictions(model, loader, device):
    #runs inference on the test set and returns true labels and predictions
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds   = torch.softmax(outputs, dim=1).argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)


def save_confusion_matrix(y_true, y_pred, class_names, save_dir):
    #saves both raw count and normalized confusion matrix as a single image
    os.makedirs(save_dir, exist_ok=True)
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted Label"); axes[0].set_ylabel("True Label")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted Label"); axes[1].set_ylabel("True Label")

    plt.suptitle("4-Class Model — Test Set Evaluation (Glass Excluded)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix_4class.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] Confusion matrix saved to {path}")


def evaluate_4class():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluate] Using device: {device}")

    #check that the 4-class model checkpoint exists before proceeding
    if not os.path.exists(FOUR_CLASS_MODEL_PATH):
        print(f"[Evaluate] ERROR: No model found at {FOUR_CLASS_MODEL_PATH}")
        print("  → Run: python src/train_4class.py first.")
        return

    _, _, test_loader, class_names = get_four_class_dataloaders(num_workers=0)

    model = build_model(num_classes=len(class_names), freeze_backbone=False)
    model.load_state_dict(torch.load(FOUR_CLASS_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"[Evaluate] Loaded model from {FOUR_CLASS_MODEL_PATH}")

    print("[Evaluate] Running inference on test set...")
    y_true, y_pred = get_predictions(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    print(f"  4-Class Overall Accuracy: {acc*100:.2f}%")
    print("\nPer-Class Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    save_confusion_matrix(y_true, y_pred, class_names, FOUR_CLASS_CM_DIR)

    print("\n[Evaluate]  4-class evaluation complete.")


if __name__ == "__main__":
    evaluate_4class()