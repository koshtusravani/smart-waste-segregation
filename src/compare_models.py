# compare_models.py
# Hema Sravani Koshtu
# April 20, 2026
# purpose: loads both trained models and produces a side-by-side comparison of 4-class vs 5-class results

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_DIR, RESULTS_DIR
from model import build_model
from dataset import get_dataloaders
from train_4class import get_four_class_dataloaders

FIVE_CLASS_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
FOUR_CLASS_MODEL_PATH = os.path.join(MODEL_DIR, "best_model_4class.pth")
COMPARISON_SAVE_DIR   = os.path.join(RESULTS_DIR, "comparison")


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


def save_comparison_chart(results_4, results_5, shared_classes, save_dir):
    #saves a grouped bar chart comparing per-class f1 scores for both models
    os.makedirs(save_dir, exist_ok=True)

    f1_4 = [results_4[c]["f1-score"] for c in shared_classes]
    f1_5 = [results_5[c]["f1-score"] for c in shared_classes]

    x = np.arange(len(shared_classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars4 = ax.bar(x - width/2, f1_4, width, label="4-Class Model", color="#4C72B0")
    bars5 = ax.bar(x + width/2, f1_5, width, label="5-Class Model", color="#DD8452")

    ax.set_xlabel("Class")
    ax.set_ylabel("F1-Score")
    ax.set_title("Per-Class F1 Score: 4-Class vs 5-Class Model")
    ax.set_xticks(x)
    ax.set_xticklabels(shared_classes)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.bar_label(bars4, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(bars5, fmt="%.3f", padding=3, fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, "f1_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Compare] Bar chart saved to {path}")


def print_comparison_table(acc_4, acc_5, report_4, report_5, shared_classes):
    #prints a formatted side-by-side comparison table to the console
    print("        4-CLASS vs 5-CLASS MODEL — FULL COMPARISON")
    print(f"  {'Metric':<32} {'4-Class':>12} {'5-Class':>12}  {'Δ':>6}")

    diff_acc = acc_4 - acc_5
    print(f"  {'Overall Accuracy':<32} {acc_4:>11.2f}% {acc_5:>11.2f}%  {diff_acc:>+6.2f}%")

    diff_macro = report_4["macro avg"]["f1-score"] - report_5["macro avg"]["f1-score"]
    print(f"  {'Macro F1':<32} {report_4['macro avg']['f1-score']:>12.4f} "
          f"{report_5['macro avg']['f1-score']:>12.4f}  {diff_macro:>+6.4f}")

    print("  Per-Class F1 (shared classes):")
    for cls in shared_classes:
        f4 = report_4[cls]["f1-score"]
        f5 = report_5[cls]["f1-score"]
        diff = f4 - f5
        #arrow indicates whether the 4-class model improved or worsened for this class
        arrow = "↑" if diff > 0.005 else ("↓" if diff < -0.005 else "~")
        print(f"    {cls:<30} {f4:>12.4f} {f5:>12.4f}  {arrow} {abs(diff):.4f}")

    print(f"  {'glass (5-class only)':<32} {'—':>12} "
          f"{report_5['glass']['f1-score']:>12.4f}")


def compare():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Compare] Using device: {device}")

    #verify both model checkpoints exist before proceeding
    for path, name in [(FIVE_CLASS_MODEL_PATH, "5-class"), (FOUR_CLASS_MODEL_PATH, "4-class")]:
        if not os.path.exists(path):
            print(f"[Compare] ERROR: {name} model not found at {path}")
            print(f"  → Run the corresponding train script first.")
            return

    print("\n[Compare] Evaluating 5-class model...")
    _, _, test_loader_5, class_names_5 = get_dataloaders(num_workers=0)
    model_5 = build_model(num_classes=5, freeze_backbone=False)
    model_5.load_state_dict(torch.load(FIVE_CLASS_MODEL_PATH, map_location=device))
    model_5.to(device)
    y_true_5, y_pred_5 = get_predictions(model_5, test_loader_5, device)
    acc_5    = accuracy_score(y_true_5, y_pred_5) * 100
    report_5 = classification_report(y_true_5, y_pred_5,
                                     target_names=class_names_5, output_dict=True)

    print("[Compare] Evaluating 4-class model...")
    _, _, test_loader_4, class_names_4 = get_four_class_dataloaders(num_workers=0)
    model_4 = build_model(num_classes=4, freeze_backbone=False)
    model_4.load_state_dict(torch.load(FOUR_CLASS_MODEL_PATH, map_location=device))
    model_4.to(device)
    y_true_4, y_pred_4 = get_predictions(model_4, test_loader_4, device)
    acc_4    = accuracy_score(y_true_4, y_pred_4) * 100
    report_4 = classification_report(y_true_4, y_pred_4,
                                     target_names=class_names_4, output_dict=True)

    #glass is excluded from shared classes since it only exists in the 5-class model
    shared_classes = [c for c in class_names_5 if c != "glass"]

    print_comparison_table(acc_4, acc_5, report_4, report_5, shared_classes)
    save_comparison_chart(report_4, report_5, shared_classes, COMPARISON_SAVE_DIR)

    print("[Compare]  Comparison complete.")


if __name__ == "__main__":
    compare()