"""
model.py
--------
CNN architecture for waste classification.

Uses MobileNetV2 as a lightweight pretrained backbone (as the proposal
specifies a "light deep learning model") with a custom classification head.

MobileNetV2 is ideal here because:
  - It's fast on CPU (important since no GPU is available)
  - It's pretrained on ImageNet → transfer learning boosts accuracy on small datasets
  - It's small enough to run real-time inference on webcam frames
"""

import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES


def build_model(num_classes=NUM_CLASSES, freeze_backbone=True):
    """
    Builds a MobileNetV2-based classifier.

    Args:
        num_classes     : number of output classes (default 5)
        freeze_backbone : if True, freeze all conv layers and only train
                          the classifier head. Set False for fine-tuning.

    Returns:
        model (nn.Module)
    """
    # Load pretrained MobileNetV2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace the default classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )

    return model


def load_model(model_path, num_classes=NUM_CLASSES, device="cpu"):
    """
    Load a saved model from disk.

    Args:
        model_path  : path to .pth checkpoint
        num_classes : number of output classes
        device      : 'cpu' or 'cuda'

    Returns:
        model (nn.Module) in eval mode
    """
    model = build_model(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[Model] Loaded from {model_path}")
    return model


def count_parameters(model):
    """Print total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params:     {total:,}")
    print(f"[Model] Trainable params: {trainable:,}")


if __name__ == "__main__":
    model = build_model()
    count_parameters(model)
    print(model.classifier)