# model.py
# Hema Sravani Koshtu
# April 20, 2026
# purpose: defines the mobilenetv2 architecture with a custom classification head

import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES


def build_model(num_classes=NUM_CLASSES, freeze_backbone=True):
    #loads pretrained mobilenetv2 and replaces the classifier head
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
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
    #loads a saved model checkpoint from disk
    model = build_model(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[Model] Loaded from {model_path}")
    return model


def count_parameters(model):
    #prints total and trainable parameter counts
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params:     {total:,}")
    print(f"[Model] Trainable params: {trainable:,}")