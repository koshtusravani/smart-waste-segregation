"""
tests/test_model.py
-------------------
Unit tests for the Smart Waste Segregation system.
Run with: python -m pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np
import pytest

from config import NUM_CLASSES, CLASS_NAMES, IMG_SIZE, CONFIDENCE_THRESHOLD
from model import build_model


class TestModelArchitecture:
    """Tests for model structure and output dimensions."""

    def setup_method(self):
        self.model = build_model(num_classes=NUM_CLASSES, freeze_backbone=True)
        self.model.eval()

    def test_output_shape(self):
        """Model should output (batch, NUM_CLASSES) tensor."""
        dummy = torch.randn(4, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            output = self.model(dummy)
        assert output.shape == (4, NUM_CLASSES), \
            f"Expected (4, {NUM_CLASSES}), got {output.shape}"

    def test_single_image_output(self):
        """Model should work with batch size 1."""
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            output = self.model(dummy)
        assert output.shape == (1, NUM_CLASSES)

    def test_softmax_sums_to_one(self):
        """Softmax probabilities should sum to ~1."""
        dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            output = self.model(dummy)
            probs  = torch.softmax(output, dim=1)
        sums = probs.sum(dim=1).numpy()
        np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-5)

    def test_trainable_params_frozen(self):
        """Backbone should be frozen when freeze_backbone=True."""
        for name, param in self.model.named_parameters():
            if "features" in name:
                assert not param.requires_grad, \
                    f"Expected {name} to be frozen."

    def test_classifier_head_trainable(self):
        """Classifier head should always be trainable."""
        for name, param in self.model.named_parameters():
            if "classifier" in name:
                assert param.requires_grad, \
                    f"Expected {name} to be trainable."


class TestConfig:
    """Tests for configuration consistency."""

    def test_class_count(self):
        assert NUM_CLASSES == 5, "Proposal specifies 5 classes."

    def test_class_names(self):
        expected = {"glass", "metal", "organic", "paper", "plastic"}
        assert set(CLASS_NAMES) == expected

    def test_confidence_threshold(self):
        assert CONFIDENCE_THRESHOLD == 0.50, \
            "Proposal specifies 50% as the warning threshold."

    def test_img_size(self):
        assert IMG_SIZE == 224, "Expected 224x224 input."


class TestPreprocessing:
    """Tests for image preprocessing pipeline."""

    def test_transform_output_shape(self):
        from dataset import get_inference_transform
        from PIL import Image
        import torchvision.transforms.functional as F

        transform = get_inference_transform()
        dummy_img = Image.fromarray(
            np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        )
        tensor = transform(dummy_img)
        assert tensor.shape == (3, IMG_SIZE, IMG_SIZE), \
            f"Expected (3, {IMG_SIZE}, {IMG_SIZE}), got {tensor.shape}"

    def test_transform_normalized(self):
        """Values should be normalized (not in 0-255 range)."""
        from dataset import get_inference_transform
        from PIL import Image

        transform = get_inference_transform()
        dummy_img = Image.fromarray(
            np.ones((224, 224, 3), dtype=np.uint8) * 128
        )
        tensor = transform(dummy_img)
        assert tensor.min() < 1.0 and tensor.max() < 2.0, \
            "Tensor should be normalized, not in 0-255 range."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])