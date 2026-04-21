# test_model.py
# Hema Sravani Koshtu
# April 20, 2026
# purpose: unit tests for model architecture, config values, and preprocessing pipeline

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np
import pytest

from config import NUM_CLASSES, CLASS_NAMES, IMG_SIZE, CONFIDENCE_THRESHOLD
from model import build_model


class TestModelArchitecture:

    def setup_method(self):
        #builds a fresh model before each test with backbone frozen
        self.model = build_model(num_classes=NUM_CLASSES, freeze_backbone=True)
        self.model.eval()

    def test_output_shape(self):
        #checks that the model outputs the correct shape for a batch of 4 images
        dummy = torch.randn(4, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            output = self.model(dummy)
        assert output.shape == (4, NUM_CLASSES), \
            f"Expected (4, {NUM_CLASSES}), got {output.shape}"

    def test_single_image_output(self):
        #checks that the model works with batch size 1
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            output = self.model(dummy)
        assert output.shape == (1, NUM_CLASSES)

    def test_softmax_sums_to_one(self):
        #verifies that softmax probabilities sum to 1 for each sample
        dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            output = self.model(dummy)
            probs  = torch.softmax(output, dim=1)
        sums = probs.sum(dim=1).numpy()
        np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-5)

    def test_trainable_params_frozen(self):
        #confirms backbone feature layers are frozen when freeze_backbone=true
        for name, param in self.model.named_parameters():
            if "features" in name:
                assert not param.requires_grad, \
                    f"Expected {name} to be frozen."

    def test_classifier_head_trainable(self):
        #confirms the custom classifier head is always trainable
        for name, param in self.model.named_parameters():
            if "classifier" in name:
                assert param.requires_grad, \
                    f"Expected {name} to be trainable."


class TestConfig:

    def test_class_count(self):
        assert NUM_CLASSES == 5, "Proposal specifies 5 classes."

    def test_class_names(self):
        expected = {"glass", "metal", "organic", "paper", "plastic"}
        assert set(CLASS_NAMES) == expected

    def test_confidence_threshold(self):
        #proposal specifies 50% as the warning threshold for low confidence
        assert CONFIDENCE_THRESHOLD == 0.50, \
            "Proposal specifies 50% as the warning threshold."

    def test_img_size(self):
        assert IMG_SIZE == 224, "Expected 224x224 input."


class TestPreprocessing:

    def test_transform_output_shape(self):
        #checks that the inference transform produces the correct tensor shape
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
        #checks that pixel values are normalized and not in the 0-255 range
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