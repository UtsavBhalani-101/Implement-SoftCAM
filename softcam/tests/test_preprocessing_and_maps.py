"""CPU tests for preprocessing shape consistency and evidence maps."""

from pathlib import Path
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from explainability.softcam_maps import (
    normalize_maps,
    predicted_class_maps,
    select_class_maps,
    upsample_maps,
)


class TinySoftCAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(1, 4, 3, padding=1)
        self.evidence = nn.Conv2d(4, 2, 1)

    def forward(self, images):
        evidence = self.evidence(self.features(images))
        return evidence.mean(dim=(2, 3)), evidence


class PreprocessingAndMapTests(unittest.TestCase):
    def test_class_selection_and_upsampling(self):
        maps = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
        selected = select_class_maps(maps, [1, 2])
        self.assertEqual(tuple(selected.shape), (2, 1, 4, 4))
        upsampled = upsample_maps(selected, (16, 16))
        self.assertEqual(tuple(upsampled.shape), (2, 1, 16, 16))

    def test_normalization_is_per_sample(self):
        maps = torch.tensor([[[[2.0, 4.0], [6.0, 8.0]]]])
        normalized = normalize_maps(maps)
        self.assertAlmostEqual(normalized.min().item(), 0.0)
        self.assertAlmostEqual(normalized.max().item(), 1.0)

    def test_predicted_maps_are_input_sized_and_finite(self):
        torch.manual_seed(0)
        model = TinySoftCAM()
        images = torch.randn(2, 1, 12, 12)
        logits, classes, maps = predicted_class_maps(model, images, (32, 32))
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(tuple(classes.shape), (2,))
        self.assertEqual(tuple(maps.shape), (2, 1, 32, 32))
        self.assertTrue(torch.isfinite(maps).all().item())
        self.assertGreaterEqual(maps.min().item(), 0.0)
        self.assertLessEqual(maps.max().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
