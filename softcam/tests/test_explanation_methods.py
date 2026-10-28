"""CPU tests for explanation metrics and post-hoc methods."""

from pathlib import Path
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.explanation_metrics import (
    activation_consistency,
    activation_precision,
    activation_sensitivity,
    deletion_audc,
    top_k_localization_precision,
)
from explainability.posthoc import (
    GradCAM,
    GuidedBackpropagation,
    IntegratedGradients,
    LayerCAM,
    ScoreCAM,
)


class TinyExplainableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.relu = nn.ReLU()
        self.classifier = nn.Conv2d(4, 2, 1)

    def forward(self, images):
        features = self.relu(self.conv(images))
        maps = self.classifier(features)
        return maps.mean(dim=(2, 3)), maps


class ExplanationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = TinyExplainableModel()
        self.images = torch.randn(1, 1, 8, 8)
        self.targets = torch.tensor([1])

    def test_localization_and_consistency_metrics(self):
        maps = torch.zeros(2, 1, 8, 8)
        maps[0, 0, 2:4, 2:4] = 1
        maps[1, 0, 2:4, 2:4] = -1
        masks = torch.zeros(2, 8, 8, dtype=torch.bool)
        masks[:, 2:4, 2:4] = True
        precision = activation_precision(maps, masks)
        sensitivity = activation_sensitivity(maps, masks)
        top1 = top_k_localization_precision(maps, masks, k=1, patch_size=4)
        disease, healthy = activation_consistency(maps[:, 0], torch.tensor([1, 0]))
        self.assertAlmostEqual(precision.item(), 0.5)
        self.assertAlmostEqual(sensitivity.item(), 0.5)
        self.assertAlmostEqual(top1.item(), 1.0)
        self.assertAlmostEqual(disease.item(), 0.0625)
        self.assertAlmostEqual(healthy.item(), 0.0625)

    def test_posthoc_methods_return_maps(self):
        methods = [
            GradCAM(self.model, self.model.conv),
            LayerCAM(self.model, self.model.conv),
            ScoreCAM(self.model, self.model.conv),
            IntegratedGradients(self.model, steps=4),
            GuidedBackpropagation(self.model),
        ]
        for method in methods:
            maps = method(self.images, self.targets)
            self.assertEqual(tuple(maps.shape), (1, 1, 8, 8))
            self.assertTrue(torch.isfinite(maps).all().item())

    def test_deletion_auc_runs(self):
        maps = torch.ones(1, 1, 8, 8)
        auc = deletion_audc(self.model, self.images, maps, self.targets, patch_size=4)
        self.assertEqual(tuple(auc.shape), (1,))
        self.assertTrue(torch.isfinite(auc).all().item())


if __name__ == "__main__":
    unittest.main()
