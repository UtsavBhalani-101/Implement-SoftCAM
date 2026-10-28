"""CPU tests for the baseline interface and SoftCAM variants."""

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.baseline import build_resnet50_baseline
from modules.regularization import evidence_penalty, variant_coefficients


class BaselineAndRegularizationTests(unittest.TestCase):
    def test_resnet_baseline_has_requested_logits(self):
        model = build_resnet50_baseline(num_classes=2, pretrained=False)
        model.eval()
        with torch.no_grad():
            logits = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(tuple(logits.shape), (1, 2))

    def test_dense_sparse_and_ridge_variants(self):
        maps = torch.ones(2, 2, 4, 4, requires_grad=True)
        self.assertEqual(variant_coefficients("dense", 0.1), (0.0, 0.0))
        self.assertEqual(variant_coefficients("sparse", 0.1), (0.1, 0.0))
        self.assertEqual(variant_coefficients("ridge", 0.1), (0.0, 0.1))

        dense = evidence_penalty(maps, 0.0, 0.0)
        sparse = evidence_penalty(maps, 0.1, 0.0)
        ridge = evidence_penalty(maps, 0.0, 0.1)
        self.assertEqual(dense.item(), 0.0)
        self.assertGreater(sparse.item(), 0.0)
        self.assertGreater(ridge.item(), 0.0)

    def test_penalty_backpropagates(self):
        maps = torch.randn(2, 2, 4, 4, requires_grad=True)
        loss = evidence_penalty(maps, 0.1, 0.1)
        loss.backward()
        self.assertIsNotNone(maps.grad)
        self.assertTrue(torch.isfinite(maps.grad).all().item())


if __name__ == "__main__":
    unittest.main()
