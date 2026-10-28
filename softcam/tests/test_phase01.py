"""CPU tests for repository stabilization and the core SoftCAM mechanism."""

from pathlib import Path
import sys
import unittest

import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.models import FullyConvModel


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU())
        self.feature_map = nn.Conv2d(4, 8, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 2)


class ModelConfig:
    class train:
        network = "resnet50"


class Phase01Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)

    def test_default_config_has_safe_test_flag(self):
        with open(ROOT / "configs" / "default.yaml", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        self.assertFalse(config["base"]["test"])

    def test_softcam_shapes_and_backward(self):
        model = FullyConvModel(ModelConfig(), TinyBackbone(), num_classes=2)
        images = torch.randn(2, 1, 16, 16)
        labels = torch.tensor([0, 1])

        logits, maps = model(images)
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(tuple(maps.shape), (2, 2, 16, 16))

        loss = (nn.CrossEntropyLoss()(logits, labels)
                + 1e-4 * maps.abs().mean()
                + 1e-4 * maps.square().mean())
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIsNotNone(model.classifier.weight.grad)

    def test_model_can_fit_tiny_synthetic_problem(self):
        model = FullyConvModel(ModelConfig(), TinyBackbone(), num_classes=2)
        images = torch.cat([
            torch.zeros(4, 1, 16, 16),
            torch.ones(4, 1, 16, 16),
        ])
        labels = torch.tensor([0] * 4 + [1] * 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            initial = criterion(model(images)[0], labels).item()

        for _ in range(30):
            logits, maps = model(images)
            loss = (criterion(logits, labels)
                    + 1e-4 * maps.abs().mean()
                    + 1e-4 * maps.square().mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            final = criterion(model(images)[0], labels).item()
        self.assertLess(final, initial)


if __name__ == "__main__":
    unittest.main()
