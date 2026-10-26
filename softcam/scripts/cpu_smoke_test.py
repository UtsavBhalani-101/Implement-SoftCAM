"""Fast CPU-only smoke test for the official SoftCAM classification head."""

from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.models import FullyConvModel


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
        self.feature_map = nn.Conv2d(8, 16, 3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 2)


class Config:
    class train:
        network = "resnet50"


def main():
    torch.manual_seed(0)
    model = FullyConvModel(Config(), TinyBackbone(), num_classes=2)
    model.train()

    images = torch.randn(2, 3, 32, 32)
    targets = torch.tensor([0, 1])
    logits, evidence_maps = model(images)

    assert logits.shape == (2, 2), logits.shape
    assert evidence_maps.shape == (2, 2, 32, 32), evidence_maps.shape

    loss = (nn.CrossEntropyLoss()(logits, targets)
            + 1e-4 * evidence_maps.abs().mean()
            + 1e-4 * evidence_maps.square().mean())
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss), loss
    print("CPU smoke test passed")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"evidence map shape: {tuple(evidence_maps.shape)}")
    print(f"loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()
