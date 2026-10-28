"""Standard classification baselines used against SoftCAM."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet50_baseline(num_classes: int, pretrained: bool = False) -> nn.Module:
    """Build the paper's ordinary ResNet-50 classification baseline."""
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
