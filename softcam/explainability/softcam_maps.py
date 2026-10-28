"""Evidence-map utilities for the SoftCAM model output."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def unpack_output(output):
    """Return logits and evidence maps from the repository model output."""
    if isinstance(output, dict):
        return output["logits"], output["evidence_maps"]
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        return output[0], output[1]
    raise TypeError("Expected (logits, evidence_maps) or a matching dictionary")


def select_class_maps(evidence_maps: torch.Tensor, class_indices=None) -> torch.Tensor:
    """Select one class map per sample, returning [B, 1, H, W]."""
    if evidence_maps.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W], got {tuple(evidence_maps.shape)}")
    batch = evidence_maps.shape[0]
    if class_indices is None:
        raise ValueError("class_indices must be provided for class-specific maps")
    indices = torch.as_tensor(class_indices, device=evidence_maps.device)
    if indices.ndim == 0:
        indices = indices.repeat(batch)
    if tuple(indices.shape) != (batch,):
        raise ValueError(f"Expected {batch} class indices, got {tuple(indices.shape)}")
    batch_indices = torch.arange(batch, device=evidence_maps.device)
    return evidence_maps[batch_indices, indices].unsqueeze(1)


def upsample_maps(maps: torch.Tensor, output_size) -> torch.Tensor:
    """Upsample maps to (height, width) using bilinear interpolation."""
    if maps.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W], got {tuple(maps.shape)}")
    return F.interpolate(maps, size=output_size, mode="bilinear", align_corners=False)


def normalize_maps(maps: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Min-max normalize each map independently to [0, 1]."""
    if maps.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W], got {tuple(maps.shape)}")
    flat = maps.flatten(2)
    low = flat.amin(dim=2).view(-1, 1, 1, 1)
    high = flat.amax(dim=2).view(-1, 1, 1, 1)
    return (maps - low) / (high - low).clamp_min(eps)


@torch.no_grad()
def predicted_class_maps(model, images, output_size=None):
    """Generate normalized maps for each sample's predicted class."""
    was_training = model.training
    model.eval()
    logits, evidence_maps = unpack_output(model(images))
    classes = logits.argmax(dim=1)
    maps = select_class_maps(evidence_maps, classes)
    if output_size is not None:
        maps = upsample_maps(maps, output_size)
    maps = normalize_maps(maps)
    if was_training:
        model.train()
    return logits, classes, maps
