"""Metrics used to evaluate SoftCAM and post-hoc explanations."""

from __future__ import annotations

import torch


def _positive_maps(maps: torch.Tensor) -> torch.Tensor:
    if maps.ndim == 4 and maps.shape[1] == 1:
        maps = maps[:, 0]
    if maps.ndim != 3:
        raise ValueError(f"Expected [B, H, W] or [B, 1, H, W], got {tuple(maps.shape)}")
    return maps.clamp_min(0)


def activation_consistency(maps: torch.Tensor, disease_labels: torch.Tensor):
    """Return positive consistency for disease and negative consistency for healthy."""
    if maps.ndim == 4 and maps.shape[1] == 1:
        maps = maps[:, 0]
    labels = disease_labels.to(device=maps.device).bool()
    positive = (maps > 0).float().flatten(1).mean(1)
    negative = (maps < 0).float().flatten(1).mean(1)
    disease_score = positive[labels].mean() if labels.any() else maps.new_tensor(0.0)
    healthy = ~labels
    healthy_score = negative[healthy].mean() if healthy.any() else maps.new_tensor(0.0)
    return disease_score, healthy_score


def activation_precision(maps: torch.Tensor, masks: torch.Tensor, eps: float = 1e-8):
    """Fraction of positive activation mass inside expert masks."""
    scores = _positive_maps(maps)
    masks = masks.to(device=scores.device).bool()
    numerator = (scores * masks).flatten(1).sum(1)
    denominator = scores.flatten(1).sum(1).clamp_min(eps)
    return (numerator / denominator).mean()


def activation_sensitivity(maps: torch.Tensor, masks: torch.Tensor, eps: float = 1e-8):
    """Weighted positive activation captured inside expert masks."""
    scores = _positive_maps(maps)
    masks = masks.to(device=scores.device).float()
    numerator = (scores * masks).flatten(1).sum(1)
    denominator = masks.flatten(1).sum(1).clamp_min(eps)
    return (numerator / denominator).mean()


def top_k_localization_precision(
    maps: torch.Tensor, masks: torch.Tensor, k: int, patch_size: int = 33
):
    """Fraction of top-scoring non-overlapping patches touching the mask."""
    scores = _positive_maps(maps)
    masks = masks.to(device=scores.device).bool()
    batch, height, width = scores.shape
    values = []
    for index in range(batch):
        patches = []
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                y2, x2 = min(y + patch_size, height), min(x + patch_size, width)
                patches.append((scores[index, y:y2, x:x2].mean(), masks[index, y:y2, x:x2].any()))
        count = min(k, len(patches))
        ranked = sorted(patches, key=lambda item: float(item[0]), reverse=True)[:count]
        values.append(torch.stack([item[1].float() for item in ranked]).mean())
    return torch.stack(values).mean()


def _logits(model, images):
    output = model(images)
    if isinstance(output, dict):
        return output["logits"]
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


@torch.no_grad()
def deletion_audc(
    model,
    images: torch.Tensor,
    maps: torch.Tensor,
    target_classes: torch.Tensor,
    patch_size: int = 33,
    baseline: float = 0.0,
):
    """Compute Area Under the Deletion Curve; lower is more faithful."""
    model_was_training = model.training
    model.eval()
    scores = _positive_maps(maps)
    if scores.shape[-2:] != images.shape[-2:]:
        scores = torch.nn.functional.interpolate(
            scores.unsqueeze(1), size=images.shape[-2:], mode="bilinear", align_corners=False
        )[:, 0]
    with torch.no_grad():
        original = _logits(model, images).gather(1, target_classes[:, None]).squeeze(1)
    curves = []
    for index in range(images.shape[0]):
        current = images[index:index + 1].clone()
        patches = []
        height, width = scores.shape[-2:]
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                y2, x2 = min(y + patch_size, height), min(x + patch_size, width)
                patches.append((scores[index, y:y2, x:x2].mean(), y, y2, x, x2))
        patches.sort(key=lambda item: float(item[0]), reverse=True)
        drops = [images.new_tensor(0.0)]
        for _, y, y2, x, x2 in patches:
            current[:, :, y:y2, x:x2] = baseline
            with torch.no_grad():
                confidence = _logits(model, current)[0, target_classes[index]]
            drops.append((original[index] - confidence) / original[index].abs().clamp_min(1e-8))
        curves.append(torch.stack(drops).clamp_min(0))
    if model_was_training:
        model.train()
    return torch.stack([torch.trapezoid(curve, dx=1.0) / max(curve.numel() - 1, 1) for curve in curves])
