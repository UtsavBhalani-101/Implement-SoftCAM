"""Regularization used by the SoftCAM evidence layer."""

from __future__ import annotations

import torch


def evidence_penalty(evidence_maps: torch.Tensor,
                     lambda_l1: float = 0.0,
                     lambda_l2: float = 0.0) -> torch.Tensor:
    """Return the repository-compatible ElasticNet evidence penalty.

    The existing training implementation uses the L1 and L2 norms of the
    evidence maps. Keeping this in one function makes dense, sparse, and ridge
    variants explicit and testable.
    """
    if evidence_maps.ndim != 4:
        raise ValueError(
            f"Expected evidence maps [B, C, H, W], got {tuple(evidence_maps.shape)}"
        )
    l1 = torch.norm(evidence_maps, p=1)
    l2 = torch.norm(evidence_maps, p=2)
    return lambda_l1 * l1 + lambda_l2 * l2


def variant_coefficients(name: str, strength: float) -> tuple[float, float]:
    """Return (lambda_l1, lambda_l2) for a named SoftCAM variant."""
    variants = {
        "dense": (0.0, 0.0),
        "sparse": (strength, 0.0),
        "ridge": (0.0, strength),
    }
    try:
        return variants[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown SoftCAM variant: {name}") from exc
