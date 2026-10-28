"""CPU-compatible post-hoc attribution methods for comparison with SoftCAM."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _logits(output):
    if isinstance(output, dict):
        return output["logits"]
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _normalize(maps, eps=1e-8):
    maps = maps.clamp_min(0)
    flat = maps.flatten(2)
    low = flat.amin(2).view(-1, 1, 1, 1)
    high = flat.amax(2).view(-1, 1, 1, 1)
    return (maps - low) / (high - low).clamp_min(eps)


class _ActivationCapture:
    def __init__(self, layer):
        self.value = None
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        self.value = output

    def close(self):
        self.handle.remove()


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def __call__(self, images, target_classes):
        activations = []
        gradients = []

        def forward_hook(_module, _inputs, output):
            activations.append(output)

        def backward_hook(_module, _grad_input, grad_output):
            gradients.append(grad_output[0])

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        try:
            self.model.zero_grad(set_to_none=True)
            logits = _logits(self.model(images))
            score = logits.gather(1, target_classes[:, None]).sum()
            score.backward()
            weights = gradients[0].mean(dim=(2, 3), keepdim=True)
            maps = (weights * activations[0]).sum(dim=1, keepdim=True).relu()
            return _normalize(maps)
        finally:
            h1.remove()
            h2.remove()


class LayerCAM(GradCAM):
    def __call__(self, images, target_classes):
        activations, gradients = [], []

        def forward_hook(_module, _inputs, output):
            activations.append(output)

        def backward_hook(_module, _grad_input, grad_output):
            gradients.append(grad_output[0])

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        try:
            self.model.zero_grad(set_to_none=True)
            logits = _logits(self.model(images))
            logits.gather(1, target_classes[:, None]).sum().backward()
            maps = (activations[0] * gradients[0].relu()).sum(dim=1, keepdim=True).relu()
            return _normalize(maps)
        finally:
            h1.remove()
            h2.remove()


class IntegratedGradients:
    def __init__(self, model, steps=32):
        self.model = model
        self.steps = steps

    def __call__(self, images, target_classes, baseline=None):
        baseline = torch.zeros_like(images) if baseline is None else baseline
        total_gradients = torch.zeros_like(images)
        for alpha in torch.linspace(0, 1, self.steps, device=images.device):
            point = (baseline + alpha * (images - baseline)).detach().requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            score = _logits(self.model(point)).gather(1, target_classes[:, None]).sum()
            score.backward()
            total_gradients += point.grad.detach()
        attribution = (images - baseline) * total_gradients / self.steps
        return _normalize(attribution.abs().sum(dim=1, keepdim=True))


class GuidedBackpropagation:
    def __init__(self, model):
        self.model = model

    def __call__(self, images, target_classes):
        point = images.detach().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        _logits(self.model(point)).gather(1, target_classes[:, None]).sum().backward()
        return _normalize(point.grad.detach().clamp_min(0).abs().sum(dim=1, keepdim=True))


class ScoreCAM:
    """Gradient-free Score-CAM using a selected convolutional feature layer."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def __call__(self, images, target_classes):
        capture = _ActivationCapture(self.target_layer)
        try:
            with torch.no_grad():
                base_logits = _logits(self.model(images))
                activations = capture.value
                maps = []
                for sample, target, feature in zip(images, target_classes, activations):
                    feature_maps = F.interpolate(
                        feature.unsqueeze(1), size=sample.shape[-2:], mode="bilinear", align_corners=False
                    )
                    feature_maps = _normalize(feature_maps)
                    masked = sample.unsqueeze(0) * feature_maps
                    repeated_logits = _logits(self.model(masked))[:, target]
                    weights = torch.softmax(repeated_logits, dim=0)
                    maps.append((weights[:, None, None] * feature_maps[:, 0]).sum(0, keepdim=True))
                return _normalize(torch.stack(maps))
        finally:
            capture.close()
