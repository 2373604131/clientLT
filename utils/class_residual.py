"""Lightweight class-residual carrier used by online SCA."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


class ClassResidualHead(nn.Module):
    """A zero-initialized feature-conditioned class residual table."""

    def __init__(self, num_classes, feature_dim, scale=10.0, clamp=3.0, use_bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_classes, feature_dim))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(num_classes))
        else:
            self.register_parameter("bias", None)
        self.scale = float(scale)
        self.clamp = float(clamp)
        self.register_buffer("active_mask", torch.zeros(num_classes, dtype=torch.bool))

    def set_active_classes(self, class_ids):
        mask = torch.zeros_like(self.active_mask)
        if class_ids:
            ids = torch.as_tensor(list(class_ids), dtype=torch.long, device=mask.device)
            if int(ids.min()) < 0 or int(ids.max()) >= mask.numel():
                raise IndexError("ClassResidualHead received an invalid class id")
            mask[ids] = True
        self.active_mask.copy_(mask)

    def forward(self, normalized_image_features):
        residual = self.scale * F.linear(
            normalized_image_features.detach().float(),
            self.weight.float(),
            self.bias.float() if self.bias is not None else None,
        )
        if self.clamp > 0:
            residual = residual.clamp(-self.clamp, self.clamp)
        mask = self.active_mask.to(residual.device, residual.dtype).view(1, -1)
        return residual * mask


def set_class_residual_active_classes(model, class_ids):
    core = unwrap_model(model)
    residual = getattr(core, "class_residual", None)
    if residual is None:
        raise AttributeError("ClipLora model has no class residual head")
    residual.set_active_classes(class_ids)


def mask_class_residual_gradients(model, labels):
    """Keep only active rows whose positive labels occur in this minibatch."""
    core = unwrap_model(model)
    residual = getattr(core, "class_residual", None)
    if residual is None:
        return
    row_mask = torch.zeros_like(residual.active_mask)
    row_mask[labels.unique()] = True
    row_mask.logical_and_(residual.active_mask)
    if residual.weight.grad is not None:
        residual.weight.grad.mul_(row_mask.to(residual.weight.grad.dtype).view(-1, 1))
    if residual.bias is not None and residual.bias.grad is not None:
        residual.bias.grad.mul_(row_mask.to(residual.bias.grad.dtype))
