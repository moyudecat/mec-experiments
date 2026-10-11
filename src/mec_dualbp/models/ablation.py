"""Exp-3 预注册消融模型头；本模块不包含训练入口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


def _backbone(input_dim: int, hidden_layers: Sequence[int]) -> tuple[nn.Sequential, int]:
    if input_dim <= 0 or not hidden_layers:
        raise ValueError("input_dim and hidden_layers must be positive")
    layers: list[nn.Module] = []
    width = input_dim
    for hidden in hidden_layers:
        if hidden <= 0:
            raise ValueError("hidden layer width must be positive")
        layers.extend((nn.Linear(width, hidden), nn.ReLU()))
        width = hidden
    return nn.Sequential(*layers), width


class UnifiedCenterRadiusMLP(nn.Module):
    """一个共享 trunk、四个原始输出的 Unified-BP。"""

    def __init__(self, input_dim: int, hidden_layers: Sequence[int],
                 time_center_scale: float, energy_center_scale: float) -> None:
        super().__init__()
        if time_center_scale <= 0 or energy_center_scale <= 0:
            raise ValueError("center scales must be positive")
        self.backbone, width = _backbone(input_dim, hidden_layers)
        self.output = nn.Linear(width, 4)
        self.register_buffer("time_center_scale", torch.tensor(float(time_center_scale), dtype=torch.float32))
        self.register_buffer("energy_center_scale", torch.tensor(float(energy_center_scale), dtype=torch.float32))

    @staticmethod
    def _pair(raw_center: Tensor, raw_radius: Tensor, scale: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        center = scale * torch.nn.functional.softplus(raw_center)
        radius = center * torch.sigmoid(raw_radius)
        return center, radius, center - radius, center + radius

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        raw = self.output(self.backbone(inputs))
        tc, tr, tl, tu = self._pair(raw[:, 0], raw[:, 1], self.time_center_scale)
        ec, er, el, eu = self._pair(raw[:, 2], raw[:, 3], self.energy_center_scale)
        return {"time_center": tc, "time_radius": tr, "time_lower": tl, "time_upper": tu,
                "energy_center": ec, "energy_radius": er, "energy_lower": el, "energy_upper": eu}


class DirectBoundsMLP(nn.Module):
    """直接输出非负 lower/upper；刻意不保证顺序，也不做事后修复。"""

    def __init__(self, input_dim: int, hidden_layers: Sequence[int], center_scale: float) -> None:
        super().__init__()
        if center_scale <= 0:
            raise ValueError("center_scale must be positive")
        self.backbone, width = _backbone(input_dim, hidden_layers)
        self.output = nn.Linear(width, 2)
        self.register_buffer("center_scale", torch.tensor(float(center_scale), dtype=torch.float32))

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        raw = self.output(self.backbone(inputs))
        lower = self.center_scale * torch.nn.functional.softplus(raw[:, 0])
        upper = self.center_scale * torch.nn.functional.softplus(raw[:, 1])
        return lower, upper


@dataclass(frozen=True, slots=True)
class AblationLossParts:
    total: Tensor
    fit: Tensor
    center_radius: Tensor
    bounds: Tensor
    safety: Tensor
    order: Tensor


def interval_loss(lower_hat: Tensor, upper_hat: Tensor, lower_true: Tensor, upper_true: Tensor,
                  center_scale: float, *, lambda_safety: float, lambda_order: float = 0.0,
                  epsilon: float = 1e-12) -> AblationLossParts:
    """Raw 同构损失；允许 safety=0，并为 Direct-Bounds 加显式 order penalty。"""
    if center_scale <= 0 or lambda_safety < 0 or lambda_order < 0:
        raise ValueError("invalid scale or loss weight")
    scale = float(center_scale) + float(epsilon)
    center_hat = (lower_hat + upper_hat) * 0.5
    radius_hat = (upper_hat - lower_hat) * 0.5
    center_true = (lower_true + upper_true) * 0.5
    radius_true = (upper_true - lower_true) * 0.5
    smooth = torch.nn.functional.smooth_l1_loss
    cr = smooth(center_hat / scale, center_true / scale) + smooth(radius_hat / scale, radius_true / scale)
    bounds = smooth(lower_hat / scale, lower_true / scale) + smooth(upper_hat / scale, upper_true / scale)
    fit = cr + 0.5 * bounds
    safety = (torch.relu(upper_true - upper_hat) / scale).mean() + (torch.relu(lower_hat - lower_true) / scale).mean()
    order = (torch.relu(lower_hat - upper_hat) / scale).mean()
    total = fit + float(lambda_safety) * safety + float(lambda_order) * order
    return AblationLossParts(total, fit, cr, bounds, safety, order)

