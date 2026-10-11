"""冻结的 DualBP 小型 MLP 与约束 center-radius 输出头。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

ARCHITECTURES: dict[str, tuple[int, ...]] = {
    "M1": (64, 32),
    "M2": (64, 64, 32),
    "M3": (128, 64, 32),
}
LAMBDA_SAFETY_CANDIDATES = (0.25, 0.50, 1.00)
TRAINING_SEEDS = (31001, 31002, 31003, 31004, 31005)


class ConstrainedCenterRadiusMLP(nn.Module):
    """ReLU MLP；对外仍输出 center/radius，rho 仅是内部参数化。"""

    def __init__(self, input_dim: int, hidden_layers: Sequence[int], center_scale: float) -> None:
        super().__init__()
        if input_dim <= 0 or not hidden_layers or center_scale <= 0:
            raise ValueError("input_dim, hidden_layers and center_scale must be positive")
        layers: list[nn.Module] = []
        width = input_dim
        for hidden in hidden_layers:
            if hidden <= 0:
                raise ValueError("hidden layer width must be positive")
            layers.extend((nn.Linear(width, hidden), nn.ReLU()))
            width = hidden
        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(width, 2)
        self.register_buffer("center_scale", torch.tensor(float(center_scale), dtype=torch.float32))

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        raw = self.output(self.backbone(inputs))
        center = self.center_scale * torch.nn.functional.softplus(raw[:, 0])
        rho = torch.sigmoid(raw[:, 1])
        radius = center * rho
        return center, radius, center - radius, center + radius


@dataclass(frozen=True, slots=True)
class LossParts:
    total: Tensor
    fit: Tensor
    center_radius: Tensor
    bounds: Tensor
    safety: Tensor


def dualbp_loss(
    center_hat: Tensor,
    radius_hat: Tensor,
    lower_hat: Tensor,
    upper_hat: Tensor,
    lower_true: Tensor,
    upper_true: Tensor,
    center_scale: float,
    lambda_safety: float,
    epsilon: float = 1e-12,
) -> LossParts:
    if center_scale <= 0 or lambda_safety <= 0:
        raise ValueError("center_scale and lambda_safety must be positive")
    scale = float(center_scale) + float(epsilon)
    center_true = (lower_true + upper_true) * 0.5
    radius_true = (upper_true - lower_true) * 0.5
    smooth = torch.nn.functional.smooth_l1_loss
    cr = smooth(center_hat / scale, center_true / scale) + smooth(radius_hat / scale, radius_true / scale)
    bounds = smooth(lower_hat / scale, lower_true / scale) + smooth(upper_hat / scale, upper_true / scale)
    fit = cr + 0.5 * bounds
    upper_under = torch.relu(upper_true - upper_hat) / scale
    lower_over = torch.relu(lower_hat - lower_true) / scale
    safety = upper_under.mean() + lower_over.mean()
    return LossParts(fit + float(lambda_safety) * safety, fit, cr, bounds, safety)


def parameter_count(input_dim: int, hidden_layers: Sequence[int]) -> int:
    widths = (input_dim, *hidden_layers, 2)
    return sum((left + 1) * right for left, right in zip(widths, widths[1:]))
