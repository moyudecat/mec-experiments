"""Improved DualBP V1: frozen physics-baseline residual interval MLP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from mec_dualbp.datasets.protocol import ENERGY_FEATURES, TIME_FEATURES
from mec_dualbp.physics.channel import downlink_rate_bit_per_s, uplink_rate_bit_per_s
from mec_dualbp.physics.energy import edge_comm_energy_point_j
from mec_dualbp.physics.latency import edge_core_latency_point_s

DESIGN_HASH = "a5c364caca08b7f800ddde6e49a9b362f3e7585855e9f4af77c5ee74061110ba"
HIDDEN_LAYERS = (128, 64, 32)
TRAINING_SEEDS = (31001, 31002, 31003, 31004, 31005)
LAMBDA_SAFETY = 0.25
TIME_DELTA_CENTER_SCALE = 0.001984608838275729
TIME_RADIUS_SCALE = 0.4359502024619836
ENERGY_DELTA_CENTER_SCALE = 0.0004869135714533095
ENERGY_RADIUS_SCALE = 0.015859765350198552
RECONSTRUCTION_EPSILON = 1e-12


def _centers(values: NDArray[np.float64], fields: Sequence[str], stem: str) -> NDArray[np.float64]:
    index = {name: i for i, name in enumerate(fields)}
    return (values[:, index[f"{stem}_lower"]] + values[:, index[f"{stem}_upper"]]) * 0.5


def time_physical_baseline_s(values: np.ndarray, fields: Sequence[str] = TIME_FEATURES) -> NDArray[np.float64]:
    """Evaluate the frozen z=0 edge-core latency point response; no GA/LHS."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(fields) or tuple(fields) != tuple(TIME_FEATURES):
        raise ValueError("frozen Time feature matrix required")
    idx = {name: i for i, name in enumerate(fields)}
    d_in, d_out = _centers(x, fields, "D_in_bit"), _centers(x, fields, "D_out_bit")
    cycles = _centers(x, fields, "C_cycle")
    i_u, i_d = _centers(x, fields, "I_u_w"), _centers(x, fields, "I_d_w")
    result = np.empty(len(x), dtype=np.float64)
    for n, row in enumerate(x):
        ru = uplink_rate_bit_per_s(row[idx["uplink_bandwidth_hz"]], row[idx["uplink_terminal_tx_power_w"]],
            row[idx["uplink_channel_gain_linear"]], row[idx["noise_power_w"]], i_u[n])
        rd = downlink_rate_bit_per_s(row[idx["downlink_bandwidth_hz"]], row[idx["downlink_bs_tx_power_w"]],
            row[idx["downlink_channel_gain_linear"]], row[idx["noise_power_w"]], i_d[n])
        result[n] = edge_core_latency_point_s(d_in[n], ru, cycles[n],
            row[idx["edge_cpu_frequency_cycle_per_s"]], d_out[n], rd)
    return result


def energy_physical_baseline_j(values: np.ndarray, fields: Sequence[str] = ENERGY_FEATURES) -> NDArray[np.float64]:
    """Evaluate the frozen z=0 terminal communication energy point response."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(fields) or tuple(fields) != tuple(ENERGY_FEATURES):
        raise ValueError("frozen Energy feature matrix required")
    idx = {name: i for i, name in enumerate(fields)}
    d_in, d_out = _centers(x, fields, "D_in_bit"), _centers(x, fields, "D_out_bit")
    i_u, i_d = _centers(x, fields, "I_u_w"), _centers(x, fields, "I_d_w")
    result = np.empty(len(x), dtype=np.float64)
    for n, row in enumerate(x):
        ru = uplink_rate_bit_per_s(row[idx["uplink_bandwidth_hz"]], row[idx["uplink_terminal_tx_power_w"]],
            row[idx["uplink_channel_gain_linear"]], row[idx["noise_power_w"]], i_u[n])
        # P_d belongs to the downlink rate; P_r belongs only to terminal energy.
        rd = downlink_rate_bit_per_s(row[idx["downlink_bandwidth_hz"]], row[idx["downlink_bs_tx_power_w"]],
            row[idx["downlink_channel_gain_linear"]], row[idx["noise_power_w"]], i_d[n])
        result[n] = edge_comm_energy_point_j(row[idx["uplink_terminal_tx_power_w"]], d_in[n], ru,
            row[idx["terminal_rx_power_w"]], d_out[n], rd)
    return result


class ImprovedResidualDualBP(nn.Module):
    """M3 ReLU trunk with signed center residual and legal radius-ratio head."""

    def __init__(self, input_dim: int, delta_center_scale: float) -> None:
        super().__init__()
        if input_dim <= 0 or delta_center_scale <= 0:
            raise ValueError("positive input_dim and delta_center_scale required")
        layers: list[nn.Module] = []
        width = input_dim
        for hidden in HIDDEN_LAYERS:
            layers.extend((nn.Linear(width, hidden), nn.ReLU()))
            width = hidden
        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(width, 2)
        self.register_buffer("delta_center_scale", torch.tensor(delta_center_scale, dtype=torch.float32))

    def forward(self, scaled_inputs: Tensor, baseline: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if scaled_inputs.ndim != 2 or baseline.ndim != 1 or len(scaled_inputs) != len(baseline):
            raise ValueError("aligned feature matrix and baseline vector required")
        raw = self.output(self.backbone(scaled_inputs))
        center = torch.clamp_min(baseline + self.delta_center_scale * raw[:, 0], RECONSTRUCTION_EPSILON)
        radius = center * torch.sigmoid(raw[:, 1])
        return center, radius, center - radius, center + radius


@dataclass(frozen=True, slots=True)
class ImprovedLossParts:
    total: Tensor
    fit: Tensor
    center_residual: Tensor
    radius: Tensor
    bounds: Tensor
    safety: Tensor


def improved_dualbp_loss(center_hat: Tensor, radius_hat: Tensor, lower_hat: Tensor, upper_hat: Tensor,
    baseline: Tensor, lower_true: Tensor, upper_true: Tensor, *, delta_center_scale: float,
    radius_scale: float, response_center_scale: float, lambda_safety: float = LAMBDA_SAFETY) -> ImprovedLossParts:
    """Frozen SmoothL1 family with independent Train-only residual scales."""
    if min(delta_center_scale, radius_scale, response_center_scale, lambda_safety) <= 0:
        raise ValueError("all scales and lambda_safety must be positive")
    center_true = (lower_true + upper_true) * 0.5
    radius_true = (upper_true - lower_true) * 0.5
    smooth = torch.nn.functional.smooth_l1_loss
    dc = smooth((center_hat - baseline) / delta_center_scale,
                (center_true - baseline) / delta_center_scale)
    radius = smooth(radius_hat / radius_scale, radius_true / radius_scale)
    bounds = smooth(lower_hat / response_center_scale, lower_true / response_center_scale) + \
        smooth(upper_hat / response_center_scale, upper_true / response_center_scale)
    fit = dc + radius + 0.5 * bounds
    upper_under = torch.relu(upper_true - upper_hat) / response_center_scale
    lower_over = torch.relu(lower_hat - lower_true) / response_center_scale
    safety = upper_under.mean() + lower_over.mean()
    return ImprovedLossParts(fit + lambda_safety * safety, fit, dc, radius, bounds, safety)


def parameter_count(input_dim: int) -> int:
    widths = (input_dim, *HIDDEN_LAYERS, 2)
    return sum((left + 1) * right for left, right in zip(widths, widths[1:]))
