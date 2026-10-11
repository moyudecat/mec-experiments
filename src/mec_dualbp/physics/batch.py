"""Vectorized float64 physical responses for Formal Dense-LHS evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from mec_dualbp.inner_ga.tasks import EdgeTaskParameters
else:
    EdgeTaskParameters = Any


def _validated_disturbances(disturbances: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(disturbances, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("disturbances must have shape (n, 6) in D_in,D_out,C,I_u,I_d,F_L order")
    if not np.all(np.isfinite(values)):
        raise ValueError("disturbances must be finite")
    if np.any(values < -1.0) or np.any(values > 1.0):
        raise ValueError("disturbances must lie in [-1, 1]")
    return values


def _recover(center: float, radius: float, z: NDArray[np.float64]) -> NDArray[np.float64]:
    return center + radius * z


def _rate(bandwidth: float, power: float, gain: float, noise: float, interference: NDArray[np.float64]) -> NDArray[np.float64]:
    denominator = noise + interference
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise ValueError("noise plus interference must be finite and positive")
    rate = bandwidth * np.log1p(power * gain / denominator) / np.log(2.0)
    if np.any(~np.isfinite(rate)) or np.any(rate <= 0.0):
        raise ArithmeticError("batch Shannon rates must be finite and positive")
    return rate


def _chunks(count: int, chunk_size: int | None):
    if chunk_size is None:
        chunk_size = max(count, 1)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer or None")
    for start in range(0, count, chunk_size):
        yield slice(start, min(start + chunk_size, count))


def evaluate_edge_core_latency_batch_si(
    disturbances: ArrayLike,
    task_parameters: EdgeTaskParameters,
    *,
    chunk_size: int | None = None,
) -> NDArray[np.float64]:
    """Evaluate ``D_in/R_u + C/f + D_out/R_d`` without any cache term."""

    z = _validated_disturbances(disturbances)
    result = np.empty(z.shape[0], dtype=np.float64)
    p = task_parameters
    for section in _chunks(z.shape[0], chunk_size):
        block = z[section]
        d_in = _recover(p.input_size_bit.center, p.input_size_bit.radius, block[:, 0])
        d_out = _recover(p.output_size_bit.center, p.output_size_bit.radius, block[:, 1])
        cycles = _recover(p.cpu_cycles.center, p.cpu_cycles.radius, block[:, 2])
        i_u = _recover(p.uplink_interference_w.center, p.uplink_interference_w.radius, block[:, 3])
        i_d = _recover(p.downlink_interference_w.center, p.downlink_interference_w.radius, block[:, 4])
        r_u = _rate(p.uplink_bandwidth_hz, p.uplink_terminal_tx_power_w, p.uplink_channel_gain_linear, p.noise_power_w, i_u)
        r_d = _rate(p.downlink_bandwidth_hz, p.downlink_bs_tx_power_w, p.downlink_channel_gain_linear, p.noise_power_w, i_d)
        response = d_in / r_u + cycles / p.edge_cpu_frequency_cycle_per_s + d_out / r_d
        if np.any(~np.isfinite(response)) or np.any(response < 0.0):
            raise ArithmeticError("batch edge core latency must be finite and nonnegative")
        result[section] = response
    return result


def evaluate_edge_comm_energy_batch_si(
    disturbances: ArrayLike,
    task_parameters: EdgeTaskParameters,
    *,
    chunk_size: int | None = None,
) -> NDArray[np.float64]:
    """Evaluate terminal ``P_u D_in/R_u + P_r D_out/R_d`` in joules."""

    z = _validated_disturbances(disturbances)
    result = np.empty(z.shape[0], dtype=np.float64)
    p = task_parameters
    for section in _chunks(z.shape[0], chunk_size):
        block = z[section]
        d_in = _recover(p.input_size_bit.center, p.input_size_bit.radius, block[:, 0])
        d_out = _recover(p.output_size_bit.center, p.output_size_bit.radius, block[:, 1])
        i_u = _recover(p.uplink_interference_w.center, p.uplink_interference_w.radius, block[:, 3])
        i_d = _recover(p.downlink_interference_w.center, p.downlink_interference_w.radius, block[:, 4])
        r_u = _rate(p.uplink_bandwidth_hz, p.uplink_terminal_tx_power_w, p.uplink_channel_gain_linear, p.noise_power_w, i_u)
        r_d = _rate(p.downlink_bandwidth_hz, p.downlink_bs_tx_power_w, p.downlink_channel_gain_linear, p.noise_power_w, i_d)
        response = p.uplink_terminal_tx_power_w * d_in / r_u + p.terminal_rx_power_w * d_out / r_d
        if np.any(~np.isfinite(response)) or np.any(response < 0.0):
            raise ArithmeticError("batch terminal communication energy must be finite and nonnegative")
        result[section] = response
    return result
