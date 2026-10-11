"""四类真实物理边界任务适配器。"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from mec_dualbp._validation import nonnegative_float, positive_float
from mec_dualbp.inner_ga.problem import Direction, InnerGAProblem
from mec_dualbp.physics.channel import downlink_rate_bit_per_s, uplink_rate_bit_per_s
from mec_dualbp.physics.energy import edge_comm_energy_point_j
from mec_dualbp.physics.latency import edge_core_latency_point_s
from mec_dualbp.uncertainty.interval_params import Interval
from mec_dualbp.uncertainty.schema import (
    EDGE_COMM_ENERGY_ACTIVE_MASK,
    EDGE_CORE_ACTIVE_MASK,
    DisturbanceIndex,
)


@dataclass(frozen=True, slots=True)
class EdgeTaskParameters:
    """四类边界任务共享的 SI 参数；区间顺序遵循冻结六维 schema。"""

    input_size_bit: Interval
    output_size_bit: Interval
    cpu_cycles: Interval
    uplink_interference_w: Interval
    downlink_interference_w: Interval
    uplink_bandwidth_hz: float
    downlink_bandwidth_hz: float
    uplink_terminal_tx_power_w: float
    downlink_bs_tx_power_w: float
    terminal_rx_power_w: float
    uplink_channel_gain_linear: float
    downlink_channel_gain_linear: float
    noise_power_w: float
    edge_cpu_frequency_cycle_per_s: float

    def __post_init__(self) -> None:
        """验证全部确定参数与区间定义域，不猜测工程单位。"""

        for name in (
            "input_size_bit",
            "output_size_bit",
            "cpu_cycles",
            "uplink_interference_w",
            "downlink_interference_w",
        ):
            interval = getattr(self, name)
            if not isinstance(interval, Interval):
                raise TypeError(f"{name} must be an Interval")
            if interval.lower < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        positive_names = (
            "uplink_bandwidth_hz",
            "downlink_bandwidth_hz",
            "uplink_channel_gain_linear",
            "downlink_channel_gain_linear",
            "noise_power_w",
            "edge_cpu_frequency_cycle_per_s",
        )
        nonnegative_names = (
            "uplink_terminal_tx_power_w",
            "downlink_bs_tx_power_w",
            "terminal_rx_power_w",
        )
        for name in positive_names:
            object.__setattr__(self, name, positive_float(name, getattr(self, name)))
        for name in nonnegative_names:
            object.__setattr__(self, name, nonnegative_float(name, getattr(self, name)))


def _recover(interval: Interval, z: NDArray[np.float64], index: DisturbanceIndex) -> float:
    """按冻结索引从区间恢复一个 SI 物理参数。"""

    return interval.recover(float(z[int(index)]))


def _build_edge_core_problem(
    parameters: EdgeTaskParameters, gamma: Real, direction: Direction
) -> InnerGAProblem:
    """构造不含缓存回源项的边缘核心时延问题。"""

    def evaluator(z: NDArray[np.float64]) -> float:
        """以真实 Shannon 速率和核心时延点评价六维可行扰动。"""

        d_in = _recover(parameters.input_size_bit, z, DisturbanceIndex.D_IN)
        d_out = _recover(parameters.output_size_bit, z, DisturbanceIndex.D_OUT)
        cycles = _recover(parameters.cpu_cycles, z, DisturbanceIndex.C)
        i_u = _recover(parameters.uplink_interference_w, z, DisturbanceIndex.I_U)
        i_d = _recover(parameters.downlink_interference_w, z, DisturbanceIndex.I_D)
        uplink_rate = uplink_rate_bit_per_s(
            parameters.uplink_bandwidth_hz,
            parameters.uplink_terminal_tx_power_w,
            parameters.uplink_channel_gain_linear,
            parameters.noise_power_w,
            i_u,
        )
        downlink_rate = downlink_rate_bit_per_s(
            parameters.downlink_bandwidth_hz,
            parameters.downlink_bs_tx_power_w,
            parameters.downlink_channel_gain_linear,
            parameters.noise_power_w,
            i_d,
        )
        return edge_core_latency_point_s(
            d_in,
            uplink_rate,
            cycles,
            parameters.edge_cpu_frequency_cycle_per_s,
            d_out,
            downlink_rate,
        )

    suffix = "lower" if direction == "minimize" else "upper"
    return InnerGAProblem(
        name=f"edge_core_latency_{suffix}",
        active_mask=EDGE_CORE_ACTIVE_MASK,
        gamma=float(gamma),
        direction=direction,
        evaluator=evaluator,
    )


def _build_edge_energy_problem(
    parameters: EdgeTaskParameters, gamma: Real, direction: Direction
) -> InnerGAProblem:
    """构造仅含终端上传发射与下载接收能耗的问题。"""

    def evaluator(z: NDArray[np.float64]) -> float:
        """以真实 Shannon 速率和终端 P_r 点评价值计算通信能耗。"""

        d_in = _recover(parameters.input_size_bit, z, DisturbanceIndex.D_IN)
        d_out = _recover(parameters.output_size_bit, z, DisturbanceIndex.D_OUT)
        i_u = _recover(parameters.uplink_interference_w, z, DisturbanceIndex.I_U)
        i_d = _recover(parameters.downlink_interference_w, z, DisturbanceIndex.I_D)
        uplink_rate = uplink_rate_bit_per_s(
            parameters.uplink_bandwidth_hz,
            parameters.uplink_terminal_tx_power_w,
            parameters.uplink_channel_gain_linear,
            parameters.noise_power_w,
            i_u,
        )
        downlink_rate = downlink_rate_bit_per_s(
            parameters.downlink_bandwidth_hz,
            parameters.downlink_bs_tx_power_w,
            parameters.downlink_channel_gain_linear,
            parameters.noise_power_w,
            i_d,
        )
        return edge_comm_energy_point_j(
            parameters.uplink_terminal_tx_power_w,
            d_in,
            uplink_rate,
            parameters.terminal_rx_power_w,
            d_out,
            downlink_rate,
        )

    suffix = "lower" if direction == "minimize" else "upper"
    return InnerGAProblem(
        name=f"edge_comm_energy_{suffix}",
        active_mask=EDGE_COMM_ENERGY_ACTIVE_MASK,
        gamma=float(gamma),
        direction=direction,
        evaluator=evaluator,
    )


def build_edge_core_latency_lower_problem(
    parameters: EdgeTaskParameters, gamma: Real
) -> InnerGAProblem:
    """构造边缘核心时延下界（最小真实响应）任务。"""

    return _build_edge_core_problem(parameters, gamma, "minimize")


def build_edge_core_latency_upper_problem(
    parameters: EdgeTaskParameters, gamma: Real
) -> InnerGAProblem:
    """构造边缘核心时延上界（最大真实响应）任务。"""

    return _build_edge_core_problem(parameters, gamma, "maximize")


def build_edge_comm_energy_lower_problem(
    parameters: EdgeTaskParameters, gamma: Real
) -> InnerGAProblem:
    """构造移动终端通信能耗下界（最小真实响应）任务。"""

    return _build_edge_energy_problem(parameters, gamma, "minimize")


def build_edge_comm_energy_upper_problem(
    parameters: EdgeTaskParameters, gamma: Real
) -> InnerGAProblem:
    """构造移动终端通信能耗上界（最大真实响应）任务。"""

    return _build_edge_energy_problem(parameters, gamma, "maximize")

