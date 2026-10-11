"""缓存回源、边缘核心和完整边缘时延点响应。"""

from __future__ import annotations

import math
from numbers import Integral, Real

from mec_dualbp._validation import finite_nonnegative_output, nonnegative_float, positive_float


def cache_miss_latency_s(
    dependency_size_bit: Real, backhaul_rate_bit_per_s: Real
) -> float:
    """计算 ``T_dep=s/R_bh``，输入为bit与bit/s，输出为s。"""

    size = nonnegative_float("dependency_size_bit", dependency_size_bit)
    rate = positive_float("backhaul_rate_bit_per_s", backhaul_rate_bit_per_s)
    return finite_nonnegative_output("cache_miss_latency_s", size / rate)


def edge_core_latency_point_s(
    input_size_bit: Real,
    uplink_rate_bit_per_s: Real,
    cpu_cycles: Real,
    edge_cpu_frequency_cycle_per_s: Real,
    output_size_bit: Real,
    downlink_rate_bit_per_s: Real,
) -> float:
    """计算不含缓存回源项的边缘核心时延，输出为s。

    公式为 ``D_in/R_u + C/f + D_out/R_d``。本函数没有缓存参数，
    其输出可作为未来时延BP标签的点响应基础。
    """

    input_size = nonnegative_float("input_size_bit", input_size_bit)
    uplink_rate = positive_float("uplink_rate_bit_per_s", uplink_rate_bit_per_s)
    cycles = nonnegative_float("cpu_cycles", cpu_cycles)
    edge_frequency = positive_float(
        "edge_cpu_frequency_cycle_per_s", edge_cpu_frequency_cycle_per_s
    )
    output_size = nonnegative_float("output_size_bit", output_size_bit)
    downlink_rate = positive_float("downlink_rate_bit_per_s", downlink_rate_bit_per_s)
    latency = input_size / uplink_rate + cycles / edge_frequency + output_size / downlink_rate
    return finite_nonnegative_output("edge_core_latency_point_s", latency)


def edge_total_latency_point_s(
    edge_core_latency_s: Real, cache_decision: Integral, cache_miss_latency_s: Real
) -> float:
    """由core时延构造完整边缘时延，输出为s。

    ``edge_total=edge_core+(1-cache_decision)*T_dep``；缓存决策必须严格为0或1。
    """

    core = nonnegative_float("edge_core_latency_s", edge_core_latency_s)
    miss_latency = nonnegative_float("cache_miss_latency_s", cache_miss_latency_s)
    if isinstance(cache_decision, bool) or not isinstance(cache_decision, Integral):
        raise TypeError("cache_decision must be integer 0 or 1")
    decision = int(cache_decision)
    if decision not in (0, 1):
        raise ValueError("cache_decision must be exactly 0 or 1")
    total = core + (1 - decision) * miss_latency
    if not math.isfinite(total):
        raise ArithmeticError("edge_total_latency_point_s must be finite")
    return finite_nonnegative_output("edge_total_latency_point_s", total)

