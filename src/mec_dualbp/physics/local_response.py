"""本地点响应及Gamma三段确定性边界求解。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from mec_dualbp._validation import finite_nonnegative_output, nonnegative_float, positive_float
from mec_dualbp.uncertainty.budget_set import validate_gamma
from mec_dualbp.uncertainty.interval_params import Interval


@dataclass(frozen=True, slots=True)
class LocalResponseBounds:
    """不可变本地时延/能耗边界结果，单位分别为s和J。"""

    local_latency_lower_s: float
    local_latency_upper_s: float
    local_energy_lower_j: float
    local_energy_upper_j: float
    method: str
    candidate_count: int

    def __post_init__(self) -> None:
        """验证不可变结果中的边界次序、有限性、方法名与候选数量。"""

        pairs = (
            ("local_latency", self.local_latency_lower_s, self.local_latency_upper_s),
            ("local_energy", self.local_energy_lower_j, self.local_energy_upper_j),
        )
        for name, lower, upper in pairs:
            finite_nonnegative_output(f"{name}_lower", lower)
            finite_nonnegative_output(f"{name}_upper", upper)
            if lower > upper:
                raise ValueError(f"{name} lower bound must not exceed upper bound")
        if self.method not in {
            "center_point",
            "deterministic_budget_2d",
            "closed_form_gamma_ge_2",
        }:
            raise ValueError("unknown local bound method")
        if isinstance(self.candidate_count, bool) or self.candidate_count <= 0:
            raise ValueError("candidate_count must be a positive integer")


def local_latency_point_s(
    cpu_cycles: Real, local_cpu_frequency_cycle_per_s: Real
) -> float:
    """计算本地时延 ``C/F_L``，输入为cycle与cycle/s，输出为s。"""

    cycles = nonnegative_float("cpu_cycles", cpu_cycles)
    frequency = positive_float(
        "local_cpu_frequency_cycle_per_s", local_cpu_frequency_cycle_per_s
    )
    return finite_nonnegative_output("local_latency_point_s", cycles / frequency)


def local_energy_point_j(
    cpu_cycles: Real,
    local_cpu_frequency_cycle_per_s: Real,
    switching_capacitance_coefficient: Real,
) -> float:
    """计算本地能耗 ``kappa*C*F_L**2``，输出为J。

    ``C>=0``、``F_L>0``、``kappa>=0``；频率平方不得省略。
    """

    cycles = nonnegative_float("cpu_cycles", cpu_cycles)
    frequency = positive_float(
        "local_cpu_frequency_cycle_per_s", local_cpu_frequency_cycle_per_s
    )
    coefficient = nonnegative_float(
        "switching_capacitance_coefficient", switching_capacitance_coefficient
    )
    energy = coefficient * cycles * frequency**2
    return finite_nonnegative_output("local_energy_point_j", energy)


def _validated_local_inputs(
    cpu_cycles_interval: Interval,
    local_cpu_frequency_interval: Interval,
    switching_capacitance_coefficient: Real,
) -> tuple[Interval, Interval, float]:
    """验证两个 SI 区间及开关电容系数，不对输入执行静默修复。"""

    if not isinstance(cpu_cycles_interval, Interval):
        raise TypeError("cpu_cycles_interval must be an Interval in cycle")
    if not isinstance(local_cpu_frequency_interval, Interval):
        raise TypeError("local_cpu_frequency_interval must be an Interval in cycle/s")
    if cpu_cycles_interval.lower < 0.0:
        raise ValueError("cpu_cycles_interval must be nonnegative")
    if local_cpu_frequency_interval.lower <= 0.0:
        raise ValueError("local_cpu_frequency_interval must be strictly positive")
    coefficient = nonnegative_float(
        "switching_capacitance_coefficient", switching_capacitance_coefficient
    )
    return cpu_cycles_interval, local_cpu_frequency_interval, coefficient


def _unique_a_candidates(values: list[float], lower: float, upper: float) -> list[float]:
    """对解析a候选做边界容差校验与近重复去重，不改变可行域。"""

    scale = max(1.0, abs(lower), abs(upper))
    tolerance = 16.0 * math.ulp(scale)
    result: list[float] = []
    for raw in values:
        if not math.isfinite(raw):
            continue
        if raw < lower - tolerance or raw > upper + tolerance:
            continue
        candidate = lower if raw < lower else upper if raw > upper else raw
        if not any(abs(candidate - existing) <= tolerance for existing in result):
            result.append(candidate)
    return result


def _validate_bounds(lower: float, upper: float, name: str) -> tuple[float, float]:
    """确认一对计算边界有限、非负且下界不大于上界。"""

    finite_nonnegative_output(f"{name}_lower", lower)
    finite_nonnegative_output(f"{name}_upper", upper)
    if lower > upper:
        raise ArithmeticError(f"{name} lower bound exceeds upper bound")
    return lower, upper


def _closed_form_bounds(
    cycles: Interval, frequency: Interval, coefficient: float
) -> LocalResponseBounds:
    """按视觉确认的 Γ≥2 闭式公式计算四个本地边界。"""

    latency_lower = local_latency_point_s(cycles.lower, frequency.upper)
    latency_upper = local_latency_point_s(cycles.upper, frequency.lower)
    energy_lower = local_energy_point_j(cycles.lower, frequency.lower, coefficient)
    energy_upper = local_energy_point_j(cycles.upper, frequency.upper, coefficient)
    return LocalResponseBounds(
        *_validate_bounds(latency_lower, latency_upper, "local_latency"),
        *_validate_bounds(energy_lower, energy_upper, "local_energy"),
        method="closed_form_gamma_ge_2",
        candidate_count=4,
    )


def _deterministic_budget_2d_bounds(
    cycles: Interval, frequency: Interval, coefficient: float, gamma: float
) -> LocalResponseBounds:
    """用有限解析候选在 0<Γ<2 的二维预算域内精确求界。"""

    a_min = max(0.0, gamma - 1.0)
    a_max = min(1.0, gamma)
    endpoint_candidates = _unique_a_candidates([a_min, a_max], a_min, a_max)
    if not endpoint_candidates:
        raise ArithmeticError("deterministic candidate set is empty")

    physical_candidates: set[tuple[str, str]] = set()

    def recover_axis(interval: Interval, z_value: float, name: str) -> float:
        """恢复一个内部解析候选，仅吸附机器舍入导致的端点微小偏差。"""

        value = interval.center + interval.radius * z_value
        scale = max(1.0, abs(interval.lower), abs(interval.upper))
        tolerance = 16.0 * math.ulp(scale)
        if not math.isfinite(value):
            raise ArithmeticError(f"recovered {name} must be finite")
        if value < interval.lower - tolerance or value > interval.upper + tolerance:
            raise ArithmeticError(f"recovered {name} lies outside the interval")
        # z is an internally validated analytic candidate.  Endpoint snapping is
        # limited to round-off (a few ULPs), never used to repair invalid input.
        if value < interval.lower:
            return interval.lower
        if value > interval.upper:
            return interval.upper
        return value

    def recover(a: float, z_cycles: float, z_frequency: float) -> tuple[float, float]:
        """恢复一对候选物理量并登记用于审计的唯一物理候选。"""

        c_value = recover_axis(cycles, z_cycles, "CPU cycles")
        f_value = recover_axis(frequency, z_frequency, "local frequency")
        if f_value <= 0.0:
            raise ArithmeticError("recovered local frequency must be finite and positive")
        physical_candidates.add((c_value.hex(), f_value.hex()))
        return c_value, f_value

    latency_upper_values: list[float] = []
    latency_lower_values: list[float] = []
    for a in endpoint_candidates:
        c_upper, f_lower = recover(a, a, -(gamma - a))
        latency_upper_values.append(local_latency_point_s(c_upper, f_lower))
        c_lower, f_upper = recover(a, -a, gamma - a)
        latency_lower_values.append(local_latency_point_s(c_lower, f_upper))

    energy_upper_a = list(endpoint_candidates)
    energy_lower_a = list(endpoint_candidates)
    if cycles.radius > 0.0 and frequency.radius > 0.0:
        upper_stationary = (
            cycles.radius * (frequency.center + frequency.radius * gamma)
            - 2.0 * frequency.radius * cycles.center
        ) / (3.0 * cycles.radius * frequency.radius)
        lower_stationary = (
            2.0 * frequency.radius * cycles.center
            - cycles.radius * (frequency.center - frequency.radius * gamma)
        ) / (3.0 * cycles.radius * frequency.radius)
        energy_upper_a = _unique_a_candidates(
            energy_upper_a + [upper_stationary], a_min, a_max
        )
        energy_lower_a = _unique_a_candidates(
            energy_lower_a + [lower_stationary], a_min, a_max
        )

    energy_upper_values: list[float] = []
    for a in energy_upper_a:
        c_value, f_value = recover(a, a, gamma - a)
        energy_upper_values.append(local_energy_point_j(c_value, f_value, coefficient))

    energy_lower_values: list[float] = []
    for a in energy_lower_a:
        c_value, f_value = recover(a, -a, -(gamma - a))
        energy_lower_values.append(local_energy_point_j(c_value, f_value, coefficient))

    latency_lower = min(latency_lower_values)
    latency_upper = max(latency_upper_values)
    energy_lower = min(energy_lower_values)
    energy_upper = max(energy_upper_values)
    return LocalResponseBounds(
        *_validate_bounds(latency_lower, latency_upper, "local_latency"),
        *_validate_bounds(energy_lower, energy_upper, "local_energy"),
        method="deterministic_budget_2d",
        candidate_count=len(physical_candidates),
    )


def local_response_bounds_si(
    cpu_cycles_interval: Interval,
    local_cpu_frequency_interval: Interval,
    switching_capacitance_coefficient: Real,
    gamma: Real,
) -> LocalResponseBounds:
    """按Gamma三段协议确定性计算本地时延和能耗边界。

    输入区间单位分别为cycle和cycle/s，系数与 ``kappa*C*F**2`` 的SI
    量纲相容，Gamma无量纲。函数不调用随机数、网格、GA、BP或SciPy。
    """

    cycles, frequency, coefficient = _validated_local_inputs(
        cpu_cycles_interval,
        local_cpu_frequency_interval,
        switching_capacitance_coefficient,
    )
    gamma_value = validate_gamma(gamma)
    if gamma_value == 0.0:
        latency = local_latency_point_s(cycles.center, frequency.center)
        energy = local_energy_point_j(cycles.center, frequency.center, coefficient)
        return LocalResponseBounds(
            local_latency_lower_s=latency,
            local_latency_upper_s=latency,
            local_energy_lower_j=energy,
            local_energy_upper_j=energy,
            method="center_point",
            candidate_count=1,
        )
    if gamma_value < 2.0:
        return _deterministic_budget_2d_bounds(cycles, frequency, coefficient, gamma_value)
    return _closed_form_bounds(cycles, frequency, coefficient)
