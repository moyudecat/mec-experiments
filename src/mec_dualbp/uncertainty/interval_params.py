"""非概率区间的中心、半径与标准化扰动恢复。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real

from mec_dualbp._validation import finite_float
from mec_dualbp.utils.numerical_audit import record_endpoint_snap


def interval_center_radius(lower: Real, upper: Real) -> tuple[float, float]:
    """返回区间中心与半径。

    输入必须为有限实数且 ``lower <= upper``；非法端点不会自动交换。
    输出与输入采用相同SI单位，退化区间的半径为0。
    """

    lower_value = finite_float("lower", lower)
    upper_value = finite_float("upper", upper)
    if lower_value > upper_value:
        raise ValueError("lower must not exceed upper")
    center = lower_value / 2.0 + upper_value / 2.0
    radius = upper_value / 2.0 - lower_value / 2.0
    if radius < 0.0:
        raise ArithmeticError("interval radius must not be negative")
    if not (float("-inf") < center < float("inf")):
        raise ArithmeticError("interval center must be finite")
    return center, radius


def recover_interval_value(lower: Real, upper: Real, z: Real) -> float:
    """由 ``p=p_center+p_radius*z`` 恢复区间内的SI数值。

    ``z`` 必须为有限实数且位于[-1,1]；恢复值越界时抛出异常，
    不执行裁剪或端点交换。
    """

    center, radius = interval_center_radius(lower, upper)
    lower_value = float(lower)
    upper_value = float(upper)
    z_value = finite_float("z", z)
    if z_value < -1.0 or z_value > 1.0:
        raise ValueError("z must lie in [-1, 1]")
    recovered = center + radius * z_value
    if not (float("-inf") < recovered < float("inf")):
        raise ArithmeticError("recovered interval value must be finite")
    scale = max(1.0, abs(lower_value), abs(upper_value))
    tolerance = 16.0 * math.ulp(scale)
    if recovered < lower_value - tolerance or recovered > upper_value + tolerance:
        raise ArithmeticError("recovered value lies outside the supplied interval")
    if recovered < lower_value:
        record_endpoint_snap((lower_value - recovered) / math.ulp(scale))
        return lower_value
    if recovered > upper_value:
        record_endpoint_snap((recovered - upper_value) / math.ulp(scale))
        return upper_value
    return recovered


@dataclass(frozen=True, slots=True)
class Interval:
    """不可变有限闭区间；四个字段使用同一内部SI单位。"""

    lower: float
    upper: float
    center: float = field(init=False)
    radius: float = field(init=False)

    def __post_init__(self) -> None:
        """验证端点并派生同一 SI 单位下的中心与非负半径。"""

        lower_value = finite_float("lower", self.lower)
        upper_value = finite_float("upper", self.upper)
        center, radius = interval_center_radius(lower_value, upper_value)
        object.__setattr__(self, "lower", lower_value)
        object.__setattr__(self, "upper", upper_value)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)

    def recover(self, z: Real) -> float:
        """使用标准化扰动 ``z∈[-1,1]`` 恢复区间值。"""

        return recover_interval_value(self.lower, self.upper, z)
