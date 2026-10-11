"""内部数值验证工具；所有物理模块均只接收有限SI数值。"""

from __future__ import annotations

import math
from numbers import Real


def finite_float(name: str, value: Real) -> float:
    """返回有限浮点数；布尔值、非实数、NaN和无穷大均被拒绝。"""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def nonnegative_float(name: str, value: Real) -> float:
    """返回有限非负浮点数。"""

    converted = finite_float(name, value)
    if converted < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return converted


def positive_float(name: str, value: Real) -> float:
    """返回有限严格正浮点数。"""

    converted = finite_float(name, value)
    if converted <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return converted


def finite_nonnegative_output(name: str, value: float) -> float:
    """验证计算结果是有限非负数，失败时不返回NaN。"""

    if not math.isfinite(value) or value < 0.0:
        raise ArithmeticError(f"{name} must be finite and nonnegative")
    return value

