"""用户级六维Gamma预算集合的严格检查，不含投影或采样。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mec_dualbp._validation import finite_float
from mec_dualbp.uncertainty.schema import DISTURBANCE_DIMENSION

USER_DISTURBANCE_DIMENSION = DISTURBANCE_DIMENSION
DEFAULT_ATOL = 1e-12
MAX_ATOL = 1e-9


def validate_gamma(gamma: Real) -> float:
    """验证无量纲预算 ``0<=Gamma<=6`` 并返回浮点值。"""

    value = finite_float("gamma", gamma)
    if value < 0.0 or value > float(USER_DISTURBANCE_DIMENSION):
        raise ValueError("gamma must lie in [0, 6]")
    return value


def _disturbance_values(z: Sequence[Real]) -> tuple[float, ...]:
    """将单用户六维有限扰动严格验证并转为不可变元组。"""

    if isinstance(z, np.ndarray):
        if z.shape != (USER_DISTURBANCE_DIMENSION,):
            raise ValueError("z must contain exactly six components")
        candidates = z.tolist()
    elif isinstance(z, Sequence) and not isinstance(z, (str, bytes)):
        candidates = z
    else:
        raise TypeError("z must be a sequence of six real values")
    if len(candidates) != USER_DISTURBANCE_DIMENSION:
        raise ValueError("z must contain exactly six components")
    return tuple(
        finite_float(f"z[{index}]", value) for index, value in enumerate(candidates)
    )


def is_user_budget_feasible(
    z: Sequence[Real], gamma: Real, atol: Real = DEFAULT_ATOL
) -> bool:
    """检查单个用户的六维扰动是否属于 ``U_{Gamma,n}``。

    ``atol`` 仅允许位于[0,1e-9]，用于浮点边界比较；函数不裁剪扰动，
    也不实现跨用户共享Gamma。
    """

    gamma_value = validate_gamma(gamma)
    atol_value = finite_float("atol", atol)
    if atol_value < 0.0 or atol_value > MAX_ATOL:
        raise ValueError("atol must lie in [0, 1e-9]")
    values = _disturbance_values(z)
    if any(value < -1.0 - atol_value or value > 1.0 + atol_value for value in values):
        return False
    return math.fsum(abs(value) for value in values) <= gamma_value + atol_value


def validate_user_disturbance(z: Sequence[Real], gamma: Real) -> tuple[float, ...]:
    """严格验证一个用户的六维扰动，非法时抛出异常。

    返回值仍是无量纲六元组，不执行投影、裁剪、LHS或系统级预算合并。
    """

    gamma_value = validate_gamma(gamma)
    values = _disturbance_values(z)
    if any(value < -1.0 or value > 1.0 for value in values):
        raise ValueError("every disturbance component must lie in [-1, 1]")
    if math.fsum(abs(value) for value in values) > gamma_value:
        raise ValueError("sum(abs(z)) must not exceed gamma")
    return values


def proportional_budget_repair(
    z: ArrayLike,
    gamma: Real,
    active_mask: Sequence[bool] | None = None,
    atol: Real = DEFAULT_ATOL,
) -> NDArray[np.float64]:
    """以径向比例规则修复一个六维扰动，并返回新数组。

    输入无量纲。无效维度先清零，有效维度裁剪至 ``[-1,1]``；若有效
    分量的 L1 范数超过 ``gamma``，整体乘以 ``gamma/L1``。这不是欧氏
    最近点投影。NaN、Inf、错误维度或非法预算会显式抛出异常。
    """

    gamma_value = validate_gamma(gamma)
    atol_value = finite_float("atol", atol)
    if atol_value < 0.0 or atol_value > MAX_ATOL:
        raise ValueError("atol must lie in [0, 1e-9]")
    try:
        values = np.asarray(z, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("z must contain six real values") from exc
    if values.shape != (USER_DISTURBANCE_DIMENSION,):
        raise ValueError("z must have shape (6,)")
    if not np.all(np.isfinite(values)):
        raise ValueError("all disturbance components must be finite")
    if active_mask is None:
        mask = np.ones(USER_DISTURBANCE_DIMENSION, dtype=bool)
    else:
        mask = np.asarray(active_mask)
        if mask.shape != (USER_DISTURBANCE_DIMENSION,) or mask.dtype.kind != "b":
            raise ValueError("active_mask must contain exactly six booleans")
        mask = mask.astype(bool, copy=True)
    repaired = np.array(values, dtype=np.float64, copy=True)
    repaired[~mask] = 0.0
    repaired[mask] = np.clip(repaired[mask], -1.0, 1.0)
    if gamma_value == 0.0:
        repaired.fill(0.0)
    else:
        norm = math.fsum(abs(float(value)) for value in repaired[mask])
        if norm > gamma_value:
            repaired[mask] *= gamma_value / norm
            repaired_norm = math.fsum(abs(float(value)) for value in repaired[mask])
            while repaired_norm > gamma_value:
                conservative_scale = np.nextafter(gamma_value / repaired_norm, 0.0)
                repaired[mask] *= conservative_scale
                repaired_norm = math.fsum(abs(float(value)) for value in repaired[mask])
    if np.any(repaired < -1.0 - atol_value) or np.any(repaired > 1.0 + atol_value):
        raise ArithmeticError("budget repair produced a component outside [-1, 1]")
    if np.any(repaired[~mask] != 0.0):
        raise ArithmeticError("budget repair did not zero inactive dimensions")
    if math.fsum(abs(float(value)) for value in repaired) > gamma_value + atol_value:
        raise ArithmeticError("budget repair did not satisfy the L1 budget")
    return repaired
