"""稳定量化缓存键与真实扰动碰撞复评。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mec_dualbp.inner_ga.problem import InnerGAProblem


@dataclass(frozen=True, slots=True)
class CacheCollisionAudit:
    """一个量化键下不同原始向量的物理响应复评结果。"""

    cache_key: tuple[float, ...]
    representative_z: tuple[float, ...]
    colliding_z: tuple[float, ...]
    cached_response: float
    direct_response: float
    absolute_difference: float
    tolerance: float
    error: bool


def quantized_cache_key(z: NDArray[np.float64], decimals: int) -> tuple[float, ...]:
    """将有限六维扰动按固定小数位量化为稳定缓存键。"""

    values = np.asarray(z, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("cache vector must be finite with shape (6,)")
    if not isinstance(decimals, int) or not 12 <= decimals <= 16:
        raise ValueError("cache decimals must lie in [12, 16]")
    return tuple(float(value) for value in np.round(values, decimals=decimals))


def audit_cache_collision(
    problem: InnerGAProblem,
    cache_key: tuple[float, ...],
    representative_z: NDArray[np.float64],
    colliding_z: NDArray[np.float64],
    cached_response: float,
    absolute_tolerance: float,
) -> CacheCollisionAudit:
    """对真实不同但量化键相同的向量重新直接执行物理评价。"""

    direct = problem.evaluate(np.asarray(colliding_z, dtype=np.float64))
    difference = abs(direct - cached_response)
    return CacheCollisionAudit(
        cache_key,
        tuple(float(value) for value in representative_z),
        tuple(float(value) for value in colliding_z),
        float(cached_response),
        direct,
        difference,
        absolute_tolerance,
        difference > absolute_tolerance,
    )
