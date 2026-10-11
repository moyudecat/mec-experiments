"""普通 Latin hypercube 采样及六维用户扰动初始化。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from mec_dualbp.uncertainty.budget_set import proportional_budget_repair, validate_gamma
from mec_dualbp.uncertainty.schema import DISTURBANCE_DIMENSION


@dataclass(frozen=True, slots=True)
class LHSDisturbanceSamples:
    """区分原始 LHS、预算修复样本与额外初始化点的不可变结果。"""

    raw_lhs: NDArray[np.float64]
    repaired_lhs: NDArray[np.float64]
    initialization_samples: NDArray[np.float64]
    sources: tuple[str, ...]

    def __post_init__(self) -> None:
        """验证数组形状并复制为只读数组，避免调用方原地篡改。"""

        arrays = []
        for name, value in (
            ("raw_lhs", self.raw_lhs),
            ("repaired_lhs", self.repaired_lhs),
            ("initialization_samples", self.initialization_samples),
        ):
            array = np.array(value, dtype=np.float64, copy=True)
            if array.ndim != 2 or array.shape[1] != DISTURBANCE_DIMENSION:
                raise ValueError(f"{name} must have shape (n, 6)")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite")
            array.setflags(write=False)
            arrays.append(array)
        if arrays[0].shape != arrays[1].shape:
            raise ValueError("raw_lhs and repaired_lhs must have equal shapes")
        if arrays[2].shape[0] != len(self.sources):
            raise ValueError("sources must align with initialization_samples")
        object.__setattr__(self, "raw_lhs", arrays[0])
        object.__setattr__(self, "repaired_lhs", arrays[1])
        object.__setattr__(self, "initialization_samples", arrays[2])


def ordinary_lhs_unit(
    n_samples: Integral, n_dimensions: Integral, rng: np.random.Generator
) -> NDArray[np.float64]:
    """在 ``[0,1)^d`` 生成普通 LHS，每维每个等宽分层恰有一个点。"""

    if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if isinstance(n_dimensions, bool) or not isinstance(n_dimensions, Integral) or n_dimensions <= 0:
        raise ValueError("n_dimensions must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    n, d = int(n_samples), int(n_dimensions)
    result = np.empty((n, d), dtype=np.float64)
    for dimension in range(d):
        points = (np.arange(n, dtype=np.float64) + rng.random(n)) / n
        result[:, dimension] = points[rng.permutation(n)]
    return result


def _boundary_anchors(gamma: float, active_indices: NDArray[np.int64]) -> list[NDArray[np.float64]]:
    """构造两个可辨识的径向预算边界锚点；无有效预算时返回空列表。"""

    if gamma <= 0.0 or active_indices.size == 0:
        return []
    positive = np.zeros(DISTURBANCE_DIMENSION, dtype=np.float64)
    remaining = min(gamma, float(active_indices.size))
    for index in active_indices:
        allocation = min(1.0, remaining)
        positive[index] = allocation
        remaining -= allocation
        if remaining <= 0.0:
            break
    return [positive, -positive]


def lhs_user_disturbances(
    n_samples: Integral,
    gamma: Real,
    active_mask: Sequence[bool],
    seed: Integral,
    include_center: bool = True,
    include_boundary_anchors: bool = False,
) -> LHSDisturbanceSamples:
    """生成普通 LHS 扰动、径向修复样本和带来源标签的初始化样本。

    ``raw_lhs`` 映射到 ``[-1,1]`` 后仍具有有效维度的 LHS 分层性质；
    ``repaired_lhs`` 满足六维 Γ 预算，但比例修复可能改变其边际分层。
    中心点与锚点是额外初始化样本，不计作 LHS 样本。
    """

    gamma_value = validate_gamma(gamma)
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    mask = np.asarray(active_mask)
    if mask.shape != (DISTURBANCE_DIMENSION,) or mask.dtype.kind != "b":
        raise ValueError("active_mask must contain exactly six booleans")
    mask = mask.astype(bool, copy=True)
    active_indices = np.flatnonzero(mask)
    if active_indices.size == 0:
        raise ValueError("active_mask must enable at least one dimension")
    rng = np.random.default_rng(int(seed))
    unit = ordinary_lhs_unit(n_samples, int(active_indices.size), rng)
    raw = np.zeros((int(n_samples), DISTURBANCE_DIMENSION), dtype=np.float64)
    raw[:, active_indices] = 2.0 * unit - 1.0
    repaired = np.vstack(
        [proportional_budget_repair(row, gamma_value, mask) for row in raw]
    )
    samples: list[NDArray[np.float64]] = []
    sources: list[str] = []
    if include_center:
        samples.append(np.zeros(DISTURBANCE_DIMENSION, dtype=np.float64))
        sources.append("center")
    if include_boundary_anchors:
        for anchor in _boundary_anchors(gamma_value, active_indices):
            samples.append(anchor)
            sources.append("boundary_anchor")
    samples.extend(repaired)
    sources.extend("ordinary_lhs_repaired" for _ in range(repaired.shape[0]))
    return LHSDisturbanceSamples(raw, repaired, np.vstack(samples), tuple(sources))
