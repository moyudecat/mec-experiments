"""所有 Exp-7 算法共享的规范解、修复器与评价入口合同。

本模块不实现任何具体 MOEA，也不加载模型。它只封住算法不得绕过的
公共入口，并拒绝 Raw DualBP 进入正式多算法评价。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

CORRECTED_DUALBP_HASH = "a15b8fb591df123b475f521ef35e4037b6ca201f4172caf804d3af571afe13e2"
TIME_MODEL_HASH = "291c5733bebdb4cb0c2cf7a4a07138ca6bb07c3ad75bffb378aa09b7f657d827"
ENERGY_MODEL_HASH = "8481dfdca0e03e3d6bebe4c4d5808259d16f7cdd67f4a65251fe7aad3e07d971"
IMPROVED_CORRECTED_DUALBP_HASH = "b505197af2f99f526d24f31ca0c4be1c6d5082612b19fe94a589f823ff1e35d7"
IMPROVED_TIME_MODEL_HASH = "75418503bc5399fcba8ef1db8b2b5b9cef14e8939fbd88532c54491c2eb61a9c"
IMPROVED_ENERGY_MODEL_HASH = "051e8ed1f9d7440a0655c7db4ff54b6b31925638f7518380b7fc4a2e356524e7"
SUPPORTED_CORRECTED_DUALBP_HASHES = frozenset((CORRECTED_DUALBP_HASH, IMPROVED_CORRECTED_DUALBP_HASH))
LEGACY_OBJECTIVE_MODE = "legacy_2d_robust_phi"
ROBUST_4D_OBJECTIVE_MODE = "robust_4d_center_radius"
OBJECTIVE_NAMES = {
    LEGACY_OBJECTIVE_MODE: ("Phi_T", "Phi_E"),
    ROBUST_4D_OBJECTIVE_MODE: ("c_T", "r_T", "c_E", "r_E"),
}


@dataclass(frozen=True)
class CanonicalSolution:
    """解码后的共同决策语义；edge_cpu 是实际 cycle/s 请求。"""

    offload: np.ndarray
    cache: np.ndarray
    edge_cpu: np.ndarray

    def __post_init__(self) -> None:
        arrays = tuple(np.asarray(v, dtype=np.float64) for v in (self.offload, self.cache, self.edge_cpu))
        if any(v.ndim != 1 for v in arrays) or len({v.size for v in arrays}) != 1 or arrays[0].size == 0:
            raise ValueError("canonical solution requires three aligned non-empty vectors")
        if not all(np.isfinite(v).all() for v in arrays):
            raise ValueError("canonical solution must be finite")
        object.__setattr__(self, "offload", arrays[0])
        object.__setattr__(self, "cache", arrays[1])
        object.__setattr__(self, "edge_cpu", arrays[2])


@dataclass(frozen=True)
class RepairContext:
    f_min: float
    f_max: float
    f_total_max: float
    cache_sizes: np.ndarray
    cache_total_max: float
    cache_benefit: np.ndarray
    offload_priority: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not (0 < self.f_min <= self.f_max <= self.f_total_max):
            raise ValueError("invalid edge CPU limits")
        sizes = np.asarray(self.cache_sizes, dtype=np.float64)
        benefit = np.asarray(self.cache_benefit, dtype=np.float64)
        if sizes.ndim != 1 or benefit.shape != sizes.shape or sizes.size == 0:
            raise ValueError("aligned cache vectors required")
        if np.any(sizes < 0) or np.any(benefit < 0) or self.cache_total_max < 0:
            raise ValueError("cache values must be nonnegative")
        priority = benefit if self.offload_priority is None else np.asarray(self.offload_priority, dtype=np.float64)
        if priority.shape != sizes.shape or not np.isfinite(priority).all():
            raise ValueError("invalid offload priority")
        object.__setattr__(self, "cache_sizes", sizes)
        object.__setattr__(self, "cache_benefit", benefit)
        object.__setattr__(self, "offload_priority", priority)


def canonical_outer_repair(solution: CanonicalSolution, context: RepairContext) -> CanonicalSolution:
    """确定性修复 C1--C6；不调用 BP、GA 或目标值。"""
    n = solution.offload.size
    if context.cache_sizes.size != n:
        raise ValueError("repair context size mismatch")
    m = (solution.offload >= 0.5).astype(np.float64)
    c = ((solution.cache >= 0.5) & (m == 1)).astype(np.float64)
    f = np.where(m == 1, np.clip(solution.edge_cpu, context.f_min, context.f_max), 0.0)

    active = np.flatnonzero(m == 1)
    max_active = int(np.floor((context.f_total_max + 1e-12) / context.f_min))
    if active.size > max_active:
        order = sorted(active.tolist(), key=lambda i: (-float(context.offload_priority[i]), i))
        drop = order[max_active:]
        m[drop] = c[drop] = f[drop] = 0.0

    cached = np.flatnonzero(c == 1)
    if float(np.dot(c, context.cache_sizes)) > context.cache_total_max + 1e-12:
        # ``cache_benefit`` is the frozen benefit density: deterministic
        # backhaul delay avoided per stored bit.  Do not divide by size again.
        density = context.cache_benefit
        keep_order = sorted(cached.tolist(), key=lambda i: (-float(density[i]), i))
        c[:] = 0.0
        used = 0.0
        for i in keep_order:
            if used + context.cache_sizes[i] <= context.cache_total_max + 1e-12:
                c[i] = 1.0
                used += float(context.cache_sizes[i])

    active = np.flatnonzero(m == 1)
    if active.size:
        f[active] = np.clip(f[active], context.f_min, context.f_max)
        excess = float(f.sum() - context.f_total_max)
        while excess > 1e-9:
            adjustable = active[f[active] > context.f_min + 1e-12]
            if adjustable.size == 0:
                break
            headroom = f[adjustable] - context.f_min
            reduction = np.minimum(headroom, excess * headroom / headroom.sum())
            f[adjustable] -= reduction
            excess = float(f.sum() - context.f_total_max)
    f[m == 0] = 0.0
    c[m == 0] = 0.0
    return CanonicalSolution(m, c, f)


def interval_possibility(lower: float, upper: float, tau: float) -> float:
    """冻结的 C7 三段可能度；退化区间按确定性条件处理。"""
    values = np.asarray([lower, upper, tau], dtype=np.float64)
    if not np.isfinite(values).all() or lower < 0 or upper < lower:
        raise ValueError("finite ordered nonnegative interval required")
    if upper == lower:
        return 1.0 if upper <= tau else 0.0
    if upper <= tau:
        return 1.0
    if tau <= lower:
        return 0.0
    return float((tau - lower) / (upper - lower))


def robust_objectives(time_lower: float, time_upper: float,
                      energy_lower: float, energy_upper: float, *,
                      rho_time: float, rho_energy: float,
                      lambda_time: float = 1.0, lambda_energy: float = 1.0) -> np.ndarray:
    """冻结的两个系统级鲁棒目标，输出方向均为 minimize。"""
    bounds = np.asarray([time_lower, time_upper, energy_lower, energy_upper], dtype=np.float64)
    if (not np.isfinite(bounds).all() or time_lower < 0 or energy_lower < 0
            or time_upper < time_lower or energy_upper < energy_lower):
        raise ValueError("finite ordered nonnegative system bounds required")
    if rho_time <= 0 or rho_energy <= 0 or lambda_time < 0 or lambda_energy < 0:
        raise ValueError("positive rho and nonnegative lambda required")
    tc, tr = (time_lower + time_upper) * .5, (time_upper - time_lower) * .5
    ec, er = (energy_lower + energy_upper) * .5, (energy_upper - energy_lower) * .5
    return np.asarray([
        np.arctan(tc / rho_time) + lambda_time * np.arctan(tr / rho_time),
        np.arctan(ec / rho_energy) + lambda_energy * np.arctan(er / rho_energy),
    ], dtype=np.float64)


def objective_payload(time_lower: float, time_upper: float,
                      energy_lower: float, energy_upper: float, *,
                      objective_mode: str = LEGACY_OBJECTIVE_MODE,
                      rho_time: float, rho_energy: float,
                      lambda_time: float = 1.0,
                      lambda_energy: float = 1.0) -> dict[str, Any]:
    """Build a named objective vector without changing the physical bounds.

    The historical two-objective mode remains the default.  The explicit 4D
    mode exposes interval center/radius values directly to the optimizer.
    """
    try:
        names = OBJECTIVE_NAMES[objective_mode]
    except KeyError as error:
        raise ValueError(f"unsupported objective mode: {objective_mode}") from error
    bounds = np.asarray(
        [time_lower, time_upper, energy_lower, energy_upper], dtype=np.float64)
    if (not np.isfinite(bounds).all() or time_lower < 0 or energy_lower < 0
            or time_upper < time_lower or energy_upper < energy_lower):
        raise ValueError("finite ordered nonnegative system bounds required")
    c_t, r_t = (time_lower + time_upper) * .5, (time_upper - time_lower) * .5
    c_e, r_e = (energy_lower + energy_upper) * .5, (energy_upper - energy_lower) * .5
    center_radius = {
        "T_lower": float(time_lower), "T_upper": float(time_upper),
        "E_lower": float(energy_lower), "E_upper": float(energy_upper),
        "c_T": float(c_t), "r_T": float(r_t),
        "c_E": float(c_e), "r_E": float(r_e),
    }
    if objective_mode == ROBUST_4D_OBJECTIVE_MODE:
        objectives = np.asarray([c_t, r_t, c_e, r_e], dtype=np.float64)
    else:
        objectives = robust_objectives(
            time_lower, time_upper, energy_lower, energy_upper,
            rho_time=rho_time, rho_energy=rho_energy,
            lambda_time=lambda_time, lambda_energy=lambda_energy)
        center_radius.update(Phi_T=float(objectives[0]), Phi_E=float(objectives[1]))
    return {
        **center_radius,
        "objective_mode": objective_mode,
        "objective_names": list(names),
        "objectives": objectives.tolist(),
    }


class CanonicalOuterEvaluator:
    """算法共享的唯一包装器；科学评价主体由后续实现注入。"""

    def __init__(self, callback: Callable[[CanonicalSolution], Mapping[str, Any]], *,
                 corrected_dualbp_hash: str, prediction_mode: str = "corr_pred",
                 cache_enabled: bool = True) -> None:
        if (corrected_dualbp_hash not in SUPPORTED_CORRECTED_DUALBP_HASHES
                or prediction_mode not in {"corr_pred", "deterministic_center"}):
            raise ValueError("Formal outer evaluator requires a registered frozen evaluator mode")
        self._callback = callback
        self.corrected_dualbp_hash = corrected_dualbp_hash
        self.prediction_mode = prediction_mode
        self.cache_enabled = bool(cache_enabled)
        self.requests = 0
        self.unique_evaluations = 0
        self.cache_hits = 0
        self._cache: dict[tuple[float, ...], dict[str, Any]] = {}

    def evaluate_outer_solution_canonical(self, solution: CanonicalSolution,
                                          context: RepairContext) -> Mapping[str, Any]:
        repaired = canonical_outer_repair(solution, context)
        self.requests += 1
        key = tuple(np.concatenate((repaired.offload, repaired.cache, repaired.edge_cpu)).tolist())
        if self.cache_enabled and key in self._cache:
            self.cache_hits += 1
            return dict(self._cache[key])
        self.unique_evaluations += 1
        result = dict(self._callback(repaired))
        if result.get("prediction_mode") != self.prediction_mode:
            raise ValueError("evaluator callback returned a mismatched prediction mode")
        if self.cache_enabled:
            self._cache[key] = dict(result)
        return result
