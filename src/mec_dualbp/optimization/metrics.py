"""Formal minimization Pareto helpers, legacy 2D indicators, and counters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable

import numpy as np


def _points(values: Iterable[Iterable[float]]) -> np.ndarray:
    x = np.asarray(list(values), dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("finite objective matrix with at least two columns required")
    return x


def nondominated_min(values: Iterable[Iterable[float]]) -> np.ndarray:
    x = _points(values)
    keep = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        if np.any(np.all(x <= x[i], axis=1) & np.any(x < x[i], axis=1)):
            keep[i] = False
    return np.unique(x[keep], axis=0)


def global_normalize(values: Iterable[Iterable[float]], lower: Iterable[float],
                     upper: Iterable[float]) -> np.ndarray:
    x = _points(values)
    lo, hi = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    if lo.shape != (x.shape[1],) or hi.shape != (x.shape[1],) or np.any(hi < lo):
        raise ValueError("invalid global normalization bounds")
    span = hi - lo
    return np.divide(x - lo, span, out=np.zeros_like(x), where=span > 0)


def hypervolume_2d_min(values: Iterable[Iterable[float]], reference: Iterable[float]) -> float:
    front = nondominated_min(values)
    ref = np.asarray(reference, dtype=np.float64)
    if front.shape[1] != 2 or ref.shape != (2,) or np.any(front > ref):
        raise ValueError("reference must be weakly worse than every point")
    front = front[np.argsort(front[:, 0], kind="stable")]
    previous_y, area = float(ref[1]), 0.0
    for x, y in front:
        if y < previous_y:
            area += float((ref[0] - x) * (previous_y - y))
            previous_y = float(y)
    return area


def pooled_reference_front(fronts: Iterable[Iterable[Iterable[float]]]) -> np.ndarray:
    arrays = [_points(front) for front in fronts]
    if not arrays:
        raise ValueError("at least one front required")
    return nondominated_min(np.vstack(arrays))


def igd(front: Iterable[Iterable[float]], reference_front: Iterable[Iterable[float]]) -> float:
    candidate, reference = _points(front), _points(reference_front)
    distances = np.linalg.norm(reference[:, None, :] - candidate[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def representative_index(values: Iterable[Iterable[float]], qos_margin: Iterable[float],
                         decision_hash: Iterable[str]) -> int:
    x = _points(values)
    margin = np.asarray(list(qos_margin), dtype=np.float64)
    hashes = list(decision_hash)
    if margin.shape != (len(x),) or len(hashes) != len(x):
        raise ValueError("representative tie fields must align")
    z = global_normalize(x, x.min(axis=0), x.max(axis=0))
    distance = np.linalg.norm(z, axis=1)
    worst = z.max(axis=1)
    return min(range(len(x)), key=lambda i: (float(distance[i]), -float(margin[i]), float(worst[i]), hashes[i]))


@dataclass
class EvaluationLedger:
    requests: int = 0
    unique_evaluations: int = 0
    cache_hits: int = 0
    _keys: set[Hashable] = field(default_factory=set, repr=False)

    def request(self, key: Hashable) -> bool:
        self.requests += 1
        if key in self._keys:
            self.cache_hits += 1
            return False
        self._keys.add(key)
        self.unique_evaluations += 1
        return True


@dataclass(frozen=True)
class HistoryRecord:
    evaluation_requests: int
    unique_evaluations: int
    generation: int
    feasible_rate: float
    nondominated_count: int
    hv_proxy: float | None
