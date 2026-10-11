"""Exp-2 独立 Test 的冻结指标与置信区间工具。"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

BOOTSTRAP_SEED = 46001
BOOTSTRAP_REPETITIONS = 5000


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("valid binomial count required")
    p = successes / total; denominator = 1 + z*z/total
    center = (p + z*z/(2*total))/denominator
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/denominator
    return max(0.0, center-half), min(1.0, center+half)


def safety_metrics(true_lower: np.ndarray, true_upper: np.ndarray,
                   pred_lower: np.ndarray, pred_upper: np.ndarray,
                   epsilon: float = 1e-12) -> dict[str, float | int]:
    tl, tu, pl, pu = map(lambda x: np.asarray(x, dtype=np.float64),
                         (true_lower, true_upper, pred_lower, pred_upper))
    if not all(x.shape == tl.shape for x in (tu, pl, pu)) or tl.size == 0:
        raise ValueError("aligned nonempty arrays required")
    upper_events, lower_events = pu < tu, pl > tl
    coverage = (pl <= tl) & (pu >= tu)
    return {"count": int(tl.size), "uer_events": int(upper_events.sum()), "ler_events": int(lower_events.sum()),
            "icr_events": int(coverage.sum()), "uer": float(upper_events.mean()), "ler": float(lower_events.mean()),
            "icr": float(coverage.mean()),
            "upper_under_relative_magnitude": float(np.mean(np.maximum(0, (tu-pu)/np.maximum(tu, epsilon)))),
            "lower_over_relative_magnitude": float(np.mean(np.maximum(0, (pl-tl)/np.maximum(tl, epsilon))))}


def paired_bootstrap_accuracy(true: np.ndarray, raw: np.ndarray, corrected: np.ndarray,
                              repetitions: int = BOOTSTRAP_REPETITIONS,
                              seed: int = BOOTSTRAP_SEED) -> dict[str, dict[str, tuple[float, float]]]:
    y, a, b = map(lambda x: np.asarray(x, dtype=np.float64), (true, raw, corrected))
    if y.shape != a.shape or y.shape != b.shape or y.ndim != 1 or y.size == 0:
        raise ValueError("aligned one-dimensional arrays required")
    rng = np.random.default_rng(seed); indices = rng.integers(0, y.size, size=(repetitions, y.size))
    yt, ar, bc = y[indices], a[indices], b[indices]
    def values(pred: np.ndarray) -> dict[str, np.ndarray]:
        error = pred-yt
        return {"mae": np.mean(np.abs(error),axis=1), "rmse": np.sqrt(np.mean(error**2,axis=1)),
                "wmape": np.sum(np.abs(error),axis=1)/np.maximum(np.sum(np.abs(yt),axis=1),1e-12)}
    output = {}
    for name, data in (("raw",values(ar)),("corrected",values(bc))):
        output[name] = {metric: tuple(map(float,np.percentile(v,[2.5,97.5]))) for metric,v in data.items()}
    return output


def test_protocol_is_frozen(protocol: dict) -> bool:
    return (protocol.get("status") == "frozen" and bool(protocol.get("test_evaluation_protocol_hash"))
            and protocol.get("target_access_allowed_after_freeze") is True
            and protocol.get("post_test_model_modification_allowed") is False
            and protocol.get("post_test_alpha_modification_allowed") is False)
