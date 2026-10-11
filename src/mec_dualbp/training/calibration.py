"""冻结 DualBP 的方向性经验保守 alpha 校准。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

CALIBRATION_DELTA = 0.05
CALIBRATION_EPSILON = 1e-12
CALIBRATION_RULE = "finite-sample conservative higher order statistic"


def upper_under_error(true_upper: np.ndarray, predicted_upper: np.ndarray,
                      epsilon: float = CALIBRATION_EPSILON) -> np.ndarray:
    true, pred = np.asarray(true_upper, dtype=np.float64), np.asarray(predicted_upper, dtype=np.float64)
    if true.shape != pred.shape or np.any(true < 0) or not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("finite aligned nonnegative upper arrays required")
    return np.maximum(0.0, (true - pred) / np.maximum(true, epsilon))


def lower_over_error(true_lower: np.ndarray, predicted_lower: np.ndarray,
                     epsilon: float = CALIBRATION_EPSILON) -> np.ndarray:
    true, pred = np.asarray(true_lower, dtype=np.float64), np.asarray(predicted_lower, dtype=np.float64)
    if true.shape != pred.shape or np.any(true < 0) or not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("finite aligned nonnegative lower arrays required")
    return np.maximum(0.0, (pred - true) / np.maximum(true, epsilon))


def conservative_rank(sample_count: int, delta: float = CALIBRATION_DELTA) -> int:
    if sample_count <= 0 or not 0.0 < delta < 1.0:
        raise ValueError("positive n and delta in (0,1) required")
    return min(math.ceil((sample_count + 1) * (1.0 - delta)), sample_count)


def conservative_alpha(errors: np.ndarray, delta: float = CALIBRATION_DELTA) -> tuple[float, int]:
    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("finite nonnegative one-dimensional errors required")
    rank = conservative_rank(values.size, delta)
    return float(np.sort(values)[rank - 1]), rank


def apply_directional_correction(lower: np.ndarray, upper: np.ndarray,
                                 alpha_minus: float, alpha_plus: float) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    if lo.shape != hi.shape or not (0 <= alpha_minus < 1 and 0 <= alpha_plus < 1):
        raise ValueError("aligned bounds and alpha in [0,1) required")
    corrected_lower = lo / (1.0 + float(alpha_minus))
    corrected_upper = hi / (1.0 - float(alpha_plus))
    if (not np.isfinite(corrected_lower).all() or not np.isfinite(corrected_upper).all()
            or np.any(corrected_lower < 0) or np.any(corrected_upper < corrected_lower)):
        raise ValueError("correction produced an invalid interval")
    return corrected_lower, corrected_upper


def interval_coverage(true_lower: np.ndarray, true_upper: np.ndarray,
                      predicted_lower: np.ndarray, predicted_upper: np.ndarray) -> float:
    return float(np.mean((np.asarray(predicted_lower) <= np.asarray(true_lower))
                         & (np.asarray(predicted_upper) >= np.asarray(true_upper))))


def protocol_is_frozen(protocol: dict[str, Any]) -> bool:
    return (protocol.get("status") == "frozen" and protocol.get("delta") == CALIBRATION_DELTA
            and protocol.get("target_access_allowed_after_freeze") is True
            and bool(protocol.get("alpha_calibration_protocol_hash")))
