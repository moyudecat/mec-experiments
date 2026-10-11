"""Exp-3 统一 Test 的冻结统计工具。"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from mec_dualbp.training.dualbp import regression_metrics
from mec_dualbp.training.test_evaluation import safety_metrics, wilson_interval

BOOTSTRAP_SEED = 47001
BOOTSTRAP_REPETITIONS = 5000


def full_test_metrics(true_lower: np.ndarray, true_upper: np.ndarray, pred_lower: np.ndarray,
                      pred_upper: np.ndarray, center_scale: float) -> dict[str, float | int]:
    result = {**regression_metrics(true_lower, true_upper, pred_lower, pred_upper, center_scale),
              **safety_metrics(true_lower, true_upper, pred_lower, pred_upper)}
    invalid = (np.asarray(pred_lower) < 0) | (np.asarray(pred_upper) < 0) | (np.asarray(pred_upper) < np.asarray(pred_lower))
    result["invalid_interval_count"] = int(invalid.sum())
    result["invalid_interval_rate"] = float(invalid.mean())
    result["nonfinite_count"] = int(np.sum(~(np.isfinite(pred_lower) & np.isfinite(pred_upper))))
    return result


def common_bootstrap_indices(size: int, repetitions: int = BOOTSTRAP_REPETITIONS,
                             seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    if size <= 0 or repetitions <= 0:
        raise ValueError("positive size and repetitions required")
    return np.random.default_rng(seed).integers(0, size, size=(repetitions, size))


def paired_metric_difference(true: np.ndarray, first: np.ndarray, second: np.ndarray,
                             indices: np.ndarray) -> dict[str, tuple[float, float]]:
    y, a, b = map(lambda value: np.asarray(value, dtype=np.float64), (true, first, second))
    if y.shape != a.shape or y.shape != b.shape or indices.shape[1] != y.size:
        raise ValueError("aligned paired arrays required")
    yt, ap, bp = y[indices], a[indices], b[indices]
    def values(pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        error = pred - yt
        return (np.mean(np.abs(error), axis=1), np.sqrt(np.mean(error ** 2, axis=1)),
                np.sum(np.abs(error), axis=1) / np.maximum(np.sum(np.abs(yt), axis=1), 1e-12))
    av, bv = values(ap), values(bp)
    return {name: tuple(map(float, np.percentile(x-y, [2.5, 97.5])))
            for name, x, y in zip(("MAE", "RMSE", "WMAPE"), av, bv)}


def wilson_rows(method: str, network: str, metrics: Mapping[str, float | int]) -> list[dict[str, object]]:
    rows = []
    for metric in ("uer", "ler", "icr"):
        low, high = wilson_interval(int(metrics[f"{metric}_events"]), int(metrics["count"]))
        rows.append({"Method": method, "Network": network, "Metric": metric.upper(),
                     "Value": metrics[metric], "CI_Low": low, "CI_High": high,
                     "Method_CI": "Wilson 95%"})
    return rows


def contribution_status(metrics: Mapping[tuple[str, str], Mapping[str, float | int]],
                        d_invariance_max: float) -> dict[str, str]:
    def m(method: str, network: str, key: str) -> float:
        return float(metrics[(method, network)][key])
    alpha = all(m("A", net, "uer") < m("B", net, "uer") and
                m("A", net, "ler") < m("B", net, "ler") and
                m("A", net, "icr") > m("B", net, "icr") for net in ("Time", "Energy"))
    safety_wins = []
    for net in ("Time", "Energy"):
        safety_wins.extend((m("B", net, "uer") < m("C", net, "uer"),
                            m("B", net, "ler") < m("C", net, "ler"),
                            m("B", net, "upper_under_relative_magnitude") < m("C", net, "upper_under_relative_magnitude"),
                            m("B", net, "lower_over_relative_magnitude") < m("C", net, "lower_over_relative_magnitude"),
                            m("B", net, "icr") > m("C", net, "icr")))
    safety_count = sum(safety_wins)
    safety = "supported" if safety_count >= 8 else "partially_supported" if safety_count >= 3 else "not_supported"
    def accuracy(first: str, second: str) -> str:
        wins = sum(m(first, net, f"{bound}_wmape") < m(second, net, f"{bound}_wmape")
                   for net in ("Time", "Energy") for bound in ("lower", "upper"))
        return "supported" if wins == 4 else "partially_supported" if wins >= 2 else "not_supported"
    e_invalid = sum(int(metrics[("E", net)]["invalid_interval_count"]) for net in ("Time", "Energy"))
    b_invalid = sum(int(metrics[("B", net)]["invalid_interval_count"]) for net in ("Time", "Energy"))
    return {"alpha_contribution": "supported" if alpha else "not_supported",
            "safety_loss_contribution": safety,
            "dual_accuracy_contribution": accuracy("B", "D"),
            "dual_physical_decoupling_contribution": "supported" if d_invariance_max > 1e-12 else "not_supported",
            "center_radius_accuracy_contribution": accuracy("B", "E"),
            "center_radius_structural_validity_contribution": "supported" if b_invalid == 0 and e_invalid > 0 else "not_supported"}

