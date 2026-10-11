"""Exp-3 数据绑定、加权和诊断合同。"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

ABLATION_SEEDS = (31001, 31002, 31003, 31004, 31005)


def stable_ordered_union(time_features: Sequence[str], energy_features: Sequence[str]) -> tuple[str, ...]:
    if len(set(time_features)) != len(time_features) or len(set(energy_features)) != len(energy_features):
        raise ValueError("source feature schemas must not contain duplicates")
    result = list(time_features)
    result.extend(field for field in energy_features if field not in set(time_features))
    return tuple(result)


def assert_ablation_data_access(role: str, *, target_access: bool) -> None:
    if role in {"Train", "Validation"}:
        return
    if role in {"Calibration", "Test"} and not target_access:
        return
    raise ValueError(f"{role} targets are sealed during Exp-3 training protocol")


def bind_unified_energy_by_base(
    time_records: Sequence[Mapping[str, Any]], energy_by_base: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """按 base 绑定 Energy；重复行权重归一为每个 base 合计 1。"""
    counts = Counter(str(row["base_sample_id"]) for row in time_records)
    if not counts or any(count <= 0 for count in counts.values()):
        raise ValueError("nonempty Time records required")
    missing = set(counts) - set(energy_by_base)
    if missing:
        raise ValueError(f"missing Energy base bindings: {sorted(missing)[:3]}")
    bound: list[dict[str, Any]] = []
    for row in time_records:
        base_id = str(row["base_sample_id"])
        energy = energy_by_base[base_id]
        if str(energy.get("role")) != str(row.get("role")):
            raise ValueError("role mismatch in Unified base binding")
        merged = dict(row)
        merged["energy_lower"] = energy["energy_lower"]
        merged["energy_upper"] = energy["energy_upper"]
        merged["energy_repeat_weight"] = 1.0 / counts[base_id]
        bound.append(merged)
    return bound


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    value_array = np.asarray(values, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    if value_array.shape != weight_array.shape or value_array.size == 0 or np.any(weight_array < 0):
        raise ValueError("aligned values and nonnegative weights required")
    total = float(weight_array.sum())
    if total <= 0:
        raise ValueError("positive total weight required")
    return float(np.dot(value_array, weight_array) / total)


def energy_f_invariance_diagnostic(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """汇总同一 base 在不同 f 下 Energy 上下界预测的相对 range。"""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        center = 0.5 * (float(row["energy_lower_pred"]) + float(row["energy_upper_pred"]))
        grouped[str(row["base_sample_id"])].append(center)
    if not grouped or any(len(values) < 2 for values in grouped.values()):
        raise ValueError("at least two f records per base required")
    stds, ranges, relative = [], [], []
    for values in grouped.values():
        array = np.asarray(values, dtype=np.float64)
        spread = float(np.ptp(array))
        stds.append(float(np.std(array)))
        ranges.append(spread)
        relative.append(spread / max(float(np.mean(np.abs(array))), 1e-12))
    rel = np.asarray(relative)
    return {"base_count": float(len(grouped)), "within_base_energy_pred_std_mean": float(np.mean(stds)),
            "within_base_energy_pred_range_mean": float(np.mean(ranges)), "relative_range_mean": float(np.mean(rel)),
            "relative_range_q95": float(np.quantile(rel, 0.95)), "relative_range_max": float(np.max(rel))}


def future_training_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for variant, networks in (("C_w_o_safety", ("Time", "Energy")),
                              ("D_unified_bp", ("Unified",)),
                              ("E_direct_bounds", ("Time", "Energy"))):
        for network in networks:
            for seed in ABLATION_SEEDS:
                tasks.append({"task_id": f"EXP3_{variant}_{network}_{seed}", "variant": variant,
                              "network": network, "training_seed": seed, "status": "planned",
                              "allowed_roles": "Train|Validation", "calibration_targets_accessed": False,
                              "test_targets_accessed": False})
    return tasks

