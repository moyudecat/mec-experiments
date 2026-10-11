"""Deterministic Time-BP Dataset V2 input expansion; no label computation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mec_dualbp.datasets.protocol import TIME_FEATURES


F_FIELD = "edge_cpu_frequency_cycle_per_s"
F_LEVELS = ("existing_upper_anchor", "new_low_stratum", "new_mid_high_stratum")
ROLES = ("Train", "Validation", "Calibration", "Test")
V1_TIME_HASH_FIELDS = (
    "sample_id", "role", "Gamma", "sample_type",
    "time_lower", "time_upper", "time_center", "time_radius",
    "time_label_route_lower", "time_label_route_upper",
    "time_b4_relative_response_std_lower", "time_b4_relative_response_std_upper",
    "time_gate_triggered_lower", "time_gate_triggered_upper",
    "time_gate_trigger_reason_lower", "time_gate_trigger_reason_upper",
    "label_protocol_name", "label_protocol_config_hash", "label_protocol_hash",
    "dataset_config_hash", "feature_schema_hash", "label_schema_hash",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_parameters(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "parameter_version": "edge_cpu_formal_parameters_v1",
        "f_min_E_cycle_per_s": 1.5e9,
        "f_max_E_cycle_per_s": 5.0e9,
        "F0_cycle_per_s": 20.0e9,
        "default_F_max_cycle_per_s": 20.0e9,
        "active_user_domain_only": True,
        "local_user_f_value": 0.0,
        "time_bp_called_when_m_equals_zero": False,
        "source_type": "author_decision",
        "decision_date": "2026-08-10",
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"invalid frozen edge CPU parameter {key}")
    if not 0.0 < value["f_min_E_cycle_per_s"] < value["f_max_E_cycle_per_s"]:
        raise ValueError("active edge CPU domain must be positive and non-degenerate")
    return value


def seed_from_namespace(namespace: str) -> int:
    """Stable 64-bit seed; deliberately independent of Python's randomized hash."""
    return int.from_bytes(hashlib.sha256(namespace.encode("ascii")).digest()[:8], "big")


def _stratified_values(count: int, lower: float, upper: float, namespace: str) -> np.ndarray:
    if count <= 0 or not 0.0 < lower < upper:
        raise ValueError("invalid one-dimensional stratum")
    rng = np.random.default_rng(seed_from_namespace(namespace))
    positions = (np.arange(count, dtype=np.float64) + rng.random(count)) / np.float64(count)
    values = np.float64(lower) + positions * np.float64(upper - lower)
    values = np.minimum(values, np.nextafter(np.float64(upper), np.float64(lower)))
    return values[rng.permutation(count)].astype(np.float64, copy=False)


def role_frequency_values(role: str, count: int, f_min: float, f_max: float) -> tuple[np.ndarray, np.ndarray]:
    if role not in ROLES or count < 2:
        raise ValueError("unsupported role or insufficient role size")
    midpoint = np.float64((f_min + f_max) / 2.0)
    low = np.empty(count, dtype=np.float64)
    low[0] = np.float64(f_min)
    interior = _stratified_values(count - 1, f_min, float(midpoint), f"TIME_V2_{role.upper()}_LOW")
    interior = np.maximum(interior, np.nextafter(np.float64(f_min), np.float64(midpoint)))
    low[1:] = interior
    mid_high = _stratified_values(count, float(midpoint), f_max, f"TIME_V2_{role.upper()}_MIDHIGH")
    return low, mid_high


def v1_time_label_hash(label: Mapping[str, Any]) -> str:
    return canonical_hash({field: label.get(field, "") for field in V1_TIME_HASH_FIELDS})


def build_manifest(
    base_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    parameters: Mapping[str, Any],
    *,
    parameter_config_hash: str,
    v2_config_hash: str,
) -> list[dict[str, Any]]:
    labels = {str(row["sample_id"]): row for row in label_rows}
    if len(base_rows) != 600 or len(labels) != 600:
        raise ValueError("V2 must bind exactly 600 base rows and labels")
    f_min = float(parameters["f_min_E_cycle_per_s"])
    f_max = float(parameters["f_max_E_cycle_per_s"])
    midpoint = (f_min + f_max) / 2.0
    by_role = {role: [row for row in base_rows if row["role"] == role] for role in ROLES}
    frequency_maps: dict[str, tuple[np.ndarray, np.ndarray]] = {
        role: role_frequency_values(role, len(rows), f_min, f_max) for role, rows in by_role.items()
    }
    rows: list[dict[str, Any]] = []
    for role in ROLES:
        low_values, mid_values = frequency_maps[role]
        for index, base in enumerate(by_role[role]):
            sample_id = str(base["sample_id"])
            label = labels[sample_id]
            label_hash = v1_time_label_hash(label)
            specs = (
                ("existing_upper_anchor", f_max, "existing_v1", "reused"),
                ("new_low_stratum", float(low_values[index]), "to_be_generated_v2", "queued"),
                ("new_mid_high_stratum", float(mid_values[index]), "to_be_generated_v2", "queued"),
            )
            for level, frequency, source, status in specs:
                row: dict[str, Any] = {
                    "time_record_id": f"{sample_id}::{level}",
                    "base_sample_id": sample_id,
                    "role": role,
                    "Gamma": float(base["Gamma"]),
                    "sample_type": str(base["sample_type"]),
                    "generation_seed": str(base["generation_seed"]),
                    "f_level_role": level,
                    "f_generation_namespace": f"TIME_V2_{role.upper()}_{'EXISTING' if level.startswith('existing') else 'LOW' if level.startswith('new_low') else 'MIDHIGH'}",
                    "edge_cpu_frequency_cycle_per_s": frequency,
                    "active_edge_user_only": True,
                    "m_zero_excluded": True,
                }
                for field in TIME_FEATURES:
                    if field != F_FIELD:
                        row[field] = float(base[field])
                existing = level == "existing_upper_anchor"
                for field in ("time_lower", "time_upper", "time_center", "time_radius"):
                    row[field] = label[field] if existing else ""
                for field in (
                    "time_label_route_lower", "time_label_route_upper",
                    "time_b4_relative_response_std_lower", "time_b4_relative_response_std_upper",
                    "time_gate_triggered_lower", "time_gate_triggered_upper",
                    "time_gate_trigger_reason_lower", "time_gate_trigger_reason_upper",
                ):
                    row[field] = label.get(field, "") if existing else ""
                row.update({
                    "label_source": source, "execution_status": status,
                    "source_v1_sample_id": sample_id, "source_v1_label_hash": label_hash,
                    "edge_cpu_parameter_config_hash": parameter_config_hash,
                    "time_bp_v2_config_hash": v2_config_hash,
                    "adaptive_protocol_hash": label["label_protocol_hash"],
                    "midpoint_cycle_per_s": midpoint,
                })
                rows.append(row)
    return rows


def audit_manifest(
    rows: Sequence[Mapping[str, Any]],
    base_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    f_min = float(parameters["f_min_E_cycle_per_s"])
    f_max = float(parameters["f_max_E_cycle_per_s"])
    midpoint = (f_min + f_max) / 2.0
    base_by_id = {str(row["sample_id"]): row for row in base_rows}
    label_by_id = {str(row["sample_id"]): row for row in label_rows}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["base_sample_id"])].append(row)
    role_leakage = sum(len({row["role"] for row in group}) != 1 for group in grouped.values())
    wrong_per_base = sum(len(group) != 3 for group in grouped.values())
    duplicate_f = sum(len({float(row[F_FIELD]) for row in group}) != len(group) for group in grouped.values())
    non_f_changed = 0
    for sample_id, group in grouped.items():
        base = base_by_id[sample_id]
        for row in group:
            if any(float(row[field]) != float(base[field]) for field in TIME_FEATURES if field != F_FIELD):
                non_f_changed += 1
    signatures = [tuple(float(row[field]) for field in TIME_FEATURES) for row in rows]
    full_duplicates = len(signatures) - len(set(signatures))
    existing = [row for row in rows if row["f_level_role"] == "existing_upper_anchor"]
    mismatches = sum(
        row["source_v1_label_hash"] != v1_time_label_hash(label_by_id[str(row["base_sample_id"])])
        or any(str(row[field]) != str(label_by_id[str(row["base_sample_id"])][field])
               for field in ("time_lower", "time_upper", "time_center", "time_radius"))
        for row in existing
    )
    result = {
        "total_rows": len(rows), "base_samples": len(grouped),
        "role_counts": dict(Counter(str(row["role"]) for row in rows)),
        "level_counts": dict(Counter(str(row["f_level_role"]) for row in rows)),
        "queued_rows": sum(row["execution_status"] == "queued" for row in rows),
        "future_time_tasks": 2 * sum(row["execution_status"] == "queued" for row in rows),
        "full_row_duplicate": full_duplicates, "duplicate_f_inside_base": duplicate_f,
        "f_below_min": sum(float(row[F_FIELD]) < f_min for row in rows),
        "f_above_max": sum(float(row[F_FIELD]) > f_max for row in rows),
        "low_outside_stratum": sum(not (f_min <= float(row[F_FIELD]) < midpoint) for row in rows if row["f_level_role"] == "new_low_stratum"),
        "mid_high_outside_stratum": sum(not (midpoint <= float(row[F_FIELD]) < f_max) for row in rows if row["f_level_role"] == "new_mid_high_stratum"),
        "m0_records": sum(not bool(row["active_edge_user_only"]) or not bool(row["m_zero_excluded"]) for row in rows),
        "role_leakage": role_leakage, "base_group_leakage": role_leakage,
        "base_missing": len(set(base_by_id) - set(grouped)), "base_extra": len(set(grouped) - set(base_by_id)),
        "wrong_records_per_base": wrong_per_base,
        "non_f_feature_changed_inside_base": non_f_changed,
        "existing_label_hash_mismatch": mismatches,
        "existing_5ghz_not_equal_fmax": sum(float(row[F_FIELD]) != f_max for row in existing),
        "lower_anchor_by_role": {role: sum(str(row["role"]) == role and float(row[F_FIELD]) == f_min for row in rows) for role in ROLES},
    }
    expected_roles = {"Train": 1260, "Validation": 180, "Calibration": 180, "Test": 180}
    expected_levels = {level: 600 for level in F_LEVELS}
    zero_keys = ("full_row_duplicate", "duplicate_f_inside_base", "f_below_min", "f_above_max",
                 "low_outside_stratum", "mid_high_outside_stratum", "m0_records", "role_leakage",
                 "base_group_leakage", "base_missing", "base_extra", "wrong_records_per_base",
                 "non_f_feature_changed_inside_base", "existing_label_hash_mismatch",
                 "existing_5ghz_not_equal_fmax")
    result["passed"] = (
        len(rows) == 1800 and len(grouped) == 600 and result["role_counts"] == expected_roles
        and result["level_counts"] == expected_levels and result["queued_rows"] == 1200
        and result["future_time_tasks"] == 2400 and all(result[key] == 0 for key in zero_keys)
        and all(value >= 1 for value in result["lower_anchor_by_role"].values())
    )
    return result


def coverage_by_role(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for role in ROLES:
        values = np.asarray([float(row[F_FIELD]) for row in rows if row["role"] == role], dtype=np.float64)
        result.append({
            "role": role, "count": len(values), "unique_count": len(np.unique(values)),
            "min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()),
            "std": float(values.std(ddof=0)), "q05": float(np.quantile(values, .05)),
            "q25": float(np.quantile(values, .25)), "median": float(np.quantile(values, .5)),
            "q75": float(np.quantile(values, .75)), "q95": float(np.quantile(values, .95)),
        })
    return result
