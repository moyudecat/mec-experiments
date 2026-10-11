"""Frozen DualBP dataset roles, feature schemas, manifests, and label schema.

This module generates inputs only.  It never starts formal label generation or
trains a neural network.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from mec_dualbp.experiments.adaptive_label_protocol import PROTOCOL_NAME
from mec_dualbp.experiments.exp1_formal_protocol import (
    SAMPLE_TYPES,
    normalized_distance,
    numeric_signature,
)
from mec_dualbp.experiments.exp1_preview_protocol import (
    PreviewSample,
    canonical_json_hash,
    preview_sample_payload,
)
from mec_dualbp.inner_ga.tasks import EdgeTaskParameters
from mec_dualbp.uncertainty.interval_params import Interval

DATASET_PROTOCOL_VERSION = "DualBP-Dataset-v1"
FEATURE_SCHEMA_VERSION = "DualBP-Features-v1"
LABEL_SCHEMA_VERSION = "DualBP-Labels-v1"
NEAR_DUPLICATE_TOLERANCE = 1e-6
GAMMAS = (1, 2, 3, 4, 5)
ROLES = ("Train", "Validation", "Calibration", "Test")
FORMAL_ROLE_COUNTS = {"Train": 420, "Validation": 60, "Calibration": 60, "Test": 60}
REPLICATES_PER_STRATUM = {"Train": 14, "Validation": 2, "Calibration": 2, "Test": 2}
ROLE_SEED_BASES = {
    "Train": 2_027_100_000,
    "Validation": 2_027_200_000,
    "Calibration": 2_027_300_000,
    "Test": 2_027_400_000,
}
SMOKE_SEED_BASE = 2_027_500_000
PROBE_SEED_BASE = 2_027_600_000

# Fourth chapter 4.2.1/4.3.1 + frozen physics contract.
TIME_FEATURES = (
    "edge_cpu_frequency_cycle_per_s",
    "D_in_bit_lower", "D_in_bit_upper",
    "D_out_bit_lower", "D_out_bit_upper",
    "C_cycle_lower", "C_cycle_upper",
    "I_u_w_lower", "I_u_w_upper",
    "I_d_w_lower", "I_d_w_upper",
    "uplink_bandwidth_hz", "downlink_bandwidth_hz",
    "uplink_terminal_tx_power_w", "downlink_bs_tx_power_w",
    "uplink_channel_gain_linear", "downlink_channel_gain_linear",
    "noise_power_w", "Gamma",
)

# Fourth chapter 4.2.1/4.3.2 + terminal energy physics.  Downlink rate uses
# base-station transmit power, while terminal energy uses terminal P_r.
ENERGY_FEATURES = (
    "D_in_bit_lower", "D_in_bit_upper",
    "D_out_bit_lower", "D_out_bit_upper",
    "I_u_w_lower", "I_u_w_upper",
    "I_d_w_lower", "I_d_w_upper",
    "uplink_bandwidth_hz", "downlink_bandwidth_hz",
    "uplink_terminal_tx_power_w", "downlink_bs_tx_power_w",
    "terminal_rx_power_w",
    "uplink_channel_gain_linear", "downlink_channel_gain_linear",
    "noise_power_w", "Gamma",
)

LABEL_FIELDS = (
    "sample_id", "role", "generation_seed", "Gamma", "sample_type",
    "time_lower", "time_upper", "time_center", "time_radius",
    "energy_lower", "energy_upper", "energy_center", "energy_radius",
    "time_label_route_lower", "time_label_route_upper",
    "energy_label_route_lower", "energy_label_route_upper",
    "time_b4_relative_response_std_lower", "time_b4_relative_response_std_upper",
    "energy_b4_relative_response_std_lower", "energy_b4_relative_response_std_upper",
    "time_gate_triggered_lower", "time_gate_triggered_upper",
    "energy_gate_triggered_lower", "energy_gate_triggered_upper",
    "time_gate_trigger_reason_lower", "time_gate_trigger_reason_upper",
    "energy_gate_trigger_reason_lower", "energy_gate_trigger_reason_upper",
    "label_protocol_name", "label_protocol_config_hash", "dataset_config_hash",
    "ga_ref_time_lower", "ga_ref_time_upper", "ga_ref_energy_lower", "ga_ref_energy_upper",
    "raw_pred_time_lower", "raw_pred_time_upper", "raw_pred_energy_lower", "raw_pred_energy_upper",
    "corr_pred_time_lower", "corr_pred_time_upper", "corr_pred_energy_lower", "corr_pred_energy_upper",
    "alpha_T_lower", "alpha_T_upper", "alpha_E_lower", "alpha_E_upper",
    "alpha_calibration_status",
)


@dataclass(frozen=True, slots=True)
class DatasetSample:
    sample_id: str
    role: str
    gamma: float
    sample_type: str
    generation_seed: str
    generation_seed_value: int
    parameters: EdgeTaskParameters
    local_cpu_frequency: Interval
    content_hash: str
    formal_eligible: bool

    def as_preview(self) -> PreviewSample:
        return PreviewSample(
            self.sample_id, self.gamma, self.sample_type, self.generation_seed_value,
            self.parameters, self.local_cpu_frequency, self.content_hash,
        )


def _positive_interval(center: float, radius_ratio: float) -> Interval:
    if not 0.0 <= radius_ratio < 1.0:
        raise ValueError("relative radius must lie in [0,1)")
    return Interval(center * (1.0 - radius_ratio), center * (1.0 + radius_ratio))


def _sample_from_seed(
    *, sample_id: str, role: str, gamma: int, sample_type: str,
    seed_name: str, seed_value: int, formal_eligible: bool,
) -> DatasetSample:
    rng = np.random.default_rng(seed_value)
    base = rng.uniform(0.805, 1.295)
    if sample_type == "narrow":
        radius = rng.uniform(0.012, 0.048)
    elif sample_type == "wide_stress":
        radius = rng.uniform(0.54, 0.73)
    elif sample_type == "near_degenerate":
        radius = rng.uniform(1.1e-7, 9.0e-7)
    else:
        radius = rng.uniform(0.14, 0.39)
    wireless_scale, compute_scale = rng.uniform(0.82, 1.28, size=2)
    wireless_radius, compute_radius = min(radius * 0.9, 0.72), min(radius * 0.8, 0.72)
    if sample_type == "high_wireless_sensitivity":
        wireless_scale *= rng.uniform(2.05, 3.25)
        wireless_radius = max(radius, 0.56)
    if sample_type == "high_compute_sensitivity":
        compute_scale *= rng.uniform(1.85, 2.65)
        compute_radius = max(radius, 0.51)
    parameters = EdgeTaskParameters(
        _positive_interval(2.1e6 * base, radius),
        _positive_interval(0.85e6 * base, min(radius * 0.85, 0.76)),
        _positive_interval(2.2e9 * compute_scale, compute_radius),
        _positive_interval(2.1e-10 * wireless_scale, wireless_radius),
        _positive_interval(2.6e-10 * wireless_scale, min(wireless_radius * 0.9, 0.76)),
        1.2e6, 2.0e6, 0.22, 1.1, 0.09, 1.1e-6, 1.8e-6, 1.0e-10, 5.0e9,
    )
    local = _positive_interval(2.3e9 * compute_scale, min(compute_radius * 0.75, 0.71))
    content_hash = canonical_json_hash(preview_sample_payload(float(gamma), sample_type, parameters, local))
    return DatasetSample(sample_id, role, float(gamma), sample_type, seed_name, seed_value,
                         parameters, local, content_hash, formal_eligible)


def generate_formal_dataset_samples() -> tuple[DatasetSample, ...]:
    samples: list[DatasetSample] = []
    for role in ROLES:
        ordinal = 0
        prefix = f"BP_{role.upper()}"
        for gamma in GAMMAS:
            for sample_type in SAMPLE_TYPES:
                for replicate in range(1, REPLICATES_PER_STRATUM[role] + 1):
                    ordinal += 1
                    seed_value = ROLE_SEED_BASES[role] + ordinal
                    samples.append(_sample_from_seed(
                        sample_id=f"{prefix}_G{gamma}_{sample_type}_{replicate:02d}", role=role,
                        gamma=gamma, sample_type=sample_type,
                        seed_name=f"{prefix}_{ordinal:06d}", seed_value=seed_value,
                        formal_eligible=True,
                    ))
    audit_role_balance(samples)
    return tuple(samples)


def generate_smoke_samples() -> tuple[DatasetSample, ...]:
    samples: list[DatasetSample] = []
    ordinal = 0
    for gamma in GAMMAS:
        for sample_type in SAMPLE_TYPES:
            for replicate in range(1, 3):
                ordinal += 1
                seed_value = SMOKE_SEED_BASE + ordinal
                samples.append(_sample_from_seed(
                    sample_id=f"BP_LABEL_SMOKE_G{gamma}_{sample_type}_{replicate:02d}", role="Smoke",
                    gamma=gamma, sample_type=sample_type,
                    seed_name=f"BP_LABEL_SMOKE_{ordinal:06d}", seed_value=seed_value,
                    formal_eligible=False,
                ))
    if len(samples) != 60:
        raise ArithmeticError("label smoke requires 60 samples")
    return tuple(samples)


def generate_probe_sample() -> DatasetSample:
    return _sample_from_seed(
        sample_id="BP_LABEL_ESCALATION_PROBE_G2_01", role="Probe", gamma=2,
        sample_type="high_wireless_sensitivity", seed_name="BP_LABEL_PROBE_000001",
        seed_value=PROBE_SEED_BASE + 1, formal_eligible=False,
    )


def audit_role_balance(samples: Sequence[DatasetSample]) -> dict[str, Any]:
    role_counts = Counter(sample.role for sample in samples)
    if dict(role_counts) != FORMAL_ROLE_COUNTS:
        raise ValueError(f"invalid formal role counts: {dict(role_counts)}")
    strata: dict[str, dict[str, int]] = {}
    for role in ROLES:
        counts = Counter((int(sample.gamma), sample.sample_type) for sample in samples if sample.role == role)
        expected = REPLICATES_PER_STRATUM[role]
        if len(counts) != 30 or any(counts[(gamma, sample_type)] != expected for gamma in GAMMAS for sample_type in SAMPLE_TYPES):
            raise ValueError(f"role {role} is not balanced")
        strata[role] = {f"Gamma={gamma}|sample_type={sample_type}": counts[(gamma, sample_type)]
                        for gamma in GAMMAS for sample_type in SAMPLE_TYPES}
    return {"passed": True, "role_counts": dict(role_counts), "strata": strata}


def _interval_values(interval: Interval) -> tuple[float, float, float, float]:
    return interval.lower, interval.upper, interval.center, interval.radius


def sample_manifest_row(sample: DatasetSample, dataset_config_hash: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_id": sample.sample_id, "role": sample.role, "Gamma": sample.gamma,
        "gamma": sample.gamma, "sample_type": sample.sample_type,
        "generation_seed": sample.generation_seed, "generation_seed_value": sample.generation_seed_value,
        "seed_namespace": sample.generation_seed.rsplit("_", 1)[0] + "_*",
        "content_hash": sample.content_hash, "units": "SI",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "dataset_protocol_version": DATASET_PROTOCOL_VERSION,
        "dataset_config_hash": dataset_config_hash,
        "label_protocol_name": PROTOCOL_NAME, "label_status": "not_generated",
        "execution_status": "queued", "not_for_formal_bp_dataset": not sample.formal_eligible,
        "not_for_bp_training": sample.role != "Train",
        "not_for_bp_validation": sample.role != "Validation",
        "not_for_bp_calibration": sample.role != "Calibration",
        "not_for_bp_test": sample.role != "Test",
    }
    for prefix, interval in (
        ("D_in_bit", sample.parameters.input_size_bit),
        ("D_out_bit", sample.parameters.output_size_bit),
        ("C_cycle", sample.parameters.cpu_cycles),
        ("I_u_w", sample.parameters.uplink_interference_w),
        ("I_d_w", sample.parameters.downlink_interference_w),
        ("F_L_cycle_per_s", sample.local_cpu_frequency),
    ):
        lower, upper, center, radius = _interval_values(interval)
        row.update({f"{prefix}_lower": lower, f"{prefix}_upper": upper,
                    f"{prefix}_center": center, f"{prefix}_radius": radius})
    p = sample.parameters
    for name in (
        "uplink_bandwidth_hz", "downlink_bandwidth_hz", "uplink_terminal_tx_power_w",
        "downlink_bs_tx_power_w", "terminal_rx_power_w", "uplink_channel_gain_linear",
        "downlink_channel_gain_linear", "noise_power_w", "edge_cpu_frequency_cycle_per_s",
    ):
        row[name] = getattr(p, name)
    return row


def sample_from_manifest_row(row: Mapping[str, Any]) -> DatasetSample:
    interval = lambda prefix: Interval(float(row[f"{prefix}_lower"]), float(row[f"{prefix}_upper"]))
    parameters = EdgeTaskParameters(
        interval("D_in_bit"), interval("D_out_bit"), interval("C_cycle"),
        interval("I_u_w"), interval("I_d_w"),
        float(row["uplink_bandwidth_hz"]), float(row["downlink_bandwidth_hz"]),
        float(row["uplink_terminal_tx_power_w"]), float(row["downlink_bs_tx_power_w"]),
        float(row["terminal_rx_power_w"]), float(row["uplink_channel_gain_linear"]),
        float(row["downlink_channel_gain_linear"]), float(row["noise_power_w"]),
        float(row["edge_cpu_frequency_cycle_per_s"]),
    )
    return DatasetSample(
        str(row["sample_id"]), str(row["role"]), float(row.get("Gamma", row.get("gamma"))),
        str(row["sample_type"]), str(row["generation_seed"]), int(row["generation_seed_value"]),
        parameters, interval("F_L_cycle_per_s"), str(row["content_hash"]),
        str(row.get("not_for_formal_bp_dataset", "False")).lower() not in ("true", "1"),
    )


def signature_from_row(row: Mapping[str, Any]) -> np.ndarray:
    names = [f"{prefix}_{bound}" for prefix in (
        "D_in_bit", "D_out_bit", "C_cycle", "I_u_w", "I_d_w", "F_L_cycle_per_s"
    ) for bound in ("lower", "upper")]
    return np.asarray([float(row[name]) for name in names], dtype=np.float64)


def pairwise_duplicate_audit(
    left: Sequence[DatasetSample], right_sets: Mapping[str, Sequence[tuple[str, np.ndarray]]],
    *, include_internal: bool = True, tolerance: float = NEAR_DUPLICATE_TOLERANCE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    left_signatures = [(sample.sample_id, numeric_signature(sample.as_preview())) for sample in left]
    duplicate_rows: list[dict[str, Any]] = []
    near_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    comparisons = ({"Current": left_signatures} if include_internal else {}) | dict(right_sets)
    for split, right in comparisons.items():
        exact = near = 0
        for index, (left_id, first) in enumerate(left_signatures):
            candidates = right[index + 1:] if split == "Current" else right
            for right_id, second in candidates:
                if np.array_equal(first, second):
                    exact += 1
                    duplicate_rows.append({"sample_id": left_id, "other_split": split, "other_id": right_id})
                else:
                    distance = normalized_distance(first, second)
                    if distance <= tolerance:
                        near += 1
                        near_rows.append({"sample_id": left_id, "other_split": split,
                                          "other_id": right_id, "normalized_distance": distance})
        summary[split] = {"exact_duplicate_pairs": exact, "near_duplicate_pairs": near}
    return duplicate_rows, near_rows, summary


def center_radius(lower: float, upper: float) -> tuple[float, float]:
    lower, upper = float(lower), float(upper)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("bounds must be finite")
    if lower < 0.0 or upper < lower:
        raise ValueError("bounds must satisfy 0 <= lower <= upper")
    center, radius = (lower + upper) / 2.0, (upper - lower) / 2.0
    if center < radius or radius < 0.0:
        raise ArithmeticError("invalid center-radius conversion")
    return center, radius


def build_label_record(
    sample: DatasetSample, task_records: Mapping[str, Mapping[str, Any]],
    *, dataset_config_hash: str, label_protocol_config_hash: str,
) -> dict[str, Any]:
    required = (
        "edge_core_latency_lower", "edge_core_latency_upper",
        "edge_comm_energy_lower", "edge_comm_energy_upper",
    )
    if any(name not in task_records for name in required):
        raise ValueError("all four task records are required")
    tl, tu, el, eu = (float(task_records[name]["adaptive_label"]) for name in required)
    tc, tr = center_radius(tl, tu)
    ec, er = center_radius(el, eu)
    row: dict[str, Any] = {
        "sample_id": sample.sample_id, "role": sample.role, "generation_seed": sample.generation_seed,
        "Gamma": sample.gamma, "sample_type": sample.sample_type,
        "time_lower": tl, "time_upper": tu, "time_center": tc, "time_radius": tr,
        "energy_lower": el, "energy_upper": eu, "energy_center": ec, "energy_radius": er,
        "label_protocol_name": PROTOCOL_NAME, "label_protocol_config_hash": label_protocol_config_hash,
        "dataset_config_hash": dataset_config_hash,
        "ga_ref_time_lower": tl, "ga_ref_time_upper": tu,
        "ga_ref_energy_lower": el, "ga_ref_energy_upper": eu,
        "alpha_T_lower": None, "alpha_T_upper": None, "alpha_E_lower": None, "alpha_E_upper": None,
        "alpha_calibration_status": "not_started",
    }
    for prefix in ("raw_pred", "corr_pred"):
        for response in ("time_lower", "time_upper", "energy_lower", "energy_upper"):
            row[f"{prefix}_{response}"] = None
    task_mapping = {
        "time_lower": required[0], "time_upper": required[1],
        "energy_lower": required[2], "energy_upper": required[3],
    }
    for label_name, task_name in task_mapping.items():
        item = task_records[task_name]
        row[f"{label_name.rsplit('_', 1)[0]}_label_route_{label_name.rsplit('_', 1)[1]}"] = item["selected_label_route"]
        row[f"{label_name.rsplit('_', 1)[0]}_b4_relative_response_std_{label_name.rsplit('_', 1)[1]}"] = item["b4_relative_response_std"]
        row[f"{label_name.rsplit('_', 1)[0]}_gate_triggered_{label_name.rsplit('_', 1)[1]}"] = item["gate_triggered"]
        row[f"{label_name.rsplit('_', 1)[0]}_gate_trigger_reason_{label_name.rsplit('_', 1)[1]}"] = json.dumps(item["gate_trigger_reasons"], ensure_ascii=False)
    missing = [field for field in LABEL_FIELDS if field not in row]
    if missing:
        raise AssertionError(f"label record fields missing: {missing}")
    return row


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    names = list(fields or (list(rows[0]) if rows else []))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_signature_csv(path: Path) -> list[tuple[str, np.ndarray]]:
    with path.open(encoding="utf-8-sig") as stream:
        return [(row["sample_id"], signature_from_row(row)) for row in csv.DictReader(stream)]
