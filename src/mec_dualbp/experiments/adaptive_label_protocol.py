"""Frozen scientific contract for Adaptive-B4-Reference-v1 confirmation."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from mec_dualbp.experiments.exp1_formal_protocol import (
    B4_FORMAL_CANDIDATE,
    DENSE_LHS_FORMAL,
    REFERENCE_GA_FORMAL,
    SAMPLE_TYPES,
    audit_formal_samples,
)
from mec_dualbp.experiments.exp1_preview_protocol import PreviewSample, canonical_json_hash, preview_sample_payload
from mec_dualbp.inner_ga.tasks import EdgeTaskParameters
from mec_dualbp.uncertainty.interval_params import Interval

PROTOCOL_NAME = "Adaptive-B4-Reference-v1"
CONFIRMATION_SAMPLE_SEED = 2026080901
CONFIRMATION_DENSE_SEED_BASE = 2026081000
FUTURE_BP_SEED_RANGE = (2027000000, 2027999999)
STABILITY_GATE_THRESHOLD = 0.01
DENOMINATOR_EPSILON = 1e-12
CONFIRMATION_THRESHOLDS = {
    "median_relative_miss_gap_max": 0.005,
    "q95_relative_miss_gap_max": 0.015,
    "max_relative_miss_gap_max": 0.03,
    "gamma_task_q95_relative_miss_gap_max": 0.02,
    "q95_effective_relative_response_std_max": 0.01,
    "max_effective_relative_response_std_max": 0.02,
    "failed_seed_runs_max": 0,
    "boundary_inversions_max": 0,
    "infeasible_best_z_max": 0,
    "cache_collision_error_max": 0,
    "endpoint_snap_max_ulp": 16,
    "reproducibility_required": True,
    "missing_units_max": 0,
    "hash_mismatch_units_max": 0,
}


def _positive_interval(center: float, radius_ratio: float) -> Interval:
    return Interval(center * (1.0 - radius_ratio), center * (1.0 + radius_ratio))


def generate_confirmation_samples(seed: int = CONFIRMATION_SAMPLE_SEED) -> tuple[PreviewSample, ...]:
    """Generate 5×6×2 new SI samples in an isolated namespace."""

    rng = np.random.default_rng(seed)
    samples: list[PreviewSample] = []
    for gamma in range(1, 6):
        for sample_type in SAMPLE_TYPES:
            for replicate in range(1, 3):
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
                payload = preview_sample_payload(float(gamma), sample_type, parameters, local)
                samples.append(PreviewSample(
                    f"CF_G{gamma}_{sample_type}_{replicate:02d}", float(gamma), sample_type,
                    seed, parameters, local, canonical_json_hash(payload),
                ))
    if len(samples) != 60 or len({sample.sample_id for sample in samples}) != 60:
        raise ArithmeticError("Confirmation requires 60 unique samples")
    if len({sample.content_hash for sample in samples}) != 60:
        raise ArithmeticError("Confirmation content hashes must be unique")
    return tuple(samples)


@dataclass(frozen=True, slots=True)
class StabilitySummary:
    mean: float
    std: float
    relative_std: float


def response_stability(responses: Sequence[float], epsilon: float = DENOMINATOR_EPSILON) -> StabilitySummary:
    values = tuple(float(value) for value in responses)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("responses must be nonempty and finite")
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    return StabilitySummary(mean, std, std / max(abs(mean), epsilon))


def aggregate_boundary(direction: Literal["minimize", "maximize"], responses: Sequence[float]) -> float:
    values = tuple(float(value) for value in responses)
    if not values:
        raise ValueError("responses must be nonempty")
    if direction == "minimize":
        return min(values)
    if direction == "maximize":
        return max(values)
    raise ValueError("direction must be minimize or maximize")


def gate_reasons(
    responses: Sequence[float], *, expected_seed_count: int = 5,
    failed_seed_runs: int = 0, infeasible_best_z: int = 0,
    boundary_direction_abnormal: bool = False, evaluation_identity_failures: int = 0,
    cache_collision_error: bool = False, endpoint_snap_max_ulp: float = 0.0,
    reproducibility_failure: bool = False,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if len(responses) != expected_seed_count:
        reasons.append("seed_result_missing")
    if failed_seed_runs:
        reasons.append("seed_execution_failed")
    if any(not math.isfinite(float(value)) for value in responses):
        reasons.append("non_finite_response")
    if infeasible_best_z:
        reasons.append("infeasible_best_z")
    if boundary_direction_abnormal:
        reasons.append("boundary_direction_abnormal")
    if evaluation_identity_failures:
        reasons.append("evaluation_identity_failure")
    if cache_collision_error:
        reasons.append("cache_collision_error")
    if endpoint_snap_max_ulp > 16:
        reasons.append("endpoint_snap_over_16_ulp")
    if reproducibility_failure:
        reasons.append("reproducibility_failure")
    if len(responses) == expected_seed_count and not any(not math.isfinite(float(value)) for value in responses):
        if response_stability(responses).relative_std > STABILITY_GATE_THRESHOLD:
            reasons.append("relative_response_std_over_0.01")
    return tuple(reasons)


def adaptive_label(direction: Literal["minimize", "maximize"], b4_label: float, reference_label: float, dense_response: float, gate_triggered: bool) -> tuple[float, str]:
    if not gate_triggered:
        return float(b4_label), "b4_stable"
    envelope = min(reference_label, dense_response) if direction == "minimize" else max(reference_label, dense_response)
    return float(envelope), "high_fidelity_approximate_label"


def protocol_config() -> dict[str, Any]:
    return {
        "protocol_name": PROTOCOL_NAME,
        "workers": 1, "execution_mode": "serial",
        "b4": B4_FORMAL_CANDIDATE,
        "reference_ga": REFERENCE_GA_FORMAL,
        "dense_lhs": DENSE_LHS_FORMAL | {"evaluation_mode": "vectorized_float64", "chunk_size": 5000},
        "stability_gate_threshold": STABILITY_GATE_THRESHOLD,
        "stability_ddof": 0, "denominator_epsilon": DENOMINATOR_EPSILON,
        "confirmation_sample_seed": CONFIRMATION_SAMPLE_SEED,
        "confirmation_dense_seed_base": CONFIRMATION_DENSE_SEED_BASE,
        "future_bp_seed_range": list(FUTURE_BP_SEED_RANGE),
        "thresholds": CONFIRMATION_THRESHOLDS,
        "not_for_bp_training": True, "not_for_bp_validation": True,
        "not_for_bp_calibration": True, "not_for_bp_test": True,
        "five_candidate_certification_used": False,
    }
