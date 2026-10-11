"""Frozen Exp-1 Formal design protocol; this module never starts the full run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mec_dualbp.experiments.exp1_preview_protocol import (
    PREREGISTERED_THRESHOLDS,
    PreviewSample,
    canonical_json_hash,
    preview_sample_payload,
)
from mec_dualbp.inner_ga.tasks import EdgeTaskParameters
from mec_dualbp.uncertainty.interval_params import Interval

FORMAL_SAMPLE_SEED = 2026080601
FORMAL_DENSE_SEED_BASE = 2026080700
FORMAL_DRY_RUN_SEED_BASE = 2026080800
BP_RESERVED_SEED_RANGE = (2027000000, 2027999999)

B4_FORMAL_CANDIDATE: dict[str, Any] = {
    "name": "B4_formal_candidate_v1",
    "population_size": 64,
    "generations": 50,
    "tournament_size": 3,
    "crossover_probability": 0.9,
    "mutation_probability": 1 / 6,
    "mutation_eta": 20,
    "elite_count": 2,
    "cache_enabled": True,
    "cache_decimals": 15,
    "absolute_tolerance": 1e-12,
    "include_boundary_anchors": True,
    "seeds": [101, 202, 303, 404, 505],
    "status": "formal_candidate_not_yet_validated",
}

REFERENCE_GA_FORMAL: dict[str, Any] = {
    "name": "Reference-GA_approximate_reference",
    "population_size": 96,
    "generations": 80,
    "tournament_size": 3,
    "crossover_probability": 0.9,
    "mutation_probability": 1 / 6,
    "mutation_eta": 20,
    "elite_count": 2,
    "cache_enabled": True,
    "cache_decimals": 15,
    "absolute_tolerance": 1e-12,
    "include_boundary_anchors": True,
    "seeds": [101, 202, 303, 404, 505, 606, 707, 808],
    "reference_status": "approximate_reference",
}

DENSE_LHS_FORMAL: dict[str, Any] = {
    "name": "Dense-LHS_approximate_reference",
    "n_samples": 50000,
    "sampling_method": "ordinary_lhs_plus_proportional_budget_repair",
    "seed_base": FORMAL_DENSE_SEED_BASE,
    "reference_status": "approximate_reference",
}

SAMPLE_TYPES = (
    "regular", "narrow", "wide_stress", "near_degenerate",
    "high_wireless_sensitivity", "high_compute_sensitivity",
)


def _positive_interval(center: float, radius_ratio: float) -> Interval:
    return Interval(center * (1.0 - radius_ratio), center * (1.0 + radius_ratio))


def generate_formal_samples(seed: int = FORMAL_SAMPLE_SEED) -> tuple[PreviewSample, ...]:
    """Generate 5×6×3 independent SI samples in a dedicated seed namespace."""

    rng = np.random.default_rng(seed)
    samples: list[PreviewSample] = []
    for gamma in range(1, 6):
        for sample_type in SAMPLE_TYPES:
            for replicate in range(1, 4):
                base = rng.uniform(0.805, 1.295)
                if sample_type == "narrow":
                    radius = rng.uniform(0.012, 0.048)
                elif sample_type == "wide_stress":
                    radius = rng.uniform(0.54, 0.73)
                elif sample_type == "near_degenerate":
                    radius = rng.uniform(1.1e-7, 9.0e-7)
                else:
                    radius = rng.uniform(0.14, 0.39)
                wireless_scale = rng.uniform(0.82, 1.28)
                compute_scale = rng.uniform(0.82, 1.28)
                wireless_radius = min(radius * 0.9, 0.72)
                compute_radius = min(radius * 0.8, 0.72)
                if sample_type == "high_wireless_sensitivity":
                    wireless_scale *= rng.uniform(2.05, 3.25)
                    wireless_radius = max(radius, 0.56)
                if sample_type == "high_compute_sensitivity":
                    compute_scale *= rng.uniform(1.85, 2.65)
                    compute_radius = max(radius, 0.51)
                p = EdgeTaskParameters(
                    _positive_interval(2.1e6 * base, radius),
                    _positive_interval(0.85e6 * base, min(radius * 0.85, 0.76)),
                    _positive_interval(2.2e9 * compute_scale, compute_radius),
                    _positive_interval(2.1e-10 * wireless_scale, wireless_radius),
                    _positive_interval(2.6e-10 * wireless_scale, min(wireless_radius * 0.9, 0.76)),
                    1.2e6, 2.0e6, 0.22, 1.1, 0.09, 1.1e-6, 1.8e-6, 1.0e-10, 5.0e9,
                )
                local = _positive_interval(2.3e9 * compute_scale, min(compute_radius * 0.75, 0.71))
                payload = preview_sample_payload(float(gamma), sample_type, p, local)
                samples.append(
                    PreviewSample(
                        f"FM_G{gamma}_{sample_type}_{replicate:02d}", float(gamma), sample_type,
                        seed, p, local, canonical_json_hash(payload),
                    )
                )
    audit_formal_samples(tuple(samples))
    return tuple(samples)


def numeric_signature(sample: PreviewSample) -> np.ndarray:
    intervals = (
        sample.parameters.input_size_bit, sample.parameters.output_size_bit,
        sample.parameters.cpu_cycles, sample.parameters.uplink_interference_w,
        sample.parameters.downlink_interference_w, sample.local_cpu_frequency,
    )
    return np.asarray([value for interval in intervals for value in (interval.lower, interval.upper)], dtype=np.float64)


def normalized_distance(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.maximum(np.maximum(np.abs(first), np.abs(second)), 1e-300)
    return float(np.max(np.abs(first - second) / denominator))


def audit_formal_samples(samples: tuple[PreviewSample, ...], near_tolerance: float = 1e-6) -> dict[str, Any]:
    if len(samples) != 90 or len({sample.sample_id for sample in samples}) != 90:
        raise ValueError("Formal requires 90 unique sample IDs")
    distribution = {
        str(gamma): {
            sample_type: sum(s.gamma == gamma and s.sample_type == sample_type for s in samples)
            for sample_type in SAMPLE_TYPES
        }
        for gamma in range(1, 6)
    }
    if any(count != 3 for group in distribution.values() for count in group.values()):
        raise ValueError("Formal samples must contain three repetitions per Gamma and type")
    hashes = [sample.content_hash for sample in samples]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Formal content hashes must be unique")
    signatures = [numeric_signature(sample) for sample in samples]
    near = sum(
        normalized_distance(first, second) <= near_tolerance
        for index, first in enumerate(signatures)
        for second in signatures[index + 1 :]
    )
    if near:
        raise ValueError("Formal samples contain normalized near duplicates")
    return {"sample_count": 90, "distribution": distribution, "exact_duplicate_pairs": 0, "near_duplicate_pairs": 0}


FormalDecision = Literal["passed", "failed", "invalid_due_to_execution"]


def formal_decision(*, execution_valid: bool, all_scientific_thresholds_passed: bool, scientific_decision_allowed: bool) -> FormalDecision | None:
    if not scientific_decision_allowed:
        return None
    if not execution_valid:
        return "invalid_due_to_execution"
    return "passed" if all_scientific_thresholds_passed else "failed"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(root: str | Path) -> str:
    base = Path(root)
    rows = []
    for path in sorted(base.rglob("*.py")):
        if "__pycache__" not in path.parts:
            rows.append({"path": path.relative_to(base).as_posix(), "sha256": file_sha256(path)})
    return canonical_json_hash(rows)


def formal_config_payload(sample_rows: list[dict[str, Any]], task_rows: list[dict[str, Any]], code_tree_hash: str, physics_contract_hash: str, preview_decision_hash: str) -> dict[str, Any]:
    return {
        "protocol_version": "exp1_formal_design_v1",
        "result_status": "formal_design",
        "formal_full_run_started": False,
        "scientific_decision_allowed": False,
        "bp_training_started": False,
        "label_ga_budget_finally_frozen": False,
        "not_for_bp_training_data": True,
        "formal_candidate": B4_FORMAL_CANDIDATE,
        "reference_ga": REFERENCE_GA_FORMAL,
        "dense_lhs": DENSE_LHS_FORMAL,
        "thresholds": PREREGISTERED_THRESHOLDS,
        "sample_generation_seed": FORMAL_SAMPLE_SEED,
        "future_bp_reserved_seed_range": list(BP_RESERVED_SEED_RANGE),
        "sample_manifest": sample_rows,
        "task_manifest": task_rows,
        "code_tree_hash": code_tree_hash,
        "physics_contract_hash": physics_contract_hash,
        "preview_decision_hash": preview_decision_hash,
    }
