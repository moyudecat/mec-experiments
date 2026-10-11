from collections import Counter

from mec_dualbp.experiments.exp1_formal_design import formal_task_manifest
from mec_dualbp.experiments.exp1_formal_protocol import (
    B4_FORMAL_CANDIDATE, BP_RESERVED_SEED_RANGE, DENSE_LHS_FORMAL,
    FORMAL_SAMPLE_SEED, REFERENCE_GA_FORMAL, audit_formal_samples,
    formal_decision, generate_formal_samples,
)


def test_b4_formal_candidate_values_are_frozen() -> None:
    assert B4_FORMAL_CANDIDATE["name"] == "B4_formal_candidate_v1"
    assert (B4_FORMAL_CANDIDATE["population_size"], B4_FORMAL_CANDIDATE["generations"]) == (64, 50)
    assert B4_FORMAL_CANDIDATE["seeds"] == [101, 202, 303, 404, 505]
    assert B4_FORMAL_CANDIDATE["status"] == "formal_candidate_not_yet_validated"


def test_reference_and_dense_formal_values_are_frozen() -> None:
    assert (REFERENCE_GA_FORMAL["population_size"], REFERENCE_GA_FORMAL["generations"]) == (96, 80)
    assert len(REFERENCE_GA_FORMAL["seeds"]) == 8
    assert DENSE_LHS_FORMAL["n_samples"] == 50000
    assert DENSE_LHS_FORMAL["reference_status"] == "approximate_reference"


def test_formal_samples_are_balanced_and_unique() -> None:
    samples = generate_formal_samples()
    audit = audit_formal_samples(samples)
    assert audit["exact_duplicate_pairs"] == audit["near_duplicate_pairs"] == 0
    counts = Counter((sample.gamma, sample.sample_type) for sample in samples)
    assert len(samples) == 90 and set(counts.values()) == {3}
    assert len({sample.content_hash for sample in samples}) == 90


def test_formal_tasks_are_complete_and_dry_selection_has_eight_tasks() -> None:
    samples = generate_formal_samples()
    dry = {samples[0].sample_id, next(s.sample_id for s in samples if s.gamma >= 3 and s.sample_type == "wide_stress")}
    rows = formal_task_manifest(samples, dry)
    assert len(rows) == 360 and len({row["task_hash"] for row in rows}) == 360
    assert sum(row["dry_run_selected"] for row in rows) == 8
    assert all(row["not_for_bp_training_data"] for row in rows)


def test_seed_namespace_isolated_from_preview_smoke_and_bp() -> None:
    assert FORMAL_SAMPLE_SEED not in (2026080401, 700001)
    assert not BP_RESERVED_SEED_RANGE[0] <= FORMAL_SAMPLE_SEED <= BP_RESERVED_SEED_RANGE[1]


def test_formal_decision_has_exact_three_states_and_dry_run_none() -> None:
    assert formal_decision(execution_valid=True, all_scientific_thresholds_passed=True, scientific_decision_allowed=True) == "passed"
    assert formal_decision(execution_valid=True, all_scientific_thresholds_passed=False, scientific_decision_allowed=True) == "failed"
    assert formal_decision(execution_valid=False, all_scientific_thresholds_passed=True, scientific_decision_allowed=True) == "invalid_due_to_execution"
    assert formal_decision(execution_valid=True, all_scientific_thresholds_passed=True, scientific_decision_allowed=False) is None

