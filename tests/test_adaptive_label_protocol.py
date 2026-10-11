import math

import pytest

from mec_dualbp.experiments.adaptive_label_protocol import (
    STABILITY_GATE_THRESHOLD, adaptive_label, aggregate_boundary,
    gate_reasons, generate_confirmation_samples, protocol_config, response_stability,
)


def test_confirmation_samples_are_balanced_and_unique() -> None:
    samples = generate_confirmation_samples()
    assert len(samples) == len({sample.sample_id for sample in samples}) == 60
    assert len({sample.content_hash for sample in samples}) == 60
    for gamma in range(1, 6):
        assert sum(sample.gamma == gamma for sample in samples) == 12
        for kind in {sample.sample_type for sample in samples}:
            assert sum(sample.gamma == gamma and sample.sample_type == kind for sample in samples) == 2


def test_stability_uses_population_std_and_mean_denominator() -> None:
    result = response_stability([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result.mean == 3.0
    assert result.std == pytest.approx(math.sqrt(2.0))
    assert result.relative_std == pytest.approx(math.sqrt(2.0) / 3.0)


def test_gate_uses_strict_greater_than_threshold() -> None:
    assert gate_reasons([1.0] * 5) == ()
    assert "relative_response_std_over_0.01" in gate_reasons([0.9, 0.9, 1.0, 1.1, 1.1])


def test_gate_collects_all_audit_reasons() -> None:
    reasons = gate_reasons([1.0, 2.0], failed_seed_runs=1, infeasible_best_z=1,
                           boundary_direction_abnormal=True, evaluation_identity_failures=1,
                           cache_collision_error=True, endpoint_snap_max_ulp=17,
                           reproducibility_failure=True)
    assert set(reasons) == {"seed_result_missing", "seed_execution_failed", "infeasible_best_z",
                            "boundary_direction_abnormal", "evaluation_identity_failure",
                            "cache_collision_error", "endpoint_snap_over_16_ulp",
                            "reproducibility_failure"}


def test_boundary_and_adaptive_routes_are_direction_aware() -> None:
    assert aggregate_boundary("minimize", [3, 1, 2]) == 1
    assert aggregate_boundary("maximize", [3, 1, 2]) == 3
    assert adaptive_label("minimize", 2, 1.5, 1.7, False) == (2.0, "b4_stable")
    assert adaptive_label("minimize", 2, 1.5, 1.7, True) == (1.5, "high_fidelity_approximate_label")
    assert adaptive_label("maximize", 2, 2.5, 2.3, True) == (2.5, "high_fidelity_approximate_label")


def test_protocol_forbids_bp_roles_and_five_candidate_logic() -> None:
    config = protocol_config()
    assert config["workers"] == 1 and config["stability_gate_threshold"] == 0.01
    assert config["dense_lhs"]["n_samples"] == 50000
    assert all(config[name] for name in ("not_for_bp_training", "not_for_bp_validation", "not_for_bp_calibration", "not_for_bp_test"))
    assert config["five_candidate_certification_used"] is False
