from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from mec_dualbp.experiments.exp7_scenario import (
    BACKHAUL_RATE, KAPPA, SCENARIOS, build_s3_payload, build_scenario_payload, sample_type_allocation,
    validate_scenario, write_s3_scenario,
)
from mec_dualbp.optimization.outer_contract import (
    CanonicalSolution, RepairContext, canonical_outer_repair,
)
from mec_dualbp.physics.local_response import local_response_bounds_si
from mec_dualbp.uncertainty.interval_params import Interval


def test_s3_payload_is_deterministic_complete_and_balanced() -> None:
    first, second = build_s3_payload(), build_s3_payload()
    assert first == second and validate_scenario(first)["passed"]
    assert first["N"] == len(first["users"]) == 20
    counts, rule = sample_type_allocation()
    assert sorted(counts.values()) == [3, 3, 3, 3, 4, 4]
    assert len(rule["extra_count_types"]) == 2
    assert first["sample_type_allocation"] == counts


def test_author_values_and_derived_rules_are_exact() -> None:
    payload = build_s3_payload()
    users = payload["users"]
    assert payload["S0_bit"] == payload["Smax_bit"]
    assert payload["Smax_bit"] == 0.5 * sum(user["s_n_bit"] for user in users)
    assert all(20e6 * 8 <= user["s_n_bit"] <= 100e6 * 8 for user in users)
    assert all(user["backhaul_rate_bit_per_s"] == BACKHAUL_RATE for user in users)
    assert all(user["kappa_SI"] == KAPPA for user in users)
    for user in users:
        expected = local_response_bounds_si(
            Interval(user["C_cycle_lower"], user["C_cycle_upper"]),
            Interval(user["F_L_cycle_per_s_lower"], user["F_L_cycle_per_s_upper"]),
            KAPPA, payload["Gamma"],
        ).local_latency_upper_s
        assert user["tau_s"] == expected


def test_tau_is_local_upper_and_independent_of_eta_for_all_scenarios() -> None:
    for scenario_id in SCENARIOS:
        payload = build_scenario_payload(scenario_id)
        assert payload["tau_rule"] == "T_local_upper from local_response_bounds_si"
        for user in payload["users"]:
            expected = local_response_bounds_si(
                Interval(user["C_cycle_lower"], user["C_cycle_upper"]),
                Interval(user["F_L_cycle_per_s_lower"], user["F_L_cycle_per_s_upper"]),
                KAPPA, payload["Gamma"],
            ).local_latency_upper_s
            assert user["tau_s"] == expected


def test_scenario_and_outer_seeds_are_disjoint() -> None:
    scenario_seeds = {spec["seed"] for spec in SCENARIOS.values()}
    assert scenario_seeds == {73001, 73002, 73003, 73004, 73005}
    assert scenario_seeds.isdisjoint({52001, 52002, 52003, 52004, 52005})


def test_scenario_file_reload_hash_is_stable(tmp_path: Path) -> None:
    config = tmp_path / "configs/scenarios/exp7_s3_formal_v1.json"
    audit = tmp_path / "outputs/05_audit/s3"
    result = write_s3_scenario(tmp_path, config, audit)
    assert result["passed"] and result["reload_hash_stable"]
    recorded = json.loads(config.with_name("exp7_s3_formal_v1_hash.json").read_text())
    assert recorded["sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()


def test_cache_repair_uses_approved_benefit_density_directly() -> None:
    context = RepairContext(
        1.0, 2.0, 6.0,
        np.asarray([1.0, 10.0, 10.0]), 10.0,
        np.asarray([3.0, 2.0, 1.0]),
    )
    raw = CanonicalSolution(np.ones(3), np.ones(3), np.ones(3))
    repaired = canonical_outer_repair(raw, context)
    assert repaired.cache.tolist() == [1.0, 0.0, 0.0]


def test_matched_initialization_starts_with_all_local_anchor() -> None:
    from mec_dualbp.optimization.adapters import matched_canonical_initial_population
    context = RepairContext(1.5e9, 5e9, 20e9, np.ones(20), 10.0, np.ones(20))
    first = matched_canonical_initial_population(52001, 50, context)
    second = matched_canonical_initial_population(52001, 50, context)
    assert np.array_equal(first, second)
    assert np.array_equal(first[0], np.zeros(60))
