from __future__ import annotations

from pathlib import Path

from mec_dualbp.experiments.oe_rvea_gate_a_runner import (
    ALGORITHMS, BUDGET, DESIGN_HASH, IMPLEMENTATION_HASH, POPULATION,
    SCENARIOS, SEEDS, preflight, protocol_hash, protocol_payload,
)


ROOT = Path(__file__).resolve().parents[1]


def test_gate_a_frozen_matrix_and_budget() -> None:
    assert ALGORITHMS == ("OE-RVEA", "RVEA")
    assert SCENARIOS == ("S1", "S2", "S3", "S4", "S5")
    assert SEEDS == (62001, 62002, 62003, 62004, 62005)
    assert POPULATION == 100 and BUDGET == 20_000


def test_gate_a_protocol_hash_is_deterministic_and_binds_both_hashes() -> None:
    payload = protocol_payload()
    assert payload["design_hash"] == DESIGN_HASH
    assert payload["implementation_hash"] == IMPLEMENTATION_HASH
    assert protocol_hash() == protocol_hash() and len(protocol_hash()) == 64


def test_gate_a_has_no_epsilon_or_extra_algorithm() -> None:
    payload = protocol_payload()
    assert payload["equivalence"] == "exact-only" and payload["epsilon"] is None
    assert payload["algorithms"] == ["OE-RVEA", "RVEA"]


def test_gate_a_read_only_preflight_passes() -> None:
    result = preflight(ROOT)
    assert result["passed"] is True and all(result["checks"].values())
    assert result["total_runs"] == 50 and result["matched_cases"] == 25
