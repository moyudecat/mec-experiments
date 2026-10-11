from pathlib import Path

from mec_dualbp.experiments.oe_rvea_final_compare_runner import (
    ALGORITHMS, ALL_ALGORITHMS, final_protocol_hash, final_protocol_payload, preflight,
)

ROOT = Path(__file__).resolve().parents[1]


def test_final_comparison_has_six_frozen_algorithms_and_four_additional() -> None:
    assert ALL_ALGORITHMS == ("OE-RVEA", "RVEA", "CorrectedDualBP-CA-CNSGA-II", "MOEA/D", "MOPSO", "Random")
    assert ALGORITHMS == ("CorrectedDualBP-CA-CNSGA-II", "MOEA/D", "MOPSO", "Random")


def test_final_protocol_binds_fair_budget_and_statistics() -> None:
    payload = final_protocol_payload()
    assert payload["population"] == 100 and payload["requests_per_run"] == 20_000
    assert payload["matched_initialization"] is True and len(final_protocol_hash()) == 64
    assert payload["workers"] == 1


def test_final_comparison_read_only_preflight_passes() -> None:
    result = preflight(ROOT)
    assert result["passed"] is True and all(result["checks"].values())
    assert result["gate_a_reused_runs"] == 50 and result["new_runs"] == 100
