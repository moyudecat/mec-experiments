from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mec_dualbp.experiments.exp7_multi_algorithm_protocol import OUTER_SEEDS, build_protocol
from mec_dualbp.optimization.metrics import (
    EvaluationLedger, HistoryRecord, global_normalize, hypervolume_2d_min, igd,
    nondominated_min, pooled_reference_front, representative_index,
)
from mec_dualbp.optimization.outer_contract import (
    CORRECTED_DUALBP_HASH, CanonicalOuterEvaluator, CanonicalSolution, RepairContext,
    canonical_outer_repair, interval_possibility,
)


def _context(n: int = 3) -> RepairContext:
    return RepairContext(1.5e9, 5e9, 5e9, np.ones(n), 2.0, np.arange(1, n + 1))


def test_corrected_hash_binding_and_raw_rejected() -> None:
    CanonicalOuterEvaluator(lambda _: {"prediction_mode": "corr_pred"}, corrected_dualbp_hash=CORRECTED_DUALBP_HASH)
    with pytest.raises(ValueError):
        CanonicalOuterEvaluator(lambda _: {}, corrected_dualbp_hash=CORRECTED_DUALBP_HASH, prediction_mode="raw_pred")


def test_canonical_schema_requires_aligned_finite_vectors() -> None:
    with pytest.raises(ValueError):
        CanonicalSolution(np.ones(2), np.ones(3), np.ones(2))
    with pytest.raises(ValueError):
        CanonicalSolution(np.array([np.nan]), np.ones(1), np.ones(1))


def test_common_repair_binary_topology_and_local_zero() -> None:
    repaired = canonical_outer_repair(CanonicalSolution([.2, .8, .9], [.9, .9, .1], [4e9, 2e9, 3e9]), _context())
    assert set(repaired.offload) <= {0.0, 1.0}
    assert set(repaired.cache) <= {0.0, 1.0}
    assert np.all(repaired.cache <= repaired.offload)
    assert np.all(repaired.edge_cpu[repaired.offload == 0] == 0)


def test_common_repair_active_domain_and_total_budget() -> None:
    repaired = canonical_outer_repair(CanonicalSolution([1, 1, 1], [0, 0, 0], [9e9, 9e9, 9e9]), _context())
    active = repaired.edge_cpu[repaired.offload == 1]
    assert np.all(active >= 1.5e9) and np.all(active <= 5e9)
    assert repaired.edge_cpu.sum() <= 5e9 + 1e-6


def test_common_repair_cache_capacity() -> None:
    repaired = canonical_outer_repair(CanonicalSolution([1, 1, 1], [1, 1, 1], [1.5e9] * 3), _context())
    assert np.dot(repaired.cache, np.ones(3)) <= 2
    assert repaired.cache.tolist() == [0.0, 1.0, 1.0]


@pytest.mark.parametrize(("lo", "hi", "tau", "expected"), [
    (1, 2, 2, 1), (1, 3, 2, .5), (1, 2, 1, 0), (1, 1, 1, 1), (1, 1, .9, 0),
])
def test_c7_possibility_formula(lo: float, hi: float, tau: float, expected: float) -> None:
    assert interval_possibility(lo, hi, tau) == pytest.approx(expected)


def test_all_algorithms_share_wrapper_and_repair() -> None:
    seen: list[CanonicalSolution] = []
    wrapper = CanonicalOuterEvaluator(lambda x: seen.append(x) or {"prediction_mode": "corr_pred", "objectives": [1, 2]}, corrected_dualbp_hash=CORRECTED_DUALBP_HASH)
    wrapper.evaluate_outer_solution_canonical(CanonicalSolution([0], [1], [5e9]), _context(1))
    assert seen[0].offload[0] == 0 and seen[0].cache[0] == 0 and seen[0].edge_cpu[0] == 0


def test_evaluation_budget_and_cache_accounting() -> None:
    ledger = EvaluationLedger()
    assert ledger.request("a") and not ledger.request("a") and ledger.request("b")
    assert (ledger.requests, ledger.unique_evaluations, ledger.cache_hits) == (3, 2, 1)


def test_hypervolume_min_orientation() -> None:
    assert hypervolume_2d_min([[1, 3], [2, 2], [3, 1]], [4, 4]) == pytest.approx(6.0)
    with pytest.raises(ValueError):
        hypervolume_2d_min([[5, 1]], [4, 4])


def test_igd_pooled_reference_and_global_normalization() -> None:
    ref = pooled_reference_front([[[1, 3], [2, 2]], [[3, 1], [4, 4]]])
    assert {tuple(x) for x in ref} == {(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)}
    z = global_normalize(ref, [1, 1], [3, 3])
    assert igd(z, z) == 0


def test_nondominated_filter_uses_minimization() -> None:
    front = nondominated_min([[1, 1], [2, 2], [1, 2]])
    assert front.tolist() == [[1.0, 1.0]]


def test_representative_rule_and_deterministic_hash_tie() -> None:
    assert representative_index([[1, 3], [2, 2], [3, 1]], [.1, .1, .1], ["c", "b", "a"]) == 1
    assert representative_index([[1, 1], [1, 1]], [.1, .1], ["b", "a"]) == 1


def test_history_uses_evaluation_count_axis() -> None:
    row = HistoryRecord(100, 80, 5, .75, 9, .4)
    assert row.evaluation_requests == 100 and row.generation == 5


def test_outer_seeds_are_frozen_and_not_bp_seeds() -> None:
    assert OUTER_SEEDS == (52001, 52002, 52003, 52004, 52005)
    assert set(OUTER_SEEDS).isdisjoint({31001, 31002, 31003, 31004, 31005})


def test_exp3_summary_hash_unchanged() -> None:
    path = Path("outputs/01_formal/exp3_ablation_test_formal_20260811_180401/summary.json")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "8ba2fb421b9792001e73eda4f372bbe8188713bfa49d09862cb99d7466522f35"


def test_protocol_generation_reports_scientific_blockers(tmp_path: Path) -> None:
    out = tmp_path / "outputs/05_audit/exp7_multi_algorithm_protocol_test"
    summary = build_protocol(Path.cwd(), out)
    protocol = json.loads((out / "exp7_multi_algorithm_protocol.json").read_text(encoding="utf-8"))
    fairness = json.loads((out / "fairness_audit.json").read_text(encoding="utf-8"))
    budget = json.loads((out / "evaluation_budget_protocol.json").read_text(encoding="utf-8"))
    assert summary["exp7_protocol_decision"] == "blocked"
    assert not summary["multi_algorithm_smoke_allowed"] and not fairness["passed"]
    assert protocol["workers"] == 1 and protocol["default_eta"] == .90
    assert protocol["evaluator"]["corrected_dualbp_hash"] == CORRECTED_DUALBP_HASH
    assert budget["maximum_difference_percent"] <= 2
    assert out.parts[-3:-1] == ("05_audit", "exp7_multi_algorithm_protocol_test")[-2:] or "05_audit" in out.parts
    required = {"README.md", "objective_definition_binding.md", "constraint_definition_binding.md", "algorithm_selection_matrix.csv", "open_questions.json", "created_files.json"}
    assert required <= {p.name for p in out.iterdir()}
