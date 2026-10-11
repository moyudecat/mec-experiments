from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mec_dualbp.experiments.exp7_multi_algorithm_protocol_v2 import (
    ENERGY_CENTER_SCALE_RULE, FORMAL_ALGORITHMS, SCENARIO_USERS, TIME_CENTER_SCALE_RULE,
    build_protocol, rho_table,
)
from mec_dualbp.optimization.adapters import (
    BaseOuterAdapter, MOPSOAdapter, OuterAlgorithmConfig, create_adapter,
    matched_canonical_initial_population,
)
from mec_dualbp.optimization.outer_contract import (
    CORRECTED_DUALBP_HASH, CanonicalOuterEvaluator, RepairContext,
    canonical_outer_repair, robust_objectives,
)


def _context() -> RepairContext:
    return RepairContext(1.5e9, 5e9, 20e9, np.ones(3), 2.0, np.array([1.0, 2.0, 3.0]))


def _results(solutions):
    return [{"objectives": [float(x.offload.sum()), float(x.edge_cpu.sum() / 1e9)],
             "constraint_violation": 0.0, "prediction_mode": "corr_pred"} for x in solutions]


def test_rho_values_for_all_scenarios() -> None:
    rows = {row["scenario"]: row for row in rho_table()}
    assert set(rows) == set(SCENARIO_USERS)
    for name, n in SCENARIO_USERS.items():
        assert rows[name]["rho_T_s"] == pytest.approx(n * TIME_CENTER_SCALE_RULE)
        assert rows[name]["rho_E_J"] == pytest.approx(n * ENERGY_CENTER_SCALE_RULE)


def test_robust_objectives_use_frozen_minimize_formula() -> None:
    base = robust_objectives(10, 12, .5, .7, rho_time=10, rho_energy=.5)
    larger = robust_objectives(20, 24, 1, 1.4, rho_time=10, rho_energy=.5)
    assert base.shape == (2,) and np.all(np.isfinite(base)) and np.all(larger > base)


def test_matched_initialization_is_seed_deterministic() -> None:
    context = _context()
    a = matched_canonical_initial_population(52001, 10, context)
    b = matched_canonical_initial_population(52001, 10, context)
    c = matched_canonical_initial_population(52002, 10, context)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_formal_set_and_mopso_parameters_frozen() -> None:
    assert FORMAL_ALGORITHMS == ("CorrectedDualBP-CNSGA-II", "MOEA/D", "RVEA", "MOPSO", "Random")
    assert (MOPSOAdapter.w, MOPSOAdapter.c1, MOPSOAdapter.c2, MOPSOAdapter.archive_max_size) == (.5, 1.5, 1.5, 100)
    assert "NSGA-III" not in FORMAL_ALGORITHMS


def test_all_adapters_close_common_interface_and_one_step() -> None:
    context, config = _context(), OuterAlgorithmConfig(10, 20, 1)
    pool = matched_canonical_initial_population(52001, 10, context)
    required = {"ask", "canonicalize", "repair", "evaluate", "tell", "get_population",
                "get_nondominated_front", "get_history", "finalize"}
    for name in FORMAL_ALGORITHMS:
        adapter = create_adapter(name, 3, context, config, 52001, pool)
        assert required <= set(dir(adapter))
        assert adapter.repair_function is canonical_outer_repair
        pending = adapter.ask()
        adapter.tell(_results(pending))
        assert adapter.generation == 1 and len(adapter.get_history()) == 1


def test_mopso_binary_transfer_reproducible_from_outer_seed() -> None:
    context, config = _context(), OuterAlgorithmConfig(10, 20, 1)
    pool = matched_canonical_initial_population(52001, 10, context)
    a = create_adapter("MOPSO", 3, context, config, 52001, pool)
    b = create_adapter("MOPSO", 3, context, config, 52001, pool)
    pa, pb = a.ask(), b.ask()
    a.tell(_results(pa)); b.tell(_results(pb))
    assert all(np.array_equal(x.offload, y.offload) and np.array_equal(x.cache, y.cache)
               and np.array_equal(x.edge_cpu, y.edge_cpu)
               for x, y in zip(a.get_population(), b.get_population()))


def test_shared_evaluator_cache_hit_still_consumes_request() -> None:
    context = _context()
    pool = matched_canonical_initial_population(52001, 2, context)
    solution = create_adapter("Random", 3, context, OuterAlgorithmConfig(2, 4, 1), 52001, pool).ask()[0]
    evaluator = CanonicalOuterEvaluator(lambda _: {"prediction_mode": "corr_pred", "objectives": [1, 2], "constraint_violation": 0}, corrected_dualbp_hash=CORRECTED_DUALBP_HASH)
    evaluator.evaluate_outer_solution_canonical(solution, context)
    evaluator.evaluate_outer_solution_canonical(solution, context)
    assert (evaluator.requests, evaluator.unique_evaluations, evaluator.cache_hits) == (2, 1, 1)


def test_v2_protocol_generation_passes_and_preserves_parent(tmp_path: Path) -> None:
    out = tmp_path / "outputs/05_audit/exp7_multi_algorithm_protocol_v2_test"
    summary = build_protocol(Path.cwd(), out, pytest_count=0)
    fairness = json.loads((out / "fairness_audit_v2.json").read_text(encoding="utf-8"))
    parent = json.loads((out / "parent_protocol_binding.json").read_text(encoding="utf-8"))
    selected = json.loads((out / "formal_algorithm_set.json").read_text(encoding="utf-8"))
    assert summary["exp7_protocol_decision"] == "passed"
    assert summary["multi_algorithm_smoke_allowed"] is True
    assert fairness["passed"] and parent["matches"]
    assert selected["NSGA_III"] == "available_not_selected"
    assert summary["remaining_blocking_open_questions"] == 0


def test_base_adapter_binds_single_repair_and_evaluator() -> None:
    assert BaseOuterAdapter.repair_function is canonical_outer_repair
    assert BaseOuterAdapter.evaluator_entry is CanonicalOuterEvaluator.evaluate_outer_solution_canonical
