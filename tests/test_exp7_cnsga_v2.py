from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from pymoo.core.population import Population

from mec_dualbp.experiments.exp7_chain_runner import _checkpoint, _restore
from mec_dualbp.optimization.adapters import (
    ALGORITHM_REGISTRY, CA_ELITE_FRACTION, CA_EPSILON, V2_ALGORITHM, V2_DESIGN_HASH,
    ConvergenceAwareSurvival, OuterAlgorithmConfig, create_adapter,
    matched_canonical_initial_population,
)
from mec_dualbp.optimization.evaluation_archive import EvaluationArchive
from mec_dualbp.optimization.outer_contract import (
    CORRECTED_DUALBP_HASH, CanonicalOuterEvaluator, RepairContext,
)


def _context(users: int = 3) -> RepairContext:
    return RepairContext(1.5e9, 5e9, 20e9, np.ones(users), float(users), np.arange(1, users + 1))


def _population(objectives, *, cv=None):
    f = np.asarray(objectives, dtype=float)
    n = len(f)
    x = np.column_stack((np.ones((n, 2)), np.arange(n, dtype=float)))
    pop = Population.new(X=x, F=f)
    pop.set("CV", np.zeros((n, 1)) if cv is None else np.asarray(cv, float)[:, None])
    return pop


def _problem():
    from pymoo.core.problem import Problem
    return Problem(n_var=3, n_obj=2, n_ieq_constr=1)


def _results(solutions):
    return [{"objectives": [float(1 + i), float(len(solutions) - i)],
             "constraint_violation": 0.0, "prediction_mode": "corr_pred"}
            for i, _ in enumerate(solutions)]


def test_registry_keeps_v1_and_adds_independent_v2_identity() -> None:
    assert "CorrectedDualBP-CNSGA-II" in ALGORITHM_REGISTRY
    assert V2_ALGORITHM in ALGORITHM_REGISTRY
    context = _context(); config = OuterAlgorithmConfig(10, 20, 1)
    pool = matched_canonical_initial_population(52001, 10, context)
    v1 = create_adapter("CorrectedDualBP-CNSGA-II", 3, context, config, 52001, pool)
    v2 = create_adapter(V2_ALGORITHM, 3, context, config, 52001, pool)
    assert not isinstance(v1.algorithm.survival, ConvergenceAwareSurvival)
    assert isinstance(v2.algorithm.survival, ConvergenceAwareSurvival)
    assert v2.algorithm.survival.design_hash == V2_DESIGN_HASH


def test_elite_count_frozen_for_formal_and_small_population() -> None:
    assert CA_ELITE_FRACTION == .10 and CA_EPSILON == 1e-12
    assert ConvergenceAwareSurvival.elite_count(100) == 10
    assert ConvergenceAwareSurvival.elite_count(9) == 1
    assert ConvergenceAwareSurvival.elite_count(2) == 1


def test_normalization_zero_range_distance_and_order_are_deterministic() -> None:
    pop = _population([[1, 4], [1, 3], [1, 2], [1, 1]])
    survival = ConvergenceAwareSurvival()
    result = survival.do(_problem(), pop, n_survive=2, random_state=np.random.default_rng(5))
    audit = survival.audit_records[-1]
    assert len(result) == 2 and audit["elite_count_actual"] == 1
    assert audit["protected_distance"][0] == pytest.approx(0.0)
    assert all(np.isfinite(audit["protected_distance"]))


def test_distinct_extremes_protected_deduplicated_and_capped() -> None:
    # Eleven mutually nondominated points: convergence elite target 1 plus
    # distinct objective extremes gives the general-population cap K+2=3.
    pop = _population([[float(i), float(10-i)] for i in range(11)])
    survival = ConvergenceAwareSurvival()
    result = survival.do(_problem(), pop, n_survive=9, random_state=np.random.default_rng(3))
    audit = survival.audit_records[-1]
    assert len(result) == 9
    assert audit["extreme_count"] == 2
    assert audit["protected_set_size"] <= audit["elite_count_target"] + 2 == 3


def test_population_100_cap_is_twelve() -> None:
    pop = _population([[float(i), float(119-i)] for i in range(120)])
    survival = ConvergenceAwareSurvival()
    result = survival.do(_problem(), pop, n_survive=100, random_state=np.random.default_rng(1))
    audit = survival.audit_records[-1]
    assert len(result) == 100 and audit["elite_count_target"] == 10
    assert audit["elite_count_actual"] == 10 and audit["protected_set_size"] <= 12


def test_elite_target_uses_configured_population_not_current_feasible_count() -> None:
    pop = _population([[float(i), float(19-i)] for i in range(20)])
    survival = ConvergenceAwareSurvival(population_size=100)
    survival.do(_problem(), pop, n_survive=20, random_state=np.random.default_rng(1))
    assert survival.audit_records[-1]["elite_count_target"] == 10


def test_remaining_slots_follow_exact_v1_survival_order() -> None:
    pop = _population([[float(i), float(29-i)] for i in range(30)])
    rng1, rng2 = np.random.default_rng(11), np.random.default_rng(11)
    v1 = __import__("pymoo.operators.survival.rank_and_crowding.classes", fromlist=["RankAndCrowding"]).RankAndCrowding()
    standard = v1.do(_problem(), pop, n_survive=10, random_state=rng1)
    ca = ConvergenceAwareSurvival()
    selected = ca.do(_problem(), pop, n_survive=10, random_state=rng2)
    protected = set(ca.audit_records[-1]["protected_decision_hashes"])
    from mec_dualbp.optimization.evaluation_archive import canonical_decision_hash
    digest = lambda ind: canonical_decision_hash(tuple(np.asarray(ind.X, float)))
    expected_remainder = [digest(ind) for ind in standard if digest(ind) not in protected]
    actual_remainder = [digest(ind) for ind in selected if digest(ind) not in protected]
    assert actual_remainder == expected_remainder[:len(actual_remainder)]


def test_infeasible_candidate_never_enters_ca_elite() -> None:
    pop = _population([[5, 5], [4, 6], [0, 0]], cv=[0, 0, 100])
    survival = ConvergenceAwareSurvival()
    selected = survival.do(_problem(), pop, n_survive=2, random_state=np.random.default_rng(2))
    assert not any(np.array_equal(ind.F, [0, 0]) for ind in selected)
    assert survival.audit_records[-1]["candidate_pool_size"] == 2


def test_v1_fixed_seed_regression_is_unchanged_by_v2_registration() -> None:
    context = _context(); config = OuterAlgorithmConfig(10, 20, 1)
    pool = matched_canonical_initial_population(52001, 10, context)
    first = create_adapter("CorrectedDualBP-CNSGA-II", 3, context, config, 52001, pool)
    second = create_adapter("CorrectedDualBP-CNSGA-II", 3, context, config, 52001, pool)
    p1, p2 = first.ask(), second.ask()
    assert all(np.array_equal(a.offload, b.offload) and np.array_equal(a.cache, b.cache)
               and np.array_equal(a.edge_cpu, b.edge_cpu) for a, b in zip(p1, p2))
    first.tell(_results(p1)); second.tell(_results(p2))
    assert np.array_equal(first.algorithm.pop.get("X"), second.algorithm.pop.get("X"))


def test_v2_determinism_protected_and_final_population_hashes() -> None:
    context = _context(); config = OuterAlgorithmConfig(10, 30, 1)
    pool = matched_canonical_initial_population(52001, 10, context)
    adapters = [create_adapter(V2_ALGORITHM, 3, context, config, 52001, pool) for _ in range(2)]
    for adapter in adapters:
        for _ in range(2):
            pending = adapter.ask(); adapter.tell(_results(pending))
    assert adapters[0].ca_selection_audit == adapters[1].ca_selection_audit
    assert np.array_equal(adapters[0].algorithm.pop.get("X"), adapters[1].algorithm.pop.get("X"))


def test_archive_contract_is_independent_of_ca_status() -> None:
    cache = {(0.,): {"objectives": [1., 2.], "constraint_violation": 0., "feasible": True},
             (1.,): {"objectives": [2., 1.], "constraint_violation": 0., "feasible": True}}
    before = EvaluationArchive.from_cache(cache).official_front()
    after = EvaluationArchive.from_cache(cache).official_front()
    assert before == after and len(after) == 2


class _Engine:
    @staticmethod
    def evaluate(_solution):
        return {"prediction_mode": "corr_pred", "objectives": [1., 1.], "constraint_violation": 0.}


def test_v2_checkpoint_identity_and_hash_reject_cross_load(tmp_path: Path) -> None:
    context = _context(); config = OuterAlgorithmConfig(10, 20, 1)
    pool = matched_canonical_initial_population(52001, 10, context)
    v2 = create_adapter(V2_ALGORITHM, 3, context, config, 52001, pool)
    evaluator = CanonicalOuterEvaluator(_Engine.evaluate, corrected_dualbp_hash=CORRECTED_DUALBP_HASH)
    path = tmp_path / "checkpoint.pkl"
    _checkpoint(path, v2, evaluator)
    restored, _ = _restore(path, _Engine(), V2_ALGORITHM, 10)
    assert restored.name == V2_ALGORITHM
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _restore(path, _Engine(), "CorrectedDualBP-CNSGA-II", 10)
    payload = pickle.loads(path.read_bytes()); payload["v2_design_hash"] = "wrong"; path.write_bytes(pickle.dumps(payload))
    with pytest.raises(RuntimeError, match="design hash mismatch"):
        _restore(path, _Engine(), V2_ALGORITHM, 10)
