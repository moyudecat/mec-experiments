from __future__ import annotations

import numpy as np
from pymoo.core.population import Population

from mec_dualbp.optimization.adapters import (
    OE_RVEA_ALGORITHM, OE_RVEA_V1_DESIGN_HASH, ObjectiveEquivalenceAwareRVEA,
    OuterAlgorithmConfig, create_adapter, exact_objective_equivalence_compression,
    matched_canonical_initial_population,
)
from mec_dualbp.optimization.evaluation_archive import canonical_decision_hash
from mec_dualbp.optimization.outer_contract import (
    CORRECTED_DUALBP_HASH, CanonicalOuterEvaluator, RepairContext,
)


def _population(objectives, decisions=None, cv=None):
    f = np.asarray(objectives, dtype=np.float64)
    n = len(f)
    x = np.arange(n * 3, dtype=np.float64).reshape(n, 3) if decisions is None else np.asarray(decisions, dtype=np.float64)
    pop = Population.new(X=x, F=f)
    pop.set("CV", np.zeros((n, 1)) if cv is None else np.asarray(cv, dtype=np.float64)[:, None])
    return pop


def _context(users=3):
    return RepairContext(1.5e9, 5e9, 20e9, np.ones(users), float(users), np.arange(1, users + 1))


def _adapter(seed=62001, budget=30):
    context = _context()
    config = OuterAlgorithmConfig(10, budget, 1)
    pool = matched_canonical_initial_population(seed, 10, context)
    return create_adapter(OE_RVEA_ALGORITHM, 3, context, config, seed, pool)


def _decision_hash(individual):
    return canonical_decision_hash(tuple(float(v) for v in np.asarray(individual.X).reshape(-1)))


def _results(pending, duplicate=False):
    rows = []
    for index, solution in enumerate(pending):
        value = float(1 + index % 2) if duplicate else float(index + 1)
        rows.append({"objectives": [value, 3.0 - value if duplicate else 20.0 - value], "constraint_violation": 0.0,
                     "prediction_mode": "corr_pred"})
    return rows


def test_algorithm_identity_and_design_hash_are_frozen() -> None:
    adapter = _adapter()
    assert adapter.name == "OE-RVEA"
    assert isinstance(adapter.algorithm, ObjectiveEquivalenceAwareRVEA)
    assert adapter.algorithm.design_hash == OE_RVEA_V1_DESIGN_HASH


def test_exact_equal_raw_float_objectives_form_one_cluster() -> None:
    reps, removed, audit = exact_objective_equivalence_compression(
        _population([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]]))
    assert len(reps) == 2 and len(removed) == 1
    assert audit == {"candidate_pool_size": 3, "unique_objective_count": 2,
                     "equivalent_cluster_count": 1, "compressed_candidate_count": 1,
                     "compression_ratio": 1 / 3}


def test_nextafter_objective_is_not_merged_without_epsilon() -> None:
    neighbor = np.nextafter(np.float64(1.0), np.float64(2.0))
    reps, removed, audit = exact_objective_equivalence_compression(
        _population([[1.0, 2.0], [neighbor, 2.0]]))
    assert len(reps) == 2 and len(removed) == 0
    assert audit["equivalent_cluster_count"] == 0


def test_no_decimal_rounding_or_string_equivalence() -> None:
    reps, _, _ = exact_objective_equivalence_compression(
        _population([[0.1 + 0.2, 1.0], [0.3, 1.0]]))
    assert len(reps) == 2


def test_representative_prefers_lower_constraint_violation() -> None:
    pop = _population([[1, 2], [1, 2]], decisions=[[9, 9, 9], [1, 1, 1]], cv=[0.2, 0.1])
    reps, _, _ = exact_objective_equivalence_compression(pop)
    assert np.array_equal(reps[0].X, [1, 1, 1])


def test_representative_tie_uses_lexicographically_smallest_canonical_hash() -> None:
    decisions = np.asarray([[9, 9, 9], [1, 1, 1], [5, 5, 5]], dtype=np.float64)
    pop = _population([[1, 2]] * 3, decisions=decisions)
    reps, _, _ = exact_objective_equivalence_compression(pop)
    expected = min((_decision_hash(ind), ind.X.copy()) for ind in pop)[1]
    assert np.array_equal(reps[0].X, expected)


def test_unique_pool_preserves_order_and_standard_rvea_input_semantics() -> None:
    pop = _population([[float(i), float(10 - i)] for i in range(10)])
    reps, removed, audit = exact_objective_equivalence_compression(pop)
    assert len(removed) == 0 and audit["compression_ratio"] == 0.0
    assert np.array_equal(reps.get("X"), pop.get("X"))


def test_underfilled_pool_uses_fallback_and_fills_population() -> None:
    adapter = _adapter()
    first = adapter.ask(); adapter.tell(_results(first, duplicate=True))
    second = adapter.ask(); adapter.tell(_results(second, duplicate=True))
    audit = adapter.objective_equivalence_audit[-1]
    assert audit["compression_fallback_triggered"] is True
    assert audit["unique_objective_count"] == 2
    assert audit["survivor_count"] == 10 and audit["fallback_added_count"] == 8


def test_fallback_is_fixed_seed_deterministic() -> None:
    adapters = [_adapter() for _ in range(2)]
    for adapter in adapters:
        first = adapter.ask(); adapter.tell(_results(first, duplicate=True))
        second = adapter.ask(); adapter.tell(_results(second, duplicate=True))
    assert adapters[0].objective_equivalence_audit == adapters[1].objective_equivalence_audit
    assert np.array_equal(adapters[0].algorithm.pop.get("X"), adapters[1].algorithm.pop.get("X"))


def test_reference_vectors_and_standard_apd_survival_are_preserved() -> None:
    oe = _adapter()
    from pymoo.algorithms.moo.rvea import APDSurvival
    assert isinstance(oe.algorithm.survival, APDSurvival)
    assert np.array_equal(oe.algorithm.ref_dirs, oe.algorithm.survival.ref_dirs)


def test_adapter_uses_canonical_repair_and_shared_evaluator_entry() -> None:
    adapter = _adapter()
    from mec_dualbp.optimization.adapters import BaseOuterAdapter
    assert adapter.repair_function is BaseOuterAdapter.repair_function
    assert adapter.evaluator_entry.__func__ is BaseOuterAdapter.evaluator_entry
    pending = adapter.ask()
    assert all(np.all(item.edge_cpu >= 0) for item in pending)


class _Engine:
    @staticmethod
    def evaluate(_solution):
        return {"objectives": [1.0, 1.0], "constraint_violation": 0.0,
                "prediction_mode": "corr_pred"}


def test_duplicate_objectives_still_consume_every_request_budget() -> None:
    adapter = _adapter(budget=20)
    pending = adapter.ask()
    evaluator = CanonicalOuterEvaluator(_Engine.evaluate, corrected_dualbp_hash=CORRECTED_DUALBP_HASH)
    results = adapter.evaluate(evaluator)
    assert len(results) == 10 and evaluator.requests == 10


def test_standard_rvea_registration_and_class_remain_unchanged() -> None:
    context = _context(); config = OuterAlgorithmConfig(10, 20, 1)
    pool = matched_canonical_initial_population(62001, 10, context)
    standard = create_adapter("RVEA", 3, context, config, 62001, pool)
    from pymoo.algorithms.moo.rvea import RVEA
    assert type(standard.algorithm) is RVEA
    assert not hasattr(standard.algorithm, "objective_equivalence_audit")


def test_audit_contains_all_required_generation_fields() -> None:
    adapter = _adapter()
    first = adapter.ask(); adapter.tell(_results(first, duplicate=True))
    second = adapter.ask(); adapter.tell(_results(second, duplicate=True))
    required = {"generation", "candidate_pool_size", "unique_objective_count",
                "equivalent_cluster_count", "compressed_candidate_count", "compression_ratio",
                "compression_fallback_triggered", "fallback_added_count", "survivor_count",
                "survivor_unique_objective_count"}
    assert required <= adapter.objective_equivalence_audit[-1].keys()
