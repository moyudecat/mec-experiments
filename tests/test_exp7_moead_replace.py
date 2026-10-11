from __future__ import annotations

import numpy as np
import pytest
from pymoo.decomposition.tchebicheff import Tchebicheff

from mec_dualbp.optimization.adapters import (
    ConstrainedMOEAD, OuterAlgorithmConfig, _moead_replacement_mask,
    _paired_decomposition_values, create_adapter, matched_canonical_initial_population,
)
from mec_dualbp.optimization.outer_contract import (
    CORRECTED_DUALBP_HASH, CanonicalOuterEvaluator, RepairContext,
)


@pytest.mark.parametrize("neighbors", [3, 7, 10])
def test_single_offspring_scalarizes_once_per_neighbor(neighbors: int) -> None:
    decomposition = Tchebicheff()
    child = np.asarray([0.3, 0.7])
    weights = np.column_stack((np.linspace(.1, .9, neighbors), np.linspace(.9, .1, neighbors)))
    observed = _paired_decomposition_values(decomposition, child, weights, np.zeros(2))
    manual = np.asarray([
        decomposition.do(child, weights=weight, ideal_point=np.zeros(2))
        for weight in weights
    ], dtype=float).reshape(-1)
    assert observed.shape == (neighbors,)
    assert np.allclose(observed, manual)


def test_paired_incumbent_batch_matches_manual_loop() -> None:
    decomposition = Tchebicheff()
    objectives = np.asarray([[.2, .8], [.5, .5], [.8, .2]])
    weights = np.asarray([[.2, .8], [.5, .5], [.8, .2]])
    observed = _paired_decomposition_values(decomposition, objectives, weights, np.zeros(2))
    manual = np.asarray([
        decomposition.do(f, weights=w, ideal_point=np.zeros(2))
        for f, w in zip(objectives, weights)
    ], dtype=float).reshape(-1)
    assert observed.shape == (3,)
    assert np.allclose(observed, manual)


def test_replacement_mask_partial_all_and_none() -> None:
    old = np.asarray([1.0, 1.0, 1.0])
    cv = np.zeros(3)
    assert _moead_replacement_mask(old, [.5, 1.5, .9], cv, 0.0).tolist() == [True, False, True]
    assert _moead_replacement_mask(old, [.5, .5, .5], cv, 0.0).all()
    assert not _moead_replacement_mask(old, [1.5, 1.5, 1.5], cv, 0.0).any()


def test_constraint_dominance_is_preserved() -> None:
    old = np.asarray([1.0, 1.0, 1.0])
    child = np.asarray([99.0, 99.0, 99.0])
    assert _moead_replacement_mask(old, child, [1.0, 0.0, 2.0], 0.0).tolist() == [True, False, True]
    assert _moead_replacement_mask(old, child, [1.0, 0.2, 2.0], 0.5).tolist() == [True, False, True]
    assert not _moead_replacement_mask(old, [0.0, 0.0, 0.0], [0.1, 0.2, 0.3], 0.5).any()


def test_moead_population10_budget100_exact_request_accounting() -> None:
    n = 3
    context = RepairContext(1.5e9, 5e9, 6e9, np.ones(n), 2.0, np.ones(n))
    config = OuterAlgorithmConfig(population_size=10, evaluation_budget=100, workers=1)
    initial = matched_canonical_initial_population(52001, 10, context)
    adapter = create_adapter("MOEA/D", n, context, config, 52001, initial)

    def callback(solution):
        return {
            "prediction_mode": "corr_pred",
            "objectives": [float(solution.offload.sum()), float(solution.edge_cpu.sum() / 1e9)],
            "constraint_violation": 0.0,
        }

    evaluator = CanonicalOuterEvaluator(callback, corrected_dualbp_hash=CORRECTED_DUALBP_HASH)
    while evaluator.requests < 100:
        pending = adapter.ask()
        before = evaluator.requests
        results = adapter.evaluate(evaluator)
        assert evaluator.requests - before == len(pending)
        adapter.tell(results)
    assert evaluator.requests == 100
    assert adapter.get_history()


def test_replace_updates_only_scalar_or_constraint_better_neighbors() -> None:
    # The exact integration behavior is exercised by the 100-request test.  This
    # mask verifies that one paid child may replace multiple, but only eligible,
    # neighbors without implying another objective evaluation.
    mask = _moead_replacement_mask(
        np.asarray([2.0, 2.0, 2.0, 2.0]),
        np.asarray([1.0, 3.0, 1.5, 4.0]),
        np.zeros(4), 0.0,
    )
    assert np.flatnonzero(mask).tolist() == [0, 2]

