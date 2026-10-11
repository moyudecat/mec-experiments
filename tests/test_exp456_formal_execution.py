from __future__ import annotations

import numpy as np

from mec_dualbp.experiments.exp456_formal_execution import (
    _average_ranks, _kendall_tau_b, _relation, _spearman,
)


def test_average_rank_ties_are_deterministic() -> None:
    np.testing.assert_array_equal(_average_ranks(np.asarray([3., 1., 1., 2.])), [4., 1.5, 1.5, 3.])


def test_primary_spearman_uses_independent_objective_order() -> None:
    truth = np.asarray([1., 2., 3., 4.])
    assert _spearman(truth, truth) == 1.0
    assert _spearman(truth, truth[::-1]) == -1.0


def test_kendall_tau_b_handles_ties_without_random_breaking() -> None:
    values = np.asarray([1., 1., 2., 3.])
    assert _kendall_tau_b(values, values) == 1.0


def test_pareto_relation_four_way_contract() -> None:
    assert _relation((1., 1.), (2., 2.)) == "a_dominates"
    assert _relation((2., 2.), (1., 1.)) == "b_dominates"
    assert _relation((1., 2.), (2., 1.)) == "incomparable"
    assert _relation((1., 1.), (1., 1.)) == "tie"
