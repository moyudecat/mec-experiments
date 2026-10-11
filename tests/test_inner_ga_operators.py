"""实数编码 GA 算子测试。"""

import numpy as np

from mec_dualbp.inner_ga.operators import (
    elite_selection,
    polynomial_mutation,
    sbx_crossover,
    tournament_selection,
)
from mec_dualbp.uncertainty.budget_set import proportional_budget_repair

MASK = (True, True, True, True, False, False)


def _repair(z: np.ndarray) -> np.ndarray:
    return proportional_budget_repair(z, 1.5, MASK)


def test_tournament_and_elite_prefer_max_fitness() -> None:
    population = np.arange(24, dtype=float).reshape(4, 6)
    fitness = np.array([1.0, 2.0, 3.0, 4.0])
    elite = elite_selection(population, fitness, 2)
    assert np.array_equal(elite, population[[3, 2]])
    selected = tournament_selection(population, fitness, 20, 4, np.random.default_rng(1))
    assert np.all(selected == population[3])


def test_sbx_children_are_repaired_and_reproducible() -> None:
    parents = (np.array([1, -1, 1, -1, 9, 9], dtype=float), np.zeros(6))
    first = sbx_crossover(*parents, np.random.default_rng(3), 1.0, 20.0, _repair)
    second = sbx_crossover(*parents, np.random.default_rng(3), 1.0, 20.0, _repair)
    for left, right in zip(first, second):
        assert np.array_equal(left, right)
        assert np.sum(np.abs(left)) <= 1.5 + 1e-12
        assert np.all(left[[4, 5]] == 0.0)


def test_polynomial_mutation_is_repaired_and_seeded() -> None:
    source = np.zeros(6)
    first = polynomial_mutation(source, np.random.default_rng(4), 1.0, 20.0, _repair)
    second = polynomial_mutation(source, np.random.default_rng(4), 1.0, 20.0, _repair)
    assert np.array_equal(first, second)
    assert np.sum(np.abs(first)) <= 1.5 + 1e-12
    assert np.all(first[[4, 5]] == 0.0)
    assert np.array_equal(source, np.zeros(6))


def test_zero_probabilities_copy_and_repair() -> None:
    a = np.array([0.2, 0.2, 0.2, 0.2, 8, 8], dtype=float)
    b = -a
    child_a, child_b = sbx_crossover(a, b, np.random.default_rng(1), 0.0, 20.0, _repair)
    mutated = polynomial_mutation(a, np.random.default_rng(1), 0.0, 20.0, _repair)
    assert np.array_equal(child_a, _repair(a))
    assert np.array_equal(child_b, _repair(b))
    assert np.array_equal(mutated, _repair(a))

