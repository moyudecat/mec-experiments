"""内层 GA 显式随机源与复现测试。"""

import random

from mec_dualbp.inner_ga.config import InnerGAConfig
from mec_dualbp.inner_ga.problem import InnerGAProblem
from mec_dualbp.inner_ga.solver import solve_inner_ga


def _problem() -> InnerGAProblem:
    return InnerGAProblem("repro", (True,) * 6, 2.0, "maximize", lambda z: float(10.0 + sum(value * value for value in z)))


def test_same_seed_reproduces_search_outputs() -> None:
    config = InnerGAConfig(population_size=12, generations=4, elite_count=2, seed=19)
    first, second = solve_inner_ga(_problem(), config), solve_inner_ga(_problem(), config)
    assert first.best_z == second.best_z
    assert first.best_response == second.best_response
    assert first.best_fitness == second.best_fitness
    assert first.history_best_response == second.history_best_response
    assert first.evaluation_requests == second.evaluation_requests
    assert first.unique_evaluations == second.unique_evaluations


def test_different_seeds_run_independently() -> None:
    first = solve_inner_ga(_problem(), InnerGAConfig(population_size=10, generations=2, elite_count=1, seed=1))
    second = solve_inner_ga(_problem(), InnerGAConfig(population_size=10, generations=2, elite_count=1, seed=2))
    assert first.status == second.status == "success"
    assert (first.seed, second.seed) == (1, 2)
    assert first.feasible and second.feasible


def test_solver_does_not_consume_python_global_random_state() -> None:
    random.seed(123)
    expected = [random.random() for _ in range(4)]
    random.seed(123)
    solve_inner_ga(_problem(), InnerGAConfig(population_size=8, generations=1, elite_count=1, seed=7))
    assert [random.random() for _ in range(4)] == expected
