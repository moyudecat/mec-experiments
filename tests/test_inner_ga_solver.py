"""单种子内层 GA 求解器测试。"""

import numpy as np

from mec_dualbp.inner_ga.config import InnerGAConfig
from mec_dualbp.inner_ga.problem import InnerGAProblem
from mec_dualbp.inner_ga.solver import solve_inner_ga


def _linear_problem(direction: str = "maximize", gamma: float = 1.0) -> InnerGAProblem:
    return InnerGAProblem(
        "linear",
        (True, False, False, False, False, False),
        gamma,
        direction,  # type: ignore[arg-type]
        lambda z: float(2.0 + z[0]),
    )


def _config(seed: int = 3, cache: bool = True) -> InnerGAConfig:
    return InnerGAConfig(population_size=10, generations=4, elite_count=2, seed=seed, cache_enabled=cache)


def test_solver_returns_feasible_monotone_history() -> None:
    result = solve_inner_ga(_linear_problem(), _config())
    assert result.status == "success" and result.feasible and result.best_z is not None
    assert sum(abs(value) for value in result.best_z) <= 1.0 + 1e-12
    assert len(result.history_best_fitness) == result.generations_completed + 1
    assert all(left <= right for left, right in zip(result.history_best_fitness, result.history_best_fitness[1:]))
    assert result.history_best_fitness[-1] >= result.history_best_fitness[0]


def test_direction_preserves_real_response_semantics() -> None:
    upper = solve_inner_ga(_linear_problem("maximize"), _config())
    lower = solve_inner_ga(_linear_problem("minimize"), _config())
    assert upper.best_response == 3.0 and upper.best_fitness == 3.0
    assert lower.best_response == 1.0 and lower.best_fitness == -1.0


def test_evaluation_accounting_with_and_without_cache() -> None:
    cached = solve_inner_ga(_linear_problem(gamma=0.0), _config(cache=True))
    uncached = solve_inner_ga(_linear_problem(gamma=0.0), _config(cache=False))
    assert cached.evaluation_requests >= cached.unique_evaluations
    assert cached.cache_hits == cached.evaluation_requests - cached.unique_evaluations
    assert uncached.unique_evaluations == uncached.evaluation_requests
    assert uncached.cache_hits == 0


def test_gamma_zero_equals_center_response() -> None:
    result = solve_inner_ga(_linear_problem(gamma=0.0), _config())
    assert result.best_z == (0.0,) * 6
    assert result.best_response == 2.0


def test_failed_evaluator_is_recorded_without_fake_optimum() -> None:
    problem = InnerGAProblem("failure", (True,) * 6, 1.0, "maximize", lambda z: (_ for _ in ()).throw(RuntimeError("boom")))
    result = solve_inner_ga(problem, _config())
    assert result.status == "failed" and not result.feasible
    assert result.best_z is result.best_response is result.best_fitness is None
    assert result.error is not None and "boom" in result.error

