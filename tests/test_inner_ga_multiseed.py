"""多种子真实响应汇总测试。"""

from mec_dualbp.inner_ga.config import InnerGAConfig
from mec_dualbp.inner_ga.multiseed import solve_boundary_multiseed
from mec_dualbp.inner_ga.problem import InnerGAProblem


def _factory(direction: str = "maximize"):
    def factory() -> InnerGAProblem:
        return InnerGAProblem("multi", (True,) + (False,) * 5, 1.0, direction, lambda z: float(2 + z[0]))  # type: ignore[arg-type]
    return factory


def test_multiseed_upper_and_lower_choose_real_response_extreme() -> None:
    config = InnerGAConfig(population_size=8, generations=2, elite_count=1)
    upper = solve_boundary_multiseed(_factory("maximize"), (1, 2), config)
    lower = solve_boundary_multiseed(_factory("minimize"), (1, 2), config)
    assert upper.best_response == max(result.best_response for result in upper.seed_results)
    assert lower.best_response == min(result.best_response for result in lower.seed_results)
    assert upper.winning_seed in (1, 2) and lower.winning_seed in (1, 2)


def test_multiseed_preserves_all_results_and_sums_counters() -> None:
    result = solve_boundary_multiseed(_factory(), (4, 5, 6), InnerGAConfig(population_size=8, generations=2, elite_count=1))
    assert len(result.seed_results) == 3
    assert result.evaluation_requests == sum(item.evaluation_requests for item in result.seed_results)
    assert result.unique_evaluations == sum(item.unique_evaluations for item in result.seed_results)
    assert result.cache_hits == sum(item.cache_hits for item in result.seed_results)


def test_all_failed_multiseed_is_explicit() -> None:
    def factory() -> InnerGAProblem:
        return InnerGAProblem("bad", (True,) * 6, 1.0, "maximize", lambda z: float("nan"))
    result = solve_boundary_multiseed(factory, (1, 2), InnerGAConfig(population_size=8, generations=2, elite_count=1))
    assert result.status == "failed" and not result.feasible
    assert result.best_z is result.best_response is result.winning_seed is None

