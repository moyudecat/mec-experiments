"""按真实响应汇总全部单种子结果。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace

from mec_dualbp.inner_ga.config import InnerGAConfig
from mec_dualbp.inner_ga.problem import InnerGAProblem
from mec_dualbp.inner_ga.result import BoundaryMultiseedResult
from mec_dualbp.inner_ga.solver import solve_inner_ga


def solve_boundary_multiseed(
    problem_factory: Callable[[], InnerGAProblem],
    seeds: Iterable[int],
    config_template: InnerGAConfig,
) -> BoundaryMultiseedResult:
    """运行全部种子并按方向取真实响应极值，不宣称数学精确极值。"""

    seed_values = tuple(seeds)
    if not seed_values or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seed_values):
        raise ValueError("seeds must be a nonempty iterable of integers")
    problem = problem_factory()
    results = tuple(
        solve_inner_ga(problem_factory(), replace(config_template, seed=seed))
        for seed in seed_values
    )
    successful = [result for result in results if result.status == "success"]
    if successful:
        chooser = max if problem.direction == "maximize" else min
        winner = chooser(successful, key=lambda result: float(result.best_response))
        best_z, best_response, winning_seed = winner.best_z, winner.best_response, winner.seed
        feasible, status = True, "success"
    else:
        best_z = best_response = winning_seed = None
        feasible, status = False, "failed"
    return BoundaryMultiseedResult(
        problem.name,
        problem.direction,
        best_z,
        best_response,
        winning_seed,
        results,
        sum(result.evaluation_requests for result in results),
        sum(result.unique_evaluations for result in results),
        sum(result.cache_hits for result in results),
        sum(result.runtime_seconds for result in results),
        feasible,
        status,
    )

