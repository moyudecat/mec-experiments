"""单种子、可缓存、可复现的开发级内层 GA 求解器。"""

from __future__ import annotations

import time
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from mec_dualbp.inner_ga.config import InnerGAConfig
from mec_dualbp.inner_ga.cache import audit_cache_collision, quantized_cache_key
from mec_dualbp.inner_ga.operators import (
    elite_selection,
    polynomial_mutation,
    sbx_crossover,
    tournament_selection,
)
from mec_dualbp.inner_ga.problem import InnerGAProblem
from mec_dualbp.inner_ga.result import InnerGAResult
from mec_dualbp.uncertainty.sampling import lhs_user_disturbances
from mec_dualbp.utils.numerical_audit import get_endpoint_snap_audit, reset_endpoint_snap_audit


def _unique_initial_population(
    candidates: Iterable[NDArray[np.float64]], problem: InnerGAProblem, size: int
) -> NDArray[np.float64]:
    """按稳定顺序保留不重复可行点；Γ=0 单点域允许必要重复。"""

    unique: list[NDArray[np.float64]] = []
    keys: set[tuple[str, ...]] = set()
    for raw in candidates:
        candidate = problem.repair(raw)
        key = tuple(float(value).hex() for value in candidate)
        if key not in keys:
            unique.append(candidate)
            keys.add(key)
        if len(unique) == size:
            break
    if not unique:
        raise ArithmeticError("initialization produced no feasible candidates")
    if len(unique) < size and problem.gamma == 0.0:
        unique.extend(unique[0].copy() for _ in range(size - len(unique)))
    if len(unique) < size:
        raise ArithmeticError("initialization did not produce enough unique candidates")
    return np.asarray(unique, dtype=np.float64)


def solve_inner_ga(problem: InnerGAProblem, config: InnerGAConfig) -> InnerGAResult:
    """运行一次开发级内层 GA；异常转为明确失败结果，不伪造最优值。

    历史长度定义为 ``generations_completed+1``：第0项是初始化种群最优，
    后续每项对应一代完成后的历史全局最优。
    """

    start = time.perf_counter()
    reset_endpoint_snap_audit()
    requests = unique_evaluations = cache_hits = completed = 0
    collision_audits = []
    history_response: list[float] = []
    history_fitness: list[float] = []
    try:
        if not isinstance(problem, InnerGAProblem) or not isinstance(config, InnerGAConfig):
            raise TypeError("problem and config must use inner_ga data classes")
        rng = np.random.default_rng(config.seed)
        sampled = lhs_user_disturbances(
            config.population_size * 2,
            problem.gamma,
            problem.active_mask,
            config.seed,
            include_center=True,
            include_boundary_anchors=config.include_boundary_anchors,
        )
        population = _unique_initial_population(
            sampled.initialization_samples, problem, config.population_size
        )
        cache: dict[tuple[float, ...], float] = {}
        cache_representatives: dict[tuple[float, ...], NDArray[np.float64]] = {}

        def evaluate_population(values: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            """评价种群并更新请求、唯一评价及缓存命中计数。"""

            nonlocal requests, unique_evaluations, cache_hits
            responses = np.empty(values.shape[0], dtype=np.float64)
            for index, candidate in enumerate(values):
                requests += 1
                key = quantized_cache_key(candidate, config.cache_decimals)
                if config.cache_enabled and key in cache:
                    cache_hits += 1
                    response = cache[key]
                    representative = cache_representatives[key]
                    if tuple(value.hex() for value in representative) != tuple(
                        value.hex() for value in candidate
                    ):
                        audit = audit_cache_collision(
                            problem,
                            key,
                            representative,
                            candidate,
                            response,
                            config.absolute_tolerance,
                        )
                        collision_audits.append(audit)
                        if audit.error:
                            raise ArithmeticError("cache_collision_error")
                else:
                    response = problem.evaluate(candidate)
                    unique_evaluations += 1
                    if config.cache_enabled:
                        cache[key] = response
                        cache_representatives[key] = candidate.copy()
                responses[index] = response
            fitness = np.asarray([problem.fitness(value) for value in responses], dtype=np.float64)
            return responses, fitness

        responses, fitness = evaluate_population(population)
        best_index = int(np.argmax(fitness))
        best_z = population[best_index].copy()
        best_response = float(responses[best_index])
        best_fitness = float(fitness[best_index])
        history_response.append(best_response)
        history_fitness.append(best_fitness)
        stagnation = 0
        for generation in range(config.generations):
            elites = elite_selection(population, fitness, config.elite_count)
            children: list[NDArray[np.float64]] = []
            while len(children) < config.population_size - config.elite_count:
                parents = tournament_selection(
                    population, fitness, 2, config.tournament_size, rng
                )
                first, second = sbx_crossover(
                    parents[0],
                    parents[1],
                    rng,
                    config.crossover_probability,
                    config.mutation_eta,
                    problem.repair,
                )
                for child in (first, second):
                    children.append(
                        polynomial_mutation(
                            child,
                            rng,
                            config.mutation_probability,
                            config.mutation_eta,
                            problem.repair,
                        )
                    )
                    if len(children) == config.population_size - config.elite_count:
                        break
            population = np.vstack([elites, np.asarray(children)])
            responses, fitness = evaluate_population(population)
            current_index = int(np.argmax(fitness))
            current_fitness = float(fitness[current_index])
            if current_fitness > best_fitness + config.absolute_tolerance:
                best_fitness = current_fitness
                best_response = float(responses[current_index])
                best_z = population[current_index].copy()
                stagnation = 0
            else:
                stagnation += 1
            completed = generation + 1
            history_response.append(best_response)
            history_fitness.append(best_fitness)
            if config.stagnation_generations is not None and stagnation >= config.stagnation_generations:
                break
        feasible = problem.is_feasible(best_z)
        if not feasible:
            raise ArithmeticError("best_z is infeasible")
        endpoint_audit = get_endpoint_snap_audit()
        return InnerGAResult(
            problem.name,
            problem.direction,
            tuple(float(value) for value in best_z),
            best_response,
            best_fitness,
            config.seed,
            config.population_size,
            completed,
            requests,
            unique_evaluations,
            cache_hits,
            time.perf_counter() - start,
            tuple(history_response),
            tuple(history_fitness),
            feasible,
            "success",
            None,
            endpoint_audit.count,
            endpoint_audit.max_ulp,
            tuple(collision_audits),
            False,
        )
    except Exception as exc:  # deliberate result boundary
        endpoint_audit = get_endpoint_snap_audit()
        return InnerGAResult(
            getattr(problem, "name", "unknown_problem"),
            getattr(problem, "direction", "unknown"),
            None,
            None,
            None,
            getattr(config, "seed", -1),
            getattr(config, "population_size", 0),
            completed,
            requests,
            unique_evaluations,
            cache_hits,
            time.perf_counter() - start,
            tuple(history_response),
            tuple(history_fitness),
            False,
            "failed",
            f"{type(exc).__name__}: {exc}",
            endpoint_audit.count,
            endpoint_audit.max_ulp,
            tuple(collision_audits),
            any(audit.error for audit in collision_audits),
        )
