"""显式 NumPy Generator 驱动的实数编码 GA 开发算子。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

RepairFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def tournament_selection(
    population: NDArray[np.float64],
    fitness: NDArray[np.float64],
    n_selected: int,
    tournament_size: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """按最大适应度执行有放回锦标赛并返回父代副本。"""

    if population.ndim != 2 or fitness.shape != (population.shape[0],):
        raise ValueError("population and fitness shapes are inconsistent")
    if not 1 <= tournament_size <= population.shape[0] or n_selected <= 0:
        raise ValueError("invalid tournament_size or n_selected")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    selected = []
    for _ in range(n_selected):
        competitors = rng.choice(population.shape[0], size=tournament_size, replace=False)
        winner = competitors[int(np.argmax(fitness[competitors]))]
        selected.append(population[winner].copy())
    return np.asarray(selected, dtype=np.float64)


def sbx_crossover(
    parent_a: NDArray[np.float64],
    parent_b: NDArray[np.float64],
    rng: np.random.Generator,
    probability: float,
    eta: float,
    repair: RepairFunction,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """执行开发默认 SBX 连续交叉；输出必经问题比例修复。

    SBX 形式和 ``eta`` 仅是可配置实现默认，不是 Formal 冻结参数。
    """

    a = np.asarray(parent_a, dtype=np.float64)
    b = np.asarray(parent_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("parents must be equal-shaped finite vectors")
    if not 0.0 <= probability <= 1.0 or eta <= 0.0:
        raise ValueError("invalid crossover probability or eta")
    if rng.random() >= probability:
        return repair(a.copy()), repair(b.copy())
    u = rng.random(a.size)
    beta = np.where(
        u <= 0.5,
        (2.0 * u) ** (1.0 / (eta + 1.0)),
        (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0)),
    )
    child_a = 0.5 * ((1.0 + beta) * a + (1.0 - beta) * b)
    child_b = 0.5 * ((1.0 - beta) * a + (1.0 + beta) * b)
    return repair(child_a), repair(child_b)


def polynomial_mutation(
    individual: NDArray[np.float64],
    rng: np.random.Generator,
    probability: float,
    eta: float,
    repair: RepairFunction,
) -> NDArray[np.float64]:
    """在 ``[-1,1]`` 上执行开发默认多项式变异，随后比例修复。

    算子形式和参数仅用于开发 Smoke，不是 Formal 冻结参数。
    """

    values = np.asarray(individual, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("individual must be a finite vector")
    if not 0.0 <= probability <= 1.0 or eta <= 0.0:
        raise ValueError("invalid mutation probability or eta")
    mutated = values.copy()
    for index in range(mutated.size):
        if rng.random() >= probability:
            continue
        x = float(np.clip(mutated[index], -1.0, 1.0))
        delta1, delta2 = (x + 1.0) / 2.0, (1.0 - x) / 2.0
        random_value = rng.random()
        power = 1.0 / (eta + 1.0)
        if random_value <= 0.5:
            value = 2.0 * random_value + (1.0 - 2.0 * random_value) * (1.0 - delta1) ** (eta + 1.0)
            delta = value**power - 1.0
        else:
            value = 2.0 * (1.0 - random_value) + 2.0 * (random_value - 0.5) * (1.0 - delta2) ** (eta + 1.0)
            delta = 1.0 - value**power
        mutated[index] = x + 2.0 * delta
    return repair(mutated)


def elite_selection(
    population: NDArray[np.float64], fitness: NDArray[np.float64], elite_count: int
) -> NDArray[np.float64]:
    """按最大适应度稳定选取精英并返回副本。"""

    if population.ndim != 2 or fitness.shape != (population.shape[0],):
        raise ValueError("population and fitness shapes are inconsistent")
    if not 0 < elite_count < population.shape[0]:
        raise ValueError("elite_count must lie within population")
    indices = np.argsort(-fitness, kind="stable")[:elite_count]
    return population[indices].copy()
