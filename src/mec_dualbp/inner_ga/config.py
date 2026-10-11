"""内层实数编码 GA 的参数化开发配置。"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InnerGAConfig:
    """开发级 GA 配置；默认值不是 Preview 或 Formal 冻结参数。"""

    population_size: int = 16
    generations: int = 8
    tournament_size: int = 3
    crossover_probability: float = 0.9
    mutation_probability: float = 1.0 / 6.0
    mutation_eta: float = 20.0
    elite_count: int = 2
    seed: int = 0
    cache_enabled: bool = True
    cache_decimals: int = 15
    stagnation_generations: int | None = None
    absolute_tolerance: float = 1e-12
    include_boundary_anchors: bool = True
    status: str = "smoke_default_not_formal"

    def __post_init__(self) -> None:
        """严格验证开发配置，不静默调整无效参数。"""

        integer_positive = {
            "population_size": self.population_size,
            "generations": self.generations,
            "tournament_size": self.tournament_size,
            "mutation_eta": self.mutation_eta,
        }
        for name, value in integer_positive.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.population_size, int) or self.population_size < 2:
            raise ValueError("population_size must be an integer >= 2")
        if not isinstance(self.generations, int):
            raise TypeError("generations must be an integer")
        if not isinstance(self.tournament_size, int) or self.tournament_size > self.population_size:
            raise ValueError("tournament_size must be an integer within population")
        if not isinstance(self.elite_count, int) or not 0 < self.elite_count < self.population_size:
            raise ValueError("elite_count must lie in [1, population_size)")
        for name, value in (
            ("crossover_probability", self.crossover_probability),
            ("mutation_probability", self.mutation_probability),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not math.isfinite(self.mutation_eta) or self.mutation_eta <= 0.0:
            raise ValueError("mutation_eta must be finite and positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(self.cache_enabled, bool):
            raise TypeError("cache_enabled must be boolean")
        if not isinstance(self.cache_decimals, int) or not 12 <= self.cache_decimals <= 16:
            raise ValueError("cache_decimals must lie in [12, 16]")
        if self.stagnation_generations is not None and (
            not isinstance(self.stagnation_generations, int) or self.stagnation_generations <= 0
        ):
            raise ValueError("stagnation_generations must be a positive integer or None")
        if not math.isfinite(self.absolute_tolerance) or self.absolute_tolerance < 0.0:
            raise ValueError("absolute_tolerance must be finite and nonnegative")
        if self.status != "smoke_default_not_formal":
            raise ValueError("GA parameter status must remain smoke_default_not_formal")

