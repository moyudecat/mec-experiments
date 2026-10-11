"""单种子与多种子内层 GA 的不可变结果对象。"""

from __future__ import annotations

from dataclasses import dataclass

from mec_dualbp.inner_ga.cache import CacheCollisionAudit


@dataclass(frozen=True, slots=True)
class InnerGAResult:
    """单种子结果；失败时最优值字段为 None，绝不伪造极值。"""

    problem_name: str
    direction: str
    best_z: tuple[float, ...] | None
    best_response: float | None
    best_fitness: float | None
    seed: int
    population_size: int
    generations_completed: int
    evaluation_requests: int
    unique_evaluations: int
    cache_hits: int
    runtime_seconds: float
    history_best_response: tuple[float, ...]
    history_best_fitness: tuple[float, ...]
    feasible: bool
    status: str
    error: str | None = None
    endpoint_snap_count: int = 0
    endpoint_snap_max_ulp: float = 0.0
    cache_collision_audits: tuple[CacheCollisionAudit, ...] = ()
    cache_collision_error: bool = False


@dataclass(frozen=True, slots=True)
class BoundaryMultiseedResult:
    """保留全部种子摘要的启发式边界汇总，不代表数学精确极值。"""

    problem_name: str
    direction: str
    best_z: tuple[float, ...] | None
    best_response: float | None
    winning_seed: int | None
    seed_results: tuple[InnerGAResult, ...]
    evaluation_requests: int
    unique_evaluations: int
    cache_hits: int
    runtime_seconds: float
    feasible: bool
    status: str
