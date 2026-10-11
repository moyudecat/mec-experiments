"""参数化、可复现的内层 GA 开发基础框架。"""

from .config import InnerGAConfig
from .cache import CacheCollisionAudit, audit_cache_collision, quantized_cache_key
from .multiseed import solve_boundary_multiseed
from .operators import elite_selection, polynomial_mutation, sbx_crossover, tournament_selection
from .problem import InnerGAProblem
from .result import BoundaryMultiseedResult, InnerGAResult
from .solver import solve_inner_ga
from .tasks import (
    EdgeTaskParameters,
    build_edge_comm_energy_lower_problem,
    build_edge_comm_energy_upper_problem,
    build_edge_core_latency_lower_problem,
    build_edge_core_latency_upper_problem,
)

__all__ = [
    "InnerGAConfig",
    "CacheCollisionAudit",
    "quantized_cache_key",
    "audit_cache_collision",
    "InnerGAProblem",
    "InnerGAResult",
    "BoundaryMultiseedResult",
    "EdgeTaskParameters",
    "tournament_selection",
    "sbx_crossover",
    "polynomial_mutation",
    "elite_selection",
    "solve_inner_ga",
    "solve_boundary_multiseed",
    "build_edge_core_latency_lower_problem",
    "build_edge_core_latency_upper_problem",
    "build_edge_comm_energy_lower_problem",
    "build_edge_comm_energy_upper_problem",
]
