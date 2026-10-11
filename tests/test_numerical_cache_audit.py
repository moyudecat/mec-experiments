"""端点吸附与缓存量化碰撞审计测试。"""

import numpy as np

from mec_dualbp.inner_ga.cache import audit_cache_collision, quantized_cache_key
from mec_dualbp.inner_ga.config import InnerGAConfig
from mec_dualbp.inner_ga.multiseed import solve_boundary_multiseed
from mec_dualbp.inner_ga.problem import InnerGAProblem
from mec_dualbp.inner_ga.solver import solve_inner_ga
from mec_dualbp.uncertainty.interval_params import Interval
from mec_dualbp.utils.numerical_audit import get_endpoint_snap_audit, reset_endpoint_snap_audit


def test_endpoint_snap_count_and_max_ulp_are_recorded() -> None:
    reset_endpoint_snap_audit()
    interval = Interval(386435980.2516156, 2178100881.4867263)
    assert interval.recover(-1.0) == interval.lower
    audit = get_endpoint_snap_audit()
    assert audit.count == 1
    assert 0.0 < audit.max_ulp <= 16.0


def test_cache_collision_direct_recheck_flags_excess_difference() -> None:
    problem = InnerGAProblem(
        "collision", (True,) + (False,) * 5, 1.0, "maximize", lambda z: float(1.0 + z[0])
    )
    representative = np.zeros(6)
    colliding = np.array([4e-13, 0, 0, 0, 0, 0], dtype=float)
    key = quantized_cache_key(representative, 12)
    assert key == quantized_cache_key(colliding, 12)
    audit = audit_cache_collision(problem, key, representative, colliding, 1.0, 1e-14)
    assert audit.error
    assert audit.absolute_difference > audit.tolerance


def test_failed_seeds_are_preserved_in_multiseed_result() -> None:
    problem = lambda: InnerGAProblem(
        "failed", (True,) * 6, 1.0, "maximize", lambda z: float("nan")
    )
    result = solve_boundary_multiseed(
        problem, (101, 202, 303), InnerGAConfig(population_size=8, generations=1, elite_count=1)
    )
    assert len(result.seed_results) == 3
    assert all(item.status == "failed" for item in result.seed_results)


def test_fixed_subset_repeats_exact_search_outputs() -> None:
    problem = InnerGAProblem(
        "repeat", (True,) + (False,) * 5, 1.0, "maximize", lambda z: float(2.0 + z[0])
    )
    config = InnerGAConfig(population_size=12, generations=3, elite_count=2, seed=101)
    first, second = solve_inner_ga(problem, config), solve_inner_ga(problem, config)
    assert first.best_z == second.best_z
    assert first.best_response == second.best_response
    assert first.history_best_response == second.history_best_response
    assert first.evaluation_requests == second.evaluation_requests
