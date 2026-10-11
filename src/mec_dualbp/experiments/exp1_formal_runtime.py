"""Engineering-only parallel runner for Exp-1 Formal benchmarks and dry-runs."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mec_dualbp.experiments.exp1_formal_protocol import B4_FORMAL_CANDIDATE, REFERENCE_GA_FORMAL
from mec_dualbp.experiments.exp1_preview_protocol import canonical_json_hash
from mec_dualbp.inner_ga.config import InnerGAConfig
from mec_dualbp.inner_ga.solver import solve_inner_ga
from mec_dualbp.inner_ga.tasks import (
    EdgeTaskParameters,
    build_edge_comm_energy_lower_problem,
    build_edge_comm_energy_upper_problem,
    build_edge_core_latency_lower_problem,
    build_edge_core_latency_upper_problem,
)
from mec_dualbp.physics.batch import evaluate_edge_comm_energy_batch_si, evaluate_edge_core_latency_batch_si
from mec_dualbp.uncertainty.sampling import lhs_user_disturbances
from mec_dualbp.utils.sharded_checkpoint import ShardedCheckpoint, ShardedUnitKey
from mec_dualbp.utils.sharded_checkpoint import _atomic_json

BUILDERS = {
    "edge_core_latency_lower": build_edge_core_latency_lower_problem,
    "edge_core_latency_upper": build_edge_core_latency_upper_problem,
    "edge_comm_energy_lower": build_edge_comm_energy_lower_problem,
    "edge_comm_energy_upper": build_edge_comm_energy_upper_problem,
}


def _ga_config(method_name: str, seed: int) -> InnerGAConfig:
    raw = B4_FORMAL_CANDIDATE if method_name.startswith("B4") else REFERENCE_GA_FORMAL
    return InnerGAConfig(
        population_size=raw["population_size"], generations=raw["generations"],
        tournament_size=raw["tournament_size"], crossover_probability=raw["crossover_probability"],
        mutation_probability=raw["mutation_probability"], mutation_eta=raw["mutation_eta"],
        elite_count=raw["elite_count"], seed=seed, cache_enabled=raw["cache_enabled"],
        cache_decimals=raw["cache_decimals"], absolute_tolerance=raw["absolute_tolerance"],
        include_boundary_anchors=raw["include_boundary_anchors"],
    )


def _ga_payload(problem, method_name: str, seed: int) -> dict[str, Any]:
    result = solve_inner_ga(problem, _ga_config(method_name, seed))
    if result.status != "success":
        raise RuntimeError(result.error or "GA failed")
    return {
        "status": "completed", "method_name": method_name, "seed": seed,
        "task_name": problem.name, "direction": problem.direction,
        "response_value": result.best_response, "best_z": result.best_z,
        "feasible": result.feasible, "evaluation_requests": result.evaluation_requests,
        "unique_evaluations": result.unique_evaluations, "cache_hits": result.cache_hits,
        "runtime_seconds": result.runtime_seconds, "endpoint_snap_max_ulp": result.endpoint_snap_max_ulp,
        "cache_collision_error": result.cache_collision_error,
    }


def _dense_payload(problem, parameters: EdgeTaskParameters, method_name: str, seed: int, count: int, chunk_size: int) -> dict[str, Any]:
    started = time.perf_counter()
    samples = lhs_user_disturbances(count, problem.gamma, problem.active_mask, seed, True, True)
    z = samples.initialization_samples
    if problem.name.startswith("edge_core_latency"):
        responses = evaluate_edge_core_latency_batch_si(z, parameters, chunk_size=chunk_size)
    else:
        responses = evaluate_edge_comm_energy_batch_si(z, parameters, chunk_size=chunk_size)
    index = int(np.argmax(responses) if problem.direction == "maximize" else np.argmin(responses))
    return {
        "status": "completed", "method_name": method_name, "seed": seed,
        "task_name": problem.name, "direction": problem.direction,
        "response_value": float(responses[index]), "best_z": tuple(float(v) for v in z[index]),
        "feasible": problem.is_feasible(z[index]), "evaluation_requests": int(z.shape[0]),
        "unique_evaluations": len({tuple(float(v).hex() for v in row) for row in z}),
        "cache_hits": 0, "runtime_seconds": time.perf_counter() - started,
        "batch_physics": True, "chunk_size": chunk_size,
    }


def execute_sharded_unit(spec: dict[str, Any]) -> dict[str, Any]:
    """Spawn-safe worker entry point; the worker writes only its own shard."""

    key = ShardedUnitKey(**spec["key"])
    checkpoint = ShardedCheckpoint(spec["checkpoint_root"], spec["stale_after_seconds"])
    checkpoint.register(key, spec["hashes"])
    if checkpoint.should_skip(key, spec["hashes"]):
        return {"unit_id": key.unit_id, "action": "skipped_completed", "result": checkpoint.load_result(key), "pid": os.getpid()}
    checkpoint.mark_running(key)
    try:
        problem = BUILDERS[key.task_name](spec["parameters"], spec["gamma"])
        if key.method_name.startswith("Dense-LHS"):
            result = _dense_payload(problem, spec["parameters"], key.method_name, key.seed, spec["dense_count"], spec["chunk_size"])
        else:
            result = _ga_payload(problem, key.method_name, key.seed)
        checkpoint.mark_completed(key, result)
        return {"unit_id": key.unit_id, "action": "completed", "result": result, "pid": os.getpid()}
    except Exception as error:
        checkpoint.mark_failed(key, error)
        return {"unit_id": key.unit_id, "action": "failed", "error": f"{type(error).__name__}: {error}", "pid": os.getpid()}


def run_parallel_units(specs: list[dict[str, Any]], workers: int) -> dict[str, Any]:
    if workers not in (1, 2, 4):
        raise ValueError("workers must be 1, 2, or 4")
    started = time.perf_counter()
    last_progress_write = started
    results = []
    progress_path = Path(specs[0]["checkpoint_root"]) / "progress.json" if specs else None
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [pool.submit(execute_sharded_unit, spec) for spec in specs]
        for future in as_completed(futures):
            results.append(future.result())
            now = time.perf_counter()
            if progress_path is not None and (len(results) % 10 == 0 or now - last_progress_write >= 30):
                _atomic_json(progress_path, {
                    "written_by": "main_process", "main_process_id": os.getpid(),
                    "workers": workers, "total_units": len(specs),
                    "finished_units": len(results),
                    "failed_units": sum(row["action"] == "failed" for row in results),
                    "final": False,
                })
                last_progress_write = now
    elapsed = time.perf_counter() - started
    results.sort(key=lambda row: row["unit_id"])
    if progress_path is not None:
        _atomic_json(progress_path, {
            "written_by": "main_process", "main_process_id": os.getpid(),
            "workers": workers, "total_units": len(specs),
            "finished_units": len(results),
            "failed_units": sum(row["action"] == "failed" for row in results),
            "final": True,
        })
    return {
        "workers": workers, "wall_time_seconds": elapsed, "completed_units": sum(row["action"] in ("completed", "skipped_completed") for row in results),
        "worker_failures": sum(row["action"] == "failed" for row in results),
        "units_per_second": len(results) / elapsed if elapsed else float("inf"),
        "process_ids": sorted({row["pid"] for row in results}), "results": results,
    }


def unit_hash_bundle(formal_config_hash: str, code_tree_hash: str, sample_hash: str, task_hash: str, method_payload: dict[str, Any]) -> dict[str, str]:
    return {
        "formal_config_hash": formal_config_hash,
        "code_tree_hash": code_tree_hash,
        "sample_content_hash": sample_hash,
        "task_hash": task_hash,
        "method_hash": canonical_json_hash(method_payload),
    }


def prepare_probe_states(checkpoint: ShardedCheckpoint, specs: list[dict[str, Any]]) -> dict[str, Any]:
    stale_spec, failed_spec = specs[0], specs[1]
    stale_key, failed_key = ShardedUnitKey(**stale_spec["key"]), ShardedUnitKey(**failed_spec["key"])
    checkpoint.register(stale_key, stale_spec["hashes"])
    checkpoint.mark_running(stale_key)
    stale_status = checkpoint.status(stale_key)
    stale_status["started_at"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    stale_status["process_id"] = 2_147_483_647
    from mec_dualbp.utils.sharded_checkpoint import _atomic_json
    _atomic_json(checkpoint.unit_dir(stale_key) / "status.json", stale_status)
    stale_marked = checkpoint.mark_stale_if_eligible(stale_key)
    checkpoint.register(failed_key, failed_spec["hashes"])
    checkpoint.mark_running(failed_key)
    try:
        raise RuntimeError("intentional Formal dry-run failed probe")
    except RuntimeError as error:
        checkpoint.mark_failed(failed_key, error)
    return {"stale_unit_id": stale_key.unit_id, "stale_marked": stale_marked, "failed_unit_id": failed_key.unit_id}
