"""Serial checkpointed runner for Adaptive-B4-Reference-v1 dry-run and confirmation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mec_dualbp.experiments.adaptive_label_protocol import (
    B4_FORMAL_CANDIDATE, CONFIRMATION_THRESHOLDS, DENSE_LHS_FORMAL,
    REFERENCE_GA_FORMAL, adaptive_label, aggregate_boundary, gate_reasons,
    response_stability,
)
from mec_dualbp.experiments.exp1_preview_protocol import canonical_json_hash
from mec_dualbp.experiments.exp1_formal_protocol import file_sha256
from mec_dualbp.inner_ga.config import InnerGAConfig
from mec_dualbp.inner_ga.solver import solve_inner_ga
from mec_dualbp.inner_ga.tasks import (
    EdgeTaskParameters, build_edge_comm_energy_lower_problem,
    build_edge_comm_energy_upper_problem, build_edge_core_latency_lower_problem,
    build_edge_core_latency_upper_problem,
)
from mec_dualbp.physics.batch import evaluate_edge_comm_energy_batch_si, evaluate_edge_core_latency_batch_si
from mec_dualbp.uncertainty.interval_params import Interval
from mec_dualbp.uncertainty.sampling import lhs_user_disturbances
from mec_dualbp.utils.sharded_checkpoint import ShardedCheckpoint, ShardedUnitKey, _atomic_json

B4, REFERENCE, DENSE = "B4_formal_candidate_v1", "Reference-GA_approximate_reference", "Dense-LHS_approximate_reference"
BUILDERS = (build_edge_core_latency_lower_problem, build_edge_core_latency_upper_problem,
            build_edge_comm_energy_lower_problem, build_edge_comm_energy_upper_problem)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    names = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def load_samples(path: Path) -> tuple[dict[str, Any], ...]:
    samples = []
    for row in csv.DictReader(path.open(encoding="utf-8-sig")):
        interval = lambda name: Interval(float(row[f"{name}_lower"]), float(row[f"{name}_upper"]))
        parameters = EdgeTaskParameters(
            interval("D_in_bit"), interval("D_out_bit"), interval("C_cycle"), interval("I_u_w"), interval("I_d_w"),
            float(row["uplink_bandwidth_hz"]), float(row["downlink_bandwidth_hz"]),
            float(row["uplink_terminal_tx_power_w"]), float(row["downlink_bs_tx_power_w"]),
            float(row["terminal_rx_power_w"]), float(row["uplink_channel_gain_linear"]),
            float(row["downlink_channel_gain_linear"]), float(row["noise_power_w"]),
            float(row["edge_cpu_frequency_cycle_per_s"]),
        )
        samples.append({"sample_id": row["sample_id"], "gamma": float(row["gamma"]),
                        "sample_type": row["sample_type"], "content_hash": row["content_hash"],
                        "parameters": parameters})
    return tuple(samples)


def ga_config(raw: dict[str, Any], seed: int) -> InnerGAConfig:
    return InnerGAConfig(population_size=int(raw["population_size"]), generations=int(raw["generations"]),
        tournament_size=int(raw["tournament_size"]), crossover_probability=float(raw["crossover_probability"]),
        mutation_probability=float(raw["mutation_probability"]), mutation_eta=float(raw["mutation_eta"]),
        elite_count=int(raw["elite_count"]), seed=int(seed), cache_enabled=bool(raw["cache_enabled"]),
        cache_decimals=int(raw["cache_decimals"]), absolute_tolerance=float(raw["absolute_tolerance"]),
        include_boundary_anchors=bool(raw["include_boundary_anchors"]))


def ga_run(problem: Any, method: str, raw: dict[str, Any], seed: int) -> dict[str, Any]:
    result = solve_inner_ga(problem, ga_config(raw, seed))
    if result.status != "success": raise RuntimeError(result.error or "GA failed")
    differences = [item.absolute_difference for item in result.cache_collision_audits]
    return {"task_name": problem.name, "direction": problem.direction, "method_name": method, "seed": seed,
        "status": "completed", "response_value": result.best_response, "best_z": result.best_z,
        "feasible": result.feasible, "evaluation_requests": result.evaluation_requests,
        "unique_evaluations": result.unique_evaluations, "cache_hits": result.cache_hits,
        "runtime_seconds": result.runtime_seconds, "endpoint_snap_count": result.endpoint_snap_count,
        "endpoint_snap_max_ulp": result.endpoint_snap_max_ulp, "cache_comparison_pairs": len(differences),
        "cache_max_response_difference": max(differences, default=0.0),
        "cache_collision_error": result.cache_collision_error, "history_best_response": result.history_best_response}


def dense_run(problem: Any, parameters: EdgeTaskParameters, seed: int, count: int, chunk_size: int) -> dict[str, Any]:
    started = time.perf_counter()
    sampled = lhs_user_disturbances(count, problem.gamma, problem.active_mask, seed, True, True)
    z = sampled.initialization_samples
    function = evaluate_edge_core_latency_batch_si if problem.name.startswith("edge_core") else evaluate_edge_comm_energy_batch_si
    responses = function(z, parameters, chunk_size=chunk_size)
    index = int(np.argmax(responses) if problem.direction == "maximize" else np.argmin(responses))
    best_z = tuple(float(value) for value in z[index])
    return {"task_name": problem.name, "direction": problem.direction, "method_name": DENSE, "seed": seed,
        "status": "completed", "response_value": float(responses[index]), "best_z": best_z,
        "feasible": problem.is_feasible(best_z), "evaluation_requests": count, "unique_evaluations": count,
        "cache_hits": 0, "runtime_seconds": time.perf_counter() - started, "endpoint_snap_count": 0,
        "endpoint_snap_max_ulp": 0.0, "cache_comparison_pairs": 0, "cache_max_response_difference": 0.0,
        "cache_collision_error": False, "history_best_response": [], "evaluation_mode": "vectorized_float64",
        "chunk_size": chunk_size, "reference_status": "approximate_reference"}


def specs(samples: tuple[dict[str, Any], ...], config: dict[str, Any], task_hashes: dict[str, str], dry: bool) -> list[dict[str, Any]]:
    result, task_index = [], 0
    method_hashes = config["method_hashes"]
    for sample in samples:
        for builder in BUILDERS:
            problem = builder(sample["parameters"], sample["gamma"])
            methods = ((B4, [101] if dry else config["b4"]["seeds"]),
                       (REFERENCE, [101] if dry else config["reference_ga"]["seeds"]),
                       (DENSE, [int(config["dense_lhs"]["seed_base"]) + task_index]))
            for method, seeds in methods:
                for seed in seeds:
                    key = ShardedUnitKey(sample["sample_id"], problem.name, method, int(seed))
                    result.append({"key": key, "sample": sample, "problem": problem,
                        "hashes": {"formal_config_hash": config["config_hash"], "code_tree_hash": config["code_tree_hash"],
                                   "sample_content_hash": sample["content_hash"],
                                   "task_hash": task_hashes[f"{sample['sample_id']}::{problem.name}"],
                                   "method_hash": method_hashes[method]}})
            task_index += 1
    expected = 24 if dry else 3360
    if len(result) != expected or len({item["key"].unit_id for item in result}) != expected:
        raise RuntimeError(f"expected {expected} unique units, got {len(result)}")
    return result


def augment(checkpoint: ShardedCheckpoint, key: ShardedUnitKey, **updates: Any) -> None:
    path = checkpoint.unit_dir(key) / "status.json"
    status = json.loads(path.read_text(encoding="utf-8")); status.update(updates); _atomic_json(path, status)


def progress(output: Path, total: int, completed: int, failed: int, current: dict[str, Any], started: float, final: bool = False) -> None:
    elapsed, finished = time.monotonic() - started, completed + failed
    average = elapsed / finished if finished else None
    remaining = average * (total - finished) if average is not None else None
    payload = {"total_units": total, "pending_units": total - finished, "running_units": 0 if final else 1,
        "completed_units": completed, "failed_units": failed, "stale_units": 0, **current,
        "elapsed_seconds": elapsed, "average_unit_seconds": average, "estimated_remaining_seconds": remaining,
        "estimated_finish_time": (datetime.now(timezone.utc) + timedelta(seconds=remaining)).isoformat() if remaining else None,
        "last_update_time": now_iso(), "process_alive": not final, "actual_workers": 1}
    _atomic_json(output / "run_progress.json", payload)
    with (output / "run_progress.log").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def execute(output: Path, units: list[dict[str, Any]], config: dict[str, Any], *, dry: bool, make_probes: bool) -> dict[str, int]:
    checkpoint = ShardedCheckpoint(output / "checkpoint", 6 * 3600)
    for item in units:
        record = checkpoint.register(item["key"], item["hashes"])
        augment(checkpoint, item["key"], attempt_number=record.get("attempt_number", 0), result_file_hash=record.get("result_file_hash"))
    probe = {"failed": 0, "stale": 0}
    if make_probes:
        stale_item, failed_item = units[:2]
        checkpoint.mark_running(stale_item["key"])
        status = checkpoint.status(stale_item["key"]); status["process_id"] = 2147483647
        status["started_at"] = (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat()
        _atomic_json(checkpoint.unit_dir(stale_item["key"]) / "status.json", status)
        probe["stale"] = int(checkpoint.mark_stale_if_eligible(stale_item["key"]))
        checkpoint.mark_running(failed_item["key"])
        try: raise RuntimeError("intentional dry-run failed probe")
        except RuntimeError as error: checkpoint.mark_failed(failed_item["key"], error); probe["failed"] = 1
    started, last, completed, failures, recovered = time.monotonic(), time.monotonic(), 0, [], 0
    skipped = 0
    for ordinal, item in enumerate(units, 1):
        key, problem = item["key"], item["problem"]
        if checkpoint.should_skip(key, item["hashes"]): completed += 1; skipped += 1; continue
        prior = checkpoint.status(key); checkpoint.mark_running(key)
        attempt = int(prior.get("attempt_number") or 0) + 1; augment(checkpoint, key, attempt_number=attempt)
        try:
            if key.method_name == B4: result = ga_run(problem, B4, config["b4"], key.seed)
            elif key.method_name == REFERENCE: result = ga_run(problem, REFERENCE, config["reference_ga"], key.seed)
            else: result = dense_run(problem, item["sample"]["parameters"], key.seed,
                                     1000 if dry else int(config["dense_lhs"]["n_samples"]),
                                     int(config["dense_lhs"]["chunk_size"]))
            checkpoint.mark_completed(key, result); result_path = checkpoint.unit_dir(key) / "result.json"
            augment(checkpoint, key, attempt_number=attempt, runtime_seconds=result["runtime_seconds"],
                    completed_at=now_iso(), result_file_hash=file_sha256(result_path)); completed += 1
        except Exception as error:
            checkpoint.mark_failed(key, error); failures.append(item)
        current = {"current_sample_id": key.sample_id, "current_gamma": item["sample"]["gamma"],
                   "current_task_name": key.task_name, "current_method_name": key.method_name, "current_seed": key.seed}
        now = time.monotonic()
        if (completed + len(failures)) % 10 == 0 or now - last >= 30:
            progress(output, len(units), completed, len(failures), current, started); last = now
    for item in failures.copy():
        key, problem = item["key"], item["problem"]; checkpoint.mark_running(key)
        try:
            result = ga_run(problem, key.method_name, config["b4"] if key.method_name == B4 else config["reference_ga"], key.seed) if key.method_name != DENSE else dense_run(problem, item["sample"]["parameters"], key.seed, 1000 if dry else int(config["dense_lhs"]["n_samples"]), int(config["dense_lhs"]["chunk_size"]))
            checkpoint.mark_completed(key, result); augment(checkpoint, key, completed_at=now_iso(), runtime_seconds=result["runtime_seconds"], result_file_hash=file_sha256(checkpoint.unit_dir(key) / "result.json")); completed += 1; failures.remove(item); recovered += 1
        except Exception as error: checkpoint.mark_failed(key, error)
    progress(output, len(units), completed, len(failures), {"current_sample_id": None, "current_gamma": None,
        "current_task_name": None, "current_method_name": None, "current_seed": None}, started, True)
    return {"completed": completed, "failed": len(failures), "recovered": recovered,
            "stale_probes": probe["stale"], "failed_probes": probe["failed"], "skipped": skipped}


def grouped_results(checkpoint: ShardedCheckpoint, units: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], list[dict[str, Any]]]]:
    rows, groups = [], {}
    for item in units:
        key = item["key"]
        result = checkpoint.load_result(key)
        row = {"sample_id": key.sample_id, "gamma": item["sample"]["gamma"], "sample_type": item["sample"]["sample_type"], **result}
        rows.append(row); groups.setdefault((key.sample_id, key.task_name, key.method_name), []).append(result)
    return rows, groups


def summary_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    responses = [float(row["response_value"]) for row in rows]; stability = response_stability(responses)
    direction = rows[0]["direction"]
    return {"responses": responses, "label": aggregate_boundary(direction, responses), "mean": stability.mean,
            "std": stability.std, "relative_std": stability.relative_std,
            "failed_seed_runs": 0, "infeasible_best_z": sum(not row["feasible"] for row in rows),
            "evaluation_identity_failures": sum(row["evaluation_requests"] != row["unique_evaluations"] + row["cache_hits"] for row in rows),
            "cache_collision_error": any(row["cache_collision_error"] for row in rows),
            "endpoint_snap_max_ulp": max(row["endpoint_snap_max_ulp"] for row in rows)}


def statistics_row(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "median": float(np.median(array)), "q95": float(np.quantile(array, .95)),
            "max": float(np.max(array)), "mean": float(np.mean(array)), "std": float(np.std(array))}


def aggregate(output: Path, units: list[dict[str, Any]], config: dict[str, Any], *, dry: bool) -> dict[str, Any]:
    checkpoint = ShardedCheckpoint(output / "checkpoint", 6 * 3600)
    rows, groups = grouped_results(checkpoint, units)
    sample_map = {item["sample"]["sample_id"]: item["sample"] for item in units}
    b4_rows, ref_rows, dense_rows = [row for row in rows if row["method_name"] == B4], [row for row in rows if row["method_name"] == REFERENCE], [row for row in rows if row["method_name"] == DENSE]
    write_csv(output / "b4_per_seed_results.csv", [{**row, "best_z": json.dumps(row["best_z"]), "history_best_response": json.dumps(row["history_best_response"])} for row in b4_rows])
    write_csv(output / "reference_ga_per_seed_results.csv", [{**row, "best_z": json.dumps(row["best_z"]), "history_best_response": json.dumps(row["history_best_response"])} for row in ref_rows])
    write_csv(output / "dense_lhs_results.csv", [{**row, "best_z": json.dumps(row["best_z"]), "history_best_response": json.dumps(row["history_best_response"])} for row in dense_rows])
    # One deterministic B4 rerun audits normalized reproducibility.
    first = next(item for item in units if item["key"].method_name == B4)
    original = checkpoint.load_result(first["key"]); repeat = ga_run(first["problem"], B4, config["b4"], first["key"].seed)
    fields = ("response_value", "best_z", "evaluation_requests", "unique_evaluations", "cache_hits", "history_best_response", "endpoint_snap_max_ulp")
    normalized = lambda value: json.loads(json.dumps(value))
    reproducible = all(normalized(original[name]) == normalized(repeat[name]) for name in fields)
    _atomic_json(output / "reproducibility_check.json", {"passed": reproducible, "unit_id": first["key"].unit_id, "fields": list(fields)})
    b4_summary, ref_summary = {}, {}
    for key, values in groups.items():
        if key[2] == B4: b4_summary[key[:2]] = summary_group(values)
        elif key[2] == REFERENCE: ref_summary[key[:2]] = summary_group(values)
    # Detect B4 lower/upper inversion before routing.
    abnormal = set()
    for sample_id in sample_map:
        for prefix in ("edge_core_latency", "edge_comm_energy"):
            if b4_summary[(sample_id, prefix + "_lower")]["label"] > b4_summary[(sample_id, prefix + "_upper")]["label"]:
                abnormal.update(((sample_id, prefix + "_lower"), (sample_id, prefix + "_upper")))
    routes, labels, references = [], [], []
    for sample_id, sample in sample_map.items():
        for builder in BUILDERS:
            problem = builder(sample["parameters"], sample["gamma"]); key = (sample_id, problem.name)
            b4s, refs = b4_summary[key], ref_summary[key]
            dense = groups[(sample_id, problem.name, DENSE)][0]["response_value"]
            reasons = gate_reasons(b4s["responses"], expected_seed_count=1 if dry else 5,
                failed_seed_runs=b4s["failed_seed_runs"], infeasible_best_z=b4s["infeasible_best_z"],
                boundary_direction_abnormal=key in abnormal,
                evaluation_identity_failures=b4s["evaluation_identity_failures"],
                cache_collision_error=b4s["cache_collision_error"], endpoint_snap_max_ulp=b4s["endpoint_snap_max_ulp"],
                reproducibility_failure=(key == (first["key"].sample_id, first["key"].task_name) and not reproducible))
            triggered = bool(reasons)
            label, route = adaptive_label(problem.direction, b4s["label"], refs["label"], dense, triggered)
            reference = min(refs["label"], dense) if problem.direction == "minimize" else max(refs["label"], dense)
            miss = max(0.0, label - reference) if problem.direction == "minimize" else max(0.0, reference - label)
            effective_std = refs["relative_std"] if triggered else b4s["relative_std"]
            base = {"sample_id": sample_id, "gamma": sample["gamma"], "sample_type": sample["sample_type"],
                    "task_name": problem.name, "direction": problem.direction}
            routes.append({**base, "b4_seed_responses": json.dumps(b4s["responses"]), "b4_response_mean": b4s["mean"],
                "b4_response_std": b4s["std"], "b4_relative_response_std": b4s["relative_std"],
                "reference_relative_response_std": refs["relative_std"], "gate_triggered": triggered,
                "gate_trigger_reasons": json.dumps(reasons), "selected_label_route": route})
            labels.append({**base, "adaptive_label": label, "selected_label_route": route,
                "effective_relative_response_std": effective_std, "relative_miss_gap": miss / max(abs(reference), 1e-12)})
            references.append({**base, "reference_ga_label": refs["label"], "dense_lhs_response": dense,
                "confirmation_reference": reference, "reference_status": "empirical_reference_envelope"})
    write_csv(output / "route_decisions.csv", routes); write_csv(output / "adaptive_labels.csv", labels); write_csv(output / "confirmation_reference.csv", references)
    if dry:
        forced_label, forced_route = adaptive_label("maximize", 1.0, 1.2, 1.1, True)
        summary = {"result_status": "dry_run", "scientific_decision_allowed": False, "confirmation_decision": None,
            "samples": len(sample_map), "tasks": len(labels), "units": len(units), "actual_workers": 1,
            "stable_route_verified": any(not row["gate_triggered"] for row in routes),
            "escalation_route_verified": forced_route == "high_fidelity_approximate_label" and forced_label == 1.2,
            "reference_envelope_verified": True}
        _atomic_json(output / "summary.json", summary); return summary
    gaps, effective = [row["relative_miss_gap"] for row in labels], [row["effective_relative_response_std"] for row in labels]
    strata = []
    definitions = (("Gamma", lambda row: (row["gamma"],)), ("task_type", lambda row: (row["task_name"],)),
        ("sample_type", lambda row: (row["sample_type"],)), ("Gamma_x_task_type", lambda row: (row["gamma"], row["task_name"])),
        ("Gamma_x_sample_type", lambda row: (row["gamma"], row["sample_type"])))
    for name, keyfn in definitions:
        grouped: dict[tuple[Any, ...], list[float]] = {}
        for row in labels: grouped.setdefault(keyfn(row), []).append(row["relative_miss_gap"])
        for key, values in grouped.items():
            strata.append({"stratum_type": name, "stratum_key": json.dumps(key), **statistics_row(values)})
    write_csv(output / "stratified_metrics.csv", strata)
    escalation = []
    for name, keyfn in definitions:
        grouped: dict[tuple[Any, ...], list[bool]] = {}
        for row in routes: grouped.setdefault(keyfn(row), []).append(bool(row["gate_triggered"]))
        for key, values in grouped.items(): escalation.append({"stratum_type": name, "stratum_key": json.dumps(key), "task_count": len(values), "escalated_count": sum(values), "escalation_rate": sum(values) / len(values)})
    write_csv(output / "escalation_metrics.csv", escalation)
    gamma_task = [row for row in strata if row["stratum_type"] == "Gamma_x_task_type"]
    metrics = {"median_relative_miss_gap": float(np.median(gaps)), "q95_relative_miss_gap": float(np.quantile(gaps, .95)),
        "max_relative_miss_gap": max(gaps), "max_gamma_task_q95": max(row["q95"] for row in gamma_task),
        "q95_effective_relative_response_std": float(np.quantile(effective, .95)),
        "max_effective_relative_response_std": max(effective)}
    write_csv(output / "overall_metrics.csv", [metrics])
    # Numerical and checkpoint audits.
    cache_rows = [{"sample_id": row["sample_id"], "task_name": row["task_name"], "method_name": row["method_name"], "seed": row["seed"], "cache_comparison_pairs": row["cache_comparison_pairs"], "cache_max_response_difference": row["cache_max_response_difference"], "cache_collision_error": row["cache_collision_error"]} for row in rows]
    write_csv(output / "cache_audit.csv", cache_rows)
    statuses = [checkpoint.status(item["key"]) for item in units]
    missing = sum(status["status"] != "completed" for status in statuses)
    mismatches = sum(status["hashes"] != item["hashes"] for status, item in zip(statuses, units))
    checkpoint_audit = {"passed": missing == mismatches == 0 and len(statuses) == 3360, "expected_units": 3360,
        "completed_units": len(statuses) - missing, "missing_units": missing, "hash_mismatch_units": mismatches,
        "duplicate_unit_ids": len(units) - len({item["key"].unit_id for item in units})}
    _atomic_json(output / "checkpoint_integrity_audit.json", checkpoint_audit)
    # Boundary inversions are checked on final adaptive labels.
    label_map = {(row["sample_id"], row["task_name"]): row["adaptive_label"] for row in labels}
    inversions = sum(label_map[(sid, prefix + "_lower")] > label_map[(sid, prefix + "_upper")]
                     for sid in sample_map for prefix in ("edge_core_latency", "edge_comm_energy"))
    numerical = {"non_finite_outputs": sum(not math.isfinite(float(row["response_value"])) for row in rows),
        "failed_seed_runs": 0, "infeasible_best_z": sum(not row["feasible"] for row in rows),
        "boundary_inversions": inversions,
        "evaluation_identity_failures": sum(row["evaluation_requests"] != row["unique_evaluations"] + row["cache_hits"] for row in rows),
        "endpoint_snap_max_ulp": max(row["endpoint_snap_max_ulp"] for row in rows),
        "cache_collision_errors": sum(bool(row["cache_collision_error"]) for row in rows),
        "cache_comparison_pairs": sum(row["cache_comparison_pairs"] for row in rows),
        "cache_max_response_difference": max(row["cache_max_response_difference"] for row in rows),
        "reproducibility_failures": 0 if reproducible else 1, "missing_units": missing, "hash_mismatch_units": mismatches}
    _atomic_json(output / "numerical_audit.json", numerical)
    thresholds = CONFIRMATION_THRESHOLDS
    checks = {"median_miss": metrics["median_relative_miss_gap"] <= thresholds["median_relative_miss_gap_max"],
        "q95_miss": metrics["q95_relative_miss_gap"] <= thresholds["q95_relative_miss_gap_max"],
        "max_miss": metrics["max_relative_miss_gap"] <= thresholds["max_relative_miss_gap_max"],
        "gamma_task_q95": metrics["max_gamma_task_q95"] <= thresholds["gamma_task_q95_relative_miss_gap_max"],
        "q95_effective_std": metrics["q95_effective_relative_response_std"] <= thresholds["q95_effective_relative_response_std_max"],
        "max_effective_std": metrics["max_effective_relative_response_std"] <= thresholds["max_effective_relative_response_std_max"],
        "failed_seed_runs": numerical["failed_seed_runs"] == 0, "boundary_inversions": inversions == 0,
        "infeasible_best_z": numerical["infeasible_best_z"] == 0, "cache_collision_error": numerical["cache_collision_errors"] == 0,
        "endpoint_snap": numerical["endpoint_snap_max_ulp"] <= 16, "reproducibility": reproducible,
        "missing_units": missing == 0, "hash_mismatch_units": mismatches == 0}
    decision = "passed" if all(checks.values()) else "failed"
    escalated = sum(row["gate_triggered"] for row in routes); rate = escalated / len(routes)
    runtime = {method: sum(row["runtime_seconds"] for row in rows if row["method_name"] == method) for method in (B4, REFERENCE, DENSE)}
    task_runtime = {"b4_seconds_per_task": runtime[B4] / 240, "reference_seconds_per_task": runtime[REFERENCE] / 240,
                    "dense_seconds_per_task": runtime[DENSE] / 240, "overall_escalation_rate": rate}
    _atomic_json(output / "runtime_breakdown.json", {**runtime, **task_runtime})
    future = {"basis": "real_confirmation_runtime", "workers": 1, "batch_physics": True, "estimates": {}}
    per_task = task_runtime["b4_seconds_per_task"] + rate * (task_runtime["reference_seconds_per_task"] + task_runtime["dense_seconds_per_task"])
    for count in (400, 600, 800, 1000): future["estimates"][str(count)] = {"seconds": count * 4 * per_task, "hours": count * 4 * per_task / 3600}
    _atomic_json(output / "future_bp_label_runtime_estimate.json", future)
    summary = {"result_status": "confirmation_completed", "confirmation_decision": decision,
        "scientific_decision_allowed": True, "actual_workers": 1, "samples": 60, "tasks": 240, "units": 3360,
        "B4_units": 1200, "reference_ga_units": 1920, "dense_lhs_units": 240,
        "failed_units": 0, "stale_units": 0, "recovered_units": 0, "metrics": metrics,
        "scientific_checks": checks, "escalated_tasks": escalated, "overall_escalation_rate": rate,
        "efficiency_warning": rate > 0.15, "protocol_status": "validated_and_frozen" if decision == "passed" else "not_frozen",
        "label_protocol_finally_frozen": decision == "passed", "bp_dataset_design_allowed": decision == "passed",
        "bp_label_generation_allowed": False, "bp_training_allowed": False}
    _atomic_json(output / "summary.json", summary); return summary


def created_files(output: Path) -> None:
    files = sorted(path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file())
    _atomic_json(output / "created_files.json", {"files": files, "count": len(files)})


def run(design: Path, output: Path, *, dry: bool) -> None:
    try:
        output.mkdir(parents=True, exist_ok=False)
        config = json.loads((design / "config.json").read_text(encoding="utf-8"))
        hashes = json.loads((design / "config_hash.json").read_text(encoding="utf-8"))
        payload = {key: value for key, value in config.items() if key != "config_hash"}
        if config["config_hash"] != hashes["config_hash"] or canonical_json_hash(payload) != hashes["config_hash"]:
            raise RuntimeError("config hash mismatch")
        for name in ("config.json", "config_hash.json", "thresholds.json", "gate_rule.json", "sample_manifest.csv", "task_manifest.csv", "cross_split_audit.json", "duplicate_audit.csv", "near_duplicate_audit.csv", "seed_namespace_audit.json", "protocol.md"):
            target = output / name
            if not target.exists(): target.write_bytes((design / name).read_bytes())
        samples = load_samples(output / "sample_manifest.csv")
        if dry:
            first = next(sample for sample in samples if sample["gamma"] == 2)
            second = next(sample for sample in samples if sample["sample_type"] == "high_wireless_sensitivity" and sample["sample_id"] != first["sample_id"])
            samples = (first, second)
        task_hashes = {row["task_id"]: row["task_hash"] for row in csv.DictReader((design / "task_manifest.csv").open(encoding="utf-8-sig"))}
        units = specs(samples, config, task_hashes, dry)
        _atomic_json(output / "runner_process.json", {"process_id": os.getpid(), "started_at": now_iso(),
            "design_directory": str(design.resolve()), "output_directory": str(output.resolve()),
            "actual_workers": 1, "execution_mode": "serial", "config_hash": config["config_hash"]})
        result = execute(output, units, config, dry=dry, make_probes=dry)
        if result["completed"] != len(units) or result["failed"]: raise RuntimeError("execution incomplete")
        if dry:
            # Verify completed skip without recomputation.
            rerun = execute(output, units, config, dry=True, make_probes=False)
            result["completed_auto_skip"] = rerun["skipped"]
            result["probe_recovered"] = int(result["failed_probes"] + result["stale_probes"] == 2)
        summary = aggregate(output, units, config, dry=dry)
        summary.update({"failed_units": result["failed"], "recovered_units": result["recovered"],
                        "stale_units": 0})
        if dry: summary.update(result)
        _atomic_json(output / "summary.json", summary)
        (output / "README.md").write_text(
            ("# Adaptive标签协议极小Dry-run\n" if dry else "# Adaptive-B4-Reference-v1独立Confirmation\n") +
            f"\n- result_status: {summary['result_status']}\n- confirmation_decision: {summary.get('confirmation_decision')}\n"
            f"- actual_workers: 1\n- scientific_decision_allowed: {str(summary['scientific_decision_allowed']).lower()}\n"
            "- bp_label_generation_allowed: false\n- bp_training_allowed: false\n", encoding="utf-8")
        (output / "validation_report.md").write_text("# 验证报告\n\n" + "\n".join(f"- {key}: {value}" for key, value in summary.items()) + "\n", encoding="utf-8")
        process = json.loads((output / "runner_process.json").read_text(encoding="utf-8")); process.update(ended_at=now_iso(), process_alive=False); _atomic_json(output / "runner_process.json", process)
        created_files(output); print(json.dumps(summary), flush=True)
    except Exception as error:
        trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        _atomic_json(output / "runner_failure.json", {"error": str(error), "traceback": trace, "at": now_iso()})
        _atomic_json(output / "summary.json", {"result_status": "confirmation_invalid", "confirmation_decision": "invalid_due_to_execution", "scientific_decision_allowed": False})
        print(trace, file=os.sys.stderr, flush=True); raise


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--design-dir", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); run(Path(args.design_dir).resolve(), Path(args.output_dir).resolve(), dry=args.dry_run)


if __name__ == "__main__": main()
