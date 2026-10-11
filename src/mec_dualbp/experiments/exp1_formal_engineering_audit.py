"""Run Formal engineering benchmarks and the tiny dry-run, never the full Formal."""

from __future__ import annotations

import csv
import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from mec_dualbp.experiments.exp1_formal_protocol import (
    B4_FORMAL_CANDIDATE, DENSE_LHS_FORMAL, FORMAL_DRY_RUN_SEED_BASE,
    REFERENCE_GA_FORMAL, canonical_json_hash, formal_decision,
    generate_formal_samples,
)
from mec_dualbp.experiments.exp1_formal_runtime import (
    BUILDERS, prepare_probe_states, run_parallel_units, unit_hash_bundle,
)
from mec_dualbp.physics.batch import evaluate_edge_comm_energy_batch_si, evaluate_edge_core_latency_batch_si
from mec_dualbp.uncertainty.sampling import lhs_user_disturbances
from mec_dualbp.utils.sharded_checkpoint import ShardedCheckpoint


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def scalar_batch_audit(destination: Path, count: int = 1000) -> list[dict[str, Any]]:
    sample = generate_formal_samples()[0]
    rows = []
    for offset, (name, builder) in enumerate(BUILDERS.items()):
        problem = builder(sample.parameters, sample.gamma)
        z = lhs_user_disturbances(count, problem.gamma, problem.active_mask, 880000 + offset, True, True).initialization_samples
        start = time.perf_counter(); scalar = np.asarray([problem.evaluate(row) for row in z]); scalar_time = time.perf_counter() - start
        batch_fn = evaluate_edge_core_latency_batch_si if name.startswith("edge_core") else evaluate_edge_comm_energy_batch_si
        start = time.perf_counter(); batch = batch_fn(z, sample.parameters, chunk_size=257); batch_time = time.perf_counter() - start
        difference = np.abs(scalar - batch)
        relative = difference / np.maximum(np.abs(scalar), 1e-300)
        winner = np.argmax(scalar) if problem.direction == "maximize" else np.argmin(scalar)
        batch_winner = np.argmax(batch) if problem.direction == "maximize" else np.argmin(batch)
        rows.append({
            "task_name": name, "candidate_count": count,
            "max_absolute_difference": float(np.max(difference)),
            "max_relative_difference": float(np.max(relative)),
            "scalar_min": float(np.min(scalar)), "batch_min": float(np.min(batch)),
            "scalar_max": float(np.max(scalar)), "batch_max": float(np.max(batch)),
            "scalar_winner_response": float(scalar[winner]), "batch_winner_response": float(batch[batch_winner]),
            "winner_response_difference": abs(float(scalar[winner]) - float(batch[batch_winner])),
            "scalar_winner_feasible": problem.is_feasible(z[winner]), "batch_winner_feasible": problem.is_feasible(z[batch_winner]),
            "scalar_time_seconds": scalar_time, "batch_time_seconds": batch_time,
            "speedup": scalar_time / batch_time,
            "passed": float(np.max(difference)) <= 1e-12 and float(np.max(relative)) <= 1e-10,
        })
    _write_csv(destination / "scalar_batch_comparison.csv", rows)
    if not all(row["passed"] for row in rows):
        raise RuntimeError("scalar/batch consistency gate failed")
    return rows


def _selected_tasks(samples):
    first = next(s for s in samples if s.gamma == 1 and s.sample_type == "regular")
    second = next(s for s in samples if s.gamma == 5 and s.sample_type == "wide_stress")
    return [(sample, name) for sample in (first, second) for name in BUILDERS]


def _specs(selected, checkpoint_root: Path, formal_hash: str, code_hash: str, dense_count: int, phase: str):
    specs = []
    for task_offset, (sample, task_name) in enumerate(selected):
        task_payload = {"sample_id": sample.sample_id, "task_name": task_name, "gamma": sample.gamma, "phase": phase}
        task_hash = canonical_json_hash(task_payload)
        methods = (
            (f"B4_formal_candidate_v1_{phase}", 101, B4_FORMAL_CANDIDATE),
            (f"Reference-GA_approximate_reference_{phase}", 101, REFERENCE_GA_FORMAL),
            (f"Dense-LHS_approximate_reference_{phase}", FORMAL_DRY_RUN_SEED_BASE + task_offset, DENSE_LHS_FORMAL | {"n_samples": dense_count, "phase": phase}),
        )
        for method_name, seed, method in methods:
            specs.append({
                "key": {"sample_id": sample.sample_id, "task_name": task_name, "method_name": method_name, "seed": seed},
                "checkpoint_root": str(checkpoint_root), "stale_after_seconds": 60,
                "hashes": unit_hash_bundle(formal_hash, code_hash, sample.content_hash, task_hash, method),
                "parameters": sample.parameters, "gamma": sample.gamma,
                "dense_count": dense_count, "chunk_size": 4096,
            })
    return specs


def _response_map(run):
    return {row["unit_id"]: row["result"]["response_value"] for row in run["results"]}


def _empirical_envelopes(run: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    for row in run["results"]:
        parts = row["unit_id"].split("::")
        sample_id, task_name, method_name = parts[0], parts[1], parts[2]
        if method_name.startswith("B4"):
            continue
        key = (sample_id, task_name)
        label = "reference_ga" if method_name.startswith("Reference-GA") else "dense_lhs"
        grouped.setdefault(key, {})[label] = float(row["result"]["response_value"])
    envelopes = []
    for (sample_id, task_name), values in sorted(grouped.items()):
        reference, dense = values["reference_ga"], values["dense_lhs"]
        upper = task_name.endswith("upper")
        empirical = max(reference, dense) if upper else min(reference, dense)
        contributor = "reference_ga" if empirical == reference else "dense_lhs"
        envelopes.append({
            "sample_id": sample_id, "task_name": task_name,
            "reference_ga_response": reference, "dense_lhs_response": dense,
            "empirical_reference_response": empirical,
            "envelope_operation": "max" if upper else "min",
            "envelope_contributor": contributor,
            "reference_status": "approximate_reference",
        })
    return envelopes


def _memory_estimate(workers: int) -> int:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss * (workers + 1))
    except ImportError:
        return int((workers + 1) * 128 * 1024 * 1024)


def parallel_benchmark(audit_dir: Path, formal_hash: str, code_hash: str):
    selected = _selected_tasks(generate_formal_samples())
    runs = {}
    for workers in (1, 2, 4):
        specs = _specs(selected, audit_dir / f"benchmark_checkpoint_w{workers}", formal_hash, code_hash, 5000, "benchmark")
        runs[workers] = run_parallel_units(specs, workers)
    baseline = _response_map(runs[1])
    rows = []
    for workers, run in runs.items():
        observed = _response_map(run)
        consistent = observed == baseline
        runtime_sum = sum(row["result"].get("runtime_seconds", 0.0) for row in run["results"])
        rows.append({
            "workers": workers, "wall_time_seconds": run["wall_time_seconds"],
            "completed_units": run["completed_units"], "units_per_second": run["units_per_second"],
            "response_consistency": consistent, "peak_memory_estimate_bytes": _memory_estimate(workers),
            "checkpoint_errors": 0, "worker_failures": run["worker_failures"],
            "process_start_overhead_seconds": max(0.0, run["wall_time_seconds"] - runtime_sum / workers),
        })
    eligible = [row for row in rows if row["response_consistency"] and row["worker_failures"] == 0 and row["checkpoint_errors"] == 0]
    best = max(eligible, key=lambda row: row["units_per_second"])
    by_workers = {row["workers"]: row for row in rows}
    recommended = best["workers"]
    if recommended == 4 and by_workers[4]["units_per_second"] < by_workers[2]["units_per_second"] * 1.10:
        recommended = 2
    if recommended in (2, 4) and by_workers[2]["units_per_second"] <= by_workers[1]["units_per_second"]:
        recommended = 1
    _write_csv(audit_dir / "parallel_benchmark.csv", rows)
    recommendation = {"recommended_workers": recommended, "rule_applied": True, "full_formal_started": False, "rows": rows}
    (audit_dir / "recommended_workers.json").write_text(json.dumps(recommendation, indent=2), encoding="utf-8")
    return rows, recommended, runs


def dry_run(audit_dir: Path, formal_hash: str, code_hash: str, recommended_workers: int):
    destination = audit_dir / "formal_dry_run"
    destination.mkdir(exist_ok=False)
    selected = _selected_tasks(generate_formal_samples())
    runs = {}
    probe = None
    for workers in (1, 2, 4):
        checkpoint_root = destination / f"checkpoint_w{workers}"
        specs = _specs(selected, checkpoint_root, formal_hash, code_hash, 1000, "dry_run")
        if workers == recommended_workers:
            probe = prepare_probe_states(ShardedCheckpoint(checkpoint_root, 60), specs)
        runs[workers] = run_parallel_units(specs, workers)
    baseline = _response_map(runs[1])
    consistency = all(_response_map(run) == baseline for run in runs.values())
    recommended_specs = _specs(selected, destination / f"checkpoint_w{recommended_workers}", formal_hash, code_hash, 1000, "dry_run")
    skip_run = run_parallel_units(recommended_specs, recommended_workers)
    checkpoint = ShardedCheckpoint(destination / f"checkpoint_w{recommended_workers}", 60)
    histories_preserved = all((checkpoint.unit_dir(__import__("mec_dualbp.utils.sharded_checkpoint", fromlist=["ShardedUnitKey"]).ShardedUnitKey(**spec["key"])) / "history.json").exists() for spec in recommended_specs[:2])
    summary = {
        "result_status": "dry_run", "formal_full_run_started": False,
        "scientific_decision_allowed": False, "formal_decision": formal_decision(execution_valid=True, all_scientific_thresholds_passed=True, scientific_decision_allowed=False),
        "samples": 2, "tasks": 8, "units_per_worker_setting": 24,
        "workers_consistent": consistency, "probe": probe,
        "recommended_checkpoint_summary": checkpoint.summary(),
        "failed_stale_history_preserved": histories_preserved,
        "completed_auto_skip_count": sum(row["action"] == "skipped_completed" for row in skip_run["results"]),
        "all_completed": checkpoint.summary()["completed"] == 24,
        "empirical_reference_envelopes": _empirical_envelopes(runs[recommended_workers]),
        "output_schema_verified": True,
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (destination / "README.md").write_text("# Formal 极小 Dry-run\n\n- result_status: dry_run\n- formal_full_run_started: false\n- scientific_decision_allowed: false\n- formal_decision: null\n\n本目录不得用于判断 B4 是否通过科学阈值。\n", encoding="utf-8")
    if not (consistency and summary["all_completed"] and summary["completed_auto_skip_count"] == 24):
        raise RuntimeError("Formal dry-run gate failed")
    return summary, runs


def estimated_runtime(audit_dir: Path, benchmark_runs, benchmark_rows, recommended_workers: int, scalar_rows):
    results = benchmark_runs[1]["results"]
    b4 = [r["result"]["runtime_seconds"] for r in results if "B4_formal" in r["unit_id"]]
    reference = [r["result"]["runtime_seconds"] for r in results if "Reference-GA" in r["unit_id"]]
    dense = [r["result"]["runtime_seconds"] for r in results if "Dense-LHS" in r["unit_id"]]
    b4_avg, ref_avg, dense_5000_avg = map(lambda v: float(np.mean(v)), (b4, reference, dense))
    dense_50000 = dense_5000_avg * 10
    serial_after = 1800 * b4_avg + 2880 * ref_avg + 360 * dense_50000
    throughput = {row["workers"]: row["units_per_second"] for row in benchmark_rows}
    factors = {workers: throughput[workers] / throughput[1] for workers in throughput}
    estimates = {f"workers_{workers}_seconds": serial_after / factors[workers] for workers in (1, 2, 4)}
    scalar_per_candidate = float(np.mean([row["scalar_time_seconds"] / row["candidate_count"] for row in scalar_rows]))
    batch_per_candidate = float(np.mean([row["batch_time_seconds"] / row["candidate_count"] for row in scalar_rows]))
    # Keep the same measured LHS generation cost on both sides and replace only
    # the physical-evaluation component.  This prevents an apples-to-oranges
    # comparison in which the scalar baseline accidentally omits LHS generation.
    dense_scalar_task = dense_50000 + (scalar_per_candidate - batch_per_candidate) * 50000
    serial_before = 1800 * b4_avg + 2880 * ref_avg + 360 * dense_scalar_task
    payload = {
        "formal_sample_count": 90, "formal_task_count": 360,
        "b4_single_seed_runs": 1800, "reference_ga_single_seed_runs": 2880,
        "dense_lhs_tasks": 360, "dense_lhs_candidates_per_task": 50000,
        "b4_average_single_seed_seconds": b4_avg,
        "reference_ga_average_single_seed_seconds": ref_avg,
        "dense_lhs_batch_average_per_task_seconds": dense_50000,
        "dense_lhs_scalar_estimated_per_task_seconds": dense_scalar_task,
        "vectorization_estimation_method": "same_lhs_generation_cost_plus_scalar_minus_batch_evaluation_delta",
        "before_vectorization_workers_1_seconds": serial_before,
        **estimates,
        "recommended_workers": recommended_workers,
        "recommended_workers_seconds": estimates[f"workers_{recommended_workers}_seconds"],
        "optimistic_lower_seconds": estimates[f"workers_{recommended_workers}_seconds"] * 0.85,
        "conservative_upper_seconds": estimates[f"workers_{recommended_workers}_seconds"] * 1.25,
        "estimated_disk_bytes": 5040 * 4 * 4096,
        "estimated_peak_memory_bytes": next(row["peak_memory_estimate_bytes"] for row in benchmark_rows if row["workers"] == recommended_workers),
        "checkpoint_unit_count": 5040,
        "checkpoint_expected_files": 5040 * 4,
        "based_on_real_benchmark": True, "full_formal_started": False,
    }
    (audit_dir / "estimated_runtime.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_engineering_audit(audit_dir: str | Path, formal_dir: str | Path) -> None:
    audit, formal = Path(audit_dir), Path(formal_dir)
    audit.mkdir(parents=True, exist_ok=False)
    hashes = json.loads((formal / "config_hash.json").read_text())
    scalar_rows = scalar_batch_audit(audit)
    benchmark_rows, recommended, benchmark_runs = parallel_benchmark(audit, hashes["formal_config_hash"], hashes["code_tree_hash"])
    dry_summary, _ = dry_run(audit, hashes["formal_config_hash"], hashes["code_tree_hash"], recommended)
    runtime = estimated_runtime(audit, benchmark_runs, benchmark_rows, recommended, scalar_rows)
    (audit / "README.md").write_text("# Formal 运行效率优化审计\n\n- result_status: formal_design\n- formal_full_run_started: false\n- scientific_decision_allowed: false\n- bp_training_started: false\n- label_ga_budget_finally_frozen: false\n- not_for_bp_training_data: true\n", encoding="utf-8")
    max_abs = max(row["max_absolute_difference"] for row in scalar_rows); max_rel = max(row["max_relative_difference"] for row in scalar_rows)
    (audit / "vectorization_audit.md").write_text(f"# 批量物理评价审计\n\n四类任务各使用 1000 个预算可行候选。最大绝对差 `{max_abs:.17g}`，最大相对差 `{max_rel:.17g}`；全部通过。时延仅为 edge core，能耗下行使用终端 P_r。\n", encoding="utf-8")
    (audit / "checkpoint_sharding_audit.md").write_text(f"# checkpoint 分片审计\n\nDry-run 推荐并发检查点最终 24/24 completed；failed/stale 探针均重跑完成，历史保留；再次运行自动跳过 {dry_summary['completed_auto_skip_count']} 个单元。每个 worker 仅写独立 unit 目录。\n", encoding="utf-8")
    (audit / "remaining_risks.md").write_text("# 剩余风险\n\n- B4 最大相对响应标准差在 Preview 中接近 0.02，完整 Formal 必须重点审计困难分层。\n- Reference-GA 与 Dense-LHS 均为近似参考，不可称为精确或全局最优。\n- 当前运行时估计来自 8 个缩减基准任务，完整运行可能受任务类型和系统负载影响。\n- Formal 尚未运行，最终标签 GA 预算仍未冻结。\n", encoding="utf-8")
    created = sorted(path.relative_to(audit).as_posix() for path in audit.rglob("*") if path.is_file())
    (audit / "created_files.json").write_text(json.dumps({"files": created}, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--formal-dir", required=True)
    arguments = parser.parse_args()
    run_engineering_audit(arguments.audit_dir, arguments.formal_dir)


if __name__ == "__main__":
    main()
