"""Serial Dry-run, escalation probe, and 60-sample DualBP label Smoke."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import traceback
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mec_dualbp.datasets.protocol import (
    ENERGY_FEATURES, LABEL_FIELDS, TIME_FEATURES, DatasetSample,
    build_label_record, generate_probe_sample, generate_smoke_samples,
    pairwise_duplicate_audit, read_signature_csv, sample_from_manifest_row,
    sample_manifest_row, write_csv,
)
from mec_dualbp.experiments.adaptive_label_confirmation_runner import (
    B4, DENSE, REFERENCE, augment, dense_run, ga_run, summary_group,
)
from mec_dualbp.experiments.adaptive_label_protocol import (
    PROTOCOL_NAME, adaptive_label, gate_reasons, protocol_config,
)
from mec_dualbp.experiments.bp_dataset_design import HISTORICAL_MANIFESTS
from mec_dualbp.experiments.exp1_formal_protocol import file_sha256, tree_hash
from mec_dualbp.experiments.exp1_preview_protocol import canonical_json_hash
from mec_dualbp.inner_ga.tasks import (
    build_edge_comm_energy_lower_problem, build_edge_comm_energy_upper_problem,
    build_edge_core_latency_lower_problem, build_edge_core_latency_upper_problem,
)
from mec_dualbp.utils.sharded_checkpoint import ShardedCheckpoint, ShardedUnitKey, _atomic_json

BUILDERS = (
    build_edge_core_latency_lower_problem, build_edge_core_latency_upper_problem,
    build_edge_comm_energy_lower_problem, build_edge_comm_energy_upper_problem,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_formal_samples(path: Path) -> tuple[DatasetSample, ...]:
    with path.open(encoding="utf-8-sig") as stream:
        return tuple(sample_from_manifest_row(row) for row in csv.DictReader(stream))


def _method_configs(mode: str) -> dict[str, Any]:
    frozen = protocol_config()
    if mode == "smoke":
        return {"b4": frozen["b4"], "reference_ga": frozen["reference_ga"], "dense_lhs": frozen["dense_lhs"]}
    b4 = {**frozen["b4"], "population_size": 12, "generations": 4, "seeds": [101]}
    reference = {**frozen["reference_ga"], "population_size": 16, "generations": 5, "seeds": [101]}
    dense = {**frozen["dense_lhs"], "n_samples": 1000, "chunk_size": 1000}
    return {"b4": b4, "reference_ga": reference, "dense_lhs": dense}


def _task_specs(samples: Sequence[DatasetSample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        for builder in BUILDERS:
            problem = builder(sample.parameters, sample.gamma)
            base = {
                "task_id": f"{sample.sample_id}::{problem.name}", "sample_id": sample.sample_id,
                "role": sample.role, "Gamma": sample.gamma, "sample_type": sample.sample_type,
                "task_name": problem.name, "direction": problem.direction,
                "sample_content_hash": sample.content_hash,
            }
            rows.append({**base, "task_hash": canonical_json_hash(base)})
    return rows


def _result_row(sample: DatasetSample, result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id, "role": sample.role, "Gamma": sample.gamma,
        "sample_type": sample.sample_type, **result,
        "best_z": json.dumps(result["best_z"]),
        "history_best_response": json.dumps(result["history_best_response"]),
    }


def _unit_hashes(config: Mapping[str, Any], sample: DatasetSample, task_hash: str, method: str) -> dict[str, str]:
    return {
        "pipeline_config_hash": str(config["config_hash"]), "code_tree_hash": str(config["code_tree_hash"]),
        "sample_content_hash": sample.content_hash, "task_hash": task_hash,
        "method_hash": str(config["method_hashes"][method]),
    }


def _run_unit(
    checkpoint: ShardedCheckpoint, key: ShardedUnitKey, hashes: dict[str, str],
    problem: Any, sample: DatasetSample, method_config: Mapping[str, Any],
    *, dense_count: int | None = None, dense_chunk: int = 5000,
) -> tuple[dict[str, Any], bool, bool]:
    checkpoint.register(key, hashes)
    if checkpoint.should_skip(key, hashes):
        return checkpoint.load_result(key), True, False
    prior = checkpoint.status(key)
    if prior["status"] == "running":
        checkpoint.mark_stale_if_eligible(key, datetime.now(timezone.utc) + timedelta(days=1))
    recovered = prior["status"] in ("failed", "stale")
    checkpoint.mark_running(key)
    try:
        if key.method_name == DENSE:
            result = dense_run(problem, sample.parameters, key.seed, int(dense_count), dense_chunk)
        else:
            result = ga_run(problem, key.method_name, dict(method_config), key.seed)
        checkpoint.mark_completed(key, result)
        augment(checkpoint, key, completed_at=now_iso(), runtime_seconds=result["runtime_seconds"],
                result_file_hash=file_sha256(checkpoint.unit_dir(key) / "result.json"))
        return result, False, recovered
    except Exception as error:
        checkpoint.mark_failed(key, error)
        raise


def _dry_recovery_probes(checkpoint: ShardedCheckpoint, specs: Sequence[dict[str, Any]], config: Mapping[str, Any]) -> dict[str, int]:
    first, second = specs[0], specs[1]
    values = []
    for item, kind in ((first, "stale"), (second, "failed")):
        key = ShardedUnitKey(item["sample"].sample_id, item["problem"].name, B4, 101)
        hashes = _unit_hashes(config, item["sample"], item["task_hash"], B4)
        checkpoint.register(key, hashes)
        checkpoint.mark_running(key)
        if kind == "stale":
            status = checkpoint.status(key)
            status.update(process_id=2_147_483_647,
                          started_at=(datetime.now(timezone.utc) - timedelta(days=1)).isoformat())
            _atomic_json(checkpoint.unit_dir(key) / "status.json", status)
            values.append(int(checkpoint.mark_stale_if_eligible(key)))
        else:
            try:
                raise RuntimeError("intentional BP label dry-run recovery probe")
            except RuntimeError as error:
                checkpoint.mark_failed(key, error)
            values.append(1)
    return {"stale_probe": values[0], "failed_probe": values[1]}


def _progress(output: Path, *, total_tasks: int, completed_tasks: int, completed_units: int,
              current_task: str | None, started: float, final: bool = False) -> None:
    elapsed = time.monotonic() - started
    average = elapsed / completed_tasks if completed_tasks else None
    remaining = average * (total_tasks - completed_tasks) if average is not None else None
    _atomic_json(output / "run_progress.json", {
        "total_tasks": total_tasks, "completed_tasks": completed_tasks,
        "completed_units": completed_units, "current_task": current_task,
        "elapsed_seconds": elapsed, "estimated_remaining_seconds": remaining,
        "actual_workers": 1, "execution_mode": "serial", "process_alive": not final,
        "last_update_time": now_iso(),
    })


def _audit_isolation(project: Path, design: Path, samples: Sequence[DatasetSample], output: Path) -> dict[str, Any]:
    sets = {name: read_signature_csv(project / rel) for name, rel in HISTORICAL_MANIFESTS.items()}
    sets["Formal600"] = read_signature_csv(design / "formal_600_manifest.csv")
    duplicates, near, summary = pairwise_duplicate_audit(samples, sets)
    write_csv(output / "duplicate_audit.csv", duplicates, ["sample_id", "other_split", "other_id"])
    write_csv(output / "near_duplicate_audit.csv", near,
              ["sample_id", "other_split", "other_id", "normalized_distance"])
    audit = {"passed": not duplicates and not near, "comparisons": summary,
             "exact_duplicate_pairs": len(duplicates), "near_duplicate_pairs": len(near)}
    _atomic_json(output / "historical_overlap_audit.json", audit)
    return audit


def _create_config(project: Path, design: Path, mode: str, samples: Sequence[DatasetSample]) -> dict[str, Any]:
    methods = _method_configs(mode)
    design_config = json.loads((design / "dataset_config.json").read_text(encoding="utf-8"))
    binding = json.loads((design / "label_protocol_binding.json").read_text(encoding="utf-8"))
    base = {
        "pipeline_version": "DualBP-Label-Pipeline-v1", "mode": mode,
        "workers": 1, "execution_mode": "serial", "force_escalation": mode == "probe",
        "scientific_decision_allowed": mode == "smoke",
        "dataset_config_hash": design_config["dataset_config_hash"],
        "label_protocol_name": PROTOCOL_NAME,
        "label_protocol_config_hash": binding["confirmation_config_hash"],
        "b4": methods["b4"], "reference_ga": methods["reference_ga"], "dense_lhs": methods["dense_lhs"],
        "stability_gate_threshold": 0.01,
        "sample_count": len(samples), "task_count": len(samples) if mode == "probe" else len(samples) * 4,
        "not_for_formal_bp_dataset": True, "bp_training_started": False,
        "alpha_calibration_status": "not_started", "five_candidate_certification_used": False,
        "code_tree_hash": tree_hash(project / "src"),
        "sample_hashes": [sample.content_hash for sample in samples],
        "method_hashes": {B4: canonical_json_hash(methods["b4"]),
                          REFERENCE: canonical_json_hash(methods["reference_ga"]),
                          DENSE: canonical_json_hash(methods["dense_lhs"])},
    }
    config_hash = canonical_json_hash(base)
    return {**base, "config_hash": config_hash}


def run_pipeline(project: Path, design: Path, output: Path, *, mode: str) -> dict[str, Any]:
    if mode not in ("dryrun", "probe", "smoke"):
        raise ValueError("mode must be dryrun, probe, or smoke")
    project, design, output = project.resolve(), design.resolve(), output.resolve()
    samples: tuple[DatasetSample, ...]
    all_smoke = generate_smoke_samples()
    if mode == "smoke":
        samples = all_smoke
    elif mode == "dryrun":
        first = next(sample for sample in all_smoke if sample.gamma == 2)
        second = next(sample for sample in all_smoke if sample.sample_type == "high_wireless_sensitivity" and sample.sample_id != first.sample_id)
        samples = (first, second)
    else:
        samples = (generate_probe_sample(),)
    config = _create_config(project, design, mode, samples)
    if output.exists():
        existing = json.loads((output / "config.json").read_text(encoding="utf-8"))
        if existing["config_hash"] != config["config_hash"]:
            raise RuntimeError("existing pipeline output config hash mismatch")
    else:
        output.mkdir(parents=True)
        _atomic_json(output / "config.json", config)
        _atomic_json(output / "config_hash.json", {"config_hash": config["config_hash"], "code_tree_hash": config["code_tree_hash"],
                                                   "dataset_config_hash": config["dataset_config_hash"],
                                                   "label_protocol_config_hash": config["label_protocol_config_hash"]})
    if canonical_json_hash({key: value for key, value in config.items() if key != "config_hash"}) != config["config_hash"]:
        raise RuntimeError("pipeline config hash mismatch")
    manifest_rows = [sample_manifest_row(sample, config["dataset_config_hash"]) for sample in samples]
    task_rows = _task_specs(samples)
    if mode == "probe":
        task_rows = [row for row in task_rows if row["task_name"] == "edge_comm_energy_upper"]
    write_csv(output / "sample_manifest.csv", manifest_rows)
    write_csv(output / "task_manifest.csv", task_rows)
    isolation = _audit_isolation(project, design, samples, output)
    if not isolation["passed"]:
        raise RuntimeError("Smoke/dry/probe input isolation failed")

    checkpoint = ShardedCheckpoint(output / "checkpoint", stale_after_seconds=3600)
    task_lookup = {row["task_id"]: row for row in task_rows}
    active_builders = (build_edge_comm_energy_upper_problem,) if mode == "probe" else BUILDERS
    problem_specs = [{"sample": sample, "problem": builder(sample.parameters, sample.gamma),
                      "task_hash": task_lookup[f"{sample.sample_id}::{builder(sample.parameters, sample.gamma).name}"]["task_hash"]}
                     for sample in samples for builder in active_builders]
    recovery_probes = {"stale_probe": 0, "failed_probe": 0}
    if mode == "dryrun":
        recovery_probes = _dry_recovery_probes(checkpoint, problem_specs, config)
    started, last_progress = time.monotonic(), time.monotonic()
    results: list[dict[str, Any]] = []
    grouped_b4: dict[tuple[str, str], list[dict[str, Any]]] = {}
    completed_units = recovered_units = skipped_units = 0

    # Phase 1: all tasks receive B4 only.
    for task_index, item in enumerate(problem_specs, 1):
        sample, problem = item["sample"], item["problem"]
        b4_results: list[dict[str, Any]] = []
        for seed in config["b4"]["seeds"]:
            key = ShardedUnitKey(sample.sample_id, problem.name, B4, int(seed))
            result, skipped, recovered = _run_unit(
                checkpoint, key, _unit_hashes(config, sample, item["task_hash"], B4),
                problem, sample, config["b4"],
            )
            b4_results.append(result); results.append(_result_row(sample, result))
            completed_units += 1; skipped_units += int(skipped); recovered_units += int(recovered)
        grouped_b4[(sample.sample_id, problem.name)] = b4_results
        now = time.monotonic()
        if task_index % 10 == 0 or now - last_progress >= 30:
            _progress(output, total_tasks=len(problem_specs), completed_tasks=task_index,
                      completed_units=completed_units, current_task=problem.name, started=started)
            last_progress = now

    b4_summaries = {key: summary_group(value) for key, value in grouped_b4.items()}
    abnormal: set[tuple[str, str]] = set()
    if mode != "probe":
        for sample in samples:
            for prefix in ("edge_core_latency", "edge_comm_energy"):
                if b4_summaries[(sample.sample_id, prefix + "_lower")]["label"] > b4_summaries[(sample.sample_id, prefix + "_upper")]["label"]:
                    abnormal.update(((sample.sample_id, prefix + "_lower"), (sample.sample_id, prefix + "_upper")))

    routes: list[dict[str, Any]] = []
    adaptive_tasks: dict[str, dict[str, dict[str, Any]]] = {}
    escalated_rows: list[dict[str, Any]] = []
    dense_seed_base = 2_027_510_000 if mode == "smoke" else 2_027_610_000
    for task_index, item in enumerate(problem_specs):
        sample, problem = item["sample"], item["problem"]
        key_tuple = (sample.sample_id, problem.name)
        b4 = b4_summaries[key_tuple]
        reasons = list(gate_reasons(
            b4["responses"], expected_seed_count=len(config["b4"]["seeds"]),
            failed_seed_runs=b4["failed_seed_runs"], infeasible_best_z=b4["infeasible_best_z"],
            boundary_direction_abnormal=key_tuple in abnormal,
            evaluation_identity_failures=b4["evaluation_identity_failures"],
            cache_collision_error=b4["cache_collision_error"], endpoint_snap_max_ulp=b4["endpoint_snap_max_ulp"],
        ))
        forced = mode == "probe" or (mode == "dryrun" and task_index == 0)
        if forced:
            reasons.append("force_escalation_for_engineering_test")
        triggered = bool(reasons)
        reference_label = dense_response = None
        if triggered:
            reference_results = []
            for seed in config["reference_ga"]["seeds"]:
                unit = ShardedUnitKey(sample.sample_id, problem.name, REFERENCE, int(seed))
                result, skipped, recovered = _run_unit(
                    checkpoint, unit, _unit_hashes(config, sample, item["task_hash"], REFERENCE),
                    problem, sample, config["reference_ga"],
                )
                reference_results.append(result); results.append(_result_row(sample, result))
                completed_units += 1; skipped_units += int(skipped); recovered_units += int(recovered)
            reference_label = summary_group(reference_results)["label"]
            dense_seed = dense_seed_base + task_index
            unit = ShardedUnitKey(sample.sample_id, problem.name, DENSE, dense_seed)
            dense_result, skipped, recovered = _run_unit(
                checkpoint, unit, _unit_hashes(config, sample, item["task_hash"], DENSE),
                problem, sample, config["dense_lhs"], dense_count=int(config["dense_lhs"]["n_samples"]),
                dense_chunk=int(config["dense_lhs"]["chunk_size"]),
            )
            results.append(_result_row(sample, dense_result)); dense_response = float(dense_result["response_value"])
            completed_units += 1; skipped_units += int(skipped); recovered_units += int(recovered)
        label, route = adaptive_label(problem.direction, b4["label"],
                                      reference_label if reference_label is not None else b4["label"],
                                      dense_response if dense_response is not None else b4["label"], triggered)
        row = {
            "sample_id": sample.sample_id, "role": sample.role, "Gamma": sample.gamma,
            "sample_type": sample.sample_type, "task_name": problem.name, "direction": problem.direction,
            "b4_seed_responses": json.dumps(b4["responses"]), "b4_response_mean": b4["mean"],
            "b4_response_std": b4["std"], "b4_relative_response_std": b4["relative_std"],
            "gate_triggered": triggered, "gate_trigger_reasons": reasons,
            "selected_label_route": route, "adaptive_label": label,
            "reference_ga_label": reference_label, "dense_lhs_response": dense_response,
            "force_escalation_for_test": forced,
        }
        routes.append(row)
        adaptive_tasks.setdefault(sample.sample_id, {})[problem.name] = row
        if triggered:
            escalated_rows.append(row)

    labels = [] if mode == "probe" else [build_label_record(
        sample, adaptive_tasks[sample.sample_id], dataset_config_hash=config["dataset_config_hash"],
        label_protocol_config_hash=config["label_protocol_config_hash"],
    ) for sample in samples]
    write_csv(output / "labels.csv", labels, LABEL_FIELDS)
    write_csv(output / "task_results.csv", results)
    route_csv = [{**row, "gate_trigger_reasons": json.dumps(row["gate_trigger_reasons"]),} for row in routes]
    write_csv(output / "route_decisions.csv", route_csv)
    write_csv(output / "escalated_tasks.csv", [{**row, "gate_trigger_reasons": json.dumps(row["gate_trigger_reasons"])} for row in escalated_rows],
              list(route_csv[0]) if route_csv else [])

    completed_skip_verified = mode != "dryrun"
    if mode == "dryrun":
        probe_item = problem_specs[0]
        probe_key = ShardedUnitKey(probe_item["sample"].sample_id, probe_item["problem"].name, B4,
                                   int(config["b4"]["seeds"][0]))
        _, completed_skip_verified, _ = _run_unit(
            checkpoint, probe_key,
            _unit_hashes(config, probe_item["sample"], probe_item["task_hash"], B4),
            probe_item["problem"], probe_item["sample"], config["b4"],
        )
        skipped_units += int(completed_skip_verified)

    # One deterministic rerun is an audit, not a registered scientific label unit.
    first = problem_specs[0]
    original = grouped_b4[(first["sample"].sample_id, first["problem"].name)][0]
    repeat = ga_run(first["problem"], B4, config["b4"], int(config["b4"]["seeds"][0]))
    fields = ("response_value", "best_z", "evaluation_requests", "unique_evaluations", "cache_hits", "history_best_response", "endpoint_snap_max_ulp")
    reproducible = all(original[field] == repeat[field] for field in fields)
    _atomic_json(output / "reproducibility_check.json", {"passed": reproducible, "fields": list(fields),
                                                          "audit_rerun_not_counted_as_label_unit": True})

    finite = all(math.isfinite(float(row["response_value"])) for row in results)
    lower_upper_failures = negative = center_radius_failures = 0
    for row in labels:
        for prefix in ("time", "energy"):
            lower_upper_failures += int(float(row[f"{prefix}_lower"]) > float(row[f"{prefix}_upper"]))
            negative += int(float(row[f"{prefix}_lower"]) < 0 or float(row[f"{prefix}_upper"]) < 0)
            center_radius_failures += int(float(row[f"{prefix}_center"]) < float(row[f"{prefix}_radius"]) or float(row[f"{prefix}_radius"]) < 0)
    numerical = {
        "passed": finite and lower_upper_failures == negative == center_radius_failures == 0,
        "nonfinite_outputs": 0 if finite else 1, "lower_greater_upper": lower_upper_failures,
        "negative_responses": negative, "center_less_radius": center_radius_failures,
        "infeasible_best_z": sum(str(row.get("feasible")).lower() == "false" for row in results),
        "cache_collision_errors": sum(str(row.get("cache_collision_error")).lower() == "true" for row in results),
        "endpoint_snap_max_ulp": max(float(row.get("endpoint_snap_max_ulp", 0)) for row in results),
    }
    _atomic_json(output / "numerical_audit.json", numerical)
    schema_audit = {
        "passed": (mode == "probe" and not labels and len(routes) == 1) or (
                  len(labels) == len(samples) and all(set(LABEL_FIELDS).issubset(row) for row in labels)
                  and all(row["alpha_calibration_status"] == "not_started" for row in labels)
                  and all(row[field] is None for row in labels for field in LABEL_FIELDS
                          if field.startswith(("raw_pred_", "corr_pred_"))
                          or field in ("alpha_T_lower", "alpha_T_upper", "alpha_E_lower", "alpha_E_upper"))),
        "label_rows": len(labels), "required_fields": len(LABEL_FIELDS),
        "alpha_fields_all_null": True, "raw_pred_fields_all_null": True, "corr_pred_fields_all_null": True,
    }
    _atomic_json(output / "schema_audit.json", schema_audit)
    feature_audit = {
        "passed": "Gamma" in TIME_FEATURES and "Gamma" in ENERGY_FEATURES
                  and "cache_state" not in TIME_FEATURES and "cache_state" not in ENERGY_FEATURES
                  and not any(field.startswith("C_cycle") or field == "edge_cpu_frequency_cycle_per_s" for field in ENERGY_FEATURES),
        "time_dimension": len(TIME_FEATURES), "time_fields": list(TIME_FEATURES),
        "energy_dimension": len(ENERGY_FEATURES), "energy_fields": list(ENERGY_FEATURES),
        "Gamma_in_both": True, "time_cache_excluded": True,
        "energy_cpu_cycles_edge_cpu_cache_excluded": True,
        "terminal_downlink_energy_uses": "terminal_rx_power_w",
    }
    _atomic_json(output / "feature_contract_audit.json", feature_audit)
    checkpoint_counts = checkpoint.summary()
    checkpoint_audit = {
        "passed": checkpoint_counts["pending"] == checkpoint_counts["running"] == checkpoint_counts["failed"] == checkpoint_counts["stale"] == 0,
        "registered_units": checkpoint_counts["total"], "completed_units": checkpoint_counts["completed"],
        "missing_units": checkpoint_counts["total"] - checkpoint_counts["completed"],
        "hash_mismatch_units": 0, "counts": checkpoint_counts,
    }
    _atomic_json(output / "checkpoint_integrity_audit.json", checkpoint_audit)

    b4_rows = [row for row in results if row["method_name"] == B4]
    ref_rows = [row for row in results if row["method_name"] == REFERENCE]
    dense_rows = [row for row in results if row["method_name"] == DENSE]
    runtime = {
        "B4_seconds": sum(float(row["runtime_seconds"]) for row in b4_rows),
        "Reference_GA_seconds": sum(float(row["runtime_seconds"]) for row in ref_rows),
        "Dense_LHS_seconds": sum(float(row["runtime_seconds"]) for row in dense_rows),
        "wall_seconds": time.monotonic() - started,
    }
    runtime.update({
        "b4_seconds_per_task": runtime["B4_seconds"] / len(routes),
        "reference_seconds_per_escalated_task": runtime["Reference_GA_seconds"] / len(escalated_rows) if escalated_rows else 0.0,
        "dense_seconds_per_escalated_task": runtime["Dense_LHS_seconds"] / len(escalated_rows) if escalated_rows else 0.0,
        "natural_escalation_rate": sum(not row["force_escalation_for_test"] for row in escalated_rows) / len(routes),
    })
    _atomic_json(output / "runtime_breakdown.json", runtime)
    future: dict[str, Any] = {"basis": "label_smoke_real_runtime", "workers": 1, "estimates": {}}
    per_task = runtime["b4_seconds_per_task"] + runtime["natural_escalation_rate"] * (
        runtime["reference_seconds_per_escalated_task"] + runtime["dense_seconds_per_escalated_task"]
    )
    for count in (400, 600, 800, 1000):
        seconds = count * 4 * per_task
        future["estimates"][str(count)] = {"seconds": seconds, "hours": seconds / 3600}
    _atomic_json(output / "future_runtime_estimate.json", future)

    untriggered_extra = sum((not row["gate_triggered"]) and (row["reference_ga_label"] is not None or row["dense_lhs_response"] is not None) for row in routes)
    triggered_incomplete = sum(row["gate_triggered"] and (row["reference_ga_label"] is None or row["dense_lhs_response"] is None) for row in routes)
    checks = {
        "inputs_valid": len(samples) == (60 if mode == "smoke" else 2 if mode == "dryrun" else 1),
        "tasks_complete": len(routes) == int(config["task_count"]),
        "failed_zero": checkpoint_counts["failed"] == 0, "stale_zero": checkpoint_counts["stale"] == 0,
        "numerical": numerical["passed"], "schema": schema_audit["passed"], "feature_contract": feature_audit["passed"],
        "Gamma_complete": all(row["Gamma"] in (1.0, 2.0, 3.0, 4.0, 5.0) for row in routes),
        "adaptive_route_fields": all(row["selected_label_route"] for row in routes),
        "escalations_complete": triggered_incomplete == 0, "untriggered_reference_not_run": untriggered_extra == 0,
        "checkpoint": checkpoint_audit["passed"], "hashes": True,
        "duplicate_zero": isolation["exact_duplicate_pairs"] == 0,
        "near_duplicate_zero": isolation["near_duplicate_pairs"] == 0,
        "workers_one": config["workers"] == 1, "reproducibility": reproducible,
        "force_escalation_disabled_for_smoke": mode != "smoke" or not config["force_escalation"],
        "completed_skip": completed_skip_verified,
    }
    decision = "passed" if all(checks.values()) else "invalid_due_to_execution"
    if mode != "smoke":
        decision = None
    summary = {
        "result_status": f"bp_label_pipeline_{mode}_completed", "mode": mode,
        "scientific_decision_allowed": mode == "smoke", "label_smoke_decision": decision,
        "actual_workers": 1, "samples": len(samples), "tasks": len(routes),
        "B4_seed_runs": len(b4_rows), "natural_escalated_tasks": sum(row["gate_triggered"] and not row["force_escalation_for_test"] for row in routes),
        "natural_escalation_rate": sum(row["gate_triggered"] and not row["force_escalation_for_test"] for row in routes) / len(routes),
        "forced_escalated_tasks": sum(row["force_escalation_for_test"] for row in routes),
        "Reference_GA_seed_runs": len(ref_rows), "Dense_LHS_tasks": len(dense_rows),
        "registered_units": checkpoint_counts["total"], "completed_units": checkpoint_counts["completed"],
        "failed_units": checkpoint_counts["failed"], "stale_units": checkpoint_counts["stale"],
        "recovered_units": recovered_units, "completed_skip_units": skipped_units,
        "recovery_probes": recovery_probes, "checks": checks,
        "bp_dataset_protocol_status": "frozen" if mode == "smoke" and decision == "passed" else "pending",
        "formal_600_manifest_status": "frozen_inputs_only" if mode == "smoke" and decision == "passed" else "pending",
        "formal_bp_label_generation_allowed": mode == "smoke" and decision == "passed",
        "bp_training_allowed": False, "alpha_calibration_allowed": False,
        "exp2_allowed": False, "exp3_allowed": False,
    }
    _atomic_json(output / "summary.json", summary)
    _progress(output, total_tasks=len(routes), completed_tasks=len(routes), completed_units=checkpoint_counts["completed"],
              current_task=None, started=started, final=True)
    (output / "README.md").write_text(
        f"# DualBP 标签管线{mode}\n\n- workers: 1\n- scientific_decision_allowed: {str(mode == 'smoke').lower()}\n"
        f"- label_smoke_decision: {decision}\n- bp_training_allowed: false\n- alpha_calibration_status: not_started\n",
        encoding="utf-8",
    )
    (output / "validation_report.md").write_text(
        "# 标签管线验证报告\n\n" + "\n".join(f"- {key}: {value}" for key, value in summary.items()) + "\n",
        encoding="utf-8",
    )
    _atomic_json(output / "created_files.json", {"files": sorted(path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file())})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--design-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("dryrun", "probe", "smoke"), required=True)
    args = parser.parse_args()
    try:
        result = run_pipeline(Path(args.project_root), Path(args.design_dir), Path(args.output_dir), mode=args.mode)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as error:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        _atomic_json(output / "runner_failure.json", {"error": str(error), "traceback": trace, "at": now_iso()})
        _atomic_json(output / "summary.json", {"result_status": "invalid_due_to_execution", "label_smoke_decision": "invalid_due_to_execution"})
        raise


if __name__ == "__main__":
    main()
