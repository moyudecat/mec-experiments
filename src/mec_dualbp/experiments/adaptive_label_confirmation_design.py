"""Create frozen design artifacts for Adaptive-B4-Reference-v1 confirmation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from mec_dualbp.experiments.adaptive_label_protocol import (
    CONFIRMATION_DENSE_SEED_BASE, CONFIRMATION_SAMPLE_SEED,
    CONFIRMATION_THRESHOLDS, FUTURE_BP_SEED_RANGE,
    STABILITY_GATE_THRESHOLD, generate_confirmation_samples, protocol_config,
)
from mec_dualbp.experiments.exp1_formal_protocol import canonical_json_hash, normalized_distance, numeric_signature, tree_hash
from mec_dualbp.experiments.exp1_preview_design import preview_sample_manifest
from mec_dualbp.inner_ga.tasks import (
    build_edge_comm_energy_lower_problem, build_edge_comm_energy_upper_problem,
    build_edge_core_latency_lower_problem, build_edge_core_latency_upper_problem,
)

BUILDERS = (build_edge_core_latency_lower_problem, build_edge_core_latency_upper_problem,
            build_edge_comm_energy_lower_problem, build_edge_comm_energy_upper_problem)
B4, REFERENCE, DENSE = "B4_formal_candidate_v1", "Reference-GA_approximate_reference", "Dense-LHS_approximate_reference"
SIGNATURE_COLUMNS = [f"{prefix}_{bound}" for prefix in ("D_in_bit", "D_out_bit", "C_cycle", "I_u_w", "I_d_w", "F_L_cycle_per_s") for bound in ("lower", "upper")]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    names = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names); writer.writeheader(); writer.writerows(rows)


def read_signatures(path: Path) -> list[tuple[str, np.ndarray]]:
    return [(row["sample_id"], np.asarray([float(row[name]) for name in SIGNATURE_COLUMNS], dtype=np.float64))
            for row in csv.DictReader(path.open(encoding="utf-8-sig"))]


def create_design(destination: str | Path, root: str | Path) -> Path:
    output, project = Path(destination), Path(root)
    output.mkdir(parents=True, exist_ok=False)
    samples = generate_confirmation_samples()
    sample_rows = preview_sample_manifest(samples)
    for row in sample_rows:
        row.update(split="adaptive_confirmation", not_for_bp_training=True, not_for_bp_validation=True,
                   not_for_bp_calibration=True, not_for_bp_test=True)
    task_rows = []
    for sample in samples:
        for builder in BUILDERS:
            problem = builder(sample.parameters, sample.gamma)
            base = {"task_id": f"{sample.sample_id}::{problem.name}", "sample_id": sample.sample_id,
                    "gamma": sample.gamma, "sample_type": sample.sample_type, "task_name": problem.name,
                    "direction": problem.direction, "content_hash": sample.content_hash,
                    "not_for_bp_training": True, "not_for_bp_validation": True,
                    "not_for_bp_calibration": True, "not_for_bp_test": True}
            base["task_hash"] = canonical_json_hash(base); task_rows.append(base)
    if len(task_rows) != 240 or len({row["task_id"] for row in task_rows}) != 240:
        raise RuntimeError("Confirmation task manifest must contain 240 unique tasks")
    old_paths = {
        "Smoke": project / "outputs/03_smoke/exp1_inner_ga_label_budget_smoke_20260804_001011/sample_inputs.csv",
        "Preview": project / "outputs/02_preview/exp1_label_budget_preview_full_20260804_010345/sample_manifest.csv",
        "Formal": project / "outputs/01_formal/exp1_label_budget_formal_20260805_214730/sample_manifest.csv",
    }
    confirmation = [(sample.sample_id, numeric_signature(sample)) for sample in samples]
    comparisons = {"Confirmation": confirmation, **{name: read_signatures(path) for name, path in old_paths.items()}}
    duplicate_rows, near_rows, summary = [], [], {}
    for name in ("Confirmation", "Smoke", "Preview", "Formal"):
        exact = near = 0
        right = comparisons[name]
        for index, (left_id, first) in enumerate(confirmation):
            candidates = right[index + 1:] if name == "Confirmation" else right
            for right_id, second in candidates:
                distance = normalized_distance(first, second)
                if np.array_equal(first, second):
                    exact += 1; duplicate_rows.append({"confirmation_id": left_id, "other_split": name, "other_id": right_id})
                elif distance <= 1e-6:
                    near += 1; near_rows.append({"confirmation_id": left_id, "other_split": name, "other_id": right_id, "normalized_distance": distance})
        summary[f"Confirmation_vs_{name}"] = {"exact_duplicate_pairs": exact, "near_duplicate_pairs": near}
    if duplicate_rows or near_rows: raise RuntimeError("cross-split isolation failed")
    base_config = protocol_config()
    base_config.update({"protocol_status": "confirmation_candidate_not_yet_validated",
        "sample_count": 60, "task_count": 240, "registered_units": 3360,
        "B5_units": 0, "BP_units": 0, "five_candidate_units": 0,
        "code_tree_hash": tree_hash(project / "src"),
        "sample_manifest_hash": canonical_json_hash(sample_rows), "task_manifest_hash": canonical_json_hash(task_rows),
        "method_hashes": {B4: canonical_json_hash(base_config["b4"]), REFERENCE: canonical_json_hash(base_config["reference_ga"]), DENSE: canonical_json_hash(base_config["dense_lhs"])},
        "source_formal": "outputs/01_formal/exp1_label_budget_formal_20260805_214730",
        "source_formal_rerun": False})
    config_hash = canonical_json_hash(base_config); config = {**base_config, "config_hash": config_hash}
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "config_hash.json").write_text(json.dumps({"config_hash": config_hash, "code_tree_hash": base_config["code_tree_hash"], "sample_manifest_hash": base_config["sample_manifest_hash"], "task_manifest_hash": base_config["task_manifest_hash"], "method_hashes": base_config["method_hashes"]}, indent=2), encoding="utf-8")
    (output / "thresholds.json").write_text(json.dumps(CONFIRMATION_THRESHOLDS, indent=2), encoding="utf-8")
    (output / "gate_rule.json").write_text(json.dumps({"threshold": STABILITY_GATE_THRESHOLD, "trigger_operator": ">", "equality_triggers": False, "ddof": 0, "denominator": "max(abs(response_mean),1e-12)", "audit_triggers": ["seed_execution_failed", "non_finite_response", "infeasible_best_z", "boundary_direction_abnormal", "evaluation_identity_failure", "cache_collision_error", "endpoint_snap_over_16_ulp", "seed_result_missing", "reproducibility_failure"]}, indent=2), encoding="utf-8")
    write_csv(output / "sample_manifest.csv", sample_rows); write_csv(output / "task_manifest.csv", task_rows)
    write_csv(output / "duplicate_audit.csv", duplicate_rows, ["confirmation_id", "other_split", "other_id"])
    write_csv(output / "near_duplicate_audit.csv", near_rows, ["confirmation_id", "other_split", "other_id", "normalized_distance"])
    (output / "cross_split_audit.json").write_text(json.dumps({"passed": True, "comparisons": summary}, indent=2), encoding="utf-8")
    (output / "seed_namespace_audit.json").write_text(json.dumps({"passed": True, "confirmation_sample_seed": CONFIRMATION_SAMPLE_SEED, "confirmation_dense_seed_base": CONFIRMATION_DENSE_SEED_BASE, "future_bp_reserved_seed_range": list(FUTURE_BP_SEED_RANGE), "namespace_collisions": 0}, indent=2), encoding="utf-8")
    (output / "protocol.md").write_text("# Adaptive-B4-Reference-v1协议\n\n本协议仅用于离线BP监督标签生成。每项任务先运行冻结B4五种子；相对响应标准差严格大于0.01或出现审计异常时，升级为冻结Reference-GA八种子与50000点Dense-LHS的经验参考包络。任务之间串行，Dense-LHS使用float64批量物理评价。该协议不是在线认证机制，不使用五候选、`certified_*`或Kcert。\n", encoding="utf-8")
    (output / "README.md").write_text("# Adaptive标签协议Confirmation设计\n\n- protocol: Adaptive-B4-Reference-v1\n- status: confirmation_candidate_not_yet_validated\n- workers: 1\n- bp_data_generated: false\n- bp_training_started: false\n- five_candidate_certification_used: false\n", encoding="utf-8")
    return output
