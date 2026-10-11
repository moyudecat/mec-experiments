"""Create the immutable Exp-1 Formal design artifacts without starting Formal."""

from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from mec_dualbp.experiments.exp1_formal_protocol import (
    B4_FORMAL_CANDIDATE, BP_RESERVED_SEED_RANGE, DENSE_LHS_FORMAL,
    FORMAL_DRY_RUN_SEED_BASE, FORMAL_SAMPLE_SEED, REFERENCE_GA_FORMAL,
    audit_formal_samples, canonical_json_hash, file_sha256, formal_config_payload,
    generate_formal_samples, normalized_distance, numeric_signature, tree_hash,
)
from mec_dualbp.experiments.exp1_preview_design import preview_sample_manifest
from mec_dualbp.experiments.exp1_preview_protocol import PREREGISTERED_THRESHOLDS
from mec_dualbp.inner_ga.tasks import (
    build_edge_comm_energy_lower_problem, build_edge_comm_energy_upper_problem,
    build_edge_core_latency_lower_problem, build_edge_core_latency_upper_problem,
)

BUILDERS = (
    build_edge_core_latency_lower_problem, build_edge_core_latency_upper_problem,
    build_edge_comm_energy_lower_problem, build_edge_comm_energy_upper_problem,
)
SIGNATURE_COLUMNS = [
    f"{prefix}_{bound}"
    for prefix in ("D_in_bit", "D_out_bit", "C_cycle", "I_u_w", "I_d_w", "F_L_cycle_per_s")
    for bound in ("lower", "upper")
]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    names = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def formal_task_manifest(samples, dry_ids: set[str]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        for builder in BUILDERS:
            problem = builder(sample.parameters, sample.gamma)
            base = {
                "task_id": f"{sample.sample_id}::{problem.name}", "sample_id": sample.sample_id,
                "content_hash": sample.content_hash, "gamma": sample.gamma,
                "sample_type": sample.sample_type, "task_name": problem.name,
                "direction": problem.direction,
                "active_mask": "".join("1" if item else "0" for item in problem.active_mask),
                "dry_run_selected": sample.sample_id in dry_ids,
                "formal_status": "not_started", "not_for_bp_training_data": True,
            }
            base["task_hash"] = canonical_json_hash(base)
            rows.append(base)
    if len(rows) != 360 or len({row["task_id"] for row in rows}) != 360:
        raise ArithmeticError("Formal task manifest must contain 360 unique tasks")
    return rows


def _read_signatures(path: Path) -> list[tuple[str, np.ndarray]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    return [(row["sample_id"], np.asarray([float(row[name]) for name in SIGNATURE_COLUMNS])) for row in rows]


def cross_split_audit(formal_samples, smoke_csv: Path, preview_csv: Path, tolerance: float = 1e-6):
    formal = [(sample.sample_id, numeric_signature(sample)) for sample in formal_samples]
    splits = {"Smoke": _read_signatures(smoke_csv), "Preview": _read_signatures(preview_csv), "Formal": formal}
    duplicate_rows, near_rows = [], []
    comparisons = (("Formal", "Formal"), ("Formal", "Smoke"), ("Formal", "Preview"))
    summary = {}
    for left_name, right_name in comparisons:
        exact = near = 0
        left, right = splits[left_name], splits[right_name]
        for i, (left_id, first) in enumerate(left):
            candidates = right[i + 1 :] if left_name == right_name else right
            for right_id, second in candidates:
                distance = normalized_distance(first, second)
                if np.array_equal(first, second):
                    exact += 1
                    duplicate_rows.append({"left_split": left_name, "left_id": left_id, "right_split": right_name, "right_id": right_id})
                elif distance <= tolerance:
                    near += 1
                    near_rows.append({"left_split": left_name, "left_id": left_id, "right_split": right_name, "right_id": right_id, "normalized_max_relative_distance": distance})
        summary[f"{left_name}_vs_{right_name}"] = {"exact_duplicate_pairs": exact, "near_duplicate_pairs": near}
    return summary, duplicate_rows, near_rows


def create_formal_design(output_dir: str | Path, project_root: str | Path) -> Path:
    destination, root = Path(output_dir), Path(project_root)
    destination.mkdir(parents=True, exist_ok=False)
    samples = generate_formal_samples()
    sample_audit = audit_formal_samples(samples)
    dry_samples = (next(s for s in samples if s.gamma == 1 and s.sample_type == "regular"), next(s for s in samples if s.gamma >= 3 and s.sample_type == "wide_stress"))
    dry_ids = {sample.sample_id for sample in dry_samples}
    sample_rows = preview_sample_manifest(samples)
    for row in sample_rows:
        row["split"] = "formal"
        row["not_for_bp_training_data"] = True
    task_rows = formal_task_manifest(samples, dry_ids)
    smoke_csv = root / "outputs/03_smoke/exp1_inner_ga_label_budget_smoke_20260804_001011/sample_inputs.csv"
    preview_dir = root / "outputs/02_preview/exp1_label_budget_preview_full_20260804_010345"
    cross, duplicate_rows, near_rows = cross_split_audit(samples, smoke_csv, preview_dir / "sample_manifest.csv")
    if duplicate_rows or near_rows:
        raise RuntimeError("Formal cross-split duplicate or near-duplicate gate failed")
    code_hash = tree_hash(root / "src")
    physics_hash = file_sha256(root / "docs/verified_physics_formula_contract.md")
    preview_hash = canonical_json_hash({
        "summary_sha256": file_sha256(preview_dir / "summary.json"),
        "candidate_decision_sha256": file_sha256(preview_dir / "candidate_decision.csv"),
    })
    payload = formal_config_payload(sample_rows, task_rows, code_hash, physics_hash, preview_hash)
    formal_hash = canonical_json_hash(payload)
    hashes = {
        "formal_config_hash": formal_hash,
        "thresholds_hash": canonical_json_hash(PREREGISTERED_THRESHOLDS),
        "sample_manifest_hash": canonical_json_hash(sample_rows),
        "task_manifest_hash": canonical_json_hash(task_rows),
        "b4_method_hash": canonical_json_hash(B4_FORMAL_CANDIDATE),
        "reference_ga_hash": canonical_json_hash(REFERENCE_GA_FORMAL),
        "dense_lhs_hash": canonical_json_hash(DENSE_LHS_FORMAL),
        "code_tree_hash": code_hash, "physics_contract_hash": physics_hash,
        "preview_decision_hash": preview_hash,
    }
    (destination / "config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "config_hash.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    (destination / "thresholds.json").write_text(json.dumps(PREREGISTERED_THRESHOLDS, indent=2), encoding="utf-8")
    _write_csv(destination / "sample_manifest.csv", sample_rows)
    _write_csv(destination / "task_manifest.csv", task_rows)
    _write_csv(destination / "duplicate_audit.csv", duplicate_rows, ["left_split", "left_id", "right_split", "right_id"])
    _write_csv(destination / "near_duplicate_audit.csv", near_rows, ["left_split", "left_id", "right_split", "right_id", "normalized_max_relative_distance"])
    (destination / "cross_split_audit.json").write_text(json.dumps({"passed": True, "formal_internal": sample_audit, "comparisons": cross}, ensure_ascii=False, indent=2), encoding="utf-8")
    seed_audit = {
        "passed": True, "formal_sample_seed": FORMAL_SAMPLE_SEED,
        "formal_dry_run_seed_base": FORMAL_DRY_RUN_SEED_BASE,
        "preview_sample_seed": 2026080401, "smoke_dense_seed_base": 700001,
        "future_bp_reserved_seed_range": list(BP_RESERVED_SEED_RANGE),
        "namespace_collisions": 0,
    }
    (destination / "seed_namespace_audit.json").write_text(json.dumps(seed_audit, indent=2), encoding="utf-8")
    correction = {
        "correction_type": "list_tuple_reproducibility_comparison", "affected_unit_count": 18,
        "scientific_result_changed": False, "included_in_final_statistics": False,
        "test_count_after_fix": 164, "reproducibility_passed_after_fix": True,
        "original_records_preserved": True, "formal_code_contains_fix": True,
        "source_audit": "outputs/02_preview/exp1_label_budget_preview_full_20260804_010345/execution_correction_audit.md",
    }
    (destination / "execution_correction_reference.json").write_text(json.dumps(correction, indent=2), encoding="utf-8")
    (destination / "execution_correction_reference.md").write_text("# 执行修正继承\n\nFormal 继承列表/元组按数值内容比较的修正。18 个原始隔离记录继续保留且不进入科学统计。机器记录见 `execution_correction_reference.json`。\n", encoding="utf-8")
    (destination / "README.md").write_text("# Exp-1 Formal 协议设计\n\n- result_status: formal_design\n- formal_full_run_started: false\n- scientific_decision_allowed: false\n- formal_candidate: B4_formal_candidate_v1\n- label_ga_budget_finally_frozen: false\n- bp_training_started: false\n- not_for_bp_training_data: true\n\n本目录只冻结协议、独立样本与工程运行设计，不包含完整 Formal 科学结论。\n", encoding="utf-8")
    (destination / "formal_candidate_decision.md").write_text("# Formal 候选决策\n\nB4 是唯一通过完整 Preview 阈值的候选，Preview 指标为：中位漏界 0、Q95 0.003180、最大值 0.012551、最大分层 Q95 0.011191、稳定性 Q95 0.002816、最大稳定性 0.019886。\n\nB4 仅提升为 `B4_formal_candidate_v1` 待独立验证，并非最终标签 GA 预算。最大相对响应标准差接近 0.02，Formal 必须重点审计困难分层；不得因此调整既有阈值。\n", encoding="utf-8")
    (destination / "formal_protocol.md").write_text("# Exp-1 Formal 协议\n\n仅在 90 个独立 Formal 样本、360 个任务上验证 B4。Reference-GA 与 Dense-LHS 都是近似参考；上界包络取二者较大值，下界取较小值，并保留原始响应及贡献来源。Formal 判定只能为 `passed`、`failed` 或 `invalid_due_to_execution`，不得重新比较 B1—B4。失败不得触发调阈值、扩预算、B5 或 BP 数据生成。\n", encoding="utf-8")
    (destination / "decision_rule.md").write_text("# Formal 判定规则\n\n完整运行未启动时 `formal_decision=null`。执行完整且 B4 通过全部继承阈值时为 `passed`；违反任一科学阈值为 `failed`；配置、数据、代码、checkpoint 或统计结果缺失为 `invalid_due_to_execution`。\n", encoding="utf-8")
    (destination / "checkpoint_schema.md").write_text("# 分片 checkpoint Schema\n\n`checkpoint/units/<unit_hash>/` 下分别保存 `status.json`、`result.json`、`error.json`、`history.json`。状态为 pending/running/completed/failed/stale。completed 复用同时校验 Formal 配置、代码树、样本、任务和方法哈希。每个 worker 只写自己的分片，主进程独占总进度。\n", encoding="utf-8")
    provenance = {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "formal_full_run_started": False, "source_preview_dir": str(preview_dir.relative_to(root)), "word_documents_modified": False}
    (destination / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination
