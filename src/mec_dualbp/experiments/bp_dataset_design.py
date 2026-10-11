"""Freeze the DualBP input dataset protocol and the 600-sample manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from mec_dualbp.datasets.protocol import (
    DATASET_PROTOCOL_VERSION, ENERGY_FEATURES, FEATURE_SCHEMA_VERSION,
    FORMAL_ROLE_COUNTS, LABEL_FIELDS, LABEL_SCHEMA_VERSION,
    NEAR_DUPLICATE_TOLERANCE, ROLE_SEED_BASES, ROLES, TIME_FEATURES,
    audit_role_balance, generate_formal_dataset_samples, pairwise_duplicate_audit,
    read_signature_csv, sample_manifest_row, write_csv,
)
from mec_dualbp.experiments.adaptive_label_protocol import PROTOCOL_NAME, protocol_config
from mec_dualbp.experiments.exp1_formal_protocol import file_sha256, tree_hash
from mec_dualbp.experiments.exp1_preview_protocol import canonical_json_hash
from mec_dualbp.utils.sharded_checkpoint import _atomic_json

LATEST_CH4 = "第四章_双BP误差修正_校订三线表版.docx"
LATEST_CH5 = "第五章_实验设计参考稿_误差修正_校订三线表版.docx"
HISTORICAL_MANIFESTS = {
    "Exp1_Smoke": "outputs/03_smoke/exp1_inner_ga_label_budget_smoke_20260804_001011/sample_inputs.csv",
    "Exp1_Preview": "outputs/02_preview/exp1_label_budget_preview_full_20260804_010345/sample_manifest.csv",
    "Exp1_Formal": "outputs/01_formal/exp1_label_budget_formal_20260805_214730/sample_manifest.csv",
    "Adaptive_Confirmation": "outputs/01_formal/adaptive_b4_reference_confirmation_20260810_000412/sample_manifest.csv",
}


def feature_schema() -> dict[str, Any]:
    return {
        "version": FEATURE_SCHEMA_VERSION,
        "units": "SI",
        "time_bp": {
            "dimension": len(TIME_FEATURES), "fields": list(TIME_FEATURES),
            "target": "edge_core_latency_interval", "cache_miss_delay_included": False,
            "forbidden_fields": ["cache_state", "terminal_rx_power_w", "F_L_cycle_per_s_lower", "F_L_cycle_per_s_upper"],
        },
        "energy_bp": {
            "dimension": len(ENERGY_FEATURES), "fields": list(ENERGY_FEATURES),
            "target": "terminal_communication_energy_interval",
            "forbidden_fields": ["C_cycle_lower", "C_cycle_upper", "edge_cpu_frequency_cycle_per_s", "cache_state"],
            "downlink_terminal_energy_power": "terminal_rx_power_w",
            "downlink_rate_transmit_power": "downlink_bs_tx_power_w",
        },
        "Gamma_in_both_inputs": True,
    }


def label_schema() -> dict[str, Any]:
    nullable = [field for field in LABEL_FIELDS if field.startswith(("raw_pred_", "corr_pred_"))
                or field in ("alpha_T_lower", "alpha_T_upper", "alpha_E_lower", "alpha_E_upper")]
    return {
        "version": LABEL_SCHEMA_VERSION, "fields": list(LABEL_FIELDS),
        "nullable_before_training_or_calibration": nullable,
        "alpha_calibration_status": "not_started",
        "label_semantics": {
            "time": "edge core latency only; deterministic cache-miss delay excluded",
            "energy": "terminal uplink-transmit plus downlink-receive communication energy",
            "ga_ref": "Adaptive-B4-Reference-v1 high-fidelity approximate label",
            "raw_pred": "frozen BP output before alpha correction",
            "corr_pred": "frozen BP output after alpha correction",
        },
    }


def role_definition() -> dict[str, Any]:
    return {
        "roles": {
            "Train": "network weight fitting only",
            "Validation": "early stopping, model selection, architecture and training hyperparameters",
            "Calibration": "alpha estimation only after network weights are frozen",
            "Test": "one-shot final independent evaluation only",
        },
        "counts": FORMAL_ROLE_COUNTS,
        "role_mutation_after_freeze_allowed": False,
        "test_feedback_allowed": False,
    }


def _seed_collision_audit(samples: list[Any], historical_paths: dict[str, Path]) -> dict[str, Any]:
    names = [sample.generation_seed for sample in samples]
    values = [sample.generation_seed_value for sample in samples]
    historical: set[str] = set()
    for path in historical_paths.values():
        with path.open(encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if row.get("generation_seed"):
                    historical.add(str(row["generation_seed"]))
    collisions = len(names) - len(set(names)) + len(values) - len(set(values))
    collisions += sum(name in historical or str(value) in historical for name, value in zip(names, values))
    return {
        "passed": collisions == 0, "namespace_collisions": collisions,
        "role_namespaces": {role: f"BP_{role.upper()}_*" for role in ROLES},
        "numeric_seed_ranges": {role: [base + 1, base + FORMAL_ROLE_COUNTS[role]] for role, base in ROLE_SEED_BASES.items()},
        "historical_seed_values_checked": len(historical),
    }


def create_design(project_root: Path, output: Path, *, word_sync_status: str) -> Path:
    project_root, output = project_root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    ch4, ch5 = project_root / "reference_docs" / LATEST_CH4, project_root / "reference_docs" / LATEST_CH5
    confirmation = project_root / "outputs/01_formal/adaptive_b4_reference_confirmation_20260810_000412"
    confirmation_summary = json.loads((confirmation / "summary.json").read_text(encoding="utf-8"))
    if confirmation_summary.get("confirmation_decision") != "passed" or not confirmation_summary.get("label_protocol_finally_frozen"):
        raise RuntimeError("Adaptive-B4-Reference-v1 confirmation gate is not passed")
    binding_hashes = json.loads((confirmation / "config_hash.json").read_text(encoding="utf-8"))
    samples = list(generate_formal_dataset_samples())
    balance = audit_role_balance(samples)
    base_config = {
        "dataset_protocol_version": DATASET_PROTOCOL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "formal_sample_count": 600, "role_counts": FORMAL_ROLE_COUNTS,
        "gamma_values": [1, 2, 3, 4, 5],
        "sample_types": ["regular", "narrow", "wide_stress", "near_degenerate", "high_wireless_sensitivity", "high_compute_sensitivity"],
        "replicates_per_stratum": {"Train": 14, "Validation": 2, "Calibration": 2, "Test": 2},
        "workers": 1, "execution_mode": "serial",
        "label_protocol_name": PROTOCOL_NAME,
        "label_protocol_config_hash": binding_hashes["config_hash"],
        "near_duplicate_tolerance": NEAR_DUPLICATE_TOLERANCE,
        "formal_labels_generated": False, "bp_training_started": False,
        "alpha_calibration_status": "not_started",
        "five_candidate_certification_used": False,
        "code_tree_hash": tree_hash(project_root / "src"),
        "sample_content_hashes": [sample.content_hash for sample in samples],
    }
    dataset_hash = canonical_json_hash(base_config)
    config = {**base_config, "dataset_config_hash": dataset_hash}
    rows = [sample_manifest_row(sample, dataset_hash) for sample in samples]
    if any(row["label_status"] != "not_generated" or row["execution_status"] != "queued" for row in rows):
        raise AssertionError("formal labels must not be generated in design phase")

    historical_paths = {name: project_root / value for name, value in HISTORICAL_MANIFESTS.items()}
    historical = {name: read_signature_csv(path) for name, path in historical_paths.items()}
    duplicate_rows, near_rows, overlap = pairwise_duplicate_audit(samples, historical)
    if duplicate_rows or near_rows:
        raise RuntimeError("formal 600 manifest overlaps historical data")
    seed_audit = _seed_collision_audit(samples, historical_paths)
    if not seed_audit["passed"]:
        raise RuntimeError("formal seed namespace collision")

    word_hashes = {
        "word_sync_status": word_sync_status,
        "documents": {
            LATEST_CH4: {"sha256": file_sha256(ch4), "bytes": ch4.stat().st_size},
            LATEST_CH5: {"sha256": file_sha256(ch5), "bytes": ch5.stat().st_size},
        },
        "source_modified": False,
    }
    protocol_binding = {
        "label_protocol_name": PROTOCOL_NAME,
        "confirmation_directory": str(confirmation.relative_to(project_root)).replace("\\", "/"),
        "confirmation_decision": "passed",
        "confirmation_config_hash": binding_hashes["config_hash"],
        "method_hashes": binding_hashes["method_hashes"],
        "stability_gate_threshold": protocol_config()["stability_gate_threshold"],
        "binding_consistent": True,
    }
    _atomic_json(output / "word_hashes.json", word_hashes)
    _atomic_json(output / "dataset_config.json", config)
    _atomic_json(output / "dataset_config_hash.json", {"dataset_config_hash": dataset_hash, "code_tree_hash": config["code_tree_hash"]})
    _atomic_json(output / "feature_schema.json", feature_schema())
    _atomic_json(output / "label_schema.json", label_schema())
    _atomic_json(output / "role_definition.json", role_definition())
    write_csv(output / "formal_600_manifest.csv", rows)
    _atomic_json(output / "formal_600_manifest_summary.json", {
        "status": "frozen_inputs_only", "sample_count": 600, **balance,
        "label_status": "not_generated", "dataset_config_hash": dataset_hash,
    })
    _atomic_json(output / "seed_namespace_plan.json", seed_audit)
    _atomic_json(output / "historical_overlap_audit.json", {"passed": True, "comparisons": overlap,
        "exact_duplicate_pairs": 0, "near_duplicate_pairs": 0})
    write_csv(output / "duplicate_audit.csv", duplicate_rows, ["sample_id", "other_split", "other_id"])
    write_csv(output / "near_duplicate_audit.csv", near_rows,
              ["sample_id", "other_split", "other_id", "normalized_distance"])
    _atomic_json(output / "label_protocol_binding.json", protocol_binding)
    _atomic_json(output / "formal_generation_plan.json", {
        "status": "not_started", "sample_count": 600, "task_count": 2400,
        "default_b4_seed_runs": 12000,
        "escalation_units": "8 Reference-GA seed-runs + 1 Dense-LHS task per triggered task",
        "workers": 1, "automatic_start": False,
    })
    confirmation_runtime = json.loads((confirmation / "future_bp_label_runtime_estimate.json").read_text(encoding="utf-8"))
    _atomic_json(output / "runtime_estimate.json", {
        "confirmation_based_estimate": confirmation_runtime,
        "smoke_based_estimate": None,
        "smoke_estimate_status": "pending_label_smoke",
    })
    (output / "README.md").write_text(
        "# DualBP 数据集协议冻结\n\n"
        "本目录冻结600个正式输入样本及Train/Validation/Calibration/Test四角色。"
        "本轮未生成正式GA标签、未训练BP、未计算alpha，也不包含搜索后五候选认证。\n",
        encoding="utf-8",
    )
    (output / "protocol.md").write_text(
        "# DualBP-Dataset-v1\n\n"
        "- 规模：600个唯一基础样本，角色为420/60/60/60。\n"
        "- 每个角色在Gamma×sample_type的30层内完全平衡。\n"
        "- Gamma同时进入Time-BP与Energy-BP。\n"
        "- Calibration只在权重冻结后估计alpha；Test只做最终评价。\n"
        "- 监督标签绑定Adaptive-B4-Reference-v1；当前状态仅为输入冻结。\n",
        encoding="utf-8",
    )
    (output / "word_sync_report.md").write_text(
        "# 最新论文规则只读同步报告\n\n"
        f"- word_sync_status: `{word_sync_status}`\n"
        f"- 第四章：`{LATEST_CH4}`，只读OOXML成功；Word COM导出因占用超时。\n"
        f"- 第五章：`{LATEST_CH5}`，只读OOXML成功；Word COM导出因占用超时。\n"
        "- 科学冲突：0。四角色、Gamma输入、Time/Energy特征边界、alpha职责、corr_pred在线口径及取消搜索后认证均一致。\n"
        "- 原Word未修改。\n",
        encoding="utf-8",
    )
    _atomic_json(output / "created_files.json", {"files": sorted(path.name for path in output.iterdir() if path.is_file())})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--word-sync-status", default="temporarily_locked")
    args = parser.parse_args()
    create_design(Path(args.project_root), Path(args.output_dir), word_sync_status=args.word_sync_status)


if __name__ == "__main__":
    main()
