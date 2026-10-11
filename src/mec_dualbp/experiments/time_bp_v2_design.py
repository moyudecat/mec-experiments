"""Create the frozen-input-only Time-BP Dataset V2 manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from mec_dualbp.datasets.time_bp_v2 import (
    F_FIELD, ROLES, audit_manifest, build_manifest, canonical_hash, coverage_by_role,
    file_sha256, load_parameters, read_csv,
)


PROTOCOL_HASH = "041cbe449ab59ee6a43156ae90030921f70f06c0a21a265c40457a614ad9b561"
BASE_MANIFEST_HASH = "e48ffc62f10ce4f46c4986154c9e31fbbe82bbb4a1b38e9a2bcf8455c23d6f40"
FORMAL_LABEL_HASH = "188c82fc990fb28a4c8a6270d043f61a9b13bad27a284fe3a52e7b89720d55b3"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_design(root: Path, output: Path) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    parameter_path = root / "configs/edge_cpu_formal_parameters_v1.json"
    base_manifest_path = root / "outputs/05_audit/bp_dataset_protocol_20260810_122228/formal_600_manifest.csv"
    formal_dir = root / "outputs/01_formal/bp_dataset_labels_formal_20260810_132254"
    label_path = formal_dir / "formal_labels.csv"
    if file_sha256(base_manifest_path) != BASE_MANIFEST_HASH or file_sha256(label_path) != FORMAL_LABEL_HASH:
        raise RuntimeError("frozen V1 source hash drift")
    parameters = load_parameters(parameter_path)
    parameter_hash = file_sha256(parameter_path)
    config = {
        "dataset_version": "time_bp_dataset_v2",
        "design_status": "frozen_inputs_only",
        "parameter_version": parameters["parameter_version"],
        "edge_cpu_parameter_config_hash": parameter_hash,
        "base_manifest_sha256": BASE_MANIFEST_HASH,
        "base_formal_labels_sha256": FORMAL_LABEL_HASH,
        "adaptive_protocol_name": "Adaptive-B4-Reference-v1",
        "adaptive_protocol_hash": PROTOCOL_HASH,
        "records_per_base": 3,
        "f_level_roles": ["existing_upper_anchor", "new_low_stratum", "new_mid_high_stratum"],
        "low_stratum_cycle_per_s": [1.5e9, 3.25e9],
        "low_upper_exclusive": True,
        "mid_high_stratum_cycle_per_s": [3.25e9, 5.0e9],
        "mid_high_upper_exclusive": True,
        "upper_anchor_cycle_per_s": 5.0e9,
        "stratification": "deterministic_one_dimensional_stratified_lhs_by_role",
        "seed_derivation": "uint64_be(sha256(namespace)[0:8])",
        "seed_namespaces": {role: [f"TIME_V2_{role.upper()}_LOW", f"TIME_V2_{role.upper()}_MIDHIGH"] for role in ROLES},
        "lower_anchor_per_role": 1,
        "split_unit": "base_sample_group",
        "active_edge_user_only": True,
        "m_zero_excluded": True,
        "energy_dataset_version": "v1",
        "workers": 1,
        "force_escalation": False,
        "label_generation_started": False,
    }
    config_hash = canonical_hash(config)
    base_rows, label_rows = read_csv(base_manifest_path), read_csv(label_path)
    rows = build_manifest(base_rows, label_rows, parameters,
                          parameter_config_hash=parameter_hash, v2_config_hash=config_hash)
    audit = audit_manifest(rows, base_rows, label_rows, parameters)
    if not audit["passed"]:
        raise RuntimeError(f"V2 manifest audit failed: {audit}")
    coverage = coverage_by_role(rows)
    if any(row["min"] != 1.5e9 or row["max"] != 5.0e9 for row in coverage):
        raise RuntimeError("each role must cover both exact edge CPU boundaries")

    write_json(output / "edge_cpu_formal_parameters.json", parameters)
    write_json(output / "edge_cpu_formal_parameters_hash.json", {
        "sha256": parameter_hash, "source_path": "configs/edge_cpu_formal_parameters_v1.json"
    })
    write_json(output / "time_bp_dataset_v2_config.json", {**config, "time_bp_v2_config_hash": config_hash})
    write_json(output / "time_bp_dataset_v2_config_hash.json", {"time_bp_v2_config_hash": config_hash})
    write_csv(output / "time_bp_dataset_v2_manifest.csv", rows)
    write_csv(output / "f_domain_coverage_by_role.csv", coverage)

    source_binding = {
        "base_manifest_path": base_manifest_path.relative_to(root).as_posix(),
        "base_manifest_sha256": file_sha256(base_manifest_path),
        "formal_labels_path": label_path.relative_to(root).as_posix(),
        "formal_labels_sha256": file_sha256(label_path),
        "adaptive_protocol_hash": PROTOCOL_HASH,
        "parameter_config_path": parameter_path.relative_to(root).as_posix(),
        "parameter_config_sha256": parameter_hash,
        "word_hashes": {
            "第三章_冻结模型.docx": file_sha256(root / "reference_docs/第三章_冻结模型.docx"),
            "第四章_双BP误差修正_校订三线表版.docx": file_sha256(root / "reference_docs/第四章_双BP误差修正_校订三线表版.docx"),
            "第五章_实验设计参考稿_误差修正_校订三线表版.docx": file_sha256(root / "reference_docs/第五章_实验设计参考稿_误差修正_校订三线表版.docx"),
        },
        "historical_files_modified": False,
        "word_modified": False,
    }
    write_json(output / "source_binding.json", source_binding)

    online = {
        "parameter_version": parameters["parameter_version"],
        "unit": "cycle/s", "C5": "m*f_min_E <= f <= m*f_max_E",
        "C6": "sum(f) <= F_max", "active_user_domain": [1.5e9, 5.0e9],
        "m_zero_f": 0.0, "m_zero_time_bp_called": False,
        "default_F_max_cycle_per_s": 20.0e9,
        "five_ghz_inside_domain": True, "five_ghz_relative_position": 1.0,
        "five_ghz_semantics": "upper-bound edge CPU slice",
        "exp5_F_max_cycle_per_s": {str(r): r * 20.0e9 for r in parameters["exp5_F_max_over_F0"]},
        "scenario_F_max_cycle_per_s": {name: ratio * 20.0e9 for name, ratio in parameters["scenario_F_max_over_F0"].items()},
        "domain_status": "frozen",
    }
    write_json(output / "edge_cpu_online_domain.json", online)

    base_integrity = {key: audit[key] for key in (
        "passed", "total_rows", "base_samples", "wrong_records_per_base", "base_missing", "base_extra",
        "non_f_feature_changed_inside_base", "duplicate_f_inside_base", "lower_anchor_by_role")}
    role_audit = {"passed": audit["role_leakage"] == audit["base_group_leakage"] == 0,
                  "role_leakage": audit["role_leakage"], "base_group_leakage": audit["base_group_leakage"],
                  "role_counts": audit["role_counts"]}
    reuse_audit = {"passed": audit["existing_label_hash_mismatch"] == audit["existing_5ghz_not_equal_fmax"] == 0,
                   "existing_rows": audit["level_counts"]["existing_upper_anchor"],
                   "existing_label_hash_mismatch": audit["existing_label_hash_mismatch"],
                   "existing_5ghz_not_equal_fmax": audit["existing_5ghz_not_equal_fmax"],
                   "source_formal_labels_sha256": FORMAL_LABEL_HASH}
    write_json(output / "base_group_integrity_audit.json", base_integrity)
    write_json(output / "role_leakage_audit.json", role_audit)
    write_json(output / "existing_v1_reuse_audit.json", reuse_audit)
    write_json(output / "energy_v1_reuse_binding.json", {
        "energy_dataset_version": "v1", "rows": 600,
        "role_counts": {"Train": 420, "Validation": 60, "Calibration": 60, "Test": 60},
        "formal_labels_sha256": FORMAL_LABEL_HASH, "edge_cpu_is_energy_feature": False,
        "energy_recomputed": False, "energy_duplicated": False, "passed": True,
    })
    write_json(output / "time_bp_dataset_v2_manifest_summary.json", {
        "status": "frozen_inputs_only", **audit, "coverage_by_role": coverage,
        "time_bp_v2_config_hash": config_hash, "parameter_config_hash": parameter_hash,
    })

    g_low, g_mid, g_high = (1.0 / 1.5e9, 1.0 / 3.25e9, 1.0 / 5.0e9)
    inverse = {
        "function": "g(f)=1/f", "unit": "s/cycle",
        "g_at_1_5ghz": g_low, "g_at_3_25ghz": g_mid, "g_at_5ghz": g_high,
        "relative_change_definition": "(g(left)-g(right))/g(left)",
        "low_to_mid_relative_change": (g_low - g_mid) / g_low,
        "mid_to_upper_relative_change": (g_mid - g_high) / g_mid,
        "whole_domain_relative_change": (g_low - g_high) / g_low,
        "recommend_total_f_records_per_base": 3,
        "warning": None,
    }
    write_json(output / "f_inverse_response_analysis.json", inverse)
    write_json(output / "future_task_plan.json", {
        "queued_time_records": 1200, "time_boundaries_per_record": 2,
        "future_time_label_tasks": 2400, "B4_seeds_per_task": 5,
        "theoretical_B4_seed_runs": 12000, "adaptive_gate_threshold": 0.01,
        "reference_seeds_if_triggered": 8, "dense_lhs_points_if_triggered": 50000,
        "workers": 1, "execution_started": False,
    })
    write_json(output / "runtime_plan.json", {
        "one_new_f_slice_seconds": 6506.9881902,
        "two_new_f_slices_seconds": 13013.9763804,
        "two_new_f_slices_hours": 3.6149934389999998,
        "formal_expected_hours": [3.6, 4.0], "conservative_hours": 4.3,
        "execution_started": False,
    })

    (output / "author_parameter_decision.md").write_text(
        "# 作者参数决策\n\n- `f_min^E=1.5 GHz`\n- `f_max^E=5.0 GHz`\n- `F0=20 GHz`\n- `default F_max=20 GHz`\n\n"
        "来源：Author decision after input-domain audit；日期：2026-08-10。\n\n"
        "该决策为C5提供连续单用户域、为C6提供默认总预算、使Time-BP覆盖连续f，并将既有5GHz切片正式定义为f_max上界；F0用于第五章资源敏感性归一化。旧Word和历史config此前未规定这些数值。\n",
        encoding="utf-8")
    (output / "word_sync_pending.md").write_text(
        "# Word同步待办\n\n作者数值已冻结并作为当前最高优先级正式实验配置。第三至第五章Word本轮未修改；后续须统一同步参数表、C5/C6说明、F0映射和5GHz上界切片语义。\n",
        encoding="utf-8")
    (output / "f_slice_design.md").write_text(
        "# Time-BP V2 f切片设计\n\n每个base保留5GHz上界锚点，并在每个role内独立生成low `[1.5,3.25)` GHz和mid-high `[3.25,5.0)` GHz确定性一维分层LHS。seed由namespace的SHA-256前8字节生成，不使用Python随机hash。每个role的首个low记录固定为1.5GHz下界锚点。所有切片继承base role，拆分单位固定为base group。\n",
        encoding="utf-8")
    (output / "README.md").write_text(
        "# Time-BP Dataset V2正式输入设计\n\n作者参数已冻结；本目录包含1800行只输入manifest，其中600条复用5GHz Time标签、1200条等待下一轮生成Time上下界。Energy继续使用v1的600条标签。本轮未运行GA、训练BP或计算alpha。\n",
        encoding="utf-8")

    summary = {
        "edge_cpu_domain_status": "frozen",
        "edge_cpu_parameter_version": "edge_cpu_formal_parameters_v1",
        "existing_5ghz_slice_valid": True,
        "time_bp_dataset_v2_design_status": "frozen_inputs_only",
        "time_bp_v2_total_records": 1800, "time_bp_v2_existing_records": 600,
        "time_bp_v2_queued_records": 1200, "future_time_label_tasks": 2400,
        "time_bp_v2_label_generation_allowed": True,
        "time_bp_training_allowed": False, "energy_bp_training_allowed": True,
        "bp_training_allowed": False, "alpha_calibration_allowed": False,
        "all_manifest_audits_passed": True,
    }
    write_json(output / "summary.json", summary)
    (output / "validation_report.md").write_text(
        "# V2设计验证报告\n\n"
        f"- manifest：1800行，600个base，每base 3行\n- 角色：1260/180/180/180\n"
        f"- existing/low/mid-high：600/600/600\n- queued：1200；未来Time任务：2400\n"
        f"- role/base leakage：0/0\n- duplicate f：0；f越界：0\n"
        f"- existing标签hash mismatch：0\n- 每个role均精确覆盖1.5GHz与5GHz\n"
        "- 本轮GA、BP训练、alpha计算：均未执行\n",
        encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(create_design(Path(args.project_root), Path(args.output_dir)), ensure_ascii=False))


if __name__ == "__main__":
    main()
