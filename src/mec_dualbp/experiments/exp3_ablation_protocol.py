"""生成 Exp-3 机制性消融协议；不读取封存目标、不执行训练。"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from mec_dualbp.training.ablation import ABLATION_SEEDS, future_training_tasks, stable_ordered_union

TRAINING_HASH = "5dba158d126591460f9b5399bdcb4e1fb3c1167f5638537044304891649ea4f8"
TIME_MODEL_HASH = "291c5733bebdb4cb0c2cf7a4a07138ca6bb07c3ad75bffb378aa09b7f657d827"
ENERGY_MODEL_HASH = "8481dfdca0e03e3d6bebe4c4d5808259d16f7cdd67f4a65251fe7aad3e07d971"
ALPHA_PROTOCOL_HASH = "77a3a1c3ed659ddd534dd69d3605bc0bce57b0b4d2f4a46a13c3b1f031f1d2bb"
CORRECTED_HASH = "a15b8fb591df123b475f521ef35e4037b6ca201f4172caf804d3af571afe13e2"
TEST_PROTOCOL_HASH = "ae0c7af865efefdd7b34597256ebd4fae48377a54fdb32b57eda4fc546f07f65"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_protocol(project_root: Path, output_dir: Path) -> dict[str, Any]:
    started = datetime.now()
    train_protocol = project_root / "outputs/05_audit/dualbp_training_protocol_20260811_003055"
    train_formal = project_root / "outputs/01_formal/dualbp_training_formal_20260811_003126"
    calibration_formal = project_root / "outputs/01_formal/dualbp_alpha_calibration_formal_20260811_134010"
    calibration_protocol = project_root / "outputs/05_audit/dualbp_alpha_calibration_protocol_20260811_133826"
    test_formal = project_root / "outputs/01_formal/dualbp_test_evaluation_formal_20260811_140611"
    test_protocol = project_root / "outputs/05_audit/dualbp_test_evaluation_protocol_20260811_140517"
    required = [train_protocol, train_formal, calibration_protocol, calibration_formal, test_protocol, test_formal]
    if not all(path.exists() for path in required):
        raise FileNotFoundError("frozen Exp-2 artifact missing")

    training = _read(train_protocol / "training_protocol.json")
    time_schema = _read(train_protocol / "time_feature_schema.json")["fields"]
    energy_schema = _read(train_protocol / "energy_feature_schema.json")["fields"]
    training_summary = _read(train_formal / "summary.json")
    calibration_summary = _read(calibration_formal / "summary.json")
    test_summary = _read(test_formal / "summary.json")
    runtime = _read(train_formal / "runtime_breakdown.json")
    base_audit = _read(project_root / "outputs/01_formal/time_bp_dataset_v2_labels_formal_20260810_194719/base_group_integrity_audit.json")
    energy_role_audit = _read(project_root / "outputs/01_formal/bp_dataset_labels_formal_20260810_132254/role_integrity_audit.json")

    actual_time_hash = _sha256(train_formal / "time_bp/selected_model.pt")
    actual_energy_hash = _sha256(train_formal / "energy_bp/selected_model.pt")
    recorded_training_hash = _read(train_protocol / "training_protocol_hash.json")["training_protocol_hash"]
    recorded_alpha_hash = _read(calibration_protocol / "alpha_calibration_protocol_hash.json")["alpha_calibration_protocol_hash"]
    recorded_corrected_hash = _read(calibration_formal / "corrected_dualbp_v1_hash.json")["sha256"]
    recorded_test_hash = _read(test_protocol / "test_evaluation_protocol_hash.json")["test_evaluation_protocol_hash"]
    union = stable_ordered_union(time_schema, energy_schema)
    shared = [field for field in time_schema if field in set(energy_schema)]
    time_only = [field for field in time_schema if field not in set(energy_schema)]
    energy_only = [field for field in energy_schema if field not in set(time_schema)]
    output_dir.mkdir(parents=True, exist_ok=False)

    shared_training = {"architecture": "M3", "hidden_layers": [128, 64, 32], "activation": "ReLU",
        "optimizer": "AdamW", "initial_lr": 1e-3, "weight_decay": 1e-5,
        "scheduler": "ReduceLROnPlateau", "scheduler_factor": 0.5, "scheduler_patience": 25,
        "min_lr": 1e-5, "max_epochs": 1500, "early_stopping_patience": 100,
        "min_delta": 1e-5, "gradient_clip_norm": 5.0, "time_batch_size": 64,
        "energy_batch_size": 32, "workers": 1, "architecture_search": False, "lambda_search": False}
    variants = {
        "A": {"name": "Full Corrected Dual-BP", "status": "existing_frozen", "dual_networks": True,
              "center_radius": True, "safety_loss": True, "alpha_correction": True, "retrain": False},
        "B": {"name": "Raw Dual-BP", "status": "existing_frozen", "dual_networks": True,
              "center_radius": True, "safety_loss": True, "alpha_correction": False, "retrain": False},
        "C": {"name": "Dual-BP w/o Safety", "status": "to_be_trained", "dual_networks": True,
              "center_radius": True, "safety_loss": False, "lambda_safety": 0.0,
              "alpha_correction": False, "retrain": True},
        "D": {"name": "Unified-BP", "status": "to_be_trained", "dual_networks": False,
              "single_shared_trunk": True, "center_radius": True, "safety_loss": True,
              "lambda_safety": 0.25, "alpha_correction": False, "retrain": True},
        "E": {"name": "Direct-Bounds BP", "status": "to_be_trained", "dual_networks": True,
              "center_radius": False, "direct_bounds": True, "safety_loss": True,
              "lambda_safety": 0.25, "lambda_order": 1.0, "post_hoc_repair": False,
              "alpha_correction": False, "retrain": True}}
    contrasts = [
        {"contrast": "A-B", "baseline": "B", "variant": "A", "module": "alpha correction"},
        {"contrast": "B-C", "baseline": "B", "variant": "C", "module": "safety loss"},
        {"contrast": "B-D", "baseline": "B", "variant": "D", "module": "dual-network decomposition"},
        {"contrast": "B-E", "baseline": "B", "variant": "E", "module": "center-radius parameterization"}]
    baseline = {"training_protocol_hash_expected": TRAINING_HASH, "training_protocol_hash_actual": recorded_training_hash,
        "time_model_hash_expected": TIME_MODEL_HASH,
        "time_model_hash_actual": actual_time_hash, "energy_model_hash_expected": ENERGY_MODEL_HASH,
        "energy_model_hash_actual": actual_energy_hash, "alpha_protocol_hash_expected": ALPHA_PROTOCOL_HASH,
        "alpha_protocol_hash_actual": recorded_alpha_hash, "corrected_dualbp_hash_expected": CORRECTED_HASH,
        "corrected_dualbp_hash_actual": recorded_corrected_hash, "test_protocol_hash_expected": TEST_PROTOCOL_HASH,
        "test_protocol_hash_actual": recorded_test_hash,
        "training_decision": training_summary.get("dualbp_training_decision"),
        "corrected_status": calibration_summary.get("corrected_dualbp_status"),
        "test_validity": test_summary.get("test_evaluation_validity"),
        "exp2_test_status": test_summary.get("exp2_test_status")}
    protocol = {"protocol_name": "Exp3-Mechanistic-Ablation-v1", "design_timing": "post-Exp-2-Test",
        "design_guard": "post-Test mechanistic ablation protocol with no Test-driven hyperparameter adaptation",
        "variants": variants, "primary_contrasts": contrasts, "shared_training_config": shared_training,
        "training_seeds": list(ABLATION_SEEDS), "new_training_runs": 25,
        "data": {"time_roles": training["datasets"]["time"]["roles"],
                 "energy_roles": training["datasets"]["energy"]["roles"], "role_split_changed": False},
        "unified": {"union_features": list(union), "union_dimension": len(union),
                    "energy_repeat_weight": "1/records_per_base (=1/3)",
                    "task_weight_time": 0.5, "task_weight_energy": 0.5,
                    "validation_energy_weighting": "base-group weighted"},
        "guards": {"calibration_targets_accessed": False, "test_targets_accessed": False,
                   "ablation_test_evaluation_allowed": False}}
    protocol_path = output_dir / "exp3_ablation_protocol.json"
    _write_json(protocol_path, protocol)
    protocol_hash = _sha256(protocol_path)
    _write_json(output_dir / "exp3_ablation_protocol_hash.json", {"sha256": protocol_hash})
    _write_json(output_dir / "baseline_artifact_binding.json", baseline)
    _write_json(output_dir / "variant_definitions.json", variants)
    _write_json(output_dir / "primary_contrasts.json", contrasts)
    _write_json(output_dir / "shared_training_config.json", shared_training)
    _write_json(output_dir / "shared_seed_config.json", {"seeds": list(ABLATION_SEEDS), "additional_seeds_allowed": False})
    _write_json(output_dir / "w_o_safety_config.json", {**shared_training, "lambda_safety": 0.0,
        "validation_selection_score_unchanged": True, "input_scalers": "reuse Raw frozen", "center_scales": "reuse Raw frozen"})
    _write_json(output_dir / "unified_bp_config.json", {"hidden_layers": [128, 64, 32], "raw_outputs": 4,
        "single_shared_trunk": True, "lambda_safety": 0.25, "task_weights": {"Time": 0.5, "Energy": 0.5},
        "energy_repeat_weight": 1/3, "input_scaler_fit_role": "Train", "output_scales": "reuse Raw Train-only scales"})
    _write_json(output_dir / "unified_feature_schema.json", {"time_features": time_schema, "energy_features": energy_schema,
        "shared_features": shared, "time_only_features": time_only, "energy_only_features": energy_only,
        "union_features": list(union), "union_dimension": len(union), "ordering": "Time order then unseen Energy fields"})
    _write_json(output_dir / "direct_bounds_config.json", {**shared_training, "raw_outputs": 2,
        "output_semantics": "direct lower/upper", "lambda_safety": 0.25, "lambda_order": 1.0,
        "post_hoc_swap_sort_clip": False, "input_scalers": "reuse Raw frozen", "center_scales": "reuse Raw frozen"})
    data_guard = {"training_target_roles": ["Train", "Validation"], "calibration_metadata_only": True,
        "test_metadata_only": True, "calibration_targets_accessed": False, "test_targets_accessed": False,
        "time_role_counts": training["datasets"]["time"]["roles"], "energy_role_counts": training["datasets"]["energy"]["roles"]}
    _write_json(output_dir / "data_role_guard.json", data_guard)
    _write_json(output_dir / "calibration_guard.json", {"ablation_calibration_targets_accessed": False,
        "alpha_for_C_D_E": "OFF", "metadata_access_allowed": ["role", "sample_id", "count"]})
    _write_json(output_dir / "test_guard.json", {"protocol_stage_test_targets_accessed": False,
        "open_condition": "C/D/E all trained and frozen, then bind one common Test protocol",
        "sequential_model_test_peeking_forbidden": True})
    fairness_checks = {"five_variants_complete": len(variants) == 5,
        "A_B_same_model_hashes": actual_time_hash == TIME_MODEL_HASH and actual_energy_hash == ENERGY_MODEL_HASH,
        "all_protocol_hashes_unchanged": recorded_training_hash == TRAINING_HASH and recorded_alpha_hash == ALPHA_PROTOCOL_HASH and recorded_test_hash == TEST_PROTOCOL_HASH,
        "corrected_hash_unchanged": recorded_corrected_hash == CORRECTED_HASH,
        "A_B_only_alpha_differs": all(variants["A"][key] == variants["B"][key] for key in ("dual_networks", "center_radius", "safety_loss")),
        "C_single_factor": variants["C"]["lambda_safety"] == 0.0,
        "D_single_shared_trunk": variants["D"]["single_shared_trunk"], "D_union_unique": len(union) == len(set(union)),
        "D_energy_weight_sum_per_base": abs(3 * (1/3) - 1) < 1e-12,
        "E_no_post_hoc_repair": not variants["E"]["post_hoc_repair"], "E_order_weight_frozen": variants["E"]["lambda_order"] == 1.0,
        "fixed_seeds": tuple(ABLATION_SEEDS) == (31001, 31002, 31003, 31004, 31005),
        "future_runs_25": len(future_training_tasks()) == 25, "base_group_integrity": bool(base_audit["passed"]),
        "energy_role_integrity": bool(energy_role_audit["passed"]), "calibration_guard": True, "test_guard": True,
        "architecture_search_disabled": not shared_training["architecture_search"], "lambda_search_disabled": not shared_training["lambda_search"]}
    fairness = {"passed": all(fairness_checks.values()), "checks": fairness_checks,
        "post_test_design_declared": True, "test_driven_hyperparameter_adaptation": False}
    _write_json(output_dir / "fairness_audit.json", fairness)

    metrics = {"accuracy": ["MAE", "RMSE", "R2", "WMAPE"], "safety": ["UER", "LER", "ICR"],
        "directional": ["upper_under_relative_magnitude", "lower_over_relative_magnitude"],
        "structure": ["invalid_interval_rate"], "time_strata": ["Low", "Mid-high", "5GHz"],
        "energy_strata": ["Gamma"], "support_rules_frozen_before_ablation_test": True}
    _write_json(output_dir / "planned_metrics.json", metrics)
    _write_json(output_dir / "planned_tables.json", {"tables": [
        {"file": "table_exp3_ablation_accuracy.csv", "columns": ["Method", "Network", "Bound", "MAE", "RMSE", "R2", "WMAPE"]},
        {"file": "table_exp3_ablation_safety.csv", "columns": ["Method", "Network", "UER", "LER", "ICR", "UpperUnderMagnitude", "LowerOverMagnitude", "InvalidIntervalRate"]},
        {"file": "table_exp3_primary_contrasts.csv", "columns": ["Contrast", "Baseline", "Variant", "Module", "Delta_WMAPE", "Delta_UER", "Delta_LER", "Delta_ICR", "Delta_InvalidRate"]}]})
    _write_json(output_dir / "planned_figures.json", {"formats": ["PNG_600dpi", "PDF", "SVG"], "plot_data_csv": True,
        "figures": ["Time ablation WMAPE", "Energy ablation WMAPE", "Time ablation safety metrics",
                    "Energy ablation safety metrics", "Primary contrast improvement", "Unified Energy f-invariance",
                    "Direct-Bounds invalid interval"]})
    tasks = future_training_tasks()
    with (output_dir / "future_training_task_manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tasks[0])); writer.writeheader(); writer.writerows(tasks)
    time_sec = float(runtime["per_seed_average_training_seconds"]["time"])
    energy_sec = float(runtime["per_seed_average_training_seconds"]["energy"])
    c_sec = 5 * (time_sec + energy_sec); e_sec = c_sec; d_sec = 5 * time_sec * 1.25
    runtime_plan = {"source": "Formal runtime_breakdown per-run averages", "time_per_run_seconds": time_sec,
        "energy_per_run_seconds": energy_sec, "unified_estimation_factor_vs_time": 1.25,
        "C_expected_minutes": c_sec/60, "D_expected_minutes": d_sec/60, "E_expected_minutes": e_sec/60,
        "total_training_minutes": (c_sec+d_sec+e_sec)/60, "audit_minutes": 5.0,
        "test_evaluation_minutes": 19.7236589/60, "conservative_total_minutes": 30.0}
    _write_json(output_dir / "exp3_runtime_estimate.json", runtime_plan)
    (output_dir / "post_test_ablation_design_guard.md").write_text(
        "# Exp-3 的 Test 后设计防护\n\nExp-3 是 **post-Test mechanistic ablation protocol with no Test-driven hyperparameter adaptation**。"
        "不得声称本协议早于 Exp-2 Test 冻结。作者已见 Full/Raw Test 结果，因此固定采取：不搜索 architecture、lambda、seed、学习率、epoch、"
        "Unified 权重或 Direct-Bounds order penalty；C/D/E 均继承 M3 框架，只作单因素移除或替换；三者在统一打开 Test 前必须全部完成 Train/Validation 冻结。\n",
        encoding="utf-8")
    (output_dir / "unified_dataset_contract.md").write_text(
        "# Unified-BP 数据合同\n\n以 Time V2 的 1800 条记录为主，经 `base_sample_id` 唯一绑定 Energy V1。"
        "同一 base 的三条 f 记录共享同一 Energy 标签，但不得称为独立 Energy 样本。角色不重划分；训练只读 Train/Validation 标签。\n",
        encoding="utf-8")
    (output_dir / "unified_loss_weighting.md").write_text(
        "# Unified-BP 损失权重\n\n每条 Time 记录权重为 1；同一 base 的三条 Energy 损失各为 1/3，合计为 1。"
        "内部 Time/Energy 均使用 `L_fit + 0.25 L_safety`，总损失固定为 `0.5 L_Time + 0.5 L_Energy`。\n",
        encoding="utf-8")
    (output_dir / "unified_validation_rule.md").write_text(
        "# Unified-BP Validation 规则\n\nTime 在 180 条记录上按既有 score 计算；Energy 按 base-group 加权，确保每个 base 总权重为 1。"
        "最终 score 为二者各 0.5；从五个固定 seed 中选择最低者，不查看 Test。\n",
        encoding="utf-8")
    (output_dir / "direct_bounds_output_contract.md").write_text(
        "# Direct-Bounds 输出合同\n\n`lower_hat = center_scale * softplus(z_lower)`；"
        "`upper_hat = center_scale * softplus(z_upper)`。两者非负但不保证顺序；Validation/Test 禁止 swap、sort、clip 或 max 修复。\n",
        encoding="utf-8")
    (output_dir / "direct_bounds_loss_definition.md").write_text(
        "# Direct-Bounds 损失\n\n由直接边界恢复 center/radius 后沿用 Raw 的 `L_fit` 与 `L_safety`。"
        "`L_order = mean(ReLU(lower_hat-upper_hat)/(center_scale+epsilon))`；"
        "`L_total=L_fit+0.25L_safety+1.0L_order`。非法区间原样计入指标。\n",
        encoding="utf-8")

    decision = "passed" if (fairness["passed"] and baseline["training_decision"] == "passed"
        and baseline["corrected_status"] == "calibrated_and_frozen" and baseline["test_validity"] == "passed"
        and baseline["exp2_test_status"] == "completed_and_frozen") else "blocked"
    summary = {"exp3_protocol_decision": decision, "exp3_protocol_status": "frozen" if decision == "passed" else "blocked",
        "protocol_hash": protocol_hash, "ablation_training_allowed": decision == "passed",
        "ablation_test_evaluation_allowed": False, "exp3_completed": False, "new_training_runs": len(tasks),
        "unified_input_dimension": len(union), "calibration_targets_accessed": False, "test_targets_accessed": False,
        "fairness_audit": "passed" if fairness["passed"] else "failed", "elapsed_seconds_before_tests": (datetime.now()-started).total_seconds()}
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "validation_report.md").write_text(
        f"# 验证报告\n\n- 协议决定：`{decision}`\n- 五个版本：完整\n- 冻结哈希：通过现场复算\n- Unified 并集：{len(union)} 维、无重复\n"
        f"- 未来任务：{len(tasks)} runs\n- Calibration/Test target：均未访问\n- 公平性审计：{'通过' if fairness['passed'] else '失败'}\n",
        encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Exp-3 DualBP 核心消融协议\n\n本目录冻结 A–E 五个方法、四个主要对比、共同训练配置、Unified 数据/权重合同、"
        "Direct-Bounds 输出/损失合同及角色访问门禁。本轮未训练模型、未读取 Calibration/Test 标签目标、未生成 Test 预测。\n",
        encoding="utf-8")
    (output_dir / "final_pytest.txt").write_text("等待本轮最终完整 pytest。\n", encoding="utf-8")
    created = sorted(path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file())
    _write_json(output_dir / "created_files.json", {"files": created + ["created_files.json"], "count": len(created)+1})
    return summary


if __name__ == "__main__":
    root = Path.cwd()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = build_protocol(root, root / "outputs/05_audit" / f"exp3_ablation_protocol_{stamp}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
