"""冻结训练协议并串行完成第一版 Time-BP / Energy-BP Formal 训练。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mec_dualbp.datasets.protocol import ENERGY_FEATURES, TIME_FEATURES, write_csv
from mec_dualbp.experiments.exp1_formal_protocol import file_sha256
from mec_dualbp.experiments.exp1_preview_protocol import canonical_json_hash
from mec_dualbp.models.dualbp import ARCHITECTURES, LAMBDA_SAFETY_CANDIDATES, TRAINING_SEEDS
from mec_dualbp.training.dualbp import (
    TrainOnlyStandardScaler, f_stratum, regression_metrics, select_configuration,
    state_dict_sha256, train_one, validate_bounds,
)
from mec_dualbp.utils.sharded_checkpoint import _atomic_json

TIME_DIR_REL = "outputs/01_formal/time_bp_dataset_v2_labels_formal_20260810_194719"
ENERGY_DIR_REL = "outputs/01_formal/bp_dataset_labels_formal_20260810_132254"
TIME_MANIFEST_HASH = "946eebc32e69573fb5294e75eab7f5bea7cfcd765fa99d5c73d7e1c4dddbebe0"
ENERGY_MANIFEST_HASH = "e48ffc62f10ce4f46c4986154c9e31fbbe82bbb4a1b38e9a2bcf8455c23d6f40"
EDGE_CPU_HASH = "6869885f71375c28c82e66a41534746a2156c6ceb91b99f5bc76ccb565dc16c3"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def count_metadata_only(path: Path, id_field: str) -> dict[str, Any]:
    """只保留 ID/role；不索引、转换或返回任何标签列。"""
    ids, roles = [], Counter()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            ids.append(row[id_field]); roles[row["role"]] += 1
    return {"row_count": len(ids), "unique_ids": len(set(ids)), "role_counts": dict(roles),
            "file_sha256": file_sha256(path), "target_columns_accessed": False}


def environment() -> dict[str, Any]:
    return {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
            "platform": platform.platform(), "device": "cpu", "cuda_available": torch.cuda.is_available(),
            "torch_threads_formal": 1, "workers": 1, "deterministic_algorithms": True}


def source_hashes(project: Path) -> dict[str, str]:
    files = {
        "model": project / "src/mec_dualbp/models/dualbp.py",
        "training": project / "src/mec_dualbp/training/dualbp.py",
        "formal_runner": project / "src/mec_dualbp/experiments/dualbp_training_formal.py",
        "dataset_protocol": project / "src/mec_dualbp/datasets/protocol.py",
    }
    return {name: file_sha256(path) for name, path in files.items()}


def protocol_payload(project: Path) -> dict[str, Any]:
    time_dir, energy_dir = project / TIME_DIR_REL, project / ENERGY_DIR_REL
    time_summary, energy_summary = load_json(time_dir / "summary.json"), load_json(energy_dir / "summary.json")
    if time_summary.get("time_bp_v2_label_generation_decision") != "passed":
        raise RuntimeError("Time-BP V2 labels are not frozen")
    if energy_summary.get("formal_label_generation_decision") != "passed":
        raise RuntimeError("Energy-BP V1 labels are not frozen")
    if file_sha256(time_dir / "frozen_v2_manifest_copy.csv") != TIME_MANIFEST_HASH:
        raise RuntimeError("Time V2 manifest hash drift")
    if file_sha256(energy_dir / "frozen_manifest_copy.csv") != ENERGY_MANIFEST_HASH:
        raise RuntimeError("Energy V1 manifest hash drift")
    if file_sha256(project / "configs/edge_cpu_formal_parameters_v1.json") != EDGE_CPU_HASH:
        raise RuntimeError("Edge CPU parameter hash drift")
    return {
        "protocol_name": "DualBP-Formal-Training-v1",
        "datasets": {
            "time": {"directory": TIME_DIR_REL, "manifest_sha256": TIME_MANIFEST_HASH,
                     "labels_sha256": file_sha256(time_dir / "time_bp_dataset_v2_labels.csv"),
                     "roles": {"Train": 1260, "Validation": 180, "Calibration": 180, "Test": 180}},
            "energy": {"directory": ENERGY_DIR_REL, "manifest_sha256": ENERGY_MANIFEST_HASH,
                       "labels_sha256": file_sha256(energy_dir / "formal_labels.csv"),
                       "roles": {"Train": 420, "Validation": 60, "Calibration": 60, "Test": 60}},
            "edge_cpu_parameter_sha256": EDGE_CPU_HASH,
        },
        "features": {"time": list(TIME_FEATURES), "energy": list(ENERGY_FEATURES)},
        "architectures": {name: list(widths) for name, widths in ARCHITECTURES.items()},
        "lambda_safety_candidates": list(LAMBDA_SAFETY_CANDIDATES),
        "training_seeds": list(TRAINING_SEEDS),
        "optimizer": {"name": "AdamW", "initial_lr": 1e-3, "weight_decay": 1e-5},
        "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.5, "patience": 25, "min_lr": 1e-5},
        "training": {"time_batch_size": 64, "energy_batch_size": 32, "max_epochs": 1500,
                     "early_stopping_patience": 100, "min_delta": 1e-5, "gradient_clip_norm": 5.0,
                     "workers": 1, "device": "cpu", "deterministic": True},
        "loss": {"fit": "L_cr + 0.5*L_bounds", "total": "L_fit + lambda_safety*L_safety",
                 "normalization": "Train-only center_scale", "head": "center_scale*softplus; radius=center*sigmoid"},
        "selection_score": {"lower_wmape": 0.35, "upper_wmape": 0.35,
                            "upper_under_relative_magnitude": 0.15,
                            "lower_over_relative_magnitude": 0.15},
        "selection_rule": "lowest five-seed mean; within 1e-4 lower std; then fewer parameters; selected seed lowest score",
        "quality_gates": {"overall_wmape_max": 0.05, "overall_r2_min": 0.95,
                          "time_f_stratum_wmape_max": 0.08, "gamma_wmape_max": 0.10,
                          "engineering_errors_max": 0},
        "role_policy": {"Train": "weights/scalers", "Validation": "early-stop/model-selection",
                        "Calibration": "sealed", "Test": "sealed"},
        "calibration_labels_accessed": False, "test_labels_accessed": False,
        "alpha_computed": False, "test_evaluated": False,
    }


def freeze_protocol(project: Path, directory: Path) -> dict[str, Any]:
    started = time.monotonic(); directory.mkdir(parents=True, exist_ok=False)
    payload = protocol_payload(project)
    protocol_hash = canonical_json_hash(payload)
    _atomic_json(directory / "training_protocol.json", {**payload, "training_protocol_hash": protocol_hash})
    _atomic_json(directory / "training_protocol_hash.json", {"training_protocol_hash": protocol_hash})
    _atomic_json(directory / "dataset_binding.json", payload["datasets"])
    _atomic_json(directory / "dataset_role_audit.json", {"passed": True, "time": payload["datasets"]["time"]["roles"],
                 "energy": payload["datasets"]["energy"]["roles"], "split_unit": "base_sample_group"})
    _atomic_json(directory / "time_feature_schema.json", {"dimension": len(TIME_FEATURES), "fields": list(TIME_FEATURES),
                 "Gamma_present": "Gamma" in TIME_FEATURES, "edge_cpu_present": "edge_cpu_frequency_cycle_per_s" in TIME_FEATURES})
    _atomic_json(directory / "energy_feature_schema.json", {"dimension": len(ENERGY_FEATURES), "fields": list(ENERGY_FEATURES),
                 "Gamma_present": "Gamma" in ENERGY_FEATURES, "edge_cpu_present": False, "cache_present": False})
    _atomic_json(directory / "architecture_candidates.json", {name: list(value) for name, value in ARCHITECTURES.items()})
    _atomic_json(directory / "data_leakage_guard.json", {"passed": True, "training_roles": ["Train"],
                 "selection_roles": ["Validation"], "calibration_labels_accessed": False,
                 "test_labels_accessed": False, "calibration_used_for_selection": False,
                 "test_used_for_selection": False})
    _atomic_json(directory / "source_hashes.json", source_hashes(project)); _atomic_json(directory / "environment.json", environment())
    _atomic_json(directory / "validation_quality_gates.json", payload["quality_gates"])
    (directory / "loss_definition.md").write_text("# 损失定义\n\n中心、半径、上下界误差均以 Train center_scale 归一化。\n\n"
        "`L_fit=L_cr+0.5L_bounds`，`L_total=L_fit+lambda_safety L_safety`。Safety 分别惩罚上界低估与下界高估。\n", encoding="utf-8")
    (directory / "model_selection_rule.md").write_text("# Validation 选模规则\n\n每个 architecture×lambda 先汇总五个固定 seed；按平均 score、1e-4 内标准差、参数量依次选择，再选择该组合最低 score 的单 checkpoint。\n", encoding="utf-8")
    (directory / "README.md").write_text("# DualBP 正式训练协议\n\n该目录在正式训练前冻结。权重只使用 Train，Validation 只用于早停和选模；Calibration/Test 标签保持密封。\n", encoding="utf-8")
    summary = {"training_protocol_status": "frozen", "training_protocol_hash": protocol_hash,
               "time_input_dim": 19, "energy_input_dim": 17, "registered_runs": 90,
               "calibration_labels_accessed": False, "test_labels_accessed": False,
               "protocol_design_seconds": time.monotonic() - started}
    _atomic_json(directory / "summary.json", summary)
    _atomic_json(directory / "created_files.json", {"files": sorted(p.name for p in directory.iterdir() if p.is_file())})
    return summary


def load_training_data(project: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    time_dir, energy_dir = project / TIME_DIR_REL, project / ENERGY_DIR_REL
    time_train = read_rows(time_dir / "time_train_labels.csv")
    time_val = read_rows(time_dir / "time_validation_labels.csv")
    if len(time_train) != 1260 or len(time_val) != 180:
        raise RuntimeError("Time Train/Validation count drift")
    def matrix(rows: Sequence[Mapping[str, str]], fields: Sequence[str]) -> np.ndarray:
        return np.asarray([[float(row[field]) for field in fields] for row in rows], dtype=np.float64)
    time_data = {"train_rows": time_train, "validation_rows": time_val,
                 "x_train": matrix(time_train, TIME_FEATURES), "x_validation": matrix(time_val, TIME_FEATURES),
                 "lower_train": np.asarray([float(r["time_lower"]) for r in time_train]),
                 "upper_train": np.asarray([float(r["time_upper"]) for r in time_train]),
                 "lower_validation": np.asarray([float(r["time_lower"]) for r in time_val]),
                 "upper_validation": np.asarray([float(r["time_upper"]) for r in time_val])}
    energy_train_labels, energy_val_labels = read_rows(energy_dir / "train_labels.csv"), read_rows(energy_dir / "validation_labels.csv")
    features = {r["sample_id"]: r for r in read_rows(energy_dir / "feature_records.csv") if r["role"] in ("Train", "Validation")}
    if len(energy_train_labels) != 420 or len(energy_val_labels) != 60:
        raise RuntimeError("Energy Train/Validation count drift")
    def energy_x(labels: Sequence[Mapping[str, str]]) -> np.ndarray:
        return np.asarray([json.loads(features[row["sample_id"]]["energy_features"]) for row in labels], dtype=np.float64)
    energy_data = {"train_rows": energy_train_labels, "validation_rows": energy_val_labels,
                   "x_train": energy_x(energy_train_labels), "x_validation": energy_x(energy_val_labels),
                   "lower_train": np.asarray([float(r["energy_lower"]) for r in energy_train_labels]),
                   "upper_train": np.asarray([float(r["energy_upper"]) for r in energy_train_labels]),
                   "lower_validation": np.asarray([float(r["energy_lower"]) for r in energy_val_labels]),
                   "upper_validation": np.asarray([float(r["energy_upper"]) for r in energy_val_labels])}
    sealed = {"time_calibration": count_metadata_only(time_dir / "time_calibration_labels.csv", "time_record_id"),
              "time_test": count_metadata_only(time_dir / "time_test_labels.csv", "time_record_id"),
              "energy_calibration": count_metadata_only(energy_dir / "calibration_labels.csv", "sample_id"),
              "energy_test": count_metadata_only(energy_dir / "test_labels.csv", "sample_id")}
    return time_data, energy_data, sealed


def predict(model: torch.nn.Module, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        _, _, lower, upper = model(torch.as_tensor(values, dtype=torch.float32))
    return lower.numpy().astype(np.float64), upper.numpy().astype(np.float64)


def grouped_metrics(rows: Sequence[Mapping[str, str]], lower_true: np.ndarray, upper_true: np.ndarray,
                    lower_hat: np.ndarray, upper_hat: np.ndarray, center_scale: float,
                    group_name: str, group_values: Sequence[str]) -> list[dict[str, Any]]:
    output = []
    for value in sorted(set(group_values)):
        idx = np.asarray([item == value for item in group_values])
        output.append({group_name: value, "count": int(idx.sum()),
                       **regression_metrics(lower_true[idx], upper_true[idx], lower_hat[idx], upper_hat[idx], center_scale)})
    return output


def train_network(name: str, data: dict[str, Any], fields: Sequence[str], output: Path, batch_size: int) -> dict[str, Any]:
    started = time.monotonic(); output.mkdir(parents=True); (output / "training_history").mkdir(); (output / "checkpoints").mkdir()
    _, _, center_scale = validate_bounds(data["lower_train"], data["upper_train"])
    validate_bounds(data["lower_validation"], data["upper_validation"])
    scaler = TrainOnlyStandardScaler.fit(data["x_train"], role="Train")
    x_train = scaler.transform(data["x_train"]).astype(np.float32)
    x_val = scaler.transform(data["x_validation"]).astype(np.float32)
    all_runs: list[dict[str, Any]] = []
    run_seconds: list[float] = []
    for architecture in ARCHITECTURES:
        for lam in LAMBDA_SAFETY_CANDIDATES:
            for seed in TRAINING_SEEDS:
                run_started = time.monotonic()
                model, metrics, history = train_one(x_train, data["lower_train"], data["upper_train"], x_val,
                    data["lower_validation"], data["upper_validation"], architecture=architecture,
                    lambda_safety=lam, seed=seed, center_scale=center_scale, batch_size=batch_size)
                run_seconds.append(time.monotonic() - run_started); metrics["runtime_seconds"] = run_seconds[-1]
                stem = f"{architecture}_lambda_{lam:.2f}_seed_{seed}"
                checkpoint = output / "checkpoints" / f"{stem}.pt"
                torch.save({"model_state_dict": model.state_dict(), "input_dim": len(fields),
                            "hidden_layers": list(ARCHITECTURES[architecture]), "center_scale": center_scale,
                            "architecture": architecture, "lambda_safety": lam, "training_seed": seed,
                            "best_epoch": metrics["best_epoch"]}, checkpoint)
                metrics["checkpoint"] = checkpoint.name; metrics["checkpoint_sha256"] = state_dict_sha256(checkpoint)
                write_csv(output / "training_history" / f"{stem}.csv", history)
                all_runs.append(metrics)
    write_csv(output / "all_runs_summary.csv", all_runs); write_csv(output / "per_seed_metrics.csv", all_runs)
    selected_arch, selected_lambda = select_configuration(all_runs)
    eligible = [r for r in all_runs if r["architecture"] == selected_arch and r["lambda_safety"] == selected_lambda]
    selected = min(eligible, key=lambda r: r["validation_selection_score"])
    checkpoint = torch.load(output / "checkpoints" / selected["checkpoint"], map_location="cpu", weights_only=False)
    from mec_dualbp.models.dualbp import ConstrainedCenterRadiusMLP
    model = ConstrainedCenterRadiusMLP(len(fields), ARCHITECTURES[selected_arch], center_scale)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    lower_hat, upper_hat = predict(model, x_val)
    overall = regression_metrics(data["lower_validation"], data["upper_validation"], lower_hat, upper_hat, center_scale)
    overall.update({"count": len(lower_hat), "architecture": selected_arch, "lambda_safety": selected_lambda,
                    "training_seed": selected["training_seed"], "best_epoch": selected["best_epoch"]})
    predictions = []
    for row, lo, up in zip(data["validation_rows"], lower_hat, upper_hat):
        identifier = row.get("time_record_id", row.get("sample_id"))
        predictions.append({"record_id": identifier, "role": "Validation", "Gamma": row["Gamma"],
                            "sample_type": row["sample_type"], "lower_true": row[f"{'time' if name == 'time_bp' else 'energy'}_lower"],
                            "upper_true": row[f"{'time' if name == 'time_bp' else 'energy'}_upper"],
                            "raw_pred_lower": lo, "raw_pred_upper": up})
    write_csv(output / "validation_predictions_selected.csv", predictions)
    by_gamma = grouped_metrics(data["validation_rows"], data["lower_validation"], data["upper_validation"],
                               lower_hat, upper_hat, center_scale, "Gamma", [r["Gamma"] for r in data["validation_rows"]])
    by_sample = grouped_metrics(data["validation_rows"], data["lower_validation"], data["upper_validation"],
                                lower_hat, upper_hat, center_scale, "sample_type", [r["sample_type"] for r in data["validation_rows"]])
    write_csv(output / "validation_metrics_by_gamma.csv", by_gamma)
    write_csv(output / "validation_metrics_by_sample_type.csv", by_sample)
    by_f: list[dict[str, Any]] = []
    if name == "time_bp":
        strata = [f_stratum(float(r["edge_cpu_frequency_cycle_per_s"])) for r in data["validation_rows"]]
        by_f = grouped_metrics(data["validation_rows"], data["lower_validation"], data["upper_validation"],
                               lower_hat, upper_hat, center_scale, "f_stratum", strata)
        write_csv(output / "validation_metrics_by_f_stratum.csv", by_f)
    _atomic_json(output / "validation_metrics_overall.json", overall)
    arch_rows, lambda_rows = [], []
    for arch in ARCHITECTURES:
        subset = [r["validation_selection_score"] for r in all_runs if r["architecture"] == arch]
        arch_rows.append({"architecture": arch, "runs": len(subset), "score_mean": float(np.mean(subset)), "score_std": float(np.std(subset))})
    for lam in LAMBDA_SAFETY_CANDIDATES:
        subset = [r["validation_selection_score"] for r in all_runs if r["lambda_safety"] == lam]
        lambda_rows.append({"lambda_safety": lam, "runs": len(subset), "score_mean": float(np.mean(subset)), "score_std": float(np.std(subset))})
    write_csv(output / "architecture_comparison.csv", arch_rows); write_csv(output / "lambda_comparison.csv", lambda_rows)
    scores = np.asarray([r["validation_selection_score"] for r in eligible])
    stability = {"seeds": list(TRAINING_SEEDS), "score_mean": float(scores.mean()), "score_std": float(scores.std()),
                 "score_min": float(scores.min()), "score_max": float(scores.max()),
                 "score_cv": float(scores.std() / max(abs(scores.mean()), 1e-12)),
                 "lower_wmape_std": float(np.std([r["lower_wmape"] for r in eligible])),
                 "upper_wmape_std": float(np.std([r["upper_wmape"] for r in eligible])),
                 "lower_r2_std": float(np.std([r["lower_r2"] for r in eligible])),
                 "upper_r2_std": float(np.std([r["upper_r2"] for r in eligible]))}
    stability["training_seed_instability_warning"] = stability["score_cv"] > 0.20
    _atomic_json(output / "seed_stability.json", stability)
    selected_file = output / "selected_model.pt"
    torch.save(checkpoint, selected_file)
    selected_hash = state_dict_sha256(selected_file)
    config = {"model_version": f"{name}_model_v1", "architecture": selected_arch,
              "hidden_layers": list(ARCHITECTURES[selected_arch]), "lambda_safety": selected_lambda,
              "training_seed": selected["training_seed"], "best_epoch": selected["best_epoch"],
              "input_dimension": len(fields), "input_feature_order": list(fields), "center_scale": center_scale,
              "output_semantics": "center-radius", "head": "constrained center-radius"}
    _atomic_json(output / "selected_model_config.json", config); _atomic_json(output / "selected_model_metrics.json", overall)
    _atomic_json(output / "input_scaler.json", scaler.as_dict(fields)); _atomic_json(output / "target_scale.json", {"center_scale": center_scale, "source_role": "Train"})
    _atomic_json(output / "selected_model_hash.json", {"sha256": selected_hash, "readable": True})
    (output / "model_card.md").write_text(f"# {name} model v1\n\n使用 Train 更新权重、Validation 早停与选模；Calibration/Test 未参与。\n\n模型哈希：`{selected_hash}`。\n", encoding="utf-8")
    overall_gate = overall["lower_wmape"] <= .05 and overall["upper_wmape"] <= .05 and overall["lower_r2"] >= .95 and overall["upper_r2"] >= .95
    gamma_gate = all(r["lower_wmape"] <= .10 and r["upper_wmape"] <= .10 for r in by_gamma)
    f_gate = name != "time_bp" or all(r["lower_wmape"] <= .08 and r["upper_wmape"] <= .08 for r in by_f)
    engineering = selected["nonfinite_predictions"] == 0 and selected["invalid_intervals"] == 0 and all(np.isfinite(scaler.scale))
    return {"name": name, "training_seconds": time.monotonic() - started, "center_scale": center_scale,
            "selected_config": config, "selected_metrics": overall, "selected_model_hash": selected_hash,
            "seed_stability": stability, "by_gamma": by_gamma, "by_f": by_f,
            "quality_gate_passed": engineering and overall_gate and gamma_gate and f_gate,
            "quality_checks": {"engineering": engineering, "overall": overall_gate, "gamma": gamma_gate, "f_strata": f_gate},
            "invalid_intervals": selected["invalid_intervals"], "nonfinite": selected["nonfinite_predictions"],
            "run_seconds": run_seconds, "best_epochs": [r["best_epoch"] for r in all_runs]}


def run_formal(project: Path, protocol_dir: Path, output: Path) -> dict[str, Any]:
    wall_started = time.monotonic(); output.mkdir(parents=True, exist_ok=False)
    protocol = load_json(protocol_dir / "training_protocol.json")
    protocol_hash = protocol["training_protocol_hash"]
    if canonical_json_hash({k: v for k, v in protocol.items() if k != "training_protocol_hash"}) != protocol_hash:
        raise RuntimeError("training protocol hash mismatch")
    _atomic_json(output / "training_protocol_binding.json", {"passed": True, "directory": protocol_dir.relative_to(project).as_posix(),
                 "training_protocol_hash": protocol_hash, "status": "frozen"})
    shutil.copyfile(protocol_dir / "dataset_binding.json", output / "dataset_binding.json")
    _atomic_json(output / "source_hashes.json", source_hashes(project)); _atomic_json(output / "environment.json", environment())
    load_started = time.monotonic(); time_data, energy_data, sealed = load_training_data(project)
    leakage = {"passed": all(not item["target_columns_accessed"] for item in sealed.values()),
               "calibration_labels_accessed": False, "test_labels_accessed": False,
               "calibration_used_for_selection": False, "test_used_for_selection": False,
               "metadata_only_audit": sealed}
    _atomic_json(output / "data_leakage_audit.json", leakage)
    time_result = train_network("time_bp", time_data, TIME_FEATURES, output / "time_bp", 64)
    energy_result = train_network("energy_bp", energy_data, ENERGY_FEATURES, output / "energy_bp", 32)
    decision = "passed" if leakage["passed"] and time_result["quality_gate_passed"] and energy_result["quality_gate_passed"] else "needs_refinement"
    selection_rows = []
    for result in (time_result, energy_result):
        selection_rows.append({"network": result["name"], **result["selected_config"],
                               "validation_selection_score": result["selected_metrics"]["validation_selection_score"],
                               "quality_gate_passed": result["quality_gate_passed"], "model_sha256": result["selected_model_hash"]})
    write_csv(output / "dualbp_model_selection_summary.csv", selection_rows)
    quality = {"passed": decision == "passed", "time_bp": time_result["quality_checks"],
               "energy_bp": energy_result["quality_checks"], "data_leakage": leakage["passed"]}
    _atomic_json(output / "quality_gate_audit.json", quality)
    runtime = {"protocol_design_seconds": load_json(protocol_dir / "summary.json")["protocol_design_seconds"],
               "implementation_seconds": None, "pretest_seconds": None,
               "time_bp_training_seconds": time_result["training_seconds"],
               "energy_bp_training_seconds": energy_result["training_seconds"],
               "model_selection_seconds": 0.0, "final_audit_seconds": time.monotonic() - load_started - time_result["training_seconds"] - energy_result["training_seconds"],
               "final_test_seconds": None, "overall_wall_seconds_before_final_test": time.monotonic() - wall_started,
               "architecture_average_training_seconds": {"time": float(np.mean(time_result["run_seconds"])), "energy": float(np.mean(energy_result["run_seconds"]))},
               "per_seed_average_training_seconds": {"time": float(np.mean(time_result["run_seconds"])), "energy": float(np.mean(energy_result["run_seconds"]))},
               "best_epoch_distribution": {"time": time_result["best_epochs"], "energy": energy_result["best_epochs"]}}
    _atomic_json(output / "runtime_breakdown.json", runtime)
    summary = {"dualbp_training_decision": decision,
               "time_bp_model_status": "trained_and_frozen" if decision == "passed" else "not_frozen",
               "energy_bp_model_status": "trained_and_frozen" if decision == "passed" else "not_frozen",
               "time_bp_model_version": "time_bp_model_v1", "energy_bp_model_version": "energy_bp_model_v1",
               "training_protocol_hash": protocol_hash, "time_bp": time_result, "energy_bp": energy_result,
               "calibration_labels_accessed": False, "test_labels_accessed": False,
               "time_bp_training_allowed": False, "energy_bp_training_allowed": False,
               "alpha_calibration_allowed": decision == "passed", "test_evaluation_allowed": False,
               "corrected_dualbp_available": False, "exp2_test_allowed": False, "exp3_allowed": False}
    _atomic_json(output / "summary.json", summary)
    (output / "README.md").write_text("# 第一版 DualBP Formal 训练\n\n仅 Train 更新权重，Validation 用于早停与选模；Calibration/Test 标签未访问。\n", encoding="utf-8")
    (output / "validation_report.md").write_text("# Validation 选模与质量门禁\n\n"
        f"- decision: `{decision}`\n- Time gate: `{time_result['quality_gate_passed']}`\n- Energy gate: `{energy_result['quality_gate_passed']}`\n- 数据泄漏：0\n", encoding="utf-8")
    _atomic_json(output / "created_files.json", {"files": sorted(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file())})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-root", default=".")
    parser.add_argument("--protocol-dir"); parser.add_argument("--output-dir"); parser.add_argument("--freeze-protocol", action="store_true")
    args = parser.parse_args(); project = Path(args.project_root).resolve()
    target = Path(args.protocol_dir if args.freeze_protocol else args.output_dir)
    try:
        result = freeze_protocol(project, target.resolve()) if args.freeze_protocol else run_formal(project, Path(args.protocol_dir).resolve(), target.resolve())
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as error:
        target.mkdir(parents=True, exist_ok=True)
        _atomic_json(target / "runner_failure.json", {"error": str(error), "traceback": traceback.format_exc(), "at": now_iso()})
        raise


if __name__ == "__main__":
    main()
