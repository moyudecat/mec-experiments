"""冻结 alpha 协议、校准 Corrected DualBP，并生成论文级诊断图。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mec_dualbp.datasets.protocol import ENERGY_FEATURES, TIME_FEATURES, write_csv
from mec_dualbp.experiments.exp1_formal_protocol import file_sha256
from mec_dualbp.experiments.exp1_preview_protocol import canonical_json_hash
from mec_dualbp.models.dualbp import ARCHITECTURES, ConstrainedCenterRadiusMLP
from mec_dualbp.training.calibration import (
    CALIBRATION_DELTA, CALIBRATION_EPSILON, CALIBRATION_RULE,
    apply_directional_correction, conservative_alpha, conservative_rank,
    interval_coverage, lower_over_error, protocol_is_frozen, upper_under_error,
)
from mec_dualbp.training.dualbp import TrainOnlyStandardScaler, regression_metrics
from mec_dualbp.utils.sharded_checkpoint import _atomic_json

TRAINING_PROTOCOL_REL = "outputs/05_audit/dualbp_training_protocol_20260811_003055"
TRAINING_FORMAL_REL = "outputs/01_formal/dualbp_training_formal_20260811_003126"
TIME_DATA_REL = "outputs/01_formal/time_bp_dataset_v2_labels_formal_20260810_194719"
ENERGY_DATA_REL = "outputs/01_formal/bp_dataset_labels_formal_20260810_132254"
TRAINING_PROTOCOL_HASH = "5dba158d126591460f9b5399bdcb4e1fb3c1167f5638537044304891649ea4f8"
TIME_MODEL_HASH = "291c5733bebdb4cb0c2cf7a4a07138ca6bb07c3ad75bffb378aa09b7f657d827"
ENERGY_MODEL_HASH = "8481dfdca0e03e3d6bebe4c4d5808259d16f7cdd67f4a65251fe7aad3e07d971"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def source_hashes(project: Path) -> dict[str, str]:
    files = {"calibration": project / "src/mec_dualbp/training/calibration.py",
             "model": project / "src/mec_dualbp/models/dualbp.py",
             "training": project / "src/mec_dualbp/training/dualbp.py",
             "runner": project / "src/mec_dualbp/experiments/dualbp_alpha_calibration.py"}
    return {name: file_sha256(path) for name, path in files.items()}


def verify_training_binding(project: Path) -> dict[str, Any]:
    formal = project / TRAINING_FORMAL_REL
    summary = load_json(formal / "summary.json")
    required = {"dualbp_training_decision": "passed", "time_bp_model_status": "trained_and_frozen",
                "energy_bp_model_status": "trained_and_frozen", "alpha_calibration_allowed": True,
                "test_evaluation_allowed": False, "corrected_dualbp_available": False}
    if any(summary.get(key) != value for key, value in required.items()):
        raise RuntimeError("frozen training gate drift")
    actual_time = file_sha256(formal / "time_bp/selected_model.pt")
    actual_energy = file_sha256(formal / "energy_bp/selected_model.pt")
    if actual_time != TIME_MODEL_HASH or actual_energy != ENERGY_MODEL_HASH:
        raise RuntimeError("frozen model hash drift")
    protocol_hash = load_json(project / TRAINING_PROTOCOL_REL / "training_protocol_hash.json")["training_protocol_hash"]
    if protocol_hash != TRAINING_PROTOCOL_HASH:
        raise RuntimeError("training protocol hash drift")
    leakage = load_json(formal / "data_leakage_audit.json")
    return {"training_formal_directory": TRAINING_FORMAL_REL, "training_protocol_directory": TRAINING_PROTOCOL_REL,
            "training_protocol_hash": protocol_hash, "time_model_hash": actual_time,
            "energy_model_hash": actual_energy,
            "time_scaler_hash": file_sha256(formal / "time_bp/input_scaler.json"),
            "energy_scaler_hash": file_sha256(formal / "energy_bp/input_scaler.json"),
            "time_model_config": load_json(formal / "time_bp/selected_model_config.json"),
            "energy_model_config": load_json(formal / "energy_bp/selected_model_config.json"),
            "prior_metadata_only_audit": leakage["metadata_only_audit"]}


def freeze_protocol(project: Path, directory: Path) -> dict[str, Any]:
    started = time.monotonic(); binding = verify_training_binding(project)
    # Calibration targets have not been opened: counts and hashes are inherited from the prior metadata-only audit.
    prior = binding["prior_metadata_only_audit"]
    if prior["time_calibration"]["row_count"] != 180 or prior["energy_calibration"]["row_count"] != 60:
        raise RuntimeError("Calibration count drift in sealed metadata binding")
    payload = {"protocol_name": "DualBP-Alpha-Calibration-v1", "status": "freeze_candidate",
               "training_protocol_hash": TRAINING_PROTOCOL_HASH,
               "time_model_hash": TIME_MODEL_HASH, "energy_model_hash": ENERGY_MODEL_HASH,
               "time_scaler_hash": binding["time_scaler_hash"], "energy_scaler_hash": binding["energy_scaler_hash"],
               "calibration_role_binding": {"time": prior["time_calibration"], "energy": prior["energy_calibration"]},
               "test_role_binding_inherited_without_file_access": {"time": prior["time_test"], "energy": prior["energy_test"]},
               "delta": CALIBRATION_DELTA, "epsilon": CALIBRATION_EPSILON,
               "rule": CALIBRATION_RULE, "rank_formula": "min(ceil((n+1)*(1-delta)), n)",
               "time_n": 180, "time_rank": conservative_rank(180),
               "energy_n": 60, "energy_rank": conservative_rank(60),
               "directional_errors": {"alpha_minus": "max(0,(pred_lower-true_lower)/max(true_lower,epsilon))",
                                      "alpha_plus": "max(0,(true_upper-pred_upper)/max(true_upper,epsilon))"},
               "correction": {"lower": "raw_lower/(1+alpha_minus)", "upper": "raw_upper/(1-alpha_plus)"},
               "target_access_allowed_after_freeze": True, "calibration_target_accessed_before_freeze": False,
               "test_labels_accessed": False, "empirical_semantics": "empirical conservative calibration"}
    protocol_hash = canonical_json_hash(payload)
    frozen = {**payload, "status": "frozen", "alpha_calibration_protocol_hash": protocol_hash,
              "frozen_before_calibration_target_access": True}
    directory.mkdir(parents=True, exist_ok=False)
    _atomic_json(directory / "training_model_binding.json", binding)
    _atomic_json(directory / "calibration_dataset_binding.json", payload["calibration_role_binding"])
    _atomic_json(directory / "alpha_calibration_protocol.json", frozen)
    _atomic_json(directory / "alpha_calibration_protocol_hash.json", {"alpha_calibration_protocol_hash": protocol_hash})
    _atomic_json(directory / "data_access_guard.json", {"passed": True,
                 "protocol_hash_generated_before_calibration_target_access": True,
                 "calibration_targets_accessed": False, "test_labels_accessed": False,
                 "test_predictions_generated": False, "test_metrics_computed": False})
    _atomic_json(directory / "source_hashes.json", source_hashes(project))
    (directory / "directional_error_definition.md").write_text("# 方向性相对误差\n\n仅校准危险方向：下界高估与上界低估；epsilon=1e-12。采用有限样本高阶order statistic，不做线性插值。\n", encoding="utf-8")
    (directory / "correction_formula.md").write_text("# 冻结修正公式\n\n`corrected_lower=raw_lower/(1+alpha_minus)`；`corrected_upper=raw_upper/(1-alpha_plus)`。\n", encoding="utf-8")
    (directory / "README.md").write_text("# DualBP α 校准协议\n\n该协议在读取任何Calibration真实上下界前冻结；Test继续密封。\n", encoding="utf-8")
    _atomic_json(directory / "created_files.json", {"files": sorted(p.name for p in directory.iterdir() if p.is_file())})
    return {"status": "frozen", "alpha_calibration_protocol_hash": protocol_hash,
            "protocol_design_seconds": time.monotonic() - started}


def load_model_and_scaler(network: str, formal: Path) -> tuple[ConstrainedCenterRadiusMLP, TrainOnlyStandardScaler, dict[str, Any]]:
    directory = formal / network
    config = load_json(directory / "selected_model_config.json")
    checkpoint = torch.load(directory / "selected_model.pt", map_location="cpu", weights_only=False)
    model = ConstrainedCenterRadiusMLP(config["input_dimension"], ARCHITECTURES[config["architecture"]], config["center_scale"])
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    raw = load_json(directory / "input_scaler.json")
    scaler = TrainOnlyStandardScaler(tuple(raw["mean"]), tuple(raw["scale"]), tuple(raw["constant_feature"]), raw["fitted_role"])
    return model, scaler, config


def infer(model: ConstrainedCenterRadiusMLP, scaler: TrainOnlyStandardScaler, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = scaler.transform(features).astype(np.float32)
    with torch.inference_mode():
        _, _, lower, upper = model(torch.as_tensor(x))
    return lower.numpy().astype(np.float64), upper.numpy().astype(np.float64)


def metrics(true_lower: np.ndarray, true_upper: np.ndarray, pred_lower: np.ndarray,
            pred_upper: np.ndarray, center_scale: float) -> dict[str, float]:
    result = regression_metrics(true_lower, true_upper, pred_lower, pred_upper, center_scale)
    result["icr"] = interval_coverage(true_lower, true_upper, pred_lower, pred_upper)
    return result


def style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 11, "legend.fontsize": 9, "axes.facecolor": "white",
                         "figure.facecolor": "white", "axes.grid": True, "grid.color": "#D9D9D9",
                         "grid.linestyle": "--", "grid.linewidth": .6, "grid.alpha": .75,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight", "svg.fonttype": "none"})


def save_figure(fig: plt.Figure, base: Path) -> list[Path]:
    outputs = []
    for suffix in ("png", "pdf", "svg"):
        path = base.with_suffix("." + suffix)
        fig.savefig(path, dpi=600 if suffix == "png" else None, facecolor="white")
        outputs.append(path)
    plt.close(fig); return outputs


def training_curve(history: Path, best_epoch: int, title: str, output: Path) -> None:
    rows = read_rows(history); epoch = np.asarray([float(r["epoch"]) for r in rows])
    train = np.asarray([float(r["train_loss"]) for r in rows]); validation = np.asarray([float(r["validation_selection_score"]) for r in rows])
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.4), sharex=True)
    axes[0].plot(epoch, train, color="#1F4E79", linewidth=1.5); axes[0].set_ylabel("Train total loss")
    axes[1].plot(epoch, validation, color="#C55A11", linewidth=1.5); axes[1].set_ylabel("Validation score"); axes[1].set_xlabel("Epoch")
    for ax in axes: ax.axvline(best_epoch, color="#7F7F7F", linestyle="--", linewidth=1.2, label=f"Best epoch = {best_epoch}"); ax.legend()
    fig.suptitle(title); fig.tight_layout(); save_figure(fig, output)


def scatter_figure(rows: Sequence[Mapping[str, str]], side: str, unit: str, title: str, output: Path) -> None:
    true = np.asarray([float(r[f"{side}_true"]) for r in rows]); pred = np.asarray([float(r[f"raw_pred_{side}"]) for r in rows])
    low, high = min(true.min(), pred.min()), max(true.max(), pred.max()); pad = (high - low) * .04
    fig, ax = plt.subplots(figsize=(5.8, 5.2)); ax.scatter(true, pred, s=24, alpha=.75, color="#2F75B5", edgecolors="white", linewidths=.35)
    ax.plot([low-pad, high+pad], [low-pad, high+pad], "--", color="#C00000", linewidth=1.3, label="y = x")
    ax.set_xlim(low-pad, high+pad); ax.set_ylim(low-pad, high+pad); ax.set_xlabel(f"Reference label ({unit})"); ax.set_ylabel(f"BP prediction ({unit})")
    ax.set_title(title); ax.legend(); fig.tight_layout(); save_figure(fig, output)


def grouped_wmape_figure(rows: Sequence[Mapping[str, str]], group_field: str, groups: Sequence[str], title: str, output: Path) -> None:
    mapped = {r[group_field]: r for r in rows}; lower = [100*float(mapped[g]["lower_wmape"]) for g in groups]; upper = [100*float(mapped[g]["upper_wmape"]) for g in groups]
    x = np.arange(len(groups)); width = .34; fig, ax = plt.subplots(figsize=(6.6, 4.5))
    bars1 = ax.bar(x-width/2, lower, width, label="Lower bound", color="#2F75B5"); bars2 = ax.bar(x+width/2, upper, width, label="Upper bound", color="#ED7D31")
    ax.bar_label(bars1, fmt="%.2f%%", padding=3, fontsize=8); ax.bar_label(bars2, fmt="%.2f%%", padding=3, fontsize=8)
    ax.set_xticks(x, groups); ax.set_ylabel("WMAPE (%)"); ax.set_title(title); ax.legend(); ax.set_ylim(0, max(lower+upper)*1.25)
    fig.tight_layout(); save_figure(fig, output)


def ecdf_figure(over: np.ndarray, under: np.ndarray, alpha_minus: float, alpha_plus: float, title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for values, color, label in ((over, "#2F75B5", "Lower overestimation"), (under, "#ED7D31", "Upper underestimation")):
        x = np.sort(values)*100; y = np.arange(1, len(x)+1)/len(x); ax.step(x, y, where="post", linewidth=1.7, color=color, label=label)
    ax.axvline(alpha_minus*100, color="#2F75B5", linestyle="--", linewidth=1.2, label=f"alpha− = {alpha_minus*100:.3f}%")
    ax.axvline(alpha_plus*100, color="#ED7D31", linestyle=":", linewidth=1.4, label=f"alpha+ = {alpha_plus*100:.3f}%")
    ax.set_xlabel("Directional relative error (%)"); ax.set_ylabel("Empirical cumulative probability"); ax.set_ylim(0, 1.02); ax.set_title(title); ax.legend()
    fig.tight_layout(); save_figure(fig, output)


def safety_figure(raw: Mapping[str, float], corrected: Mapping[str, float], title: str, output: Path) -> None:
    names = ("UER", "LER", "ICR"); rv = [100*raw[k.lower()] for k in names]; cv = [100*corrected[k.lower()] for k in names]
    x = np.arange(3); width=.34; fig, ax = plt.subplots(figsize=(6.2, 4.5))
    a=ax.bar(x-width/2,rv,width,label="Raw BP",color="#7F8C8D"); b=ax.bar(x+width/2,cv,width,label="Corrected BP",color="#2F75B5")
    ax.bar_label(a,fmt="%.1f%%",padding=3,fontsize=8); ax.bar_label(b,fmt="%.1f%%",padding=3,fontsize=8)
    ax.set_xticks(x,names); ax.set_ylabel("Rate (%)"); ax.set_ylim(0, max(rv+cv+[100])*1.12); ax.set_title(title); ax.legend()
    fig.tight_layout(); save_figure(fig, output)


def run_formal(project: Path, protocol_dir: Path, output: Path) -> dict[str, Any]:
    started=time.monotonic(); protocol=load_json(protocol_dir/"alpha_calibration_protocol.json")
    protocol_hash=load_json(protocol_dir/"alpha_calibration_protocol_hash.json")["alpha_calibration_protocol_hash"]
    check_payload={k:v for k,v in protocol.items() if k not in ("status","alpha_calibration_protocol_hash","frozen_before_calibration_target_access")}
    check_payload["status"]="freeze_candidate"
    if canonical_json_hash(check_payload)!=protocol_hash or not protocol_is_frozen(protocol): raise RuntimeError("alpha protocol binding invalid")
    output.mkdir(parents=True,exist_ok=False); formal=project/TRAINING_FORMAL_REL; binding=verify_training_binding(project)
    before={"time_model":file_sha256(formal/"time_bp/selected_model.pt"),"energy_model":file_sha256(formal/"energy_bp/selected_model.pt"),
            "time_scaler":file_sha256(formal/"time_bp/input_scaler.json"),"energy_scaler":file_sha256(formal/"energy_bp/input_scaler.json")}
    _atomic_json(output/"protocol_binding.json",{"passed":True,"alpha_calibration_protocol_hash":protocol_hash,"frozen_before_target_access":True})
    _atomic_json(output/"model_binding.json",binding); _atomic_json(output/"dataset_binding.json",protocol["calibration_role_binding"])
    # First Calibration target access occurs here, after the frozen protocol hash has been verified and copied.
    time_rows=read_rows(project/TIME_DATA_REL/"time_calibration_labels.csv")
    energy_rows=read_rows(project/ENERGY_DATA_REL/"calibration_labels.csv")
    if len(time_rows)!=180 or len(energy_rows)!=60 or any(r["role"]!="Calibration" for r in time_rows+energy_rows): raise RuntimeError("Calibration role/count drift")
    time_model,time_scaler,time_config=load_model_and_scaler("time_bp",formal); energy_model,energy_scaler,energy_config=load_model_and_scaler("energy_bp",formal)
    tx=np.asarray([[float(r[f]) for f in TIME_FEATURES] for r in time_rows]); tl=np.asarray([float(r["time_lower"]) for r in time_rows]); tu=np.asarray([float(r["time_upper"]) for r in time_rows])
    feature_rows={r["sample_id"]:r for r in read_rows(project/ENERGY_DATA_REL/"feature_records.csv") if r["role"]=="Calibration"}
    ex=np.asarray([json.loads(feature_rows[r["sample_id"]]["energy_features"]) for r in energy_rows]); el=np.asarray([float(r["energy_lower"]) for r in energy_rows]); eu=np.asarray([float(r["energy_upper"]) for r in energy_rows])
    trl,tru=infer(time_model,time_scaler,tx); erl,eru=infer(energy_model,energy_scaler,ex)
    t_over,t_under=lower_over_error(tl,trl),upper_under_error(tu,tru); e_over,e_under=lower_over_error(el,erl),upper_under_error(eu,eru)
    a_tm,k_t=conservative_alpha(t_over); a_tp,k_t2=conservative_alpha(t_under); a_em,k_e=conservative_alpha(e_over); a_ep,k_e2=conservative_alpha(e_under)
    if k_t!=k_t2 or k_e!=k_e2 or any(not 0<=a<1 for a in (a_tm,a_tp,a_em,a_ep)): raise RuntimeError("alpha calibration failed")
    tcl,tcu=apply_directional_correction(trl,tru,a_tm,a_tp); ecl,ecu=apply_directional_correction(erl,eru,a_em,a_ep)
    def pred_rows(rows,lt,ut,pl,pu): return [{"record_id":r.get("time_record_id",r.get("sample_id")),"role":"Calibration","true_lower":a,"true_upper":b,"raw_lower":c,"raw_upper":d} for r,a,b,c,d in zip(rows,lt,ut,pl,pu)]
    t_raw=pred_rows(time_rows,tl,tu,trl,tru); e_raw=pred_rows(energy_rows,el,eu,erl,eru); write_csv(output/"time_calibration_predictions_raw.csv",t_raw); write_csv(output/"energy_calibration_predictions_raw.csv",e_raw)
    t_err=[{"record_id":r["record_id"],"lower_over_relative_error":a,"upper_under_relative_error":b} for r,a,b in zip(t_raw,t_over,t_under)]
    e_err=[{"record_id":r["record_id"],"lower_over_relative_error":a,"upper_under_relative_error":b} for r,a,b in zip(e_raw,e_over,e_under)]
    write_csv(output/"time_calibration_directional_errors.csv",t_err); write_csv(output/"energy_calibration_directional_errors.csv",e_err)
    def corrected_rows(raw,cl,cu): return [{**r,"corrected_lower":a,"corrected_upper":b} for r,a,b in zip(raw,cl,cu)]
    write_csv(output/"time_calibration_predictions_corrected.csv",corrected_rows(t_raw,tcl,tcu)); write_csv(output/"energy_calibration_predictions_corrected.csv",corrected_rows(e_raw,ecl,ecu))
    raw_metrics={"time":metrics(tl,tu,trl,tru,time_config["center_scale"]),"energy":metrics(el,eu,erl,eru,energy_config["center_scale"])}
    corr_metrics={"time":metrics(tl,tu,tcl,tcu,time_config["center_scale"]),"energy":metrics(el,eu,ecl,ecu,energy_config["center_scale"])}
    _atomic_json(output/"raw_calibration_metrics.json",raw_metrics); _atomic_json(output/"corrected_calibration_metrics.json",corr_metrics)
    safety=[]
    for net in ("time","energy"):
        for variant,source in (("raw",raw_metrics),("corrected",corr_metrics)): safety.append({"network":net,"variant":variant,"uer":source[net]["uer"],"ler":source[net]["ler"],"icr":source[net]["icr"]})
    write_csv(output/"calibration_safety_metrics.csv",safety)
    alpha={"delta":CALIBRATION_DELTA,"time_order_statistic_rank":k_t,"energy_order_statistic_rank":k_e,
           "alpha_T_minus":a_tm,"alpha_T_plus":a_tp,"alpha_E_minus":a_em,"alpha_E_plus":a_ep,"all_alpha_less_than_one":True}
    _atomic_json(output/"alpha_values.json",alpha)
    corrected={"version":"corrected_dualbp_v1","definition":"Frozen Raw DualBP + Frozen directional alpha correction",
               "time_model_hash":TIME_MODEL_HASH,"energy_model_hash":ENERGY_MODEL_HASH,**alpha,
               "calibration_rule":CALIBRATION_RULE,"calibration_role_hash":canonical_json_hash(protocol["calibration_role_binding"]),
               "alpha_calibration_protocol_hash":protocol_hash,"training_protocol_hash":TRAINING_PROTOCOL_HASH,
               "correction_formula_version":"multiplicative-directional-v1"}
    corrected_hash=canonical_json_hash(corrected); _atomic_json(output/"corrected_dualbp_v1.json",corrected); _atomic_json(output/"corrected_dualbp_v1_hash.json",{"sha256":corrected_hash})
    fig_tv=output/"figures/training_validation"; fig_cd=output/"figures/calibration_diagnostic"; fig_tv.mkdir(parents=True); fig_cd.mkdir(parents=True); style(); manifest=[]
    def register(name,category,sources):
        for fmt in ("png","pdf","svg"): manifest.append({"figure":name,"category":category,"format":fmt,"path":f"figures/{category}/{name}.{fmt}","data_role":"Train/Validation" if category=="training_validation" else "Calibration diagnostic","source_files":";".join(sources),"test_source_used":False})
    th=formal/"time_bp/training_history/M3_lambda_0.25_seed_31004.csv"; eh=formal/"energy_bp/training_history/M3_lambda_0.25_seed_31001.csv"
    training_curve(th,1298,"Time-BP training and Validation diagnostic",fig_tv/"time_training_curve"); register("time_training_curve","training_validation",[str(th)])
    training_curve(eh,928,"Energy-BP training and Validation diagnostic",fig_tv/"energy_training_curve"); register("energy_training_curve","training_validation",[str(eh)])
    tv=read_rows(formal/"time_bp/validation_predictions_selected.csv"); ev=read_rows(formal/"energy_bp/validation_predictions_selected.csv")
    for side in ("lower","upper"):
        scatter_figure(tv,side,"s",f"Time-BP Validation: {side} bound",fig_tv/f"time_validation_{side}_scatter"); register(f"time_validation_{side}_scatter","training_validation",["time_bp/validation_predictions_selected.csv"])
        scatter_figure(ev,side,"J",f"Energy-BP Validation: {side} bound",fig_tv/f"energy_validation_{side}_scatter"); register(f"energy_validation_{side}_scatter","training_validation",["energy_bp/validation_predictions_selected.csv"])
    tf=read_rows(formal/"time_bp/validation_metrics_by_f_stratum.csv"); grouped_wmape_figure(tf,"f_stratum",["Low","Mid-high","Upper-anchor"],"Time-BP Validation WMAPE by f stratum",fig_tv/"time_validation_f_stratum_wmape"); register("time_validation_f_stratum_wmape","training_validation",["time_bp/validation_metrics_by_f_stratum.csv"])
    eg=read_rows(formal/"energy_bp/validation_metrics_by_gamma.csv"); grouped_wmape_figure(eg,"Gamma",["1.0","2.0","3.0","4.0","5.0"],"Energy-BP Validation WMAPE by Gamma",fig_tv/"energy_validation_gamma_wmape"); register("energy_validation_gamma_wmape","training_validation",["energy_bp/validation_metrics_by_gamma.csv"])
    ecdf_figure(t_over,t_under,a_tm,a_tp,"Time Calibration directional-error ECDF",fig_cd/"time_directional_error_ecdf"); register("time_directional_error_ecdf","calibration_diagnostic",["time_calibration_directional_errors.csv"])
    ecdf_figure(e_over,e_under,a_em,a_ep,"Energy Calibration directional-error ECDF",fig_cd/"energy_directional_error_ecdf"); register("energy_directional_error_ecdf","calibration_diagnostic",["energy_calibration_directional_errors.csv"])
    safety_figure(raw_metrics["time"],corr_metrics["time"],"Time Calibration safety diagnostic",fig_cd/"time_raw_vs_corrected_safety"); register("time_raw_vs_corrected_safety","calibration_diagnostic",["calibration_safety_metrics.csv"])
    safety_figure(raw_metrics["energy"],corr_metrics["energy"],"Energy Calibration safety diagnostic",fig_cd/"energy_raw_vs_corrected_safety"); register("energy_raw_vs_corrected_safety","calibration_diagnostic",["calibration_safety_metrics.csv"])
    write_csv(output/"figure_manifest.csv",manifest)
    after={"time_model":file_sha256(formal/"time_bp/selected_model.pt"),"energy_model":file_sha256(formal/"energy_bp/selected_model.pt"),"time_scaler":file_sha256(formal/"time_bp/input_scaler.json"),"energy_scaler":file_sha256(formal/"energy_bp/input_scaler.json")}
    integrity={"passed":before==after,"before":before,"after":after,"model_weights_updated":False,"scaler_refit":False}; _atomic_json(output/"model_hash_integrity.json",integrity)
    leakage={"passed":True,"protocol_hash_generated_before_calibration_target_access":True,"calibration_labels_accessed":True,"test_labels_accessed":False,"test_predictions_generated":False,"test_metrics_computed":False,"test_used_for_alpha":False,"test_used_for_model_selection":False}; _atomic_json(output/"data_leakage_audit.json",leakage)
    invalid=int(np.sum((tcl<0)|(tcu<tcl)))+int(np.sum((ecl<0)|(ecu<ecl))); nonfinite=int(not all(np.isfinite(x).all() for x in (tcl,tcu,ecl,ecu)))
    numerical={"passed":invalid==0 and nonfinite==0,"corrected_invalid_intervals":invalid,"nonfinite":nonfinite,"all_alpha_finite":all(math.isfinite(x) for x in (a_tm,a_tp,a_em,a_ep)),"all_alpha_in_range":all(0<=x<1 for x in (a_tm,a_tp,a_em,a_ep))}; _atomic_json(output/"numerical_audit.json",numerical)
    decision="passed" if integrity["passed"] and leakage["passed"] and numerical["passed"] else "failed"
    runtime={"overall_wall_seconds":time.monotonic()-started,"protocol_design_seconds":load_json(protocol_dir/"summary.json").get("protocol_design_seconds"),"calibration_and_inference_seconds":None,"figure_generation_included":True}; _atomic_json(output/"runtime_breakdown.json",runtime)
    summary={"alpha_calibration_decision":decision,"raw_dualbp_status":"trained_and_frozen","corrected_dualbp_status":"calibrated_and_frozen" if decision=="passed" else "not_frozen","corrected_dualbp_available":decision=="passed","alpha_calibration_allowed":False,"model_training_allowed":False,"test_evaluation_allowed":decision=="passed","exp2_test_allowed":decision=="passed","exp3_allowed":False,"alpha_values":alpha,"raw_metrics":raw_metrics,"corrected_metrics":corr_metrics,"corrected_dualbp_hash":corrected_hash,"test_labels_accessed":False,"test_predictions_generated":False,"test_metrics_computed":False,"figure_count":12,"figure_files":36}; _atomic_json(output/"summary.json",summary)
    (output/"README.md").write_text("# DualBP α 正式校准\n\nCorrected DualBP = 冻结Raw DualBP + 冻结方向性alpha修正。本目录仅使用Calibration；Test仍密封。\n",encoding="utf-8")
    (output/"validation_report.md").write_text(f"# α校准审计\n\n- decision: `{decision}`\n- Test标签访问：false\n- 模型与scaler哈希保持：{integrity['passed']}\n- 术语：经验保守误差修正。\n",encoding="utf-8")
    _atomic_json(output/"created_files.json",{"files":sorted(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file())})
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--project-root",default="."); parser.add_argument("--protocol-dir"); parser.add_argument("--output-dir"); parser.add_argument("--freeze-protocol",action="store_true"); args=parser.parse_args(); project=Path(args.project_root).resolve(); target=Path(args.protocol_dir if args.freeze_protocol else args.output_dir).resolve()
    try:
        result=freeze_protocol(project,target) if args.freeze_protocol else run_formal(project,Path(args.protocol_dir).resolve(),target)
        if args.freeze_protocol: _atomic_json(target/"summary.json",result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except Exception as error:
        target.mkdir(parents=True,exist_ok=True); _atomic_json(target/"runner_failure.json",{"error":str(error),"traceback":traceback.format_exc()}); raise


if __name__=="__main__": main()
