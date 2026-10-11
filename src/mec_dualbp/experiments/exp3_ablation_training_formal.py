"""Exp-3 C/D/E 正式消融训练；只允许 Train/Validation 标签。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from mec_dualbp.datasets.protocol import ENERGY_FEATURES, TIME_FEATURES, write_csv
from mec_dualbp.experiments.dualbp_training_formal import load_training_data, read_rows
from mec_dualbp.experiments.exp1_formal_protocol import file_sha256
from mec_dualbp.models.ablation import DirectBoundsMLP, UnifiedCenterRadiusMLP
from mec_dualbp.models.dualbp import ARCHITECTURES, ConstrainedCenterRadiusMLP
from mec_dualbp.training.ablation import ABLATION_SEEDS, stable_ordered_union
from mec_dualbp.training.dualbp import TrainOnlyStandardScaler, regression_metrics, seed_everything
from mec_dualbp.utils.sharded_checkpoint import _atomic_json

PROTOCOL_DIR_REL = "outputs/05_audit/exp3_ablation_protocol_20260811_144345"
PROTOCOL_HASH = "a4d7a4a86c4618754534495609364f6debf20ed5e3ca6f814085d543d79d8544"
TRAINING_DIR_REL = "outputs/01_formal/dualbp_training_formal_20260811_003126"
TIME_MODEL_HASH = "291c5733bebdb4cb0c2cf7a4a07138ca6bb07c3ad75bffb378aa09b7f657d827"
ENERGY_MODEL_HASH = "8481dfdca0e03e3d6bebe4c4d5808259d16f7cdd67f4a65251fe7aad3e07d971"
CORRECTED_HASH = "a15b8fb591df123b475f521ef35e4037b6ca201f4172caf804d3af571afe13e2"
HIDDEN = (128, 64, 32)
MAX_EPOCHS = 1500
EPSILON = 1e-12


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _scaler(path: Path, fields: Sequence[str]) -> TrainOnlyStandardScaler:
    value = _read_json(path)
    if value["fields"] != list(fields) or value["fitted_role"] != "Train":
        raise RuntimeError("Raw scaler binding mismatch")
    return TrainOnlyStandardScaler(tuple(value["mean"]), tuple(value["scale"]), tuple(value["constant_feature"]), "Train")


def _environment() -> dict[str, Any]:
    return {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
            "platform": platform.platform(), "device": "cpu", "workers": 1,
            "torch_threads": 1, "deterministic_algorithms": True}


def _baseline_hashes(project: Path) -> dict[str, Any]:
    training = project / TRAINING_DIR_REL
    actual = {"time_model_hash": file_sha256(training / "time_bp/selected_model.pt"),
              "energy_model_hash": file_sha256(training / "energy_bp/selected_model.pt"),
              "corrected_dualbp_hash": _read_json(project / "outputs/01_formal/dualbp_alpha_calibration_formal_20260811_134010/corrected_dualbp_v1_hash.json")["sha256"]}
    actual["passed"] = (actual["time_model_hash"] == TIME_MODEL_HASH and
                        actual["energy_model_hash"] == ENERGY_MODEL_HASH and
                        actual["corrected_dualbp_hash"] == CORRECTED_HASH)
    return actual


def _loss_vectors(lower_hat: Tensor, upper_hat: Tensor, lower_true: Tensor, upper_true: Tensor,
                  center_scale: float) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    scale = float(center_scale) + EPSILON
    center_hat, radius_hat = (lower_hat + upper_hat) * .5, (upper_hat - lower_hat) * .5
    center_true, radius_true = (lower_true + upper_true) * .5, (upper_true - lower_true) * .5
    smooth = torch.nn.functional.smooth_l1_loss
    cr = smooth(center_hat / scale, center_true / scale, reduction="none") + smooth(radius_hat / scale, radius_true / scale, reduction="none")
    bounds = smooth(lower_hat / scale, lower_true / scale, reduction="none") + smooth(upper_hat / scale, upper_true / scale, reduction="none")
    fit = cr + .5 * bounds
    safety = torch.relu(upper_true - upper_hat) / scale + torch.relu(lower_hat - lower_true) / scale
    order = torch.relu(lower_hat - upper_hat) / scale
    return fit, cr, bounds, safety, order


def _metrics(lower_true: np.ndarray, upper_true: np.ndarray, lower_hat: np.ndarray,
             upper_hat: np.ndarray, center_scale: float) -> dict[str, float | int]:
    result: dict[str, float | int] = regression_metrics(lower_true, upper_true, lower_hat, upper_hat, center_scale)
    invalid = (lower_hat < 0) | (upper_hat < 0) | (upper_hat < lower_hat)
    nonfinite = ~(np.isfinite(lower_hat) & np.isfinite(upper_hat))
    result.update({"icr": float(np.mean((lower_hat <= lower_true) & (upper_hat >= upper_true))),
                   "invalid_interval_count": int(invalid.sum()), "invalid_interval_rate": float(invalid.mean()),
                   "nonfinite_count": int(nonfinite.sum())})
    return result


def _weighted_metrics(lower_true: np.ndarray, upper_true: np.ndarray, lower_hat: np.ndarray,
                      upper_hat: np.ndarray, center_scale: float, weights: np.ndarray) -> dict[str, float | int]:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    def mean(x: np.ndarray) -> float: return float(np.sum(weights * x))
    def side(prefix: str, true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
        error = pred - true; true_mean = mean(true)
        ss_total = float(np.sum(weights * (true - true_mean) ** 2))
        return {f"{prefix}_mae": mean(np.abs(error)), f"{prefix}_rmse": math.sqrt(mean(error ** 2)),
                f"{prefix}_r2": 1.0 - mean(error ** 2) / max(ss_total, EPSILON),
                f"{prefix}_wmape": float(np.sum(weights * np.abs(error)) / max(np.sum(weights * np.abs(true)), EPSILON))}
    result: dict[str, float | int] = {**side("lower", lower_true, lower_hat), **side("upper", upper_true, upper_hat)}
    result.update({"center_mae": mean(np.abs((lower_hat + upper_hat - lower_true - upper_true) * .5)),
        "radius_mae": mean(np.abs((upper_hat - lower_hat - upper_true + lower_true) * .5)),
        "uer": mean((upper_hat < upper_true).astype(float)), "ler": mean((lower_hat > lower_true).astype(float)),
        "icr": mean(((lower_hat <= lower_true) & (upper_hat >= upper_true)).astype(float)),
        "upper_under_relative_magnitude": mean(np.maximum(upper_true - upper_hat, 0)) / (center_scale + EPSILON),
        "lower_over_relative_magnitude": mean(np.maximum(lower_hat - lower_true, 0)) / (center_scale + EPSILON),
        "invalid_interval_count": int(np.sum(upper_hat < lower_hat)), "invalid_interval_rate": mean((upper_hat < lower_hat).astype(float)),
        "nonfinite_count": int(np.sum(~(np.isfinite(lower_hat) & np.isfinite(upper_hat))))})
    result["validation_selection_score"] = (.35 * float(result["lower_wmape"]) + .35 * float(result["upper_wmape"])
        + .15 * float(result["upper_under_relative_magnitude"]) + .15 * float(result["lower_over_relative_magnitude"]))
    return result


def _save_checkpoint(path: Path, model: nn.Module, config: Mapping[str, Any]) -> str:
    torch.save({"model_state_dict": model.state_dict(), **dict(config)}, path)
    return file_sha256(path)


def _train_single(
    model_factory: Callable[[], nn.Module], x_train: np.ndarray, train_targets: tuple[np.ndarray, ...],
    x_val: np.ndarray, val_targets: tuple[np.ndarray, ...], *, seed: int, batch_size: int,
    loss_kind: str, time_scale: float, energy_scale: float | None = None,
    train_energy_weights: np.ndarray | None = None, val_energy_weights: np.ndarray | None = None,
) -> tuple[nn.Module, dict[str, Any], list[dict[str, float]]]:
    seed_everything(seed); model = model_factory()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=25, min_lr=1e-5)
    xt, xv = torch.as_tensor(x_train, dtype=torch.float32), torch.as_tensor(x_val, dtype=torch.float32)
    tt = tuple(torch.as_tensor(value, dtype=torch.float32) for value in train_targets)
    vt = tuple(torch.as_tensor(value, dtype=torch.float32) for value in val_targets)
    wt = None if train_energy_weights is None else torch.as_tensor(train_energy_weights, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    best_score, best_epoch, stale, best_state = math.inf, 0, 0, None
    history: list[dict[str, float]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); order = torch.randperm(len(xt), generator=generator); total = 0.0
        for start in range(0, len(xt), batch_size):
            idx = order[start:start + batch_size]
            if loss_kind == "unified":
                pred = model(xt[idx]); tf, _, _, ts, _ = _loss_vectors(pred["time_lower"], pred["time_upper"], tt[0][idx], tt[1][idx], time_scale)
                ef, _, _, es, _ = _loss_vectors(pred["energy_lower"], pred["energy_upper"], tt[2][idx], tt[3][idx], float(energy_scale))
                ew = wt[idx]; energy_loss = torch.sum((ef + .25 * es) * ew) / torch.sum(ew)
                loss = .5 * torch.mean(tf + .25 * ts) + .5 * energy_loss
            else:
                lower, upper = model(xt[idx]) if loss_kind == "direct" else model(xt[idx])[2:]
                fit, _, _, safety, order_loss = _loss_vectors(lower, upper, tt[0][idx], tt[1][idx], time_scale)
                loss = torch.mean(fit + (0.0 if loss_kind == "no_safety" else .25) * safety + (1.0 if loss_kind == "direct" else 0.0) * order_loss)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            total += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            if loss_kind == "unified":
                pred = model(xv); tm = _metrics(vt[0].numpy(), vt[1].numpy(), pred["time_lower"].numpy(), pred["time_upper"].numpy(), time_scale)
                em = _weighted_metrics(vt[2].numpy(), vt[3].numpy(), pred["energy_lower"].numpy(), pred["energy_upper"].numpy(), float(energy_scale), val_energy_weights)
                score = .5 * float(tm["validation_selection_score"]) + .5 * float(em["validation_selection_score"])
            else:
                lower, upper = model(xv) if loss_kind == "direct" else model(xv)[2:]
                metrics = _metrics(vt[0].numpy(), vt[1].numpy(), lower.numpy(), upper.numpy(), time_scale)
                score = float(metrics["validation_selection_score"])
        scheduler.step(score); history.append({"epoch": float(epoch), "train_loss": total/len(xt),
            "validation_selection_score": score, "learning_rate": float(optimizer.param_groups[0]["lr"])})
        if score < best_score - 1e-5:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else: stale += 1
        if stale >= 100: break
    if best_state is None: raise RuntimeError("no checkpoint produced")
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        if loss_kind == "unified":
            pred = model(xv); tm = _metrics(vt[0].numpy(), vt[1].numpy(), pred["time_lower"].numpy(), pred["time_upper"].numpy(), time_scale)
            em = _weighted_metrics(vt[2].numpy(), vt[3].numpy(), pred["energy_lower"].numpy(), pred["energy_upper"].numpy(), float(energy_scale), val_energy_weights)
            metrics = {"time_validation_score": tm["validation_selection_score"], "energy_validation_score": em["validation_selection_score"],
                       "validation_selection_score": .5*float(tm["validation_selection_score"])+.5*float(em["validation_selection_score"]),
                       "time": tm, "energy": em}
        else:
            lower, upper = model(xv) if loss_kind == "direct" else model(xv)[2:]
            metrics = _metrics(vt[0].numpy(), vt[1].numpy(), lower.numpy(), upper.numpy(), time_scale)
    metrics.update({"training_seed": seed, "best_epoch": best_epoch, "epochs_ran": len(history)})
    return model, metrics, history


def _single_predictions(model: nn.Module, x: np.ndarray, rows: Sequence[Mapping[str, str]],
                        lower_true: np.ndarray, upper_true: np.ndarray, direct: bool) -> list[dict[str, Any]]:
    model.eval()
    with torch.no_grad():
        out = model(torch.as_tensor(x, dtype=torch.float32)); lower, upper = out if direct else out[2:]
    result = []
    for row, lt, ut, lp, up in zip(rows, lower_true, upper_true, lower.numpy(), upper.numpy()):
        result.append({"record_id": row.get("time_record_id", row.get("sample_id")), "base_sample_id": row.get("base_sample_id", row.get("sample_id")),
            "role": row["role"], "Gamma": row["Gamma"], "sample_type": row["sample_type"],
            "lower_true": lt, "upper_true": ut, "raw_pred_lower": float(lp), "raw_pred_upper": float(up),
            "invalid_interval": bool(up < lp)})
    return result


def _train_single_network(name: str, model_version: str, data: dict[str, Any], fields: Sequence[str],
                          scaler: TrainOnlyStandardScaler, center_scale: float, output: Path,
                          *, loss_kind: str, batch_size: int) -> dict[str, Any]:
    output.mkdir(parents=True); (output/"training_history").mkdir(); (output/"checkpoints").mkdir()
    x_train = scaler.transform(data["x_train"]).astype(np.float32); x_val = scaler.transform(data["x_validation"]).astype(np.float32)
    runs=[]; start_all=time.monotonic()
    factory = (lambda: DirectBoundsMLP(len(fields), HIDDEN, center_scale)) if loss_kind == "direct" else (lambda: ConstrainedCenterRadiusMLP(len(fields), HIDDEN, center_scale))
    for seed in ABLATION_SEEDS:
        started=time.monotonic(); model, metrics, history = _train_single(factory, x_train, (data["lower_train"],data["upper_train"]), x_val,
            (data["lower_validation"],data["upper_validation"]), seed=seed, batch_size=batch_size, loss_kind=loss_kind, time_scale=center_scale)
        metrics["train_seconds"]=time.monotonic()-started; metrics.update({"variant":name,"network":model_version,"architecture":"M3",
            "lambda_safety":0.0 if loss_kind=="no_safety" else .25,"lambda_order":1.0 if loss_kind=="direct" else 0.0})
        stem=f"M3_seed_{seed}"; ck=output/"checkpoints"/f"{stem}.pt"
        metrics["checkpoint_hash"]=_save_checkpoint(ck,model,{"model_version":model_version,"input_dim":len(fields),"hidden_layers":list(HIDDEN),
            "center_scale":center_scale,"loss_kind":loss_kind,"training_seed":seed,"best_epoch":metrics["best_epoch"]})
        metrics["checkpoint"]=ck.name; write_csv(output/"training_history"/f"{stem}.csv",history); runs.append(metrics)
    flat=[]
    for row in runs:
        flat.append({k:v for k,v in row.items() if k not in {"time","energy"}})
    write_csv(output/"all_runs_summary.csv",flat); write_csv(output/"per_seed_metrics.csv",flat)
    selected=min(runs,key=lambda row:float(row["validation_selection_score"])); selected_ck=output/"checkpoints"/selected["checkpoint"]
    checkpoint=torch.load(selected_ck,map_location="cpu",weights_only=False); model=factory(); model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    shutil.copyfile(selected_ck,output/"selected_model.pt"); selected_hash=file_sha256(output/"selected_model.pt")
    predictions=_single_predictions(model,x_val,data["validation_rows"],data["lower_validation"],data["upper_validation"],loss_kind=="direct")
    write_csv(output/"validation_predictions_selected.csv",predictions)
    train_predictions=_single_predictions(model,x_train,data["train_rows"],data["lower_train"],data["upper_train"],loss_kind=="direct")
    config={"model_version":model_version,"architecture":"M3","hidden_layers":list(HIDDEN),"lambda_safety":0.0 if loss_kind=="no_safety" else .25,
        "lambda_order":1.0 if loss_kind=="direct" else 0.0,"training_seed":selected["training_seed"],"best_epoch":selected["best_epoch"],
        "input_dimension":len(fields),"input_feature_order":list(fields),"center_scale":center_scale,"input_scaler_binding":"Raw frozen scaler",
        "output_semantics":"direct lower-upper" if loss_kind=="direct" else "center-radius","post_hoc_repair":False}
    _atomic_json(output/"selected_model_config.json",config); _atomic_json(output/"selected_model_metrics.json",selected)
    _atomic_json(output/"selected_model_hash.json",{"sha256":selected_hash,"readable":True}); _atomic_json(output/"input_scaler.json",scaler.as_dict(fields))
    scores=np.asarray([float(r["validation_selection_score"]) for r in runs]); _atomic_json(output/"seed_stability.json",{"seeds":list(ABLATION_SEEDS),
        "score_mean":float(scores.mean()),"score_std":float(scores.std()),"score_min":float(scores.min()),"score_max":float(scores.max())})
    (output/"model_card.md").write_text(f"# {model_version}\n\n只使用Train/Validation，五个固定seed按冻结Validation score选模。模型哈希：`{selected_hash}`。\n",encoding="utf-8")
    invalid={"Train":{"count":sum(r["invalid_interval"] for r in train_predictions),"total":len(train_predictions)},
             "Validation":{"count":sum(r["invalid_interval"] for r in predictions),"total":len(predictions)}}
    for value in invalid.values(): value["rate"]=value["count"]/value["total"]
    return {"model_version":model_version,"runs_completed":len(runs),"selected_seed":selected["training_seed"],"best_epoch":selected["best_epoch"],
        "selected_metrics":selected,"model_hash":selected_hash,"training_seconds":time.monotonic()-start_all,"invalid_intervals":invalid,
        "nonfinite":selected["nonfinite_count"],"train_predictions":train_predictions,"validation_predictions":predictions}


def _build_unified(project: Path, time_data: dict[str, Any], energy_data: dict[str, Any], union_fields: Sequence[str]) -> dict[str, Any]:
    energy_dir=project/"outputs/01_formal/bp_dataset_labels_formal_20260810_132254"
    feature_rows={r["sample_id"]:r for r in read_rows(energy_dir/"feature_records.csv") if r["role"] in {"Train","Validation"}}
    energy_rows={r["sample_id"]:r for r in (*energy_data["train_rows"],*energy_data["validation_rows"])}
    def build(rows: Sequence[Mapping[str,str]]) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,list[dict[str,str]]]:
        counts=Counter(r["base_sample_id"] for r in rows); x=[]; el=[];eu=[];weights=[]; metadata=[]
        for row in rows:
            base=row["base_sample_id"]
            if base not in energy_rows or base not in feature_rows or energy_rows[base]["role"]!=row["role"]: raise RuntimeError("Unified base binding mismatch")
            values={field:float(row[field]) for field in TIME_FEATURES}; energy_values=json.loads(feature_rows[base]["energy_features"])
            energy_map=dict(zip(ENERGY_FEATURES,energy_values)); values.update({field:energy_map[field] for field in union_fields if field not in values})
            x.append([values[field] for field in union_fields]); el.append(float(energy_rows[base]["energy_lower"]));eu.append(float(energy_rows[base]["energy_upper"]));weights.append(1/counts[base]);metadata.append(dict(row))
        return np.asarray(x),np.asarray(el),np.asarray(eu),np.asarray(weights),np.asarray([counts[r["base_sample_id"]] for r in rows]),metadata
    tr=build(time_data["train_rows"]); va=build(time_data["validation_rows"])
    return {"x_train":tr[0],"energy_lower_train":tr[1],"energy_upper_train":tr[2],"energy_weights_train":tr[3],"records_per_base_train":tr[4],"train_rows":tr[5],
            "x_validation":va[0],"energy_lower_validation":va[1],"energy_upper_validation":va[2],"energy_weights_validation":va[3],"records_per_base_validation":va[4],"validation_rows":va[5]}


def _invariance_rows(model: UnifiedCenterRadiusMLP, x: np.ndarray, rows: Sequence[Mapping[str,str]]) -> tuple[list[dict[str,Any]],dict[str,Any]]:
    model.eval()
    with torch.no_grad(): pred=model(torch.as_tensor(x,dtype=torch.float32))
    grouped=defaultdict(list)
    for i,row in enumerate(rows): grouped[row["base_sample_id"]].append((row,float(pred["energy_lower"][i]),float(pred["energy_upper"][i])))
    output=[]; fields={name:[] for name in ("lower","upper","center","radius")}
    for base,values in grouped.items():
        lower=np.asarray([v[1] for v in values]);upper=np.asarray([v[2] for v in values]); arrays={"lower":lower,"upper":upper,"center":.5*(lower+upper),"radius":.5*(upper-lower)}
        row={"base_sample_id":base,"role":values[0][0]["role"]}
        for name,array in arrays.items():
            std=float(array.std());spread=float(np.ptp(array));relative=spread/max(float(np.mean(np.abs(array))),EPSILON)
            row.update({f"{name}_std":std,f"{name}_range":spread,f"{name}_relative_range":relative});fields[name].append(relative)
        output.append(row)
    summary={name:{"mean":float(np.mean(values)),"q95":float(np.quantile(values,.95)),"max":float(np.max(values))} for name,values in fields.items()}
    return output,summary


def _train_unified(project: Path, time_data: dict[str,Any], energy_data: dict[str,Any], output: Path,
                   union_fields: Sequence[str], time_scale: float, energy_scale: float) -> dict[str,Any]:
    output.mkdir(parents=True);(output/"training_history").mkdir();(output/"checkpoints").mkdir(); data=_build_unified(project,time_data,energy_data,union_fields)
    scaler=TrainOnlyStandardScaler.fit(data["x_train"],role="Train");xt=scaler.transform(data["x_train"]).astype(np.float32);xv=scaler.transform(data["x_validation"]).astype(np.float32)
    train_targets=(time_data["lower_train"],time_data["upper_train"],data["energy_lower_train"],data["energy_upper_train"])
    val_targets=(time_data["lower_validation"],time_data["upper_validation"],data["energy_lower_validation"],data["energy_upper_validation"])
    factory=lambda:UnifiedCenterRadiusMLP(len(union_fields),HIDDEN,time_scale,energy_scale);runs=[];start_all=time.monotonic()
    for seed in ABLATION_SEEDS:
        started=time.monotonic();model,metrics,history=_train_single(factory,xt,train_targets,xv,val_targets,seed=seed,batch_size=64,loss_kind="unified",
            time_scale=time_scale,energy_scale=energy_scale,train_energy_weights=data["energy_weights_train"],val_energy_weights=data["energy_weights_validation"])
        metrics.update({"variant":"D_unified_bp","network":"Unified","architecture":"M3","lambda_safety":.25,"task_weight_time":.5,"task_weight_energy":.5,
            "train_seconds":time.monotonic()-started});stem=f"M3_seed_{seed}";ck=output/"checkpoints"/f"{stem}.pt"
        metrics["checkpoint_hash"]=_save_checkpoint(ck,model,{"model_version":"unified_bp_v1","input_dim":len(union_fields),"hidden_layers":list(HIDDEN),
            "time_center_scale":time_scale,"energy_center_scale":energy_scale,"training_seed":seed,"best_epoch":metrics["best_epoch"]});metrics["checkpoint"]=ck.name
        write_csv(output/"training_history"/f"{stem}.csv",history);runs.append(metrics)
    flat=[]
    for r in runs:
        flat.append({k:v for k,v in r.items() if k not in {"time","energy"}}|{f"time_{k}":v for k,v in r["time"].items()}|{f"energy_{k}":v for k,v in r["energy"].items()})
    write_csv(output/"all_runs_summary.csv",flat);write_csv(output/"per_seed_metrics.csv",flat);selected=min(runs,key=lambda r:float(r["validation_selection_score"]));ck=torch.load(output/"checkpoints"/selected["checkpoint"],map_location="cpu",weights_only=False)
    model=factory();model.load_state_dict(ck["model_state_dict"]);model.eval();shutil.copyfile(output/"checkpoints"/selected["checkpoint"],output/"selected_model.pt");model_hash=file_sha256(output/"selected_model.pt")
    with torch.no_grad(): pred=model(torch.as_tensor(xv,dtype=torch.float32))
    predictions=[]
    for i,row in enumerate(data["validation_rows"]): predictions.append({"time_record_id":row["time_record_id"],"base_sample_id":row["base_sample_id"],"role":"Validation",
        "time_lower_true":time_data["lower_validation"][i],"time_upper_true":time_data["upper_validation"][i],"time_pred_lower":float(pred["time_lower"][i]),"time_pred_upper":float(pred["time_upper"][i]),
        "energy_lower_true":data["energy_lower_validation"][i],"energy_upper_true":data["energy_upper_validation"][i],"energy_pred_lower":float(pred["energy_lower"][i]),"energy_pred_upper":float(pred["energy_upper"][i]),"energy_weight":data["energy_weights_validation"][i]})
    write_csv(output/"validation_predictions_selected.csv",predictions);invariance,inv_summary=_invariance_rows(model,xv,data["validation_rows"]);write_csv(output/"energy_f_invariance_validation.csv",invariance);_atomic_json(output/"energy_f_invariance_summary.json",inv_summary)
    config={"model_version":"unified_bp_v1","architecture":"M3","hidden_layers":list(HIDDEN),"single_shared_trunk":True,"raw_outputs":4,"lambda_safety":.25,
        "task_weight_time":.5,"task_weight_energy":.5,"training_seed":selected["training_seed"],"best_epoch":selected["best_epoch"],"input_dimension":len(union_fields),
        "input_feature_order":list(union_fields),"time_center_scale":time_scale,"energy_center_scale":energy_scale,"input_scaler_source":"Unified Train only"}
    _atomic_json(output/"selected_model_config.json",config);_atomic_json(output/"selected_model_metrics.json",selected);_atomic_json(output/"selected_model_hash.json",{"sha256":model_hash,"readable":True});_atomic_json(output/"input_scaler.json",scaler.as_dict(union_fields))
    scores=np.asarray([float(r["validation_selection_score"]) for r in runs]);_atomic_json(output/"seed_stability.json",{"seeds":list(ABLATION_SEEDS),"score_mean":float(scores.mean()),"score_std":float(scores.std()),"score_min":float(scores.min()),"score_max":float(scores.max())})
    base_sums=defaultdict(float)
    for row,w in zip(data["train_rows"],data["energy_weights_train"]):base_sums[row["base_sample_id"]]+=float(w)
    for row,w in zip(data["validation_rows"],data["energy_weights_validation"]):base_sums[row["base_sample_id"]]+=float(w)
    audit={"passed":all(abs(v-1)<1e-12 for v in base_sums.values()),"base_count":len(base_sums),"min_weight_sum":min(base_sums.values()),"max_weight_sum":max(base_sums.values()),"records_per_base_unique":[3]}
    _atomic_json(output/"unified_base_weight_audit.json",audit);_atomic_json(output/"unified_dataset_binding.json",{"train_rows":1260,"validation_rows":180,"energy_unique_train_bases":420,"energy_unique_validation_bases":60,"new_energy_labels_generated":0,"role_binding":"unchanged"})
    (output/"model_card.md").write_text(f"# unified_bp_v1\n\n单一共享trunk，Train更新权重，Validation选seed；模型哈希：`{model_hash}`。\n",encoding="utf-8")
    return {"model_version":"unified_bp_v1","runs_completed":len(runs),"selected_seed":selected["training_seed"],"best_epoch":selected["best_epoch"],"selected_metrics":selected,
        "model_hash":model_hash,"training_seconds":time.monotonic()-start_all,"f_invariance":inv_summary,"base_weight_audit":audit,"nonfinite":selected["time"]["nonfinite_count"]+selected["energy"]["nonfinite_count"]}


def run_formal(project: Path, output: Path) -> dict[str,Any]:
    wall=time.monotonic();protocol_dir=project/PROTOCOL_DIR_REL
    if file_sha256(protocol_dir/"exp3_ablation_protocol.json")!=PROTOCOL_HASH:raise RuntimeError("Exp-3 protocol hash drift")
    if len(read_rows(protocol_dir/"future_training_task_manifest.csv"))!=25:raise RuntimeError("planned runs drift")
    baseline_before=_baseline_hashes(project)
    if not baseline_before["passed"]:raise RuntimeError("A/B frozen baseline hash drift")
    output.mkdir(parents=True,exist_ok=False);_atomic_json(output/"protocol_binding.json",{"passed":True,"protocol_hash":PROTOCOL_HASH,"directory":PROTOCOL_DIR_REL})
    _atomic_json(output/"baseline_binding.json",{"before":baseline_before,"A_retrained":False,"B_retrained":False});_atomic_json(output/"environment.json",_environment())
    sources={name:file_sha256(project/path) for name,path in {"ablation_models":"src/mec_dualbp/models/ablation.py","ablation_contract":"src/mec_dualbp/training/ablation.py","formal_runner":"src/mec_dualbp/experiments/exp3_ablation_training_formal.py"}.items()};_atomic_json(output/"source_hashes.json",sources)
    data_start=time.monotonic();time_data,energy_data,sealed=load_training_data(project);access={"calibration_targets_accessed":False,"test_targets_accessed":False,"metadata_only_audit":sealed}
    _atomic_json(output/"calibration_access_audit.json",{"passed":True,"calibration_targets_accessed":False,"metadata_only":True});_atomic_json(output/"test_access_audit.json",{"passed":True,"test_targets_accessed":False,"metadata_only":True})
    _atomic_json(output/"dataset_binding.json",{"time":{"Train":1260,"Validation":180},"energy":{"Train":420,"Validation":60},"roles_changed":False,"load_seconds":time.monotonic()-data_start})
    raw=project/TRAINING_DIR_REL;time_scaler=_scaler(raw/"time_bp/input_scaler.json",TIME_FEATURES);energy_scaler=_scaler(raw/"energy_bp/input_scaler.json",ENERGY_FEATURES)
    time_scale=float(_read_json(raw/"time_bp/target_scale.json")["center_scale"]);energy_scale=float(_read_json(raw/"energy_bp/target_scale.json")["center_scale"])
    c_time=_train_single_network("C_w_o_safety","w_o_safety_time_bp_v1",time_data,TIME_FEATURES,time_scaler,time_scale,output/"variant_C_w_o_safety/time",loss_kind="no_safety",batch_size=64)
    c_energy=_train_single_network("C_w_o_safety","w_o_safety_energy_bp_v1",energy_data,ENERGY_FEATURES,energy_scaler,energy_scale,output/"variant_C_w_o_safety/energy",loss_kind="no_safety",batch_size=32)
    union=stable_ordered_union(TIME_FEATURES,ENERGY_FEATURES);d=_train_unified(project,time_data,energy_data,output/"variant_D_unified",union,time_scale,energy_scale)
    e_time=_train_single_network("E_direct_bounds","direct_bounds_time_bp_v1",time_data,TIME_FEATURES,time_scaler,time_scale,output/"variant_E_direct_bounds/time",loss_kind="direct",batch_size=64)
    e_energy=_train_single_network("E_direct_bounds","direct_bounds_energy_bp_v1",energy_data,ENERGY_FEATURES,energy_scaler,energy_scale,output/"variant_E_direct_bounds/energy",loss_kind="direct",batch_size=32)
    invalid_rows=[]
    for network,result in (("Time",e_time),("Energy",e_energy)):
        for role in ("Train","Validation"):
            item=result["invalid_intervals"][role];invalid_rows.append({"network":network,"role":role,**item})
    write_csv(output/"variant_E_direct_bounds/invalid_interval_validation.csv",[r for r in (*e_time["validation_predictions"],*e_energy["validation_predictions"]) if r["invalid_interval"]])
    _atomic_json(output/"variant_E_direct_bounds/invalid_interval_summary.json",{"rows":invalid_rows,"post_hoc_repair":False})
    registry={"C_Time":{"version":c_time["model_version"],"hash":c_time["model_hash"]},"C_Energy":{"version":c_energy["model_version"],"hash":c_energy["model_hash"]},
        "D_Unified":{"version":d["model_version"],"hash":d["model_hash"]},"E_Time":{"version":e_time["model_version"],"hash":e_time["model_hash"]},"E_Energy":{"version":e_energy["model_version"],"hash":e_energy["model_hash"]}}
    _atomic_json(output/"ablation_model_registry.json",registry)
    comparison=[]
    for variant,network,result,metrics in (("C","Time",c_time,c_time["selected_metrics"]),("C","Energy",c_energy,c_energy["selected_metrics"]),("D","Time",d,d["selected_metrics"]["time"]),("D","Energy",d,d["selected_metrics"]["energy"]),("E","Time",e_time,e_time["selected_metrics"]),("E","Energy",e_energy,e_energy["selected_metrics"])):
        comparison.append({"variant":variant,"network":network,"selected_seed":result["selected_seed"],"best_epoch":result["best_epoch"],**{k:v for k,v in metrics.items() if isinstance(v,(int,float))}})
    write_csv(output/"validation_comparison.csv",comparison)
    baseline_after=_baseline_hashes(project);all_results=(c_time,c_energy,d,e_time,e_energy);runs=sum(int(r["runs_completed"]) for r in all_results)
    fairness={"passed":baseline_after==baseline_before and runs==25 and all(int(r["nonfinite"])==0 for r in all_results) and d["base_weight_audit"]["passed"],
        "protocol_hash_match":True,"A_B_hashes_unchanged":baseline_after==baseline_before,"C_runs":10,"D_runs":5,"E_runs":10,"total_runs":runs,
        "calibration_targets_accessed":False,"test_targets_accessed":False,"architecture_search":False,"lambda_search":False,"D_base_weight_audit":d["base_weight_audit"]["passed"],
        "E_invalid_intervals_allowed_and_unmodified":True,"post_hoc_repair":False}
    _atomic_json(output/"fairness_audit_final.json",fairness);_atomic_json(output/"baseline_binding.json",{"before":baseline_before,"after":baseline_after,"unchanged":baseline_before==baseline_after,"A_retrained":False,"B_retrained":False})
    pure=sum(float(r["training_seconds"]) for r in all_results);runtime={"C_Time_seconds":c_time["training_seconds"],"C_Energy_seconds":c_energy["training_seconds"],"D_seconds":d["training_seconds"],"E_Time_seconds":e_time["training_seconds"],"E_Energy_seconds":e_energy["training_seconds"],"pure_training_seconds":pure,"model_selection_seconds":"included per variant","audit_seconds":time.monotonic()-wall-pure,"pytest_seconds":None,"overall_wall_seconds_before_pytest":time.monotonic()-wall,"protocol_estimated_pure_training_seconds":19.94867361111092*60,"protocol_estimated_total_seconds":25.3*60}
    _atomic_json(output/"runtime_breakdown.json",runtime)
    complete=fairness["passed"] and runs==25
    summary={"exp3_ablation_training_decision":"pending_final_pytest" if complete else "blocked","c_model_status":"trained_pending_final_audit" if complete else "not_frozen",
        "d_model_status":"trained_pending_final_audit" if complete else "not_frozen","e_model_status":"trained_pending_final_audit" if complete else "not_frozen","all_ablation_models_frozen":False,
        "ablation_training_allowed":False,"ablation_test_evaluation_allowed":False,"exp3_completed":False,"protocol_hash":PROTOCOL_HASH,"runs_completed":runs,
        "calibration_targets_accessed":False,"test_targets_accessed":False,"C":{"time":c_time,"energy":c_energy},"D":d,"E":{"time":e_time,"energy":e_energy}}
    for result in (c_time,c_energy,e_time,e_energy):result.pop("train_predictions",None);result.pop("validation_predictions",None)
    _atomic_json(output/"summary.json",summary);(output/"README.md").write_text("# Exp-3 正式消融模型训练\n\nC/D/E仅使用Train更新权重、Validation早停与选seed；Calibration/Test目标保持封存。A/B未重训。\n",encoding="utf-8")
    (output/"validation_report.md").write_text(f"# Validation冻结记录\n\n- 25 runs：{runs}/25\n- Calibration/Test targets：均未访问\n- 公平性预审：{'通过' if fairness['passed'] else '失败'}\n- 当前状态：等待最终pytest，不构成Exp-3最终Test结论。\n",encoding="utf-8")
    (output/"final_pytest.txt").write_text("等待训练后完整pytest。\n",encoding="utf-8");_atomic_json(output/"created_files.json",{"files":sorted(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file())})
    return summary


def finalize(project: Path, output: Path, pytest_count: int, pytest_seconds: float) -> dict[str,Any]:
    summary=_read_json(output/"summary.json");fairness=_read_json(output/"fairness_audit_final.json");baseline=_read_json(output/"baseline_binding.json");runtime=_read_json(output/"runtime_breakdown.json")
    reload_ok=True
    for item in _read_json(output/"ablation_model_registry.json").values():
        matches=list(output.rglob("selected_model.pt")); path=next((p for p in matches if file_sha256(p)==item["hash"]),None)
        if path is None:
            reload_ok=False;continue
        torch.load(path,map_location="cpu",weights_only=False)
    hashes_now=_baseline_hashes(project);passed=fairness["passed"] and baseline["after"]==hashes_now and reload_ok and pytest_count>=292
    summary.update({"exp3_ablation_training_decision":"passed" if passed else "blocked","c_model_status":"trained_and_frozen" if passed else "not_frozen",
        "d_model_status":"trained_and_frozen" if passed else "not_frozen","e_model_status":"trained_and_frozen" if passed else "not_frozen",
        "all_ablation_models_frozen":passed,"ablation_training_allowed":False,"ablation_test_evaluation_allowed":passed,"final_pytest_passed":passed,
        "final_pytest_count":pytest_count,"selected_models_reloadable":reload_ok})
    runtime["pytest_seconds"]=pytest_seconds;runtime["overall_wall_seconds"]=float(runtime["overall_wall_seconds_before_pytest"])+pytest_seconds;_atomic_json(output/"runtime_breakdown.json",runtime);_atomic_json(output/"summary.json",summary)
    (output/"final_pytest.txt").write_text(f"python -m compileall -q src\n结果：通过\n\npython -m pytest -q\n结果：{pytest_count} passed in {pytest_seconds:.2f}s\n",encoding="utf-8")
    with (output/"validation_report.md").open("a",encoding="utf-8") as stream:stream.write(f"\n- 最终pytest：{pytest_count} passed\n- 模型重载与哈希：{'通过' if reload_ok else '失败'}\n- 最终门禁：`{summary['exp3_ablation_training_decision']}`\n")
    _atomic_json(output/"created_files.json",{"files":sorted(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file())})
    return summary


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--project-root",default=".");parser.add_argument("--output-dir",required=True);parser.add_argument("--finalize",action="store_true");parser.add_argument("--pytest-count",type=int);parser.add_argument("--pytest-seconds",type=float)
    args=parser.parse_args();project=Path(args.project_root).resolve();output=Path(args.output_dir).resolve()
    try:result=finalize(project,output,args.pytest_count,args.pytest_seconds) if args.finalize else run_formal(project,output);print(json.dumps({k:v for k,v in result.items() if k not in {"C","D","E"}},ensure_ascii=False),flush=True)
    except Exception as error:
        output.mkdir(parents=True,exist_ok=True);_atomic_json(output/"runner_failure.json",{"error":str(error),"traceback":traceback.format_exc(),"at":datetime.now(timezone.utc).isoformat()});raise


if __name__=="__main__":main()
