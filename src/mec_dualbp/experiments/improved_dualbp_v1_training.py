"""Gate-chain smoke and five-seed Train/Validation-only Improved DualBP V1 training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mec_dualbp.datasets.protocol import ENERGY_FEATURES, TIME_FEATURES, write_csv
from mec_dualbp.experiments.exp1_formal_protocol import file_sha256
from mec_dualbp.models.improved_dualbp import (
    DESIGN_HASH, ENERGY_DELTA_CENTER_SCALE, ENERGY_RADIUS_SCALE, HIDDEN_LAYERS,
    LAMBDA_SAFETY, TIME_DELTA_CENTER_SCALE, TIME_RADIUS_SCALE, TRAINING_SEEDS,
    ImprovedResidualDualBP, energy_physical_baseline_j, parameter_count,
    time_physical_baseline_s,
)
from mec_dualbp.training.dualbp import TrainOnlyStandardScaler, regression_metrics, state_dict_sha256, validate_bounds
from mec_dualbp.training.improved_dualbp import train_improved_one, train_only_rms
from mec_dualbp.utils.sharded_checkpoint import _atomic_json

TIME_DIR = "outputs/01_formal/time_bp_dataset_v2_labels_formal_20260810_194719"
ENERGY_DIR = "outputs/01_formal/bp_dataset_labels_formal_20260810_132254"
DESIGN_CONTRACT = "outputs/05_audit/improved_dualbp_v1_design_freeze_20260818_230450/improved_dualbp_v1_contract.json"
ROLE_HASHES = {
 "time_train":"8f3a1900f78311ced38b506ddf5b862082531a3e3c09b59aa6881caafc8aabfd",
 "time_validation":"656b34d7d039f95273afce4da66ce0b61930ee8a6c60e76b0a609655640d93ff",
 "time_calibration":"98ffd70d01a188794ace3b2fe988727c6aac8ee58963c0dea1c31fe001b89ed7",
 "time_test":"ff1ba0c130274c7df17574088095d51a6bfe231bd745386e7048c1c196002b0e",
 "energy_train":"9f04013e5285816fc84079a46104b27e1d5fe5f5cef4d19f527d9d5953d80243",
 "energy_validation":"9efe2e903e44149cbed171433b8407c5cf14f6c1907c4de372042a5e38a23d89",
 "energy_calibration":"dfb7ddd70300651f434e5804ec6d7e61dfacc1bfeb36a7f6ef22d1657a940de9",
 "energy_test":"c74ff147cdf395e9866e363e4ccd2b2555c32b8a963a0975cf8f8df931a7c856"}
ORIGINAL_FORMAL_HASHES = {"Time":"291c5733bebdb4cb0c2cf7a4a07138ca6bb07c3ad75bffb378aa09b7f657d827",
                          "Energy":"8481dfdca0e03e3d6bebe4c4d5808259d16f7cdd67f4a65251fe7aad3e07d971"}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f: return list(csv.DictReader(f))


def _matrix(rows: Sequence[Mapping[str,str]], fields: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(r[f]) for f in fields] for r in rows], dtype=np.float64)


def _model_content_hash(model: torch.nn.Module) -> str:
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode()); a=tensor.detach().cpu().contiguous().numpy(); h.update(str(a.dtype).encode()); h.update(np.asarray(a.shape,dtype=np.int64).tobytes()); h.update(a.tobytes())
    return h.hexdigest()


def _verify_hash(path: Path, expected: str) -> None:
    actual=file_sha256(path)
    if actual != expected: raise RuntimeError(f"hash drift: {path}: {actual}")


def load_data(project: Path) -> dict[str, dict[str, Any]]:
    td,ed=project/TIME_DIR,project/ENERGY_DIR
    paths={"time_train":td/"time_train_labels.csv","time_validation":td/"time_validation_labels.csv",
      "time_calibration":td/"time_calibration_labels.csv","time_test":td/"time_test_labels.csv",
      "energy_train":ed/"train_labels.csv","energy_validation":ed/"validation_labels.csv",
      "energy_calibration":ed/"calibration_labels.csv","energy_test":ed/"test_labels.csv"}
    for key,path in paths.items(): _verify_hash(path,ROLE_HASHES[key])
    tr,tv=_rows(paths["time_train"]),_rows(paths["time_validation"])
    er,ev=_rows(paths["energy_train"]),_rows(paths["energy_validation"])
    # Feature records contain no labels; retain Train/Validation rows only.
    feature={r["sample_id"]:json.loads(r["energy_features"]) for r in _rows(ed/"feature_records.csv") if r["role"] in ("Train","Validation")}
    def pack(rows,fields,prefix,x):
        return {"rows":rows,"fields":fields,"x":x,"lower":np.asarray([float(r[f"{prefix}_lower"]) for r in rows]),"upper":np.asarray([float(r[f"{prefix}_upper"]) for r in rows])}
    return {"time_train":pack(tr,TIME_FEATURES,"time",_matrix(tr,TIME_FEATURES)),
      "time_validation":pack(tv,TIME_FEATURES,"time",_matrix(tv,TIME_FEATURES)),
      "energy_train":pack(er,ENERGY_FEATURES,"energy",np.asarray([feature[r["sample_id"]] for r in er],dtype=np.float64)),
      "energy_validation":pack(ev,ENERGY_FEATURES,"energy",np.asarray([feature[r["sample_id"]] for r in ev],dtype=np.float64))}


def prepare_network(data: dict[str,dict[str,Any]], network: str) -> dict[str,Any]:
    tr,va=data[f"{network}_train"],data[f"{network}_validation"]
    baseline_fn=time_physical_baseline_s if network=="time" else energy_physical_baseline_j
    btr,bva=baseline_fn(tr["x"]),baseline_fn(va["x"])
    center=(tr["lower"]+tr["upper"])*.5; radius=(tr["upper"]-tr["lower"])*.5
    ds=train_only_rms(center-btr,role="Train"); rs=train_only_rms(radius,role="Train")
    expected_ds=TIME_DELTA_CENTER_SCALE if network=="time" else ENERGY_DELTA_CENTER_SCALE
    expected_rs=TIME_RADIUS_SCALE if network=="time" else ENERGY_RADIUS_SCALE
    if not np.isclose(ds,expected_ds,rtol=0,atol=1e-15) or not np.isclose(rs,expected_rs,rtol=0,atol=1e-15): raise RuntimeError("frozen output scale mismatch")
    _,_,response_scale=validate_bounds(tr["lower"],tr["upper"]); validate_bounds(va["lower"],va["upper"])
    scaler=TrainOnlyStandardScaler.fit(tr["x"],role="Train")
    return {"train":tr,"validation":va,"baseline_train":btr,"baseline_validation":bva,
      "x_train":scaler.transform(tr["x"]).astype(np.float32),"x_validation":scaler.transform(va["x"]).astype(np.float32),
      "scaler":scaler,"delta_scale":ds,"radius_scale":rs,"response_scale":response_scale}


def run_smoke(project: Path, report: Path) -> dict[str,Any]:
    started=time.monotonic(); data=load_data(project); checks={}
    for network,batch in (("time",32),("energy",16)):
        p=prepare_network(data,network); ntr,nv=(64,24) if network=="time" else (48,12)
        model,metrics,history=train_improved_one(p["x_train"][:ntr],p["baseline_train"][:ntr],p["train"]["lower"][:ntr],p["train"]["upper"][:ntr],p["x_validation"][:nv],p["baseline_validation"][:nv],p["validation"]["lower"][:nv],p["validation"]["upper"][:nv],seed=31001,delta_center_scale=p["delta_scale"],radius_scale=p["radius_scale"],response_center_scale=p["response_scale"],batch_size=batch,max_epochs=30,early_stopping_patience=30)
        with tempfile.TemporaryDirectory() as temp:
            ck=Path(temp)/"smoke.pt"; torch.save({"model_state_dict":model.state_dict()},ck)
            clone=ImprovedResidualDualBP(len(p["train"]["fields"]),p["delta_scale"]); clone.load_state_dict(torch.load(ck,weights_only=False)["model_state_dict"])
            resumed=_model_content_hash(clone)==_model_content_hash(model)
        checks[network]={"passed":metrics["nonfinite_predictions"]==0 and metrics["invalid_intervals"]==0 and resumed and min(h["train_loss"] for h in history)<history[0]["train_loss"],"epochs":len(history),"first_loss":history[0]["train_loss"],"minimum_loss":min(h["train_loss"] for h in history),"validation_selection_score":metrics["validation_selection_score"],"checkpoint_resume":resumed,"model_hash":_model_content_hash(model)}
    passed=all(v["passed"] for v in checks.values()); payload={"passed":passed,"Train_Validation_only":True,"Calibration_used":False,"Test_used":False,"checks":checks,"seconds":time.monotonic()-started}
    report.write_text("# Improved DualBP V1工程Smoke\n\n```json\n"+json.dumps(payload,ensure_ascii=False,indent=2)+"\n```\n",encoding="utf-8")
    if not passed: raise RuntimeError("implementation smoke failed")
    return payload


def run_formal(project: Path, output: Path, implementation_hash: str) -> dict[str,Any]:
    started=time.monotonic(); output.mkdir(parents=True,exist_ok=False); (output/"checkpoints").mkdir()
    if file_sha256(project/DESIGN_CONTRACT)!=DESIGN_HASH: raise RuntimeError("design hash mismatch")
    data=load_data(project); all_rows=[]; hash_rows=[]
    for network,batch in (("time",64),("energy",32)):
        p=prepare_network(data,network)
        for seed in TRAINING_SEEDS:
            run_start=time.monotonic(); model,metrics,_=train_improved_one(p["x_train"],p["baseline_train"],p["train"]["lower"],p["train"]["upper"],p["x_validation"],p["baseline_validation"],p["validation"]["lower"],p["validation"]["upper"],seed=seed,delta_center_scale=p["delta_scale"],radius_scale=p["radius_scale"],response_center_scale=p["response_scale"],batch_size=batch)
            stem=f"improved_{network}_seed_{seed}"; ck=output/"checkpoints"/f"{stem}.pt"
            torch.save({"model_state_dict":model.state_dict(),"network":network,"seed":seed,"best_epoch":metrics["best_epoch"],"validation_selection_score":metrics["validation_selection_score"],"input_dim":len(p["train"]["fields"]),"hidden_layers":list(HIDDEN_LAYERS),"delta_center_scale":p["delta_scale"],"radius_scale":p["radius_scale"],"response_center_scale":p["response_scale"],"design_hash":DESIGN_HASH,"implementation_hash":implementation_hash,"input_scaler":p["scaler"].as_dict(p["train"]["fields"])},ck)
            checkpoint_hash=state_dict_sha256(ck); model_hash=_model_content_hash(model)
            row={"network":network.title(),"seed":seed,"best_epoch":metrics["best_epoch"],"epochs_ran":metrics["epochs_ran"],"validation_selection_score":metrics["validation_selection_score"],"lower_MAE":metrics["lower_mae"],"lower_RMSE":metrics["lower_rmse"],"lower_WMAPE":metrics["lower_wmape"],"lower_R2":metrics["lower_r2"],"upper_MAE":metrics["upper_mae"],"upper_RMSE":metrics["upper_rmse"],"upper_WMAPE":metrics["upper_wmape"],"upper_R2":metrics["upper_r2"],"parameter_count":metrics["parameter_count"],"delta_center_scale":p["delta_scale"],"radius_scale":p["radius_scale"],"checkpoint_path":ck.relative_to(output).as_posix(),"checkpoint_sha256":checkpoint_hash,"model_hash":model_hash,"design_hash":DESIGN_HASH,"implementation_hash":implementation_hash,"runtime_seconds":time.monotonic()-run_start,"status":"trained_and_frozen"}
            all_rows.append(row); hash_rows.append({k:row[k] for k in ("network","seed","checkpoint_path","checkpoint_sha256","model_hash")})
    write_csv(output/"per_seed_training_summary.csv",all_rows); write_csv(output/"improved_models_5seed_manifest.csv",all_rows)
    validation=[]; candidates={}
    for network in ("Time","Energy"):
        rows=[r for r in all_rows if r["network"]==network]
        validation.extend(rows); selected=min(rows,key=lambda r:(r["validation_selection_score"],r["seed"]))
        candidates[network]={k:selected[k] for k in ("checkpoint_path","seed","best_epoch","validation_selection_score","checkpoint_sha256","model_hash")}
    write_csv(output/"validation_summary.csv",validation); _atomic_json(output/"formal_candidate_models.json",candidates)
    _atomic_json(output/"checkpoint_hash_manifest.json",{"models":hash_rows,"count":len(hash_rows)})
    _atomic_json(output/"training_protocol_binding.json",{"design_hash":DESIGN_HASH,"implementation_hash":implementation_hash,"seeds":list(TRAINING_SEEDS),"architecture":"128-64-32 ReLU","lambda_safety":LAMBDA_SAFETY,"workers":1,"roles_used":["Train","Validation"],"Calibration_used":False,"Test_used":False,"Original_retrained":False})
    original_now={"Time":file_sha256(project/"outputs/01_formal/dualbp_training_formal_20260811_003126/time_bp/selected_model.pt"),"Energy":file_sha256(project/"outputs/01_formal/dualbp_training_formal_20260811_003126/energy_bp/selected_model.pt")}
    gate={"design_hash_match":True,"implementation_hash_bound":bool(implementation_hash),"models_trained":len(all_rows),"model_hashes_frozen":len(hash_rows),"time_candidate_frozen":"Time" in candidates,"energy_candidate_frozen":"Energy" in candidates,"test_unblinded":False,"Calibration_used":False,"sampling_changed":False,"Original_retrained":False,"Original_hashes_unchanged":original_now==ORIGINAL_FORMAL_HASHES}
    gate["blind_test_ready"]=all([gate["design_hash_match"],gate["implementation_hash_bound"],gate["models_trained"]==10,gate["model_hashes_frozen"]==10,gate["time_candidate_frozen"],gate["energy_candidate_frozen"],not gate["test_unblinded"],not gate["Calibration_used"],not gate["sampling_changed"],not gate["Original_retrained"],gate["Original_hashes_unchanged"]])
    gate["runtime_seconds"]=time.monotonic()-started; _atomic_json(output/"pre_test_gate.json",gate)
    (output/"README.md").write_text("# Improved DualBP V1五种子正式训练\n\n仅使用Train更新权重、Validation选择checkpoint；Calibration和Test均未用于本轮。全部模型绑定冻结design与implementation hash。\n",encoding="utf-8")
    files=sorted(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()); _atomic_json(output/"created_files.json",{"files":files+["created_files.json"],"count":len(files)+1})
    if not gate["blind_test_ready"]: raise RuntimeError("pre-Test gate failed")
    return gate


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--project-root",default=".");parser.add_argument("--smoke-report");parser.add_argument("--output-dir");parser.add_argument("--implementation-hash")
    args=parser.parse_args();project=Path(args.project_root).resolve()
    result=run_smoke(project,Path(args.smoke_report).resolve()) if args.smoke_report else run_formal(project,Path(args.output_dir).resolve(),str(args.implementation_hash))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__": main()
