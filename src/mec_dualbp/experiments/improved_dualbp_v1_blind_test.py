"""One-shot blind Test, raw gate, and conditional calibration for Improved DualBP V1."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mec_dualbp.datasets.protocol import ENERGY_FEATURES, TIME_FEATURES, write_csv
from mec_dualbp.experiments.exp1_formal_protocol import file_sha256
from mec_dualbp.experiments.exp1_preview_protocol import canonical_json_hash
from mec_dualbp.models.dualbp import ARCHITECTURES, ConstrainedCenterRadiusMLP
from mec_dualbp.models.improved_dualbp import (
    DESIGN_HASH, ImprovedResidualDualBP, energy_physical_baseline_j,
    time_physical_baseline_s,
)
from mec_dualbp.training.calibration import (
    apply_directional_correction, conservative_alpha, interval_coverage,
    lower_over_error, upper_under_error,
)
from mec_dualbp.training.dualbp import TrainOnlyStandardScaler, f_stratum, regression_metrics
from mec_dualbp.utils.sharded_checkpoint import _atomic_json

IMPLEMENTATION_HASH = "c69188a81519a2eac7959c9e95cd5509ebe8a545f2a4983299f9538b3164b5da"
TRAINING_REL = "outputs/01_formal/improved_dualbp_v1_5seed_training_20260818_235906"
ORIGINAL_REL = "outputs/01_formal/dualbp_training_formal_20260811_003126"
TIME_DATA_REL = "outputs/01_formal/time_bp_dataset_v2_labels_formal_20260810_194719"
ENERGY_DATA_REL = "outputs/01_formal/bp_dataset_labels_formal_20260810_132254"
SEEDS = (31001, 31002, 31003, 31004, 31005)
ORIGINAL_FORMAL_HASHES = {"Time":"291c5733bebdb4cb0c2cf7a4a07138ca6bb07c3ad75bffb378aa09b7f657d827",
                          "Energy":"8481dfdca0e03e3d6bebe4c4d5808259d16f7cdd67f4a65251fe7aad3e07d971"}
TEST_HASHES = {"Time":"ff1ba0c130274c7df17574088095d51a6bfe231bd745386e7048c1c196002b0e",
               "Energy":"c74ff147cdf395e9866e363e4ccd2b2555c32b8a963a0975cf8f8df931a7c856"}
CAL_HASHES = {"Time":"98ffd70d01a188794ace3b2fe988727c6aac8ee58963c0dea1c31fe001b89ed7",
              "Energy":"dfb7ddd70300651f434e5804ec6d7e61dfacc1bfeb36a7f6ef22d1657a940de9"}


def _rows(path: Path) -> list[dict[str,str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream: return list(csv.DictReader(stream))


def _matrix(rows: Sequence[Mapping[str,str]], fields: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(row[field]) for field in fields] for row in rows], dtype=np.float64)


def _scaler(payload: Mapping[str,Any]) -> TrainOnlyStandardScaler:
    return TrainOnlyStandardScaler(tuple(payload["mean"]),tuple(payload["scale"]),tuple(payload["constant_feature"]),payload["fitted_role"])


def _metric_side(true: np.ndarray, pred: np.ndarray) -> dict[str,float]:
    error=pred-true; sst=float(np.sum((true-true.mean())**2))
    return {"MAE":float(np.mean(np.abs(error))),"RMSE":float(np.sqrt(np.mean(error**2))),
      "WMAPE":float(np.sum(np.abs(error))/np.sum(np.abs(true))),
      "R2":1-float(np.sum(error**2))/max(sst,1e-12),"one_minus_R2":float(np.sum(error**2))/max(sst,1e-12)}


def _load_original(project: Path, network: str, seed: int) -> tuple[torch.nn.Module,TrainOnlyStandardScaler]:
    folder="time_bp" if network=="Time" else "energy_bp"; base=project/ORIGINAL_REL/folder
    checkpoint=torch.load(base/"checkpoints"/f"M3_lambda_0.25_seed_{seed}.pt",map_location="cpu",weights_only=False)
    model=ConstrainedCenterRadiusMLP(checkpoint["input_dim"],ARCHITECTURES["M3"],checkpoint["center_scale"])
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    return model,_scaler(json.loads((base/"input_scaler.json").read_text(encoding="utf-8")))


def _load_improved(project: Path, network: str, seed: int) -> tuple[torch.nn.Module,TrainOnlyStandardScaler]:
    path=project/TRAINING_REL/"checkpoints"/f"improved_{network.lower()}_seed_{seed}.pt"
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    model=ImprovedResidualDualBP(checkpoint["input_dim"],checkpoint["delta_center_scale"])
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    return model,_scaler(checkpoint["input_scaler"])


def _infer_original(model,scaler,x):
    with torch.inference_mode(): _,_,lo,up=model(torch.as_tensor(scaler.transform(x).astype(np.float32)))
    return lo.numpy().astype(np.float64),up.numpy().astype(np.float64)


def _infer_improved(model,scaler,x,network):
    baseline=(time_physical_baseline_s if network=="Time" else energy_physical_baseline_j)(x)
    with torch.inference_mode(): _,_,lo,up=model(torch.as_tensor(scaler.transform(x).astype(np.float32)),torch.as_tensor(baseline,dtype=torch.float32))
    return lo.numpy().astype(np.float64),up.numpy().astype(np.float64)


def _dataset(project: Path, role: str) -> dict[str,dict[str,Any]]:
    td,ed=project/TIME_DATA_REL,project/ENERGY_DATA_REL
    time_file=td/f"time_{role.lower()}_labels.csv"; energy_file=ed/f"{role.lower()}_labels.csv"
    time_rows,energy_rows=_rows(time_file),_rows(energy_file)
    features={r["sample_id"]:json.loads(r["energy_features"]) for r in _rows(ed/"feature_records.csv") if r["role"]==role}
    return {"Time":{"rows":time_rows,"x":_matrix(time_rows,TIME_FEATURES),"lower":np.asarray([float(r["time_lower"]) for r in time_rows]),"upper":np.asarray([float(r["time_upper"]) for r in time_rows])},
      "Energy":{"rows":energy_rows,"x":np.asarray([features[r["sample_id"]] for r in energy_rows],dtype=np.float64),"lower":np.asarray([float(r["energy_lower"]) for r in energy_rows]),"upper":np.asarray([float(r["energy_upper"]) for r in energy_rows])}}


def freeze_unblinding(project: Path, output: Path) -> dict[str,Any]:
    output.mkdir(parents=True,exist_ok=False); training=project/TRAINING_REL
    design_actual=file_sha256(project/"outputs/05_audit/improved_dualbp_v1_design_freeze_20260818_230450/improved_dualbp_v1_contract.json")
    implementation=json.loads((project/"outputs/06_implementation/improved_dualbp_v1_implementation_20260818_235330/implementation_hash.json").read_text(encoding="utf-8"))["improved_dualbp_v1_implementation_hash"]
    manifest=json.loads((training/"checkpoint_hash_manifest.json").read_text(encoding="utf-8"))["models"]
    for row in manifest:
        if file_sha256(training/row["checkpoint_path"])!=row["checkpoint_sha256"]: raise RuntimeError("Improved checkpoint hash drift")
    candidates=json.loads((training/"formal_candidate_models.json").read_text(encoding="utf-8"))
    binding={"test_unblinding_started":True,"timestamp":datetime.now(timezone.utc).isoformat(),"design_hash":design_actual,
      "implementation_hash":implementation,"improved_model_hashes":[{"network":r["network"],"seed":r["seed"],"model_hash":r["model_hash"],"checkpoint_sha256":r["checkpoint_sha256"]} for r in manifest],
      "formal_candidates":candidates,"test_dataset_hashes":TEST_HASHES,"model_modification_allowed_after_unblinding":False,
      "candidate_change_allowed_after_unblinding":False,"Original_retraining_allowed":False}
    if design_actual!=DESIGN_HASH or implementation!=IMPLEMENTATION_HASH or len(manifest)!=10: raise RuntimeError("frozen binding mismatch")
    _atomic_json(output/"blind_test_binding.json",binding)
    return binding


def _aggregate_by_seed(rows: list[dict[str,Any]]) -> tuple[list[dict[str,Any]],list[dict[str,Any]],int]:
    main=[];paired=[];passed=0
    for network in ("Time","Energy"):
      for boundary in ("lower","upper"):
       for metric in ("MAE","RMSE","WMAPE","R2"):
        o=np.asarray([r["value"] for r in rows if r["network"]==network and r["boundary"]==boundary and r["metric"]==metric and r["model"]=="Original BP"])
        i=np.asarray([r["value"] for r in rows if r["network"]==network and r["boundary"]==boundary and r["metric"]==metric and r["model"]=="Improved BP"])
        direction=bool(i.mean()>o.mean()) if metric=="R2" else bool(i.mean()<o.mean()); passed+=int(direction)
        relative=(float((o.mean()-i.mean())/o.mean()*100) if metric!="R2" else float(((1-o.mean())-(1-i.mean()))/(1-o.mean())*100))
        main.append({"network":network,"boundary":boundary,"metric":metric,"Original_mean":o.mean(),"Original_std":o.std(),"Original_median":np.median(o),"Improved_mean":i.mean(),"Improved_std":i.std(),"Improved_median":np.median(i),"absolute_difference":i.mean()-o.mean(),"relative_improvement_percent":relative,"direction_pass":direction})
        diff=i-o;paired.append({"network":network,"boundary":boundary,"metric":metric,"paired_difference_mean":diff.mean(),"paired_difference_std":diff.std(),"paired_improved_seed_count":int(np.sum(diff>0) if metric=="R2" else np.sum(diff<0)),"seed_count":5})
    return main,paired,passed


def _group_summary(data,predictions,selector_name,selector_values):
    output=[]
    for network in predictions:
      values=np.asarray(selector_values[network]);
      for group in sorted(set(values)):
       idx=values==group
       for model in ("Original BP","Improved BP"):
        for boundary in ("lower","upper"):
         per=[]
         for seed in SEEDS: per.append(_metric_side(data[network][boundary][idx],predictions[network][model][seed][boundary][idx]))
         for metric in ("MAE","RMSE","WMAPE","R2"):
          a=np.asarray([x[metric] for x in per]); output.append({"network":network,selector_name:group,"boundary":boundary,"model":model,"metric":metric,"mean":a.mean(),"std":a.std(),"seed_count":5})
    return output


def run(project: Path, output: Path) -> dict[str,Any]:
    started=time.monotonic(); binding=freeze_unblinding(project,output)
    # First semantic access to Test labels occurs only after binding is durable.
    data=_dataset(project,"Test"); predictions={"Time":{"Original BP":{},"Improved BP":{}},"Energy":{"Original BP":{},"Improved BP":{}}}; by_seed=[]
    for network in ("Time","Energy"):
      for seed in SEEDS:
       om,os=_load_original(project,network,seed); im,iscl=_load_improved(project,network,seed)
       olo,oup=_infer_original(om,os,data[network]["x"]); ilo,iup=_infer_improved(im,iscl,data[network]["x"],network)
       predictions[network]["Original BP"][seed]={"lower":olo,"upper":oup};predictions[network]["Improved BP"][seed]={"lower":ilo,"upper":iup}
       for model,lo,up in (("Original BP",olo,oup),("Improved BP",ilo,iup)):
        for boundary,true,pred in (("lower",data[network]["lower"],lo),("upper",data[network]["upper"],up)):
         m=_metric_side(true,pred)
         for metric in ("MAE","RMSE","WMAPE","R2"): by_seed.append({"network":network,"boundary":boundary,"seed":seed,"model":model,"metric":metric,"value":m[metric],"one_minus_R2":m["one_minus_R2"] if metric=="R2" else ""})
    write_csv(output/"raw_accuracy_by_seed.csv",by_seed)
    main,paired,passed=_aggregate_by_seed(by_seed);write_csv(output/"raw_accuracy_main_table.csv",main);write_csv(output/"paired_improvement_summary.csv",paired)
    paper=[];r2=[]
    for network in ("Time","Energy"):
     for boundary in ("lower","upper"):
      for model in ("Original BP","Improved BP"):
       record={"网络":network,"边界":boundary,"模型":model}
       for metric in ("MAE","RMSE","WMAPE","R2"):
        row=next(r for r in main if r["network"]==network and r["boundary"]==boundary and r["metric"]==metric); prefix="Original" if model=="Original BP" else "Improved"
        record[metric]=f"{row[prefix+'_mean']:.12g} ± {row[prefix+'_std']:.12g}"
       paper.append(record)
      row=next(r for r in main if r["network"]==network and r["boundary"]==boundary and r["metric"]=="R2")
      r2.append({"network":network,"boundary":boundary,"Original_R2":row["Original_mean"],"Improved_R2":row["Improved_mean"],"Delta_R2":row["Improved_mean"]-row["Original_mean"],"Original_one_minus_R2":1-row["Original_mean"],"Improved_one_minus_R2":1-row["Improved_mean"],"relative_unexplained_variance_reduction_percent":row["relative_improvement_percent"],"direction_pass":row["direction_pass"]})
    write_csv(output/"table_original_vs_improved_dualbp.csv",paper);write_csv(output/"table_r2_improvement.csv",r2)
    candidate=[]
    for network in ("Time","Energy"):
     seed=31005
     for boundary in ("lower","upper"):
      m=_metric_side(data[network][boundary],predictions[network]["Improved BP"][seed][boundary])
      candidate.append({"network":network,"boundary":boundary,"seed":seed,**m})
    write_csv(output/"formal_candidate_test_metrics.csv",candidate)
    near_values={n:np.asarray([r["sample_type"] for r in data[n]["rows"]]) for n in data}
    near=_group_summary(data,predictions,"sample_type",near_values);near=[r for r in near if r["sample_type"]=="near_degenerate"]
    write_csv(output/"near_degenerate_test_summary.csv",near)
    cpu_values={"Time":np.asarray([f_stratum(float(r["edge_cpu_frequency_cycle_per_s"])) for r in data["Time"]["rows"]])}
    cpu=_group_summary({"Time":data["Time"]},{"Time":predictions["Time"]},"f_stratum",cpu_values)
    write_csv(output/"time_cpu_strata_test_summary.csv",cpu)
    failures=[r for r in main if not r["direction_pass"]];gate={"raw_accuracy_gate":"PASS" if passed==16 else "FAIL","direction_checks_passed":passed,"direction_checks_total":16,"failures":failures,"five_seed_paired_complete":True,"test_unblinded":True,"candidate_seeds_unchanged":True,"Calibration_allowed":passed==16}
    _atomic_json(output/"raw_accuracy_gate.json",gate)
    if passed!=16:
      (output/"README.md").write_text("# Improved DualBP V1 Blind Test\n\n五种子配对Test已完成；Raw联合精度门失败，按协议未访问Calibration、未运行Safety Correction。\n",encoding="utf-8")
      files=sorted(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file());_atomic_json(output/"created_files.json",{"files":files+["created_files.json"],"count":len(files)+1})
      return {**gate,"Calibration_used":False,"Safety_evaluated":False,"runtime_seconds":time.monotonic()-started}
    # Gate passed: use only frozen candidates and the unchanged calibration rule.
    cal=_dataset(project,"Calibration"); alpha={}; cal_pred={}
    for network in ("Time","Energy"):
      model,scaler=_load_improved(project,network,31005);lo,up=_infer_improved(model,scaler,cal[network]["x"],network);cal_pred[network]=(lo,up)
      am,rank_m=conservative_alpha(lower_over_error(cal[network]["lower"],lo));ap,rank_p=conservative_alpha(upper_under_error(cal[network]["upper"],up))
      alpha[network]={"alpha_minus":am,"alpha_plus":ap,"rank_minus":rank_m,"rank_plus":rank_p,"n":len(lo)}
    (output/"calibration").mkdir();_atomic_json(output/"calibration/new_alpha.json",{"delta":.05,"rule":"finite-sample conservative higher order statistic","Time":alpha["Time"],"Energy":alpha["Energy"],"old_alpha_reused":False,"candidate_seed":31005,"calibration_role_hashes":CAL_HASHES})
    safety=[];safety_table=[];freeze_metrics={}
    for network in ("Time","Energy"):
      raw_lo=predictions[network]["Improved BP"][31005]["lower"];raw_up=predictions[network]["Improved BP"][31005]["upper"]
      cor_lo,cor_up=apply_directional_correction(raw_lo,raw_up,alpha[network]["alpha_minus"],alpha[network]["alpha_plus"])
      freeze_metrics[network]={}
      for variant,lo,up in (("Improved Raw",raw_lo,raw_up),("Improved Corrected",cor_lo,cor_up)):
       m=regression_metrics(data[network]["lower"],data[network]["upper"],lo,up,float(np.median((data[network]["lower"]+data[network]["upper"])*.5)))
       m.pop("uer", None);m.pop("ler", None)
       m.update({"UER":float(np.mean(up<data[network]["upper"])),"LER":float(np.mean(lo>data[network]["lower"])),"ICR":interval_coverage(data[network]["lower"],data[network]["upper"],lo,up)})
       safety.append({"network":network,"variant":variant,**m});safety_table.append({"network":network,"variant":variant,"UER":m["UER"],"LER":m["LER"],"ICR":m["ICR"]});freeze_metrics[network][variant]=m
    (output/"safety").mkdir();write_csv(output/"safety/improved_raw_vs_corrected.csv",safety);write_csv(output/"safety/table_improved_safety.csv",safety_table)
    corrected_identity={"version":"improved_corrected_dualbp_v1","design_hash":DESIGN_HASH,"implementation_hash":IMPLEMENTATION_HASH,"time_model_hash":binding["formal_candidates"]["Time"]["model_hash"],"energy_model_hash":binding["formal_candidates"]["Energy"]["model_hash"],"alpha":alpha,"calibration_role_hashes":CAL_HASHES,"correction_formula":"lower/(1+alpha_minus); upper/(1-alpha_plus)"}
    corrected_hash=canonical_json_hash(corrected_identity);(output/"final_freeze").mkdir();_atomic_json(output/"final_freeze/improved_corrected_dualbp_hash.json",{"improved_corrected_dualbp_hash":corrected_hash,"identity":corrected_identity})
    formal_freeze={"status":"formally_frozen","design_hash":DESIGN_HASH,"implementation_hash":IMPLEMENTATION_HASH,"model_hashes":{"Time":binding["formal_candidates"]["Time"]["model_hash"],"Energy":binding["formal_candidates"]["Energy"]["model_hash"]},"dataset_hashes":{"Test":TEST_HASHES,"Calibration":CAL_HASHES},"alpha":alpha,"raw_accuracy_gate":gate,"formal_candidate_test_metrics":candidate,"safety_metrics":freeze_metrics,"improved_corrected_dualbp_hash":corrected_hash,"models_retrained":False,"sampling_changed":False,"downstream_optimization_allowed":False}
    _atomic_json(output/"final_freeze/improved_dualbp_formal_freeze.json",formal_freeze)
    (output/"README.md").write_text("# Improved DualBP V1 Blind Test与正式冻结\n\n五种子配对Test一次性揭盲；Raw 16项联合精度门通过后，使用冻结Calibration规则重新估计alpha并完成Safety评价。未训练模型、未修改采样。\n",encoding="utf-8")
    files=sorted(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file());_atomic_json(output/"created_files.json",{"files":files+["created_files.json"],"count":len(files)+1})
    return {**gate,"Calibration_used":True,"Safety_evaluated":True,"alpha":alpha,"improved_corrected_dualbp_hash":corrected_hash,"runtime_seconds":time.monotonic()-started}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--project-root",default=".");parser.add_argument("--output-dir",required=True);args=parser.parse_args()
    print(json.dumps(run(Path(args.project_root).resolve(),Path(args.output_dir).resolve()),ensure_ascii=False),flush=True)


if __name__=="__main__":main()
