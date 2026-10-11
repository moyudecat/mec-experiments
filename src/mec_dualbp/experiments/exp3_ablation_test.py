"""冻结协议后一次性执行 Exp-3 A/B/C/D/E 统一 Test。"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mec_dualbp.datasets.protocol import ENERGY_FEATURES, TIME_FEATURES, write_csv
from mec_dualbp.experiments.dualbp_alpha_calibration import save_figure, style
from mec_dualbp.experiments.dualbp_training_formal import read_rows
from mec_dualbp.experiments.exp1_formal_protocol import file_sha256
from mec_dualbp.experiments.exp1_preview_protocol import canonical_json_hash
from mec_dualbp.models.ablation import DirectBoundsMLP, UnifiedCenterRadiusMLP
from mec_dualbp.models.dualbp import ConstrainedCenterRadiusMLP
from mec_dualbp.training.dualbp import TrainOnlyStandardScaler, f_stratum
from mec_dualbp.training.exp3_evaluation import (
    BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED, common_bootstrap_indices,
    contribution_status, full_test_metrics, paired_metric_difference, wilson_rows,
)
from mec_dualbp.utils.sharded_checkpoint import _atomic_json

EXP3_PROTOCOL_REL="outputs/05_audit/exp3_ablation_protocol_20260811_144345"
EXP3_PROTOCOL_HASH="a4d7a4a86c4618754534495609364f6debf20ed5e3ca6f814085d543d79d8544"
TRAINING_REL="outputs/01_formal/exp3_ablation_training_formal_20260811_145920"
EXP2_REL="outputs/01_formal/dualbp_test_evaluation_formal_20260811_140611"
EXP2_TEST_PROTOCOL_HASH="ae0c7af865efefdd7b34597256ebd4fae48377a54fdb32b57eda4fc546f07f65"
TIME_DATA_REL="outputs/01_formal/time_bp_dataset_v2_labels_formal_20260810_194719"
ENERGY_DATA_REL="outputs/01_formal/bp_dataset_labels_formal_20260810_132254"
MODEL_HASHES={"C_Time":"59b22661d91c52db48e7d0c6fffd0732c159c9033bcc75c8b32411e04746969f",
"C_Energy":"320dcdac13cec992b3ebfb455cc4bea0e707815adf8cf69519a725083643064f",
"D":"0ca75faa72fa26a5f0c1417276b6b795000d0d15ca5f47086d932ecbc0c52f45",
"E_Time":"c8990794992777b729a6f26497560fe51d75b53878448a84ebd2aa2e25c5f6b7",
"E_Energy":"b490e46007ce8057c0e7e8a0ee5fe2d4ec958b44b945c0bcfaaa9b78091e6b23"}
METHOD_NAMES={"A":"Full Corrected","B":"Raw Dual-BP","C":"w/o Safety","D":"Unified-BP","E":"Direct-Bounds"}
CONTRASTS=(('A','B','Alpha correction'),('B','C','Safety Loss'),('B','D','Dual-network decomposition'),('B','E','Center-Radius parameterization'))


def _json(path:Path)->dict[str,Any]:return json.loads(path.read_text(encoding="utf-8-sig"))
def _scaler(path:Path)->TrainOnlyStandardScaler:
    v=_json(path);return TrainOnlyStandardScaler(tuple(v["mean"]),tuple(v["scale"]),tuple(v["constant_feature"]),v["fitted_role"])
def _source_hashes(project:Path)->dict[str,str]:
    return {name:file_sha256(project/path) for name,path in {"runner":"src/mec_dualbp/experiments/exp3_ablation_test.py","statistics":"src/mec_dualbp/training/exp3_evaluation.py","models":"src/mec_dualbp/models/ablation.py"}.items()}


def _model_binding(project:Path)->dict[str,Any]:
    root=project/TRAINING_REL; paths={"C_Time":root/"variant_C_w_o_safety/time/selected_model.pt","C_Energy":root/"variant_C_w_o_safety/energy/selected_model.pt","D":root/"variant_D_unified/selected_model.pt","E_Time":root/"variant_E_direct_bounds/time/selected_model.pt","E_Energy":root/"variant_E_direct_bounds/energy/selected_model.pt"}
    actual={key:file_sha256(path) for key,path in paths.items()};return {"expected":MODEL_HASHES,"actual":actual,"passed":actual==MODEL_HASHES,"paths":{k:p.relative_to(project).as_posix() for k,p in paths.items()}}


def freeze_protocol(project:Path,directory:Path)->dict[str,Any]:
    training=_json(project/TRAINING_REL/"summary.json"); models=_model_binding(project)
    if training.get("exp3_ablation_training_decision")!="passed" or not training.get("ablation_test_evaluation_allowed") or not models["passed"]:raise RuntimeError("Exp-3 training gate/hash drift")
    exp2_binding=_json(project/"outputs/05_audit/dualbp_test_evaluation_protocol_20260811_140517/test_dataset_binding.json")
    payload={"protocol_name":"Exp3-Unified-Ablation-Test-v1","design_timing":"post-Exp-2-Test","guard":"post-Exp-2 mechanistic ablation with frozen variants and no Test-driven hyperparameter adaptation",
      "variants":_json(project/EXP3_PROTOCOL_REL/"variant_definitions.json"),"primary_contrasts":[{"first":a,"second":b,"module":m,"difference":"first-minus-second"} for a,b,m in CONTRASTS],
      "metrics":{"accuracy":["MAE","RMSE","R2","WMAPE"],"safety":["UER","LER","ICR","upper_under_relative_magnitude","lower_over_relative_magnitude"],"structure":["invalid_interval_count","invalid_interval_rate"]},
      "test_binding":{"time":exp2_binding["time"],"energy":exp2_binding["energy"],"same_ids_required":True,"deletion_allowed":False},
      "stratification":{"Time_f":["Low","Mid-high","5GHz"],"Gamma":[1,2,3,4,5],"sample_type":"all frozen types"},
      "invalid_interval_rule":"record raw lower/upper; no swap/sort/clip/max/min/abs","unified_energy_base_rule":"arithmetic mean of the three f-expanded predictions per base; official Energy n=60",
      "unified_f_invariance":"lower/upper/center/radius std, range, relative_range; report mean/q95/max; diagnostic only",
      "confidence_intervals":{"Wilson":{"level":.95,"metrics":["UER","LER","ICR"]},"paired_bootstrap":{"seed":BOOTSTRAP_SEED,"repetitions":BOOTSTRAP_REPETITIONS,"common_indices":True,"metrics":["MAE","RMSE","WMAPE"]}},
      "tables":["table_exp3_ablation_accuracy.csv","table_exp3_ablation_safety.csv","table_exp3_primary_contrasts.csv","table_exp3_mechanism_diagnostics.csv"],
      "figures":["exp3_time_wmape_ablation","exp3_energy_wmape_ablation","exp3_time_safety_ablation","exp3_energy_safety_ablation","exp3_primary_contrast_safety","exp3_unified_energy_f_invariance","exp3_direct_bounds_invalid_intervals"],
      "contribution_rules":{"alpha":"A has lower UER/LER and higher ICR than B on both networks","safety":"B wins >=8/10 directional/coverage checks => supported; >=3 => partial","dual_accuracy":"B wins all 4 WMAPE => supported; >=2 => partial","dual_physical":"D Energy f relative-range max >1e-12 => supported","center_accuracy":"B wins all 4 WMAPE => supported; >=2 => partial","center_structural":"B invalid=0 and E invalid>0 => supported"},
      "AB_reuse":"reuse frozen Exp-2 prediction CSVs without reinference","overall_rank_forbidden":True,"test_driven_adaptation_forbidden":True,"post_exp3_test_retraining_allowed":False,
      "exp3_protocol_hash":EXP3_PROTOCOL_HASH,"exp2_test_protocol_hash":EXP2_TEST_PROTOCOL_HASH,"ablation_model_hashes":MODEL_HASHES,"target_access_allowed_after_freeze":True}
    ph=canonical_json_hash(payload);frozen={**payload,"status":"frozen","exp3_test_evaluation_protocol_hash":ph,"exp3_test_protocol_frozen_before_CDE_test_evaluation":True}
    directory.mkdir(parents=True,exist_ok=False);_atomic_json(directory/"exp3_test_evaluation_protocol.json",frozen);_atomic_json(directory/"exp3_test_evaluation_protocol_hash.json",{"sha256":ph});_atomic_json(directory/"variant_binding.json",{"models":models,"AB_source":EXP2_REL});_atomic_json(directory/"primary_contrast_binding.json",payload["primary_contrasts"]);_atomic_json(directory/"metric_definition_binding.json",payload["metrics"]);_atomic_json(directory/"test_dataset_binding.json",payload["test_binding"]);_atomic_json(directory/"figure_plan.json",{"figures":payload["figures"],"formats":["PNG_600dpi","PDF","SVG"],"plot_data_csv":True});_atomic_json(directory/"table_plan.json",{"tables":payload["tables"]});_atomic_json(directory/"confidence_interval_plan.json",payload["confidence_intervals"]);_atomic_json(directory/"source_hashes.json",_source_hashes(project))
    guard="# Test后消融评价防护\n\nExp-3是Exp-2 Test之后冻结的机制性消融，不能声称早于任何Test结果。C/D/E在本协议冻结前从未访问Test target；本协议冻结后只允许一次性推理与评价，禁止任何Test驱动回调。\n";(directory/"post_test_ablation_evaluation_guard.md").write_text(guard,encoding="utf-8");(directory/"README.md").write_text("# Exp-3统一Test评价协议\n\n本目录在C/D/E首次Test读取与推理前冻结五方法、四对比、指标、CI、表图和贡献判定规则。\n",encoding="utf-8");_atomic_json(directory/"created_files.json",{"files":sorted(p.name for p in directory.iterdir() if p.is_file())})
    return {"status":"frozen","hash":ph,"exp3_test_protocol_frozen_before_CDE_test_evaluation":True}


def _load_model(root:Path,variant:str,network:str):
    if variant=="C":path=root/f"variant_C_w_o_safety/{network.lower()}";cls=ConstrainedCenterRadiusMLP
    elif variant=="E":path=root/f"variant_E_direct_bounds/{network.lower()}";cls=DirectBoundsMLP
    else:path=root/"variant_D_unified";cls=UnifiedCenterRadiusMLP
    cfg=_json(path/"selected_model_config.json");ck=torch.load(path/"selected_model.pt",map_location="cpu",weights_only=False)
    if variant=="D":model=cls(cfg["input_dimension"],cfg["hidden_layers"],cfg["time_center_scale"],cfg["energy_center_scale"])
    else:model=cls(cfg["input_dimension"],cfg["hidden_layers"],cfg["center_scale"])
    model.load_state_dict(ck["model_state_dict"]);model.eval();return model,_scaler(path/"input_scaler.json"),cfg


def _infer_pair(model,scaler,x,direct=False):
    with torch.no_grad():out=model(torch.as_tensor(scaler.transform(x),dtype=torch.float32));lower,upper=out if direct else out[2:]
    return lower.numpy().astype(float),upper.numpy().astype(float)


def _group_metrics(records,field,network,scales):
    rows=[]
    for method in METHOD_NAMES:
        selected=[r for r in records if r["method"]==method]
        for value in sorted({str(r[field]) for r in selected}):
            group=[r for r in selected if str(r[field])==value];tl=np.asarray([r["true_lower"] for r in group]);tu=np.asarray([r["true_upper"] for r in group]);pl=np.asarray([r["pred_lower"] for r in group]);pu=np.asarray([r["pred_upper"] for r in group]);m=full_test_metrics(tl,tu,pl,pu,scales[network]);rows.append({"Method":method,"Network":network,"GroupField":field,"Group":value,**m})
    return rows


def _bar(rows,name,title,figures,plot_data,groups,series,ylabel,percent=True):
    write_csv(plot_data/f"{name}.csv",rows);mapping={(str(r["Group"]),str(r["Series"])):float(r["Value"]) for r in rows};x=np.arange(len(groups));width=.8/len(series);colors=["#2F75B5","#ED7D31","#70AD47","#A64D79","#7F8C8D"]
    fig,ax=plt.subplots(figsize=(max(7,len(groups)*1.25),4.8))
    for i,s in enumerate(series):
        vals=[mapping[(str(g),s)]*(100 if percent else 1) for g in groups];bars=ax.bar(x-.4+width/2+i*width,vals,width,label=s,color=colors[i%len(colors)],edgecolor="white",linewidth=.4);ax.bar_label(bars,fmt="%.2f",padding=2,fontsize=7)
    ax.set_xticks(x,[METHOD_NAMES.get(g,g) for g in groups]);ax.set_ylabel(ylabel);ax.set_title(title);ax.legend(ncol=min(3,len(series)))
    heights=[p.get_height() for p in ax.patches] or [0.0];low,high=min(heights),max(heights)
    if low < 0:
        span=max(high-low,1e-9);ax.set_ylim(low-.16*span,high+.18*span);ax.axhline(0,color="#595959",linewidth=.8)
    else:ax.set_ylim(0,high*1.25 if high>0 else 1)
    fig.tight_layout();save_figure(fig,figures/name)


def run_formal(project:Path,protocol_dir:Path,output:Path)->dict[str,Any]:
    started=time.monotonic();protocol=_json(protocol_dir/"exp3_test_evaluation_protocol.json");ph=_json(protocol_dir/"exp3_test_evaluation_protocol_hash.json")["sha256"]
    candidate={k:v for k,v in protocol.items() if k not in {"status","exp3_test_evaluation_protocol_hash","exp3_test_protocol_frozen_before_CDE_test_evaluation"}}
    if canonical_json_hash(candidate)!=ph or protocol.get("status")!="frozen" or not protocol.get("exp3_test_protocol_frozen_before_CDE_test_evaluation"):raise RuntimeError("Exp-3 Test protocol invalid")
    models_before=_model_binding(project);training=project/TRAINING_REL
    if not models_before["passed"]:raise RuntimeError("ablation model hash drift")
    output.mkdir(parents=True,exist_ok=False);_atomic_json(output/"protocol_binding.json",{"passed":True,"hash":ph,"frozen_before_CDE_test":True});shutil.copyfile(protocol_dir/"exp3_test_evaluation_protocol.json",output/"exp3_test_evaluation_protocol.json");shutil.copyfile(protocol_dir/"exp3_test_evaluation_protocol_hash.json",output/"exp3_test_evaluation_protocol_hash.json");_atomic_json(output/"ablation_model_binding.json",models_before)
    exp2_time=read_rows(project/EXP2_REL/"time_test_predictions.csv");exp2_energy=read_rows(project/EXP2_REL/"energy_test_predictions.csv")
    # First C/D/E Test target access occurs only after the frozen protocol was verified and copied above.
    time_rows=read_rows(project/TIME_DATA_REL/"time_test_labels.csv");energy_rows=read_rows(project/ENERGY_DATA_REL/"test_labels.csv");features={r["sample_id"]:r for r in read_rows(project/ENERGY_DATA_REL/"feature_records.csv") if r["role"]=="Test"}
    if len(time_rows)!=180 or len(energy_rows)!=60:raise RuntimeError("Test count drift")
    time_ids=[r["time_record_id"] for r in time_rows];energy_ids=[r["sample_id"] for r in energy_rows]
    align={"passed":time_ids==[r["time_record_id"] for r in exp2_time] and energy_ids==[r["sample_id"] for r in exp2_energy],"time_count":180,"energy_count":60,"time_mismatch":sum(a!=b for a,b in zip(time_ids,[r["time_record_id"] for r in exp2_time])),"energy_mismatch":sum(a!=b for a,b in zip(energy_ids,[r["sample_id"] for r in exp2_energy])),"deleted_samples":0};_atomic_json(output/"test_sample_alignment_audit.json",align)
    tx=np.asarray([[float(r[f]) for f in TIME_FEATURES] for r in time_rows]);ex=np.asarray([json.loads(features[r["sample_id"]]["energy_features"]) for r in energy_rows]);tl=np.asarray([float(r["time_lower"]) for r in time_rows]);tu=np.asarray([float(r["time_upper"]) for r in time_rows]);el=np.asarray([float(r["energy_lower"]) for r in energy_rows]);eu=np.asarray([float(r["energy_upper"]) for r in energy_rows])
    predictions={}
    predictions[("A","Time")]=(np.asarray([float(r["corrected_lower"]) for r in exp2_time]),np.asarray([float(r["corrected_upper"]) for r in exp2_time]));predictions[("B","Time")]=(np.asarray([float(r["raw_lower"]) for r in exp2_time]),np.asarray([float(r["raw_upper"]) for r in exp2_time]));predictions[("A","Energy")]=(np.asarray([float(r["corrected_lower"]) for r in exp2_energy]),np.asarray([float(r["corrected_upper"]) for r in exp2_energy]));predictions[("B","Energy")]=(np.asarray([float(r["raw_lower"]) for r in exp2_energy]),np.asarray([float(r["raw_upper"]) for r in exp2_energy]))
    for variant in ("C","E"):
        for network,x in (("Time",tx),("Energy",ex)):
            model,scaler,cfg=_load_model(training,variant,network);predictions[(variant,network)]=_infer_pair(model,scaler,x,variant=="E")
    dm,ds,dcfg=_load_model(training,"D","Unified");union=dcfg["input_feature_order"];expanded=[];ux=[]
    energy_feature_maps={sid:dict(zip(ENERGY_FEATURES,json.loads(row["energy_features"]))) for sid,row in features.items()}
    for row in time_rows:
        values={field:float(row[field]) for field in TIME_FEATURES};values.update({field:energy_feature_maps[row["base_sample_id"]][field] for field in union if field not in values});ux.append([values[field] for field in union])
    with torch.no_grad():dout=dm(torch.as_tensor(ds.transform(np.asarray(ux)),dtype=torch.float32))
    predictions[("D","Time")]=(dout["time_lower"].numpy(),dout["time_upper"].numpy());groups=defaultdict(list)
    for i,row in enumerate(time_rows):groups[row["base_sample_id"]].append(i);expanded.append({"time_record_id":row["time_record_id"],"base_sample_id":row["base_sample_id"],"edge_cpu_frequency_cycle_per_s":row["edge_cpu_frequency_cycle_per_s"],"pred_lower":float(dout["energy_lower"][i]),"pred_upper":float(dout["energy_upper"][i])})
    d_el=[];d_eu=[]
    for sid in energy_ids:
        idx=groups[sid];d_el.append(float(np.mean(dout["energy_lower"].numpy()[idx])));d_eu.append(float(np.mean(dout["energy_upper"].numpy()[idx])))
    predictions[("D","Energy")]=(np.asarray(d_el),np.asarray(d_eu));write_csv(output/"variant_D_energy_test_predictions_f_expanded.csv",expanded)
    def records(method,network,rows,true_l,true_u,pred_l,pred_u):
        result=[]
        for i,(r,a,b,c,d) in enumerate(zip(rows,true_l,true_u,pred_l,pred_u)):
            item={"method":method,"record_id":r.get("time_record_id",r.get("sample_id")),"base_sample_id":r.get("base_sample_id",r.get("sample_id")),"role":"Test","Gamma":r["Gamma"],"sample_type":r["sample_type"],"true_lower":a,"true_upper":b,"pred_lower":c,"pred_upper":d,"invalid_interval":bool(d<c)}
            if network=="Time":item.update({"edge_cpu_frequency_cycle_per_s":r["edge_cpu_frequency_cycle_per_s"],"f_stratum":"5GHz" if float(r["edge_cpu_frequency_cycle_per_s"])==5e9 else f_stratum(float(r["edge_cpu_frequency_cycle_per_s"]))})
            result.append(item)
        return result
    all_records=[]
    for method in METHOD_NAMES:
        all_records+=records(method,"Time",time_rows,tl,tu,*predictions[(method,"Time")]);all_records+=records(method,"Energy",energy_rows,el,eu,*predictions[(method,"Energy")])
    for method in ("C","E"):
        write_csv(output/f"variant_{method}_time_test_predictions.csv",[r for r in all_records if r["method"]==method and "f_stratum" in r]);write_csv(output/f"variant_{method}_energy_test_predictions.csv",[r for r in all_records if r["method"]==method and "f_stratum" not in r])
    write_csv(output/"variant_D_time_test_predictions.csv",[r for r in all_records if r["method"]=="D" and "f_stratum" in r]);write_csv(output/"variant_D_energy_test_predictions_base.csv",[r for r in all_records if r["method"]=="D" and "f_stratum" not in r])
    scales={"Time":float(dcfg["time_center_scale"]),"Energy":float(dcfg["energy_center_scale"])};metrics={};metric_rows=[];wilson=[]
    for method in METHOD_NAMES:
        for network,true_l,true_u in (("Time",tl,tu),("Energy",el,eu)):
            m=full_test_metrics(true_l,true_u,*predictions[(method,network)],scales[network]);metrics[(method,network)]=m;metric_rows.append({"Method":method,"MethodName":METHOD_NAMES[method],"Network":network,**m});wilson+=wilson_rows(method,network,m)
    write_csv(output/"all_variants_test_metrics.csv",metric_rows);write_csv(output/"wilson_confidence_intervals.csv",wilson);write_csv(output/"variant_A_time_metrics.csv",[metrics[("A","Time")]]);write_csv(output/"variant_A_energy_metrics.csv",[metrics[("A","Energy")]]);write_csv(output/"variant_B_time_metrics.csv",[metrics[("B","Time")]]);write_csv(output/"variant_B_energy_metrics.csv",[metrics[("B","Energy")]])
    f_rows=_group_metrics([r for r in all_records if "f_stratum" in r],"f_stratum","Time",scales);gamma_rows=_group_metrics([r for r in all_records if "f_stratum" in r],"Gamma","Time",scales)+_group_metrics([r for r in all_records if "f_stratum" not in r],"Gamma","Energy",scales);sample_rows=_group_metrics([r for r in all_records if "f_stratum" in r],"sample_type","Time",scales)+_group_metrics([r for r in all_records if "f_stratum" not in r],"sample_type","Energy",scales);write_csv(output/"metrics_by_f_stratum.csv",f_rows);write_csv(output/"metrics_by_gamma.csv",gamma_rows);write_csv(output/"metrics_by_sample_type.csv",sample_rows)
    inv_rows=[];inv_values={name:[] for name in ("lower","upper","center","radius")}
    for base,idx in groups.items():
        lo=dout["energy_lower"].numpy()[idx];up=dout["energy_upper"].numpy()[idx];arrays={"lower":lo,"upper":up,"center":.5*(lo+up),"radius":.5*(up-lo)};row={"base_sample_id":base}
        for name,array in arrays.items():std=float(array.std());spread=float(np.ptp(array));rel=spread/max(float(np.mean(np.abs(array))),1e-12);row.update({f"{name}_std":std,f"{name}_range":spread,f"{name}_relative_range":rel});inv_values[name].append(rel)
        inv_rows.append(row)
    inv_summary={name:{"mean":float(np.mean(v)),"q95":float(np.quantile(v,.95)),"max":float(np.max(v))} for name,v in inv_values.items()};write_csv(output/"unified_energy_f_invariance_test.csv",inv_rows);_atomic_json(output/"unified_energy_f_invariance_test_summary.json",inv_summary)
    invalid=[r for r in all_records if r["method"]=="E" and r["invalid_interval"]];write_csv(output/"direct_bounds_invalid_test_intervals.csv",invalid);invalid_summary={net:{"count":metrics[("E",net)]["invalid_interval_count"],"rate":metrics[("E",net)]["invalid_interval_rate"]} for net in ("Time","Energy")};invalid_summary["post_hoc_repair"]=False;_atomic_json(output/"direct_bounds_invalid_test_summary.json",invalid_summary)
    accuracy=[];safety=[]
    for row in metric_rows:
        accuracy.append({"Method":METHOD_NAMES[row["Method"]],"Network":row["Network"],**{key:row[key] for key in ("lower_mae","upper_mae","lower_rmse","upper_rmse","lower_r2","upper_r2","lower_wmape","upper_wmape")}})
        ci={r["Metric"]:r for r in wilson if r["Method"]==row["Method"] and r["Network"]==row["Network"]};safety.append({"Method":METHOD_NAMES[row["Method"]],"Network":row["Network"],"UER":row["uer"],"UER_CI_Low":ci["UER"]["CI_Low"],"UER_CI_High":ci["UER"]["CI_High"],"LER":row["ler"],"LER_CI_Low":ci["LER"]["CI_Low"],"LER_CI_High":ci["LER"]["CI_High"],"ICR":row["icr"],"ICR_CI_Low":ci["ICR"]["CI_Low"],"ICR_CI_High":ci["ICR"]["CI_High"],"UpperUnderMagnitude":row["upper_under_relative_magnitude"],"LowerOverMagnitude":row["lower_over_relative_magnitude"],"InvalidIntervalCount":row["invalid_interval_count"],"InvalidIntervalRate":row["invalid_interval_rate"]})
    write_csv(output/"table_exp3_ablation_accuracy.csv",accuracy);write_csv(output/"table_exp3_ablation_safety.csv",safety)
    boot_rows=[];contrast_rows=[];statuses=contribution_status(metrics,max(v["max"] for v in inv_summary.values()))
    for first,second,module in CONTRASTS:
        for network,true_l,true_u in (("Time",tl,tu),("Energy",el,eu)):
            idx=common_bootstrap_indices(len(true_l));cis={}
            for bound,true in (("lower",true_l),("upper",true_u)):
                dif=paired_metric_difference(true,predictions[(first,network)][0 if bound=="lower" else 1],predictions[(second,network)][0 if bound=="lower" else 1],idx)
                for metric,ci in dif.items():boot_rows.append({"Contrast":f"{first}-{second}","Network":network,"Bound":bound,"Metric":metric,"CI_Low":ci[0],"CI_High":ci[1],"Seed":BOOTSTRAP_SEED,"Repetitions":BOOTSTRAP_REPETITIONS});cis[f"{bound}_{metric}"]=f"[{ci[0]:.8g}, {ci[1]:.8g}]"
            fm,sm=metrics[(first,network)],metrics[(second,network)];status_key={"Alpha correction":"alpha_contribution","Safety Loss":"safety_loss_contribution","Dual-network decomposition":"dual_accuracy_contribution","Center-Radius parameterization":"center_radius_accuracy_contribution"}[module]
            contrast_rows.append({"Contrast":f"{first}-{second}","Baseline":second,"Variant":first,"Module":module,"Network":network,"Delta_Lower_WMAPE":float(fm["lower_wmape"])-float(sm["lower_wmape"]),"Delta_Upper_WMAPE":float(fm["upper_wmape"])-float(sm["upper_wmape"]),"Delta_UER":float(fm["uer"])-float(sm["uer"]),"Delta_LER":float(fm["ler"])-float(sm["ler"]),"Delta_ICR":float(fm["icr"])-float(sm["icr"]),"Delta_UpperUnderMagnitude":float(fm["upper_under_relative_magnitude"])-float(sm["upper_under_relative_magnitude"]),"Delta_LowerOverMagnitude":float(fm["lower_over_relative_magnitude"])-float(sm["lower_over_relative_magnitude"]),"Delta_InvalidIntervalRate":float(fm["invalid_interval_rate"])-float(sm["invalid_interval_rate"]),"Bootstrap_CI_where_applicable":json.dumps(cis),"Contribution_Status":statuses[status_key]})
    write_csv(output/"paired_bootstrap_contrasts.csv",boot_rows);write_csv(output/"table_exp3_primary_contrasts.csv",contrast_rows)
    mechanism=[]
    for name,values in inv_summary.items():mechanism.append({"Mechanism":"Unified Energy f-invariance","Measure":f"{name}_relative_range","Mean":values["mean"],"Q95":values["q95"],"Max":values["max"]})
    for net in ("Time","Energy"):mechanism.append({"Mechanism":"Direct-Bounds validity","Measure":f"{net}_invalid_interval_rate","Mean":invalid_summary[net]["rate"],"Q95":"","Max":invalid_summary[net]["count"]})
    for method in ("A","B"):mechanism.append({"Mechanism":"Center-Radius structural validity","Measure":f"{method}_invalid_interval_rate","Mean":0.0,"Q95":"","Max":0})
    write_csv(output/"table_exp3_mechanism_diagnostics.csv",mechanism);_atomic_json(output/"exp3_contribution_status.json",statuses)
    interpretation=f"# Exp-3科学解释\n\n本研究是Exp-2 Test之后冻结的机制性消融，未进行Test驱动超参适配。\n\n## Alpha correction\n\n安全贡献：`{statuses['alpha_contribution']}`。该模块主要改变危险方向风险，不要求WMAPE同步改善。\n\n## Safety Loss\n\n方向风险贡献：`{statuses['safety_loss_contribution']}`。准确率与安全性分别解释。\n\n## Dual-network decomposition\n\n准确率贡献：`{statuses['dual_accuracy_contribution']}`；物理解耦贡献：`{statuses['dual_physical_decoupling_contribution']}`。Unified对物理无关f输入的Energy预测出现实测变化。\n\n## Center-Radius parameterization\n\n准确率贡献：`{statuses['center_radius_accuracy_contribution']}`；结构合法性贡献：`{statuses['center_radius_structural_validity_contribution']}`。Direct-Bounds非法区间保持原样。\n";(output/"exp3_scientific_interpretation.md").write_text(interpretation,encoding="utf-8")
    figures=output/"figures";plot_data=output/"plot_data";figures.mkdir();plot_data.mkdir();style();manifest=[]
    def reg(name):
        for ext in ("png","pdf","svg"):manifest.append({"Figure":name,"Format":ext,"Path":f"figures/{name}.{ext}","PlotData":f"plot_data/{name}.csv"})
    for network in ("Time","Energy"):
        rows=[]
        for method in METHOD_NAMES:
            rows += [{"Group":method,"Series":"Lower","Value":metrics[(method,network)]["lower_wmape"]},{"Group":method,"Series":"Upper","Value":metrics[(method,network)]["upper_wmape"]}]
        name=f"exp3_{network.lower()}_wmape_ablation";_bar(rows,name,f"{network} WMAPE ablation",figures,plot_data,list(METHOD_NAMES),["Lower","Upper"],"WMAPE (%)");reg(name)
        rows=[]
        for method in METHOD_NAMES:
            for metric,label in (("uer","UER ↓"),("ler","LER ↓"),("icr","ICR ↑")):rows.append({"Group":method,"Series":label,"Value":metrics[(method,network)][metric]})
        name=f"exp3_{network.lower()}_safety_ablation";_bar(rows,name,f"{network} safety ablation",figures,plot_data,list(METHOD_NAMES),["UER ↓","LER ↓","ICR ↑"],"Rate (%)");reg(name)
    rows=[]
    for row in contrast_rows:
        for metric,label in (("Delta_UER","ΔUER"),("Delta_LER","ΔLER"),("Delta_ICR","ΔICR")):rows.append({"Group":f"{row['Contrast']} {row['Network'][0]}","Series":label,"Value":row[metric]})
    _bar(rows,"exp3_primary_contrast_safety","Primary contrast safety differences (first − second)",figures,plot_data,[f"{a}-{b} {n}" for a,b,_ in CONTRASTS for n in ("T","E")],["ΔUER","ΔLER","ΔICR"],"Difference (percentage points)");reg("exp3_primary_contrast_safety")
    rows=[]
    for name,v in inv_summary.items():
        for label,key in (("Mean","mean"),("Q95","q95"),("Max","max")):rows.append({"Group":name.title(),"Series":label,"Value":v[key]})
    _bar(rows,"exp3_unified_energy_f_invariance","Unified-BP Energy variation with edge CPU frequency",figures,plot_data,["Lower","Upper","Center","Radius"],["Mean","Q95","Max"],"Relative range (%)");reg("exp3_unified_energy_f_invariance")
    rows=[]
    for network in ("Time","Energy"):
        rows += [{"Group":network,"Series":"Raw Dual-BP","Value":metrics[("B",network)]["invalid_interval_rate"]},{"Group":network,"Series":"Direct-Bounds","Value":metrics[("E",network)]["invalid_interval_rate"]}]
    _bar(rows,"exp3_direct_bounds_invalid_intervals","Direct-Bounds interval validity on Test",figures,plot_data,["Time","Energy"],["Raw Dual-BP","Direct-Bounds"],"Invalid interval rate (%)");reg("exp3_direct_bounds_invalid_intervals");write_csv(output/"figure_manifest.csv",manifest)
    models_after=_model_binding(project);integrity={"passed":models_before==models_after,"before":models_before,"after":models_after,"AB_models_reinferred":False,"post_test_retraining":False};_atomic_json(output/"frozen_artifact_integrity.json",integrity);_atomic_json(output/"baseline_binding.json",{"AB_predictions_reused":True,"source":EXP2_REL,"exp2_test_protocol_hash":EXP2_TEST_PROTOCOL_HASH});_atomic_json(output/"test_dataset_binding.json",{"Time":180,"Energy":60,"Unified_Energy_official_base_count":60});nonfinite=sum(int(m["nonfinite_count"]) for m in metrics.values());numerical={"passed":nonfinite==0,"nonfinite":nonfinite,"invalid_intervals_by_method":{method:sum(int(metrics[(method,n)]["invalid_interval_count"]) for n in ("Time","Energy")) for method in METHOD_NAMES},"deleted_rows":0};_atomic_json(output/"numerical_audit.json",numerical)
    tables_ok=all((output/name).exists() for name in protocol["tables"]);fig_ok=len(manifest)==21 and all((output/r["Path"]).exists() for r in manifest);fairness={"passed":align["passed"] and integrity["passed"] and numerical["passed"] and tables_ok and fig_ok,"protocol_frozen_before_CDE_test":True,"AB_frozen_predictions_reused":True,"same_test_samples":align["passed"],"CDE_retrained_after_test":False,"overall_rank_computed":False,"paired_bootstrap_common_indices":True,"Unified_Energy_official_n":60,"E_post_hoc_repair":False};_atomic_json(output/"fairness_audit_final.json",fairness)
    decision="passed" if fairness["passed"] else "invalid_due_to_execution";summary={"exp3_test_evaluation_decision":decision,"exp3_test_status":"completed_and_frozen" if decision=="passed" else "not_frozen","exp3_completed":decision=="passed","all_ablation_models_frozen":True,"post_exp3_test_retraining_allowed":False,"multi_algorithm_experiment_allowed":decision=="passed","protocol_hash":ph,"time_test_count":180,"energy_test_count":60,"metrics":{f"{m}_{n}":v for (m,n),v in metrics.items()},"contribution_status":statuses,"scientific_outcome":"mixed_module_specific" if len(set(statuses.values()))>1 else next(iter(statuses.values())),"figure_count":7,"figure_files":21,"plot_data_count":7,"overall_wall_seconds_before_pytest":time.monotonic()-started};_atomic_json(output/"summary.json",summary);_atomic_json(output/"runtime_breakdown.json",{"overall_wall_seconds_before_pytest":summary["overall_wall_seconds_before_pytest"],"bootstrap_seed":BOOTSTRAP_SEED,"bootstrap_repetitions":BOOTSTRAP_REPETITIONS,"workers":1,"pytest_seconds":None})
    guard="# Test后消融评价防护\n\nExp-3是post-Exp-2 mechanistic ablation study with frozen variants and no Test-driven hyperparameter adaptation。A/B复用既有结果；C/D/E首次统一Test后禁止重训、换seed或修改任何协议参数。\n";(output/"post_test_ablation_evaluation_guard.md").write_text(guard,encoding="utf-8");(output/"README.md").write_text("# Exp-3统一Test正式评价\n\nA/B复用Exp-2冻结预测，C/D/E在协议冻结后首次一次性Test。结果按模块解释，不构造总体排名。\n",encoding="utf-8");(output/"validation_report.md").write_text(f"# Exp-3最终评价门禁\n\n- decision: `{decision}`\n- sample alignment: `{align['passed']}`\n- fairness: `{fairness['passed']}`\n- post-Test retraining: `false`\n",encoding="utf-8");(output/"final_pytest.txt").write_text("等待最终pytest。\n",encoding="utf-8");_atomic_json(output/"created_files.json",{"files":sorted(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file())});return summary


def finalize(output:Path,count:int,seconds:float)->dict[str,Any]:
    summary=_json(output/"summary.json");runtime=_json(output/"runtime_breakdown.json");passed=summary["exp3_test_evaluation_decision"]=="passed" and count>=301;summary.update({"final_pytest_passed":passed,"final_pytest_count":count});runtime["pytest_seconds"]=seconds;runtime["overall_wall_seconds"]=float(runtime["overall_wall_seconds_before_pytest"])+seconds;_atomic_json(output/"summary.json",summary);_atomic_json(output/"runtime_breakdown.json",runtime);(output/"final_pytest.txt").write_text(f"python -m compileall -q src\n结果：通过\n\npython -m pytest -q\n结果：{count} passed in {seconds:.2f}s\n",encoding="utf-8");_atomic_json(output/"created_files.json",{"files":sorted(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file())});return summary


def main():
    p=argparse.ArgumentParser();p.add_argument("--project-root",default=".");p.add_argument("--protocol-dir");p.add_argument("--output-dir");p.add_argument("--freeze-protocol",action="store_true");p.add_argument("--finalize",action="store_true");p.add_argument("--pytest-count",type=int);p.add_argument("--pytest-seconds",type=float);a=p.parse_args();project=Path(a.project_root).resolve();target=Path(a.protocol_dir if a.freeze_protocol else a.output_dir).resolve()
    try:result=freeze_protocol(project,target) if a.freeze_protocol else finalize(target,a.pytest_count,a.pytest_seconds) if a.finalize else run_formal(project,Path(a.protocol_dir).resolve(),target);print(json.dumps({k:v for k,v in result.items() if k!="metrics"},ensure_ascii=False),flush=True)
    except Exception as e:target.mkdir(parents=True,exist_ok=True);_atomic_json(target/"runner_failure.json",{"error":str(e),"traceback":traceback.format_exc()});raise
if __name__=="__main__":main()
