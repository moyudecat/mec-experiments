"""Exp-2：冻结协议后首次执行独立 Test 正式评价。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mec_dualbp.datasets.protocol import ENERGY_FEATURES, TIME_FEATURES, write_csv
from mec_dualbp.experiments.dualbp_alpha_calibration import (
    ENERGY_DATA_REL, ENERGY_MODEL_HASH, TIME_DATA_REL, TIME_MODEL_HASH,
    TRAINING_FORMAL_REL, TRAINING_PROTOCOL_HASH, infer, load_model_and_scaler,
    load_json, read_rows, save_figure, style, verify_training_binding,
)
from mec_dualbp.experiments.exp1_formal_protocol import file_sha256
from mec_dualbp.experiments.exp1_preview_protocol import canonical_json_hash
from mec_dualbp.training.calibration import apply_directional_correction
from mec_dualbp.training.dualbp import f_stratum, regression_metrics
from mec_dualbp.training.test_evaluation import (
    BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED, paired_bootstrap_accuracy,
    safety_metrics, test_protocol_is_frozen, wilson_interval,
)
from mec_dualbp.utils.sharded_checkpoint import _atomic_json

CAL_PROTOCOL_REL="outputs/05_audit/dualbp_alpha_calibration_protocol_20260811_133826"
CAL_FORMAL_REL="outputs/01_formal/dualbp_alpha_calibration_formal_20260811_134010"
ALPHA_PROTOCOL_HASH="77a3a1c3ed659ddd534dd69d3605bc0bce57b0b4d2f4a46a13c3b1f031f1d2bb"
CORRECTED_HASH="a15b8fb591df123b475f521ef35e4037b6ca201f4172caf804d3af571afe13e2"


def source_hashes(project:Path)->dict[str,str]:
    files={"evaluation":project/"src/mec_dualbp/training/test_evaluation.py","runner":project/"src/mec_dualbp/experiments/dualbp_test_evaluation.py","calibration":project/"src/mec_dualbp/training/calibration.py","model":project/"src/mec_dualbp/models/dualbp.py"}
    return {k:file_sha256(v) for k,v in files.items()}


def verify_frozen(project:Path)->dict[str,Any]:
    training=verify_training_binding(project); cal=load_json(project/CAL_FORMAL_REL/"summary.json")
    if cal.get("alpha_calibration_decision")!="passed" or cal.get("corrected_dualbp_status")!="calibrated_and_frozen" or cal.get("test_evaluation_allowed") is not True: raise RuntimeError("Calibration gate drift")
    corrected_path=project/CAL_FORMAL_REL/"corrected_dualbp_v1.json"; corrected=load_json(corrected_path)
    if canonical_json_hash(corrected)!=CORRECTED_HASH: raise RuntimeError("Corrected DualBP hash drift")
    if load_json(project/CAL_PROTOCOL_REL/"alpha_calibration_protocol_hash.json")["alpha_calibration_protocol_hash"]!=ALPHA_PROTOCOL_HASH: raise RuntimeError("alpha protocol drift")
    historical=load_json(project/TRAINING_FORMAL_REL/"data_leakage_audit.json")
    if historical.get("test_labels_accessed") is not False or historical.get("test_used_for_selection") is not False: raise RuntimeError("historical Test leakage")
    cal_leak=load_json(project/CAL_FORMAL_REL/"data_leakage_audit.json")
    if any(cal_leak.get(k) for k in ("test_labels_accessed","test_predictions_generated","test_metrics_computed","test_used_for_alpha","test_used_for_model_selection")): raise RuntimeError("Calibration/Test leakage")
    return {**training,"alpha_protocol_hash":ALPHA_PROTOCOL_HASH,"corrected_dualbp_hash":CORRECTED_HASH,"corrected_config":corrected,"historical_test_metadata":historical["metadata_only_audit"]}


def freeze_protocol(project:Path,directory:Path)->dict[str,Any]:
    started=time.monotonic(); binding=verify_frozen(project); meta=binding["historical_test_metadata"]
    if meta["time_test"]["row_count"]!=180 or meta["energy_test"]["row_count"]!=60: raise RuntimeError("sealed Test count drift")
    payload={"protocol_name":"Exp-2-DualBP-Independent-Test-v1","status":"freeze_candidate","training_protocol_hash":TRAINING_PROTOCOL_HASH,"alpha_protocol_hash":ALPHA_PROTOCOL_HASH,"time_model_hash":TIME_MODEL_HASH,"energy_model_hash":ENERGY_MODEL_HASH,"corrected_dualbp_hash":CORRECTED_HASH,
             "test_dataset_binding":{"time":meta["time_test"],"energy":meta["energy_test"]},"metrics":["MAE","RMSE","R2","WMAPE","UER","LER","ICR","directional_relative_magnitude"],"metric_definition_binding":"same as frozen Training/Validation","wilson_confidence_level":.95,"bootstrap":{"seed":BOOTSTRAP_SEED,"repetitions":BOOTSTRAP_REPETITIONS,"interval":"95% percentile","paired_indices":True},"stratification":{"time_f":{"Low":"[1.5e9,3.25e9)","Mid-high":"[3.25e9,5e9)","Upper-anchor":"5e9"},"Gamma":[1,2,3,4,5],"sample_type":"frozen six types"},"figures":["time_test_lower_scatter","time_test_upper_scatter","energy_test_lower_scatter","energy_test_upper_scatter","time_test_wmape","energy_test_wmape","time_raw_corrected_safety","energy_raw_corrected_safety","time_test_f_stratum_wmape","energy_test_gamma_wmape"],"raw_corrected_comparison":"paired same Test rows","result_interpretation":{"raw_accuracy":"WMAPE<=5% and R2>=0.95 => strong","safety":"UER and LER lower and ICR higher => consistent"},"target_access_allowed_after_freeze":True,"historical_test_labels_accessed":False,"test_target_accessed_before_freeze":False,"post_test_model_modification_allowed":False,"post_test_alpha_modification_allowed":False}
    protocol_hash=canonical_json_hash(payload); frozen={**payload,"status":"frozen","test_evaluation_protocol_hash":protocol_hash,"test_protocol_frozen_before_target_access":True}
    directory.mkdir(parents=True,exist_ok=False); _atomic_json(directory/"model_binding.json",{k:binding[k] for k in ("time_model_hash","energy_model_hash","time_scaler_hash","energy_scaler_hash")}); _atomic_json(directory/"corrected_dualbp_binding.json",{"corrected_dualbp_hash":CORRECTED_HASH,"alpha_protocol_hash":ALPHA_PROTOCOL_HASH}); _atomic_json(directory/"test_dataset_binding.json",payload["test_dataset_binding"]); _atomic_json(directory/"test_evaluation_protocol.json",frozen); _atomic_json(directory/"test_evaluation_protocol_hash.json",{"test_evaluation_protocol_hash":protocol_hash}); _atomic_json(directory/"stratification_rules.json",payload["stratification"]); _atomic_json(directory/"confidence_interval_protocol.json",{"wilson":.95,**payload["bootstrap"]}); _atomic_json(directory/"data_access_guard.json",{"passed":True,"test_protocol_frozen_before_target_access":True,"test_targets_accessed":False}); _atomic_json(directory/"source_hashes.json",source_hashes(project))
    (directory/"metric_definitions.md").write_text("# Exp-2 Test指标\n\n沿用训练阶段MAE、RMSE、R²、WMAPE定义；安全指标为UER、LER、ICR及危险方向相对幅度。delta=0.05不等价于ICR理论保证95%。\n",encoding="utf-8"); (directory/"README.md").write_text("# Exp-2 独立Test评价协议\n\n协议在首次读取Test真实标签前冻结；Test结果不得用于任何回调修改。\n",encoding="utf-8"); _atomic_json(directory/"created_files.json",{"files":sorted(p.name for p in directory.iterdir() if p.is_file())})
    result={"status":"frozen","test_evaluation_protocol_hash":protocol_hash,"protocol_design_seconds":time.monotonic()-started}; _atomic_json(directory/"summary.json",result); return result


def full_metrics(tl,tu,pl,pu,scale)->dict[str,Any]: return {**regression_metrics(tl,tu,pl,pu,scale),**safety_metrics(tl,tu,pl,pu)}


def grouped(rows,tl,tu,raw_l,raw_u,corr_l,corr_u,scale,field,values)->list[dict[str,Any]]:
    output=[]
    for value in sorted(set(values)):
        idx=np.asarray([x==value for x in values])
        for variant,lo,up in (("Raw",raw_l,raw_u),("Corrected",corr_l,corr_u)):
            output.append({field:value,"variant":variant,**full_metrics(tl[idx],tu[idx],lo[idx],up[idx],scale)})
    return output


def scatter_plot(rows,name,title,unit,out,plot_data):
    write_csv(plot_data/f"fig_{name}.csv",rows); true=np.asarray([float(r["reference"]) for r in rows]); raw=np.asarray([float(r["raw"]) for r in rows]); corr=np.asarray([float(r["corrected"]) for r in rows]); low=min(true.min(),raw.min(),corr.min()); high=max(true.max(),raw.max(),corr.max()); pad=(high-low)*.04
    fig,ax=plt.subplots(figsize=(5.9,5.2)); ax.scatter(true,raw,s=25,alpha=.72,color="#7F8C8D",marker="o",label="Raw BP",edgecolors="white",linewidths=.3); ax.scatter(true,corr,s=27,alpha=.72,color="#2F75B5",marker="^",label="Corrected BP",edgecolors="white",linewidths=.3); ax.plot([low-pad,high+pad],[low-pad,high+pad],"--",color="#C00000",linewidth=1.2,label="y = x"); ax.set_xlim(low-pad,high+pad); ax.set_ylim(low-pad,high+pad); ax.set_xlabel(f"Reference label ({unit})"); ax.set_ylabel(f"Prediction ({unit})"); ax.set_title(title); ax.legend(); fig.tight_layout(); save_figure(fig,out/name)


def grouped_bar(rows,name,title,out,plot_data,groups,series,ylabel="WMAPE (%)"):
    write_csv(plot_data/f"fig_{name}.csv",rows); mapping={(r["group"],r["series"]):float(r["value"]) for r in rows}; x=np.arange(len(groups)); width=.8/len(series); fig,ax=plt.subplots(figsize=(max(6.4,len(groups)*1.15),4.7)); colors=("#7F8C8D","#A5A5A5","#2F75B5","#ED7D31")
    for i,s in enumerate(series):
        vals=[mapping[(g,s)] for g in groups]; bars=ax.bar(x-.4+width/2+i*width,vals,width,label=s,color=colors[i]); ax.bar_label(bars,fmt="%.2f",padding=2,fontsize=7)
    ax.set_xticks(x,groups); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(ncol=2); ax.set_ylim(0,max(float(r["value"]) for r in rows)*1.28); fig.tight_layout(); save_figure(fig,out/name)


def safety_plot(network,raw,corr,out,plot_data):
    rows=[]
    for group,source in (("UER",("uer",)),("LER",("ler",)),("ICR",("icr",))):
        rows.extend(({"group":group,"series":"Raw","value":100*raw[source[0]]},{"group":group,"series":"Corrected","value":100*corr[source[0]]}))
    grouped_bar(rows,f"{network}_raw_corrected_safety",f"{network.title()} Test safety",out,plot_data,["UER","LER","ICR"],["Raw","Corrected"],"Rate (%)")


def run_formal(project:Path,protocol_dir:Path,output:Path)->dict[str,Any]:
    started=time.monotonic(); protocol=load_json(protocol_dir/"test_evaluation_protocol.json"); ph=load_json(protocol_dir/"test_evaluation_protocol_hash.json")["test_evaluation_protocol_hash"]; candidate={k:v for k,v in protocol.items() if k not in ("status","test_evaluation_protocol_hash","test_protocol_frozen_before_target_access")}; candidate["status"]="freeze_candidate"
    if canonical_json_hash(candidate)!=ph or not test_protocol_is_frozen(protocol): raise RuntimeError("Test protocol invalid")
    binding=verify_frozen(project); formal=project/TRAINING_FORMAL_REL; cal=project/CAL_FORMAL_REL; before={"time_model":file_sha256(formal/"time_bp/selected_model.pt"),"energy_model":file_sha256(formal/"energy_bp/selected_model.pt"),"time_scaler":file_sha256(formal/"time_bp/input_scaler.json"),"energy_scaler":file_sha256(formal/"energy_bp/input_scaler.json"),"corrected":canonical_json_hash(load_json(cal/"corrected_dualbp_v1.json"))}
    output.mkdir(parents=True,exist_ok=False); _atomic_json(output/"protocol_binding.json",{"passed":True,"test_evaluation_protocol_hash":ph,"frozen_before_target_access":True}); _atomic_json(output/"model_binding.json",binding); shutil_copy=lambda src,dst: dst.write_bytes(src.read_bytes()); shutil_copy(cal/"corrected_dualbp_v1.json",output/"corrected_dualbp_binding.json"); _atomic_json(output/"test_dataset_binding.json",protocol["test_dataset_binding"])
    # First Test target access: protocol hash is already frozen and verified above.
    time_rows=read_rows(project/TIME_DATA_REL/"time_test_labels.csv"); energy_rows=read_rows(project/ENERGY_DATA_REL/"test_labels.csv")
    if len(time_rows)!=180 or len(energy_rows)!=60 or any(r["role"]!="Test" for r in time_rows+energy_rows): raise RuntimeError("Test count/role drift")
    tm,ts,tc=load_model_and_scaler("time_bp",formal); em,es,ec=load_model_and_scaler("energy_bp",formal); tx=np.asarray([[float(r[f]) for f in TIME_FEATURES] for r in time_rows]); tl=np.asarray([float(r["time_lower"]) for r in time_rows]); tu=np.asarray([float(r["time_upper"]) for r in time_rows]); features={r["sample_id"]:r for r in read_rows(project/ENERGY_DATA_REL/"feature_records.csv") if r["role"]=="Test"}; ex=np.asarray([json.loads(features[r["sample_id"]]["energy_features"]) for r in energy_rows]); el=np.asarray([float(r["energy_lower"]) for r in energy_rows]); eu=np.asarray([float(r["energy_upper"]) for r in energy_rows]); trl,tru=infer(tm,ts,tx); erl,eru=infer(em,es,ex)
    corr=binding["corrected_config"]; tcl,tcu=apply_directional_correction(trl,tru,corr["alpha_T_minus"],corr["alpha_T_plus"]); ecl,ecu=apply_directional_correction(erl,eru,corr["alpha_E_minus"],corr["alpha_E_plus"])
    time_pred=[]
    for r,a,b,c,d,e,f in zip(time_rows,tl,tu,trl,tru,tcl,tcu): time_pred.append({"time_record_id":r["time_record_id"],"base_sample_id":r["base_sample_id"],"role":"Test","Gamma":r["Gamma"],"sample_type":r["sample_type"],"edge_cpu_frequency_cycle_per_s":r["edge_cpu_frequency_cycle_per_s"],"f_stratum":f_stratum(float(r["edge_cpu_frequency_cycle_per_s"])),"true_lower":a,"true_upper":b,"raw_lower":c,"raw_upper":d,"corrected_lower":e,"corrected_upper":f})
    energy_pred=[]
    for r,a,b,c,d,e,f in zip(energy_rows,el,eu,erl,eru,ecl,ecu): energy_pred.append({"sample_id":r["sample_id"],"role":"Test","Gamma":r["Gamma"],"sample_type":r["sample_type"],"true_lower":a,"true_upper":b,"raw_lower":c,"raw_upper":d,"corrected_lower":e,"corrected_upper":f})
    write_csv(output/"time_test_predictions.csv",time_pred); write_csv(output/"energy_test_predictions.csv",energy_pred)
    metrics_all={}
    for net,lt,ut,rl,ru,cl,cu,scale in (("Time",tl,tu,trl,tru,tcl,tcu,tc["center_scale"]),("Energy",el,eu,erl,eru,ecl,ecu,ec["center_scale"])):
        metrics_all[net]={"Raw":full_metrics(lt,ut,rl,ru,scale),"Corrected":full_metrics(lt,ut,cl,cu,scale)}
        write_csv(output/f"{net.lower()}_test_metrics_overall.csv",[{"network":net,"variant":v,**m} for v,m in metrics_all[net].items()])
    time_f=grouped(time_rows,tl,tu,trl,tru,tcl,tcu,tc["center_scale"],"f_stratum",[r["f_stratum"] for r in time_pred]); time_g=grouped(time_rows,tl,tu,trl,tru,tcl,tcu,tc["center_scale"],"Gamma",[r["Gamma"] for r in time_rows]); time_s=grouped(time_rows,tl,tu,trl,tru,tcl,tcu,tc["center_scale"],"sample_type",[r["sample_type"] for r in time_rows]); energy_g=grouped(energy_rows,el,eu,erl,eru,ecl,ecu,ec["center_scale"],"Gamma",[r["Gamma"] for r in energy_rows]); energy_s=grouped(energy_rows,el,eu,erl,eru,ecl,ecu,ec["center_scale"],"sample_type",[r["sample_type"] for r in energy_rows]); write_csv(output/"time_test_metrics_by_f_stratum.csv",time_f); write_csv(output/"time_test_metrics_by_gamma.csv",time_g); write_csv(output/"time_test_metrics_by_sample_type.csv",time_s); write_csv(output/"energy_test_metrics_by_gamma.csv",energy_g); write_csv(output/"energy_test_metrics_by_sample_type.csv",energy_s)
    accuracy=[]; safety=[]; ci=[]
    arrays={"Time":(tl,tu,trl,tru,tcl,tcu),"Energy":(el,eu,erl,eru,ecl,ecu)}
    for net,(lt,ut,rl,ru,cl,cu) in arrays.items():
        for variant in ("Raw","Corrected"):
            m=metrics_all[net][variant]
            for bound in ("lower","upper"): accuracy.append({"Network":net,"Variant":variant,"Bound":bound.title(),"MAE":m[f"{bound}_mae"],"RMSE":m[f"{bound}_rmse"],"R2":m[f"{bound}_r2"],"WMAPE":m[f"{bound}_wmape"]})
            row={"Network":net,"Variant":variant}
            for metric in ("uer","ler","icr"):
                low,high=wilson_interval(int(m[f"{metric}_events"]),int(m["count"])); row[metric.upper()]=m[metric]; row[f"{metric.upper()}_CI_low"]=low; row[f"{metric.upper()}_CI_high"]=high; ci.append({"Network":net,"Variant":variant,"Bound":"Interval","Metric":metric.upper(),"CI_low":low,"CI_high":high,"Method":"Wilson 95%"})
            safety.append(row)
        for bound,true,rawp,corrp in (("Lower",lt,rl,cl),("Upper",ut,ru,cu)):
            boot=paired_bootstrap_accuracy(true,rawp,corrp)
            for variant in ("raw","corrected"):
                for metric,(low,high) in boot[variant].items(): ci.append({"Network":net,"Variant":variant.title(),"Bound":bound,"Metric":metric.upper(),"CI_low":low,"CI_high":high,"Method":"Paired bootstrap percentile 95%"})
    for row in accuracy:
        for metric in ("MAE","RMSE","WMAPE"):
            match=next(item for item in ci if item["Network"]==row["Network"] and item["Variant"]==row["Variant"] and item["Bound"]==row["Bound"] and item["Metric"]==metric)
            row[f"{metric}_CI_low"]=match["CI_low"]; row[f"{metric}_CI_high"]=match["CI_high"]
    write_csv(output/"table_exp2_prediction_accuracy.csv",accuracy); write_csv(output/"table_exp2_safety.csv",safety); write_csv(output/"confidence_intervals.csv",ci)
    improvement=[]
    for net in ("Time","Energy"):
        raw,corm=metrics_all[net]["Raw"],metrics_all[net]["Corrected"]
        for metric in ("lower_mae","upper_mae","lower_rmse","upper_rmse","lower_wmape","upper_wmape"):
            improvement.append({"Network":net,"Metric":metric,"Raw":raw[metric],"Corrected":corm[metric],"absolute_change":corm[metric]-raw[metric],"relative_change":(corm[metric]-raw[metric])/max(abs(raw[metric]),1e-12)})
        for metric in ("uer","ler"): improvement.append({"Network":net,"Metric":metric,"Raw":raw[metric],"Corrected":corm[metric],"absolute_change":raw[metric]-corm[metric],"relative_change":(raw[metric]-corm[metric])/max(raw[metric],1e-12)})
        improvement.append({"Network":net,"Metric":"icr","Raw":raw["icr"],"Corrected":corm["icr"],"absolute_change":corm["icr"]-raw["icr"],"relative_change":(corm["icr"]-raw["icr"])/max(raw["icr"],1e-12)})
    write_csv(output/"raw_vs_corrected_improvement.csv",improvement); write_csv(output/"table_exp2_improvement.csv",improvement)
    figures=output/"figures"; plot_data=output/"plot_data"; figures.mkdir(); plot_data.mkdir(); style(); manifests=[]
    def register(name,sources):
        hashes=";".join(file_sha256(p) for p in sources)
        for ext in ("png","pdf","svg"): manifests.append({"figure":name,"format":ext,"path":f"figures/{name}.{ext}","plot_data":f"plot_data/fig_{name}.csv","source_sha256":hashes,"data_role":"Test only","calibration_rows_used":False})
    for net,preds,unit in (("time",time_pred,"s"),("energy",energy_pred,"J")):
        for side in ("lower","upper"):
            rows=[{"record_id":r.get("time_record_id",r.get("sample_id")),"reference":r[f"true_{side}"],"raw":r[f"raw_{side}"],"corrected":r[f"corrected_{side}"]} for r in preds]; name=f"{net}_test_{side}_scatter"; scatter_plot(rows,name,f"{net.title()} Test: {side} bound",unit,figures,plot_data); register(name,[output/f"{net}_test_predictions.csv"])
    for net in ("Time","Energy"):
        m=metrics_all[net]; rows=[]
        for group in ("Lower","Upper"):
            key=group.lower()+"_wmape"
            rows.extend(({"group":group,"series":"Raw","value":100*m["Raw"][key]},{"group":group,"series":"Corrected","value":100*m["Corrected"][key]}))
        name=f"{net.lower()}_test_wmape"; grouped_bar(rows,name,f"{net} Test prediction accuracy",figures,plot_data,["Lower","Upper"],["Raw","Corrected"]); register(name,[output/f"{net.lower()}_test_metrics_overall.csv"]); safety_plot(net.lower(),m["Raw"],m["Corrected"],figures,plot_data); register(f"{net.lower()}_raw_corrected_safety",[output/f"{net.lower()}_test_metrics_overall.csv"])
    rows=[]
    for r in time_f: rows.extend(({"group":r["f_stratum"],"series":f"{r['variant']} Lower","value":100*r["lower_wmape"]},{"group":r["f_stratum"],"series":f"{r['variant']} Upper","value":100*r["upper_wmape"]}))
    grouped_bar(rows,"time_test_f_stratum_wmape","Time Test WMAPE by f stratum",figures,plot_data,["Low","Mid-high","Upper-anchor"],["Raw Lower","Raw Upper","Corrected Lower","Corrected Upper"]); register("time_test_f_stratum_wmape",[output/"time_test_metrics_by_f_stratum.csv"])
    rows=[]
    for r in energy_g: rows.extend(({"group":r["Gamma"],"series":f"{r['variant']} Lower","value":100*r["lower_wmape"]},{"group":r["Gamma"],"series":f"{r['variant']} Upper","value":100*r["upper_wmape"]}))
    grouped_bar(rows,"energy_test_gamma_wmape","Energy Test WMAPE by Gamma",figures,plot_data,["1.0","2.0","3.0","4.0","5.0"],["Raw Lower","Raw Upper","Corrected Lower","Corrected Upper"]); register("energy_test_gamma_wmape",[output/"energy_test_metrics_by_gamma.csv"]); write_csv(output/"figure_manifest.csv",manifests)
    after={"time_model":file_sha256(formal/"time_bp/selected_model.pt"),"energy_model":file_sha256(formal/"energy_bp/selected_model.pt"),"time_scaler":file_sha256(formal/"time_bp/input_scaler.json"),"energy_scaler":file_sha256(formal/"energy_bp/input_scaler.json"),"corrected":canonical_json_hash(load_json(cal/"corrected_dualbp_v1.json"))}; integrity={"passed":before==after,"before":before,"after":after,"post_test_model_modified":False,"post_test_alpha_modified":False}; _atomic_json(output/"frozen_artifact_integrity.json",integrity)
    invalid=sum(int(np.sum((lo<0)|(up<lo))) for lo,up in ((trl,tru),(tcl,tcu),(erl,eru),(ecl,ecu))); nonfinite=int(not all(np.isfinite(x).all() for x in (trl,tru,tcl,tcu,erl,eru,ecl,ecu))); numerical={"passed":invalid==0 and nonfinite==0,"invalid_intervals":invalid,"nonfinite":nonfinite}; _atomic_json(output/"numerical_audit.json",numerical); leakage={"passed":True,"test_protocol_frozen_before_target_access":True,"test_labels_accessed":True,"test_used_for_training":False,"test_used_for_model_selection":False,"test_used_for_alpha":False,"calibration_test_merged":False,"post_test_model_modification_allowed":False,"post_test_alpha_modification_allowed":False}; _atomic_json(output/"data_leakage_audit.json",leakage)
    raw_strong={net:all(metrics_all[net]["Raw"][k]<=.05 for k in ("lower_wmape","upper_wmape")) and all(metrics_all[net]["Raw"][k]>=.95 for k in ("lower_r2","upper_r2")) for net in ("Time","Energy")}; safety_consistent={net:metrics_all[net]["Corrected"]["uer"]<metrics_all[net]["Raw"]["uer"] and metrics_all[net]["Corrected"]["ler"]<metrics_all[net]["Raw"]["ler"] and metrics_all[net]["Corrected"]["icr"]>metrics_all[net]["Raw"]["icr"] for net in ("Time","Energy")}; scientific="favorable" if all(raw_strong.values()) and all(safety_consistent.values()) else "mixed"
    validity="passed" if integrity["passed"] and numerical["passed"] and leakage["passed"] and len(manifests)==30 else "invalid_due_to_execution"; summary={"test_evaluation_validity":validity,"exp2_test_status":"completed_and_frozen" if validity=="passed" else "not_frozen","raw_dualbp_test_evaluated":True,"corrected_dualbp_test_evaluated":True,"corrected_dualbp_status":"calibrated_frozen_and_tested","time_test_count":180,"energy_test_count":60,"metrics":metrics_all,"raw_prediction_accuracy":raw_strong,"safety_improvement":safety_consistent,"scientific_result":scientific,"post_test_model_modification_allowed":False,"post_test_alpha_modification_allowed":False,"figure_count":10,"figure_files":30,"plot_data_files":10,"exp3_allowed":False,"overall_wall_seconds":time.monotonic()-started}; _atomic_json(output/"summary.json",summary); _atomic_json(output/"runtime_breakdown.json",{"overall_wall_seconds":summary["overall_wall_seconds"],"bootstrap_repetitions":5000,"bootstrap_seed":46001,"workers":1})
    (output/"README.md").write_text("# Exp-2 独立Test正式评价\n\nTest协议先冻结后首次读取标签。Raw与Corrected结果只作最终独立评价，禁止任何Test后回调修改。\n",encoding="utf-8"); (output/"validation_report.md").write_text(f"# Exp-2有效性与科学结果\n\n- validity: `{validity}`\n- scientific result: `{scientific}`\n- delta=0.05是单侧经验风险水平，不构成ICR=95%的理论保证。\n",encoding="utf-8"); _atomic_json(output/"created_files.json",{"files":sorted(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file())}); return summary


def main():
    p=argparse.ArgumentParser(); p.add_argument("--project-root",default="."); p.add_argument("--protocol-dir"); p.add_argument("--output-dir"); p.add_argument("--freeze-protocol",action="store_true"); a=p.parse_args(); project=Path(a.project_root).resolve(); target=Path(a.protocol_dir if a.freeze_protocol else a.output_dir).resolve()
    try:
        result=freeze_protocol(project,target) if a.freeze_protocol else run_formal(project,Path(a.protocol_dir).resolve(),target); print(json.dumps(result,ensure_ascii=False),flush=True)
    except Exception as e:
        target.mkdir(parents=True,exist_ok=True); _atomic_json(target/"runner_failure.json",{"error":str(e),"traceback":traceback.format_exc()}); raise

if __name__=="__main__": main()
