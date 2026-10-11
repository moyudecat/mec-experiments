"""Offline finalization of the 150-run OE-RVEA six-algorithm comparison."""
from __future__ import annotations

import argparse, csv, hashlib, json, pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from mec_dualbp.experiments.oe_rvea_final_compare_runner import (
    ALL_ALGORITHMS, ALGORITHMS as ADDITIONAL_ALGORITHMS, BUDGET, DESIGN_HASH,
    IMPLEMENTATION_HASH, SCENARIOS, SEEDS, CA_IMPLEMENTATION_HASH,
    final_protocol_hash, final_protocol_payload,
)
from mec_dualbp.optimization.adapters import V2_DESIGN_HASH
from mec_dualbp.optimization.evaluation_archive import EvaluationArchive
from mec_dualbp.optimization.metrics import global_normalize, hypervolume_2d_min
from mec_dualbp.optimization.outer_contract import IMPROVED_CORRECTED_DUALBP_HASH


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def jout(path: Path, value: Any) -> None: path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def cout(path: Path, rows: list[dict[str,Any]]) -> None:
    fields=list(rows[0]) if rows else ["status"]
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)


def independent_front(records: tuple[dict[str,Any],...]) -> set[str]:
    groups:dict[float,list[dict[str,Any]]]={}
    for row in records: groups.setdefault(float(row["Phi_T"]),[]).append(row)
    selected:set[str]=set();best=np.inf
    for t in sorted(groups):
        minimum=min(float(row["Phi_E"]) for row in groups[t])
        if minimum<best:
            selected.update(row["decision_hash"] for row in groups[t] if float(row["Phi_E"])==minimum);best=minimum
    return selected


def objective_front(rows:list[dict[str,Any]]) -> np.ndarray:
    unique=sorted({(float(row["Phi_T"]),float(row["Phi_E"])) for row in rows});selected=[];best=np.inf
    for point in unique:
        if point[1]<best: selected.append(point);best=point[1]
    return np.asarray(selected,dtype=np.float64).reshape(-1,2)


def holm(items:list[tuple[int,float]]) -> dict[int,float]:
    ordered=sorted(items,key=lambda row:row[1]);out={};running=0.;count=len(items)
    for rank,(index,p) in enumerate(ordered):
        running=max(running,min(1.,(count-rank)*p));out[index]=running
    return out


def paired(difference:np.ndarray) -> tuple[float,float]:
    nonzero=difference[difference!=0]
    if not len(nonzero): return 1.,0.
    p=float(wilcoxon(difference,zero_method="wilcox",alternative="two-sided").pvalue)
    ranks=rankdata(np.abs(nonzero));effect=float(np.sum(ranks*np.sign(nonzero))/np.sum(ranks))
    return p,effect


def finalize(project:Path,gate:Path,additional:Path,mirror:Path,audit:Path)->dict[str,Any]:
    audit.mkdir(parents=True,exist_ok=True)
    add_summary=json.loads((additional/"summary.json").read_text(encoding="utf-8"))
    if not(add_summary["completed_runs"]==add_summary["persisted_completed_runs"]==add_summary["mirror_valid_completed_runs"]==100 and add_summary["failed_runs"]==0):
        raise RuntimeError("additional Formal completion gate failed")
    manifests={}
    for path in (mirror/"manifests").glob("*.json"):
        row=json.loads(path.read_text(encoding="utf-8"));identity=row["identity"]
        manifests[(identity["scenario"],identity["algorithm"],identity["seed"])]=row
    runs={};reuse=[];additional_rows=[];init_rows=[];oracle=[];feasibility=[];source_manifest=[]
    for scenario in SCENARIOS:
      for seed in SEEDS:
        initial=[]
        for algorithm in ALL_ALGORITHMS:
          source=gate if algorithm in ("OE-RVEA","RVEA") else additional
          run=source/"scenarios"/scenario/"runs"/f"{algorithm.replace('/','_')}__seed_{seed}"
          status=json.loads((run/"status.json").read_text(encoding="utf-8"));payload=pickle.loads((run/"checkpoint.pkl").read_bytes())
          archive=EvaluationArchive.from_cache(payload["evaluator"]["cache"]);front=archive.official_front();production={row["decision_hash"] for row in front};independent=independent_front(archive.feasible_records())
          required=all((run/name).is_file() for name in ("status.json","checkpoint.pkl","evaluation_archive_metadata.json","final_front.csv","history.csv"))
          common={"scenario":scenario,"algorithm":algorithm,"seed":seed,"status":status["status"],"requests":status["requests"],"required_artifacts":required,"evaluator_match":status.get("evaluator_hash")==IMPROVED_CORRECTED_DUALBP_HASH,"nonfinite_count":status["nonfinite_count"]}
          if algorithm in ("OE-RVEA","RVEA"):
            common["source"]="Gate-A immutable reuse";reuse.append(common)
          else:
            common["source"]="Additional Formal";common["protocol_match"]=status.get("effective_protocol_hash")==final_protocol_hash();additional_rows.append(common)
            manifest=manifests[(scenario,algorithm,seed)];valid=manifest["source_artifact_hashes"]==manifest["mirror_artifact_hashes"] and all(sha(run/name)==digest for name,digest in manifest["source_artifact_hashes"].items())
            if not valid: raise RuntimeError("additional mirror mismatch")
          initial.append(status["initial_pool_hash"])
          oracle.append({"scenario":scenario,"algorithm":algorithm,"seed":seed,"production_front_size":len(production),"independent_front_size":len(independent),"oracle_pass":production==independent})
          has_feasible=len(archive.feasible_records())>0;feasibility.append({"scenario":scenario,"algorithm":algorithm,"seed":seed,"has_feasible":has_feasible,"feasible_rate":status["feasible_rate"]})
          hashes={p.relative_to(run).as_posix():sha(p) for p in sorted(run.rglob("*")) if p.is_file()};source_manifest.append({"source_directory":source.relative_to(project).as_posix(),"scenario":scenario,"algorithm":algorithm,"seed":seed,"artifact_count":len(hashes),"artifact_manifest_sha256":hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(",",":")).encode()).hexdigest()})
          runs[(scenario,algorithm,seed)]={"status":status,"archive":archive,"front":front}
        init_rows.append({"scenario":scenario,"seed":seed,"algorithms":6,"unique_initialization_hashes":len(set(initial)),"all_six_match":len(set(initial))==1,"initialization_hash":initial[0]})
    cout(audit/"gate_a_reuse_audit.csv",reuse);cout(audit/"additional_formal_completion.csv",additional_rows);cout(audit/"initialization_fairness_audit.csv",init_rows);cout(audit/"evaluation_archive_oracle.csv",oracle);cout(audit/"feasibility_audit.csv",feasibility)
    if not all(row["status"]=="completed" and row["requests"]==BUDGET and row["required_artifacts"] and row["evaluator_match"] and row["nonfinite_count"]==0 for row in reuse+additional_rows): raise RuntimeError("150-run artifact gate failed")
    if not all(row.get("protocol_match",True) for row in additional_rows) or not all(row["all_six_match"] for row in init_rows) or not all(row["oracle_pass"] for row in oracle): raise RuntimeError("protocol/initialization/oracle gate failed")
    jout(audit/"source_formal_binding.json",{"gate_a":{"directory":gate.relative_to(project).as_posix(),"runs":50,"immutable_reuse":True},"additional":{"directory":additional.relative_to(project).as_posix(),"runs":100},"artifact_manifest":source_manifest})

    metrics=[];pooled_summary=[]
    for scenario in SCENARIOS:
        feasible_rows=[row for algorithm in ALL_ALGORITHMS for seed in SEEDS for row in runs[(scenario,algorithm,seed)]["archive"].feasible_records()]
        values=np.asarray([[row["Phi_T"],row["Phi_E"]] for row in feasible_rows]);lo,hi=values.min(0),values.max(0);safe=np.where(hi>lo,hi,lo+1.)
        pooled=objective_front([row for algorithm in ALL_ALGORITHMS for seed in SEEDS for row in runs[(scenario,algorithm,seed)]["front"]]);ref=global_normalize(pooled,lo,safe);refpoint=global_normalize([hi+.1*(hi-lo)],lo,safe)[0];refpoint=np.where(hi>lo,refpoint,.1*np.maximum(np.abs(hi),1.))
        pooled_summary.append({"scenario":scenario,"source_fronts":30,"pooled_front_size":len(pooled),"normalization_lower_T":lo[0],"normalization_lower_E":lo[1],"normalization_upper_T":hi[0],"normalization_upper_E":hi[1]})
        for algorithm in ALL_ALGORITHMS:
          for seed in SEEDS:
            item=runs[(scenario,algorithm,seed)];front=np.asarray([[row["Phi_T"],row["Phi_E"]] for row in item["front"]]);z=global_normalize(front,lo,safe);distance=np.linalg.norm(ref[:,None,:]-z[None,:,:],axis=2)
            metrics.append({"scenario":scenario,"algorithm":algorithm,"seed":seed,"HV":hypervolume_2d_min(z,refpoint),"IGD":float(distance.min(1).mean()),"front_size":len(front)})
    cout(audit/"pooled_reference_front_summary.csv",pooled_summary);cout(audit/"hv_all_runs.csv",[{k:r[k] for k in ("scenario","algorithm","seed","HV")} for r in metrics]);cout(audit/"igd_all_runs.csv",[{k:r[k] for k in ("scenario","algorithm","seed","IGD")} for r in metrics])
    ranked=[]
    for scenario in SCENARIOS:
      for seed in SEEDS:
        block=[next(row for row in metrics if row["scenario"]==scenario and row["seed"]==seed and row["algorithm"]==algorithm) for algorithm in ALL_ALGORITHMS];hr=rankdata([-row["HV"] for row in block],method="average");ir=rankdata([row["IGD"] for row in block],method="average")
        for row,h,i in zip(block,hr,ir): ranked.append({**row,"HV_rank":float(h),"IGD_rank":float(i)})
    algorithm_summary=[];average_rank=[];scenario_summary=[]
    for algorithm in ALL_ALGORITHMS:
        subset=[row for row in ranked if row["algorithm"]==algorithm]
        algorithm_summary.append({"algorithm":algorithm,"HV_mean":np.mean([r["HV"] for r in subset]),"HV_std":np.std([r["HV"] for r in subset],ddof=1),"IGD_mean":np.mean([r["IGD"] for r in subset]),"IGD_std":np.std([r["IGD"] for r in subset],ddof=1)})
        average_rank.append({"algorithm":algorithm,"HV_average_rank":np.mean([r["HV_rank"] for r in subset]),"IGD_average_rank":np.mean([r["IGD_rank"] for r in subset])})
        for scenario in SCENARIOS:
            local=[r for r in subset if r["scenario"]==scenario];scenario_summary.append({"scenario":scenario,"algorithm":algorithm,"HV_mean":np.mean([r["HV"] for r in local]),"HV_std":np.std([r["HV"] for r in local],ddof=1),"IGD_mean":np.mean([r["IGD"] for r in local]),"IGD_std":np.std([r["IGD"] for r in local],ddof=1),"HV_mean_rank":np.mean([r["HV_rank"] for r in local]),"IGD_mean_rank":np.mean([r["IGD_rank"] for r in local])})
    cout(audit/"algorithm_summary.csv",algorithm_summary);cout(audit/"scenario_summary.csv",scenario_summary);cout(audit/"average_rank.csv",average_rank)

    vectors={metric:{algorithm:np.asarray([next(r[metric] for r in metrics if r["scenario"]==scenario and r["seed"]==seed and r["algorithm"]==algorithm) for scenario in SCENARIOS for seed in SEEDS]) for algorithm in ALL_ALGORITHMS} for metric in ("HV","IGD")}
    stats=[];wtl=[];effects=[];friedman_p={}
    for metric in ("HV","IGD"):
        test=friedmanchisquare(*[vectors[metric][algorithm] for algorithm in ALL_ALGORITHMS]);friedman_p[metric]=float(test.pvalue);stats.append({"test":"Friedman","metric":metric,"baseline":"ALL","statistic":float(test.statistic),"raw_p":float(test.pvalue),"holm_adjusted_p":""})
        raw=[];temporary=[];oe=vectors[metric]["OE-RVEA"]
        for index,baseline in enumerate(ALL_ALGORITHMS[1:]):
            base=vectors[metric][baseline];difference=oe-base if metric=="HV" else base-oe;p,effect=paired(difference);raw.append((index,p));temporary.append((baseline,difference,p,effect))
        adjusted=holm(raw) if test.pvalue<.05 else {index:p for index,p in raw}
        for index,(baseline,difference,p,effect) in enumerate(temporary):
            wtl.append({"metric":metric,"baseline":baseline,"win":int(np.sum(difference>0)),"tie":int(np.sum(difference==0)),"loss":int(np.sum(difference<0))});stats.append({"test":"paired Wilcoxon-Holm","metric":metric,"baseline":baseline,"statistic":"","raw_p":p,"holm_adjusted_p":adjusted[index]});effects.append({"metric":metric,"baseline":baseline,"rank_biserial":effect,"positive_favors":"OE-RVEA"})
    cout(audit/"pairwise_wtl.csv",wtl);cout(audit/"friedman_holm.csv",stats);cout(audit/"effect_size.csv",effects)
    rank_oe=next(row for row in average_rank if row["algorithm"]=="OE-RVEA");rvea_hv=next(row for row in wtl if row["metric"]=="HV" and row["baseline"]=="RVEA");rvea_igd=next(row for row in wtl if row["metric"]=="IGD" and row["baseline"]=="RVEA");rvea_tests={row["metric"]:row for row in stats if row["test"]=="paired Wilcoxon-Holm" and row["baseline"]=="RVEA"};rvea_effects={row["metric"]:row for row in effects if row["baseline"]=="RVEA"}
    gates={"A_OE_HV_average_rank_is_1":rank_oe["HV_average_rank"]==1.,"B_OE_IGD_average_rank_is_1":rank_oe["IGD_average_rank"]==1.,"C_vs_RVEA_HV_win_gt_loss":rvea_hv["win"]>rvea_hv["loss"],"D_vs_RVEA_IGD_win_gt_loss":rvea_igd["win"]>rvea_igd["loss"],"E_HV_or_IGD_Holm_significant_favorable":any(rvea_tests[m]["holm_adjusted_p"]<.05 and rvea_effects[m]["rank_biserial"]>0 for m in ("HV","IGD")),"F_other_metric_not_Holm_significant_favorable_RVEA":not any(rvea_tests[m]["holm_adjusted_p"]<.05 and rvea_effects[m]["rank_biserial"]<0 for m in ("HV","IGD")),"G_no_run_level_feasibility_regression":all(row["has_feasible"] for row in feasibility)}
    gates={name:bool(value) for name,value in gates.items()}
    gate_result={**gates,"FINAL_OE_RVEA_GATE":"PASS" if all(gates.values()) else "FAIL","automatic_tuning":False};jout(audit/"final_oe_rvea_gate.json",gate_result)
    jout(audit/"final_protocol.json",final_protocol_payload());jout(audit/"protocol_hash.json",{"oe_rvea_final_comparison_protocol_hash":final_protocol_hash()})
    result={"additional_formal":str(additional),"final_audit":str(audit),"protocol_hash":final_protocol_hash(),"gate_a_reused_runs":50,"additional_runs":100,"total_unique_runs":150,"additional_persisted":100,"failed":0,"budget_violations":0,"initialization_cases_passed":sum(row["all_six_match"] for row in init_rows),"oracle_passed":sum(row["oracle_pass"] for row in oracle),"pooled_front_sizes":{row["scenario"]:row["pooled_front_size"] for row in pooled_summary},"algorithm_summary":algorithm_summary,"average_rank":average_rank,"wtl":wtl,"friedman_p":friedman_p,"holm":[row for row in stats if row["test"]=="paired Wilcoxon-Holm"],"effects":effects,"gate":gate_result,"additional_elapsed_seconds":add_summary["elapsed_seconds"],"scientific_parameters_modified":False};jout(audit/"final_result_manifest.json",result)
    (audit/"README.md").write_text("# OE-RVEA最终六算法正式比较\n\nGate-A 50个raw runs不可变复用，新增100个runs；全部150个指标基于六算法pooled reference重新计算。\n",encoding="utf-8")
    files=sorted(path.name for path in audit.iterdir() if path.is_file());jout(audit/"created_files.json",{"count":len(files)+1,"files":files+["created_files.json"]});return result


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--project-root",default=".");parser.add_argument("--gate",required=True);parser.add_argument("--additional",required=True);parser.add_argument("--mirror",required=True);parser.add_argument("--audit",required=True);args=parser.parse_args();print(json.dumps(finalize(Path(args.project_root).resolve(),Path(args.gate).resolve(),Path(args.additional).resolve(),Path(args.mirror).resolve(),Path(args.audit).resolve()),ensure_ascii=False),flush=True)
if __name__=="__main__":main()
