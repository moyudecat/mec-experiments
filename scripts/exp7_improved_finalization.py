"""Offline-only finalization of the persistence-recovery Exp-7 Formal."""
from __future__ import annotations

import argparse, csv, hashlib, json, pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from mec_dualbp.optimization.evaluation_archive import EvaluationArchive
from mec_dualbp.optimization.metrics import global_normalize, hypervolume_2d_min

ALGORITHMS = ["CorrectedDualBP-CA-CNSGA-II", "MOEA/D", "RVEA", "MOPSO", "Random"]
SCENARIOS = ["S1", "S2", "S3", "S4", "S5"]
SEEDS = [52001, 52002, 52003, 52004, 52005]
MAIN = ALGORITHMS[0]
PROTOCOL = "edc029f40cf45c1d99f63c44d988f655b2b13cc078e291e94ea66287e6752b2e"
EVALUATOR = "b505197af2f99f526d24f31ca0c4be1c6d5082612b19fe94a589f823ff1e35d7"


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)


def independent_front(records: tuple[dict[str, Any], ...]) -> set[str]:
    groups: dict[float, list[dict[str, Any]]] = {}
    for row in records: groups.setdefault(float(row["Phi_T"]), []).append(row)
    selected: set[str] = set(); best_e = np.inf
    for time_value in sorted(groups):
        group = groups[time_value]; group_min = min(float(r["Phi_E"]) for r in group)
        if group_min < best_e:
            selected.update(r["decision_hash"] for r in group if float(r["Phi_E"]) == group_min)
            best_e = group_min
    return selected


def nd_objectives(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[float,float],dict[str,Any]] = {}
    for row in rows: unique.setdefault((float(row["Phi_T"]),float(row["Phi_E"])),row)
    ordered=sorted(unique.values(),key=lambda r:(float(r["Phi_T"]),float(r["Phi_E"])))
    out=[];best=np.inf
    for row in ordered:
        energy=float(row["Phi_E"])
        if energy < best: out.append(row);best=energy
    return out


def holm(values: list[tuple[int,float]]) -> dict[int,float]:
    ordered=sorted(values,key=lambda x:x[1]);out={};running=0.;m=len(values)
    for rank,(index,p) in enumerate(ordered):
        running=max(running,min(1.,(m-rank)*p));out[index]=running
    return out


def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument("--formal",required=True);ap.add_argument("--mirror",required=True);ap.add_argument("--audit",required=True)
    args=ap.parse_args();formal=Path(args.formal).resolve();mirror=Path(args.mirror).resolve();audit=Path(args.audit).resolve();audit.mkdir(parents=True,exist_ok=True)
    summary=json.loads((formal/"summary.json").read_text(encoding="utf-8"))
    if not (summary["completed_runs"]==summary["persisted_completed_runs"]==summary["mirror_valid_completed_runs"]==125 and summary["failed_runs"]==0): raise RuntimeError("persistence completion gate failed")
    completion=[];oracle=[];mirror_rows=[];runs={}
    manifests={tuple(json.loads(p.read_text(encoding="utf-8"))["identity"].get(k) for k in ("scenario","algorithm","seed")):json.loads(p.read_text(encoding="utf-8")) for p in (mirror/"manifests").glob("*.json")}
    for scenario in SCENARIOS:
      for algorithm in ALGORITHMS:
       for seed in SEEDS:
        run=formal/"scenarios"/scenario/"runs"/f"{algorithm.replace('/','_')}__seed_{seed}"
        status=json.loads((run/"status.json").read_text(encoding="utf-8"));payload=pickle.loads((run/"checkpoint.pkl").read_bytes());archive=EvaluationArchive.from_cache(payload["evaluator"]["cache"]);front=archive.official_front()
        production={r["decision_hash"] for r in front};independent=independent_front(archive.feasible_records());oracle_ok=production==independent
        required=all((run/name).is_file() for name in ("status.json","checkpoint.pkl","evaluation_archive_metadata.json","final_front.csv","history.csv"))
        completion.append({"scenario":scenario,"algorithm":algorithm,"seed":seed,"status":status["status"],"requests":status["requests"],"required_artifacts":required,"evaluator_match":status.get("evaluator_hash")==EVALUATOR,"protocol_match":status.get("effective_protocol_hash")==PROTOCOL,"nonfinite_count":status["nonfinite_count"],"front_size":status["front_size"]})
        oracle.append({"scenario":scenario,"algorithm":algorithm,"seed":seed,"production_front_size":len(production),"independent_front_size":len(independent),"decision_hashes_equal":oracle_ok})
        manifest=manifests[(scenario,algorithm,seed)];mirror_valid=manifest["source_artifact_hashes"]==manifest["mirror_artifact_hashes"] and all(sha(run/path)==digest for path,digest in manifest["source_artifact_hashes"].items())
        mirror_rows.append({"scenario":scenario,"algorithm":algorithm,"seed":seed,"mirror_valid":mirror_valid,"artifact_count":len(manifest["source_artifact_hashes"])})
        runs[(scenario,algorithm,seed)]={"status":status,"archive":archive,"front":front}
    write_csv(audit/"formal_completion_audit.csv",completion);write_csv(audit/"mirror_integrity_audit.csv",mirror_rows);write_csv(audit/"evaluation_archive_oracle.csv",oracle)
    if not all(r["status"]=="completed" and r["requests"]==20000 and r["required_artifacts"] and r["evaluator_match"] and r["protocol_match"] and r["nonfinite_count"]==0 for r in completion): raise RuntimeError("Formal artifact gate failed")
    if not all(r["mirror_valid"] for r in mirror_rows) or not all(r["decision_hashes_equal"] for r in oracle): raise RuntimeError("mirror/oracle gate failed")
    metrics=[];pooled_sizes={}
    for scenario in SCENARIOS:
        all_feasible=[r for a in ALGORITHMS for s in SEEDS for r in runs[(scenario,a,s)]["archive"].feasible_records()];all_values=np.asarray([[r["Phi_T"],r["Phi_E"]] for r in all_feasible]);lo,hi=all_values.min(0),all_values.max(0);safe_hi=np.where(hi>lo,hi,lo+1.)
        pooled=nd_objectives([r for a in ALGORITHMS for s in SEEDS for r in runs[(scenario,a,s)]["front"]]);pooled_sizes[scenario]=len(pooled);ref_z=global_normalize([[r["Phi_T"],r["Phi_E"]] for r in pooled],lo,safe_hi);ref_point=global_normalize([hi+.1*(hi-lo)],lo,safe_hi)[0];ref_point=np.where(hi>lo,ref_point,.1*np.maximum(np.abs(hi),1.))
        for algorithm in ALGORITHMS:
          for seed in SEEDS:
            item=runs[(scenario,algorithm,seed)];z=global_normalize([[r["Phi_T"],r["Phi_E"]] for r in item["front"]],lo,safe_hi);dist=np.sqrt(((ref_z[:,None,:]-z[None,:,:])**2).sum(2));status=item["status"]
            metrics.append({"scenario":scenario,"algorithm":algorithm,"seed":seed,"HV":hypervolume_2d_min(z,ref_point),"IGD":float(dist.min(1).mean()),"front_size":len(item["front"]),"feasible_rate":status["feasible_rate"],"has_feasible":len(item["archive"].feasible_records())>0,"wall_seconds":status["wall_seconds"],"unique_candidates":status["unique_evaluations"],"cache_hits":status["cache_hits"]})
    rank_rows=[]
    for scenario in SCENARIOS:
      for seed in SEEDS:
        block=[next(r for r in metrics if r["scenario"]==scenario and r["seed"]==seed and r["algorithm"]==a) for a in ALGORITHMS];hr=rankdata([-r["HV"] for r in block],method="average");ir=rankdata([r["IGD"] for r in block],method="average")
        for row,h,i in zip(block,hr,ir):rank_rows.append({**row,"HV_rank":float(h),"IGD_rank":float(i)})
    hv_summary=[];igd_summary=[];front_summary=[];efficiency=[]
    for algorithm in ALGORITHMS:
        subset=[r for r in rank_rows if r["algorithm"]==algorithm]
        hv_summary.append({"algorithm":algorithm,"mean":float(np.mean([r["HV"] for r in subset])),"std":float(np.std([r["HV"] for r in subset],ddof=1)),"median":float(np.median([r["HV"] for r in subset])),"average_rank":float(np.mean([r["HV_rank"] for r in subset]))})
        igd_summary.append({"algorithm":algorithm,"mean":float(np.mean([r["IGD"] for r in subset])),"std":float(np.std([r["IGD"] for r in subset],ddof=1)),"median":float(np.median([r["IGD"] for r in subset])),"average_rank":float(np.mean([r["IGD_rank"] for r in subset]))})
        sizes=[r["front_size"] for r in subset];front_summary.append({"algorithm":algorithm,"mean":float(np.mean(sizes)),"median":float(np.median(sizes)),"min":min(sizes),"max":max(sizes),"front_size_1_count":sum(x==1 for x in sizes)})
        efficiency.append({"algorithm":algorithm,"wall_seconds_mean":float(np.mean([r["wall_seconds"] for r in subset])),"unique_candidates_mean":float(np.mean([r["unique_candidates"] for r in subset])),"cache_hits_mean":float(np.mean([r["cache_hits"] for r in subset])),"feasible_rate_mean":float(np.mean([r["feasible_rate"] for r in subset]))})
    write_csv(audit/"hv_summary.csv",hv_summary);write_csv(audit/"igd_summary.csv",igd_summary);write_csv(audit/"front_size_summary.csv",front_summary);write_csv(audit/"efficiency_summary.csv",efficiency)
    stats=[];wtl=[];effects=[]
    for metric in ("HV","IGD"):
        vectors=[[next(r[metric] for r in metrics if r["scenario"]==sc and r["seed"]==sd and r["algorithm"]==a) for sc in SCENARIOS for sd in SEEDS] for a in ALGORITHMS];fr=friedmanchisquare(*vectors);stats.append({"test":"Friedman","metric":metric,"baseline":"ALL","statistic":float(fr.statistic),"raw_p":float(fr.pvalue),"holm_adjusted_p":""})
        raw=[];temporary=[];main=np.asarray(vectors[0])
        for index,algorithm in enumerate(ALGORITHMS[1:]):
            base=np.asarray(vectors[index+1]);difference=main-base if metric=="HV" else base-main;test=wilcoxon(difference,zero_method="wilcox",alternative="two-sided");nonzero=difference[difference!=0];ranks=rankdata(np.abs(nonzero));rbc=float(np.sum(ranks*np.sign(nonzero))/np.sum(ranks)) if len(ranks) else 0.;raw.append((index,float(test.pvalue)));temporary.append((algorithm,difference,float(test.pvalue),rbc))
        adjusted=holm(raw)
        for index,(algorithm,difference,p,rbc) in enumerate(temporary):
            w,t,l=int(np.sum(difference>0)),int(np.sum(difference==0)),int(np.sum(difference<0));stats.append({"test":"Wilcoxon-Holm","metric":metric,"baseline":algorithm,"statistic":"","raw_p":p,"holm_adjusted_p":adjusted[index]});wtl.append({"metric":metric,"baseline":algorithm,"win":w,"tie":t,"loss":l});effects.append({"metric":metric,"baseline":algorithm,"rank_biserial_correlation":rbc,"positive_favors_main":True})
    write_csv(audit/"pairwise_wtl.csv",wtl);write_csv(audit/"friedman_holm.csv",stats);write_csv(audit/"effect_size.csv",effects)
    hv_rank=next(r["average_rank"] for r in hv_summary if r["algorithm"]==MAIN);igd_rank=next(r["average_rank"] for r in igd_summary if r["algorithm"]==MAIN);rvea_hv=next(r for r in wtl if r["metric"]=="HV" and r["baseline"]=="RVEA");rvea_igd=next(r for r in wtl if r["metric"]=="IGD" and r["baseline"]=="RVEA");significant=any(r["test"]=="Wilcoxon-Holm" and r["holm_adjusted_p"]<.05 for r in stats);feasible=all(r["has_feasible"] for r in metrics)
    gate={"HV_average_rank_is_1":hv_rank==1.,"IGD_average_rank_is_1":igd_rank==1.,"vs_RVEA_HV_win_gt_loss":rvea_hv["win"]>rvea_hv["loss"],"vs_RVEA_IGD_win_gt_loss":rvea_igd["win"]>rvea_igd["loss"],"at_least_one_Holm_adjusted_p_lt_0_05":significant,"no_run_level_feasibility_regression":feasible};gate["FINAL_V2_GATE"]="PASS" if all(gate.values()) else "FAIL";write_json(audit/"final_v2_gate.json",gate)
    write_json(audit/"failed_formal_binding.json",{"path":"outputs/01_formal/exp7_improved_dualbp_final_20260819_200944","status":"INVALID_DUE_TO_PERSISTENCE_LOSS","included_in_statistics":False});write_json(audit/"scientific_protocol_binding.json",{"scientific_protocol_hash":PROTOCOL,"evaluator_hash":EVALUATOR,"scientific_parameters_modified":False});rerun=json.loads((formal/"rerun_binding.json").read_text(encoding="utf-8"));write_json(audit/"persistence_guard_binding.json",rerun)
    result={"formal":str(formal),"audit":str(audit),"completed":125,"persisted":125,"mirror_valid":125,"failed":0,"budget_violations":0,"second_persistence_anomaly":summary["recovered_from_mirror_count"]>0,"oracle_passed":sum(r["decision_hashes_equal"] for r in oracle),"pooled_front_sizes":pooled_sizes,"front_size_1_count":sum(r["front_size"]==1 for r in metrics),"feasibility_regression":not feasible,"hv_average_ranks":{r["algorithm"]:r["average_rank"] for r in hv_summary},"igd_average_ranks":{r["algorithm"]:r["average_rank"] for r in igd_summary},"gate":gate,"automatic_tuning":False,"formal_elapsed_seconds":summary["elapsed_seconds"]}
    write_json(audit/"final_result_manifest.json",result);(audit/"README.md").write_text("# Exp-7 Improved DualBP 最终冻结审计\n\n仅使用 persistence-loss recovery rerun 的125个正式run；旧失败Formal完全排除。\n",encoding="utf-8")
    created=sorted(p.relative_to(audit).as_posix() for p in audit.rglob("*") if p.is_file());write_json(audit/"created_files.json",{"count":len(created)+1,"files":created+["created_files.json"]});print(json.dumps(result,ensure_ascii=False))


if __name__ == "__main__": main()
