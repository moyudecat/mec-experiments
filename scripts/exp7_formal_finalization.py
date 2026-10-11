"""Offline-only Exp-7 Formal finalization from persisted paid evaluation caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from mec_dualbp.optimization.evaluation_archive import EvaluationArchive
from mec_dualbp.optimization.metrics import global_normalize, hypervolume_2d_min


ALGORITHMS = ["CorrectedDualBP-CNSGA-II", "MOEA/D", "RVEA", "MOPSO", "Random"]
SCENARIOS = ["S1", "S2", "S3", "S4", "S5"]
SEEDS = [52001, 52002, 52003, 52004, 52005]
MAIN = ALGORITHMS[0]
COLORS = {MAIN: "#0072B2", "MOEA/D": "#E69F00", "RVEA": "#009E73", "MOPSO": "#CC79A7", "Random": "#D55E00"}
MARKERS = {MAIN: "o", "MOEA/D": "s", "RVEA": "^", "MOPSO": "D", "Random": "X"}
LINES = {MAIN: "-", "MOEA/D": "--", "RVEA": "-.", "MOPSO": ":", "Random": (0, (3, 1, 1, 1))}
HATCHES = {MAIN: "", "MOEA/D": "//", "RVEA": "\\\\", "MOPSO": "xx", "Random": ".."}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else ["status"])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def independent_o_n2_hashes(rows: tuple[dict, ...]) -> set[str]:
    """Independent blocked O(n^2) oracle checking every dominance pair."""
    values = np.asarray([[r["Phi_T"], r["Phi_E"]] for r in rows], dtype=np.float64)
    dominated = np.zeros(len(rows), dtype=bool)
    block = 256
    for start in range(0, len(rows), block):
        target = values[start:start + block]
        weak = np.all(values[:, None, :] <= target[None, :, :], axis=2)
        strict = np.any(values[:, None, :] < target[None, :, :], axis=2)
        dominated[start:start + len(target)] = np.any(weak & strict, axis=0)
    return {row["decision_hash"] for row, is_dominated in zip(rows, dominated) if not is_dominated}


def nd_rows(rows: list[dict]) -> list[dict]:
    if not rows: return []
    # Objective duplicates are equivalent for metric reference purposes.
    unique = {}
    for row in rows: unique.setdefault((row["Phi_T"], row["Phi_E"]), row)
    cache = {tuple([float(i)]): {"objectives": [r["Phi_T"], r["Phi_E"]], "constraint_violation": 0,
             "feasible": True} for i, r in enumerate(unique.values())}
    selected = EvaluationArchive.from_cache(cache).official_front()
    keys = {(r["Phi_T"], r["Phi_E"]) for r in selected}
    return [r for r in unique.values() if (r["Phi_T"], r["Phi_E"]) in keys]


def representative(front: tuple[dict, ...]) -> dict:
    values = np.asarray([[r["Phi_T"], r["Phi_E"]] for r in front])
    lo, hi = values.min(0), values.max(0)
    z = np.divide(values-lo, hi-lo, out=np.zeros_like(values), where=hi>lo)
    distance, worst = np.linalg.norm(z, axis=1), z.max(axis=1)
    margin = np.asarray([float(r["qos_min_possibility"] or 0) for r in front])
    i = min(range(len(front)), key=lambda k: (distance[k], -margin[k], worst[k], front[k]["decision_hash"]))
    return front[i]


def describe(values: list[float]) -> dict:
    x=np.asarray(values,float)
    return {"mean":float(x.mean()),"std":float(x.std(ddof=1)),"median":float(np.median(x)),
            "iqr":float(np.percentile(x,75)-np.percentile(x,25))}


def holm(raw: list[tuple[int,float]]) -> dict[int,float]:
    ordered=sorted(raw,key=lambda x:x[1]); result={}; running=0.0; m=len(raw)
    for rank,(idx,p) in enumerate(ordered):
        running=max(running,min(1.0,(m-rank)*p)); result[idx]=running
    return result


def save_figure(fig, base: Path) -> None:
    for ext in ("png","pdf","svg"):
        fig.savefig(base.with_suffix("."+ext), dpi=600 if ext=="png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--project",default="."); ap.add_argument("--formal",required=True); ap.add_argument("--audit",required=True)
    args=ap.parse_args(); root=Path(args.project).resolve(); formal=Path(args.formal).resolve(); audit=Path(args.audit).resolve()
    final=formal/"final_analysis"; tables=final/"tables"; figures=final/"figures"; stats_dir=final/"statistics"; pareto=final/"pareto_fronts"; reps=final/"representative_solutions"; audit_local=final/"audit"
    for p in (tables,figures,stats_dir,pareto,reps,audit_local,audit): p.mkdir(parents=True,exist_ok=True)
    integrity=[]; runs={}; oracle=[]
    for scenario in SCENARIOS:
      for algorithm in ALGORITHMS:
       for seed in SEEDS:
        run_dir=formal/"scenarios"/scenario/"runs"/f"{algorithm.replace('/','_')}__seed_{seed}"
        status=json.loads((run_dir/"status.json").read_text(encoding="utf-8")); payload=pickle.loads((run_dir/"checkpoint.pkl").read_bytes())
        archive=EvaluationArchive.from_cache(payload["evaluator"]["cache"]); front=archive.official_front()
        production={r["decision_hash"] for r in front}; reference=independent_o_n2_hashes(archive.feasible_records())
        consistent=production==reference
        metadata=archive.metadata(paid_requests=payload["evaluator"]["requests"],cache_hits=payload["evaluator"]["cache_hits"])
        front_rows=[{"scenario":scenario,"algorithm":algorithm,"seed":seed,**r} for r in front]
        write_csv(run_dir/"official_front.csv",front_rows)
        write_json(run_dir/"evaluation_archive_metadata.json",metadata)
        front_hash=hashlib.sha256(json.dumps(sorted(production),separators=(",",":")).encode()).hexdigest()
        integrity.append({"scenario":scenario,"algorithm":algorithm,"seed":seed,"status":status.get("status"),
          "total_requests":payload["evaluator"]["requests"],"raw_record_count":payload["evaluator"]["requests"],
          "canonical_unique_count":len(archive.records),"feasible_unique_count":len(archive.feasible_records()),"official_front_size":len(front),
          "checkpoint_present":True,"summary_present":(run_dir/"status.json").exists(),"raw_records_present":bool(payload["evaluator"]["cache"]),"official_front_hash":front_hash})
        oracle.append({"scenario":scenario,"algorithm":algorithm,"seed":seed,"production_front_size":len(production),"o_n2_oracle_front_size":len(reference),"decision_hashes_equal":consistent})
        runs[(scenario,algorithm,seed)]={"status":status,"payload":payload,"archive":archive,"front":front,"front_hash":front_hash,"run_dir":run_dir}
    write_csv(audit/"formal_integrity_audit.csv",integrity); write_csv(audit/"evaluation_archive_oracle_validation.csv",oracle)
    gate=(len(integrity)==125 and all(r["status"]=="completed" and r["total_requests"]==20000 and r["checkpoint_present"] and r["raw_records_present"] for r in integrity) and all(r["decision_hashes_equal"] for r in oracle))
    if not gate: raise RuntimeError("Formal integrity or EvaluationArchive oracle gate failed")

    # Scenario-specific bounds are required because S1-S5 are distinct optimization problems.
    metrics=[]; references={}; bounds={}; tradeoff=[]
    for scenario in SCENARIOS:
        all_feasible=[r for a in ALGORITHMS for s in SEEDS for r in runs[(scenario,a,s)]["archive"].feasible_records()]
        all_obj=np.asarray([[r["Phi_T"],r["Phi_E"]] for r in all_feasible]); lo,hi=all_obj.min(0),all_obj.max(0); safe_hi=np.where(hi>lo,hi,lo+1.0)
        ref_raw=nd_rows([r for a in ALGORITHMS for s in SEEDS for r in runs[(scenario,a,s)]["front"]]); references[scenario]=ref_raw; bounds[scenario]=(lo,safe_hi)
        write_csv(pareto/f"reference_front_{scenario}.csv",[{"Phi_T":r["Phi_T"],"Phi_E":r["Phi_E"]} for r in ref_raw],["Phi_T","Phi_E"])
        ref_z=global_normalize([[r["Phi_T"],r["Phi_E"]] for r in ref_raw],lo,safe_hi)
        ref_point=global_normalize([hi+0.1*(hi-lo)],lo,safe_hi)[0]
        ref_point=np.where(hi>lo,ref_point,0.1*np.maximum(np.abs(hi),1.0))
        for algorithm in ALGORITHMS:
          for seed in SEEDS:
            item=runs[(scenario,algorithm,seed)]; front=item["front"]; z=global_normalize([[r["Phi_T"],r["Phi_E"]] for r in front],lo,safe_hi)
            hv=hypervolume_2d_min(z,ref_point)
            distances=np.sqrt(((ref_z[:,None,:]-z[None,:,:])**2).sum(2)); igd=float(distances.min(1).mean())
            st=item["status"]; metrics.append({"scenario":scenario,"algorithm":algorithm,"seed":seed,"HV":hv,"IGD":igd,
              "front_size":len(front),"feasible_unique":len(item["archive"].feasible_records()),"canonical_unique":len(item["archive"].records),
              "wall_seconds":st["wall_seconds"],"requests_per_second":st["requests_per_second"],"cache_hits":item["payload"]["evaluator"]["cache_hits"],"cache_hit_rate":item["payload"]["evaluator"]["cache_hits"]/20000})
        ref_obj=np.asarray([[r["Phi_T"],r["Phi_E"]] for r in ref_raw]); tradeoff.append({"scenario":scenario,"pooled_reference_front_size":len(ref_raw),
          "phi_t_range":float(np.ptp(ref_obj[:,0])),"phi_e_range":float(np.ptp(ref_obj[:,1])),"tradeoff_pair_exists":len(ref_raw)>1})
    write_csv(stats_dir/"formal_hv_per_run.csv",metrics,["scenario","algorithm","seed","HV"]); write_csv(stats_dir/"formal_igd_per_run.csv",metrics,["scenario","algorithm","seed","IGD"])
    write_csv(audit/"formal_tradeoff_audit.csv",tradeoff)

    rep_rows=[]
    for key,item in runs.items():
        scenario,algorithm,seed=key; r=representative(item["front"]); n=len(r["canonical_key"])//3; keyv=r["canonical_key"]
        rep_rows.append({"scenario":scenario,"algorithm":algorithm,"seed":seed,"decision_hash":r["decision_hash"],"Phi_T":r["Phi_T"],"Phi_E":r["Phi_E"],
          "offload_count":int(sum(keyv[:n])),"cache_count":int(sum(keyv[n:2*n])),"total_edge_cpu":float(sum(keyv[2*n:])),"feasible":True,"front_size":len(item["front"])})
    write_csv(reps/"formal_representative_solutions.csv",rep_rows); write_csv(tables/"table_exp7_representative_solution.csv",rep_rows)

    summary=[]
    for scenario in SCENARIOS:
      for algorithm in ALGORITHMS:
        subset=[r for r in metrics if r["scenario"]==scenario and r["algorithm"]==algorithm]
        row={"scenario":scenario,"algorithm":algorithm}
        for field in ("HV","IGD","front_size","feasible_unique","wall_seconds","requests_per_second","cache_hits","cache_hit_rate"):
            d=describe([r[field] for r in subset]); row.update({f"{field}_{k}":v for k,v in d.items() if k in ({"mean","std","median","iqr"} if field in {"HV","IGD"} else {"mean","std","median"})})
        summary.append(row)
    # Per-scenario ranks by means.
    for scenario in SCENARIOS:
      subset=[r for r in summary if r["scenario"]==scenario]
      for field,reverse in (("HV_mean",True),("IGD_mean",False)):
        vals=np.asarray([r[field] for r in subset]); rr=rankdata(-vals if reverse else vals,method="average")
        for r,rank in zip(subset,rr):r[field.replace("_mean","_rank")]=float(rank)
    write_csv(tables/"scenario_algorithm_summary.csv",summary)

    # Block ranks and global rank summary.
    rank_records=[]
    for scenario in SCENARIOS:
      for seed in SEEDS:
        block=[next(r for r in metrics if r["scenario"]==scenario and r["seed"]==seed and r["algorithm"]==a) for a in ALGORITHMS]
        hv_rank=rankdata([-r["HV"] for r in block],method="average"); igd_rank=rankdata([r["IGD"] for r in block],method="average")
        for r,hr,ir in zip(block,hv_rank,igd_rank): rank_records.append({"scenario":scenario,"seed":seed,"algorithm":r["algorithm"],"HV_rank":hr,"IGD_rank":ir})
    rank_summary=[]
    for a in ALGORITHMS:
      x=[r for r in rank_records if r["algorithm"]==a]
      row={"algorithm":a}
      for metric in ("HV","IGD"):
        vals=np.asarray([r[f"{metric}_rank"] for r in x]); row.update({f"{metric}_average_rank":vals.mean(),f"{metric}_median_rank":np.median(vals),
          f"{metric}_rank1_count":int(np.sum(vals==1)),f"{metric}_top2_count":int(np.sum(vals<=2))})
      rank_summary.append(row)
    write_csv(tables/"overall_rank_summary.csv",rank_summary); write_csv(tables/"table_exp7_rank_summary.csv",rank_summary)

    friedman=[]; pairwise=[]; effects=[]; wtl=[]
    for metric in ("HV","IGD"):
        vectors=[[next(r[metric] for r in metrics if r["scenario"]==sc and r["seed"]==sd and r["algorithm"]==a) for sc in SCENARIOS for sd in SEEDS] for a in ALGORITHMS]
        fr=friedmanchisquare(*vectors); friedman.append({"metric":metric,"n_blocks":25,"n_algorithms":5,"statistic":fr.statistic,"p_value":fr.pvalue,"significant_at_0.05":fr.pvalue<.05})
        raw=[]; temp=[]
        main=np.asarray(vectors[0])
        for i,a in enumerate(ALGORITHMS[1:]):
            base=np.asarray(vectors[i+1]); favorable=main-base if metric=="HV" else base-main
            test=wilcoxon(favorable,zero_method="wilcox",alternative="two-sided")
            ranks=rankdata(np.abs(favorable[favorable!=0])); signs=np.sign(favorable[favorable!=0]); rbc=float(np.sum(ranks*signs)/np.sum(ranks)) if len(ranks) else 0.0
            raw.append((i,float(test.pvalue))); temp.append((a,favorable,float(test.pvalue),rbc))
        adjusted=holm(raw)
        for i,(a,d,p,rbc) in enumerate(temp):
            wins=int(np.sum(d>0));ties=int(np.sum(d==0));losses=int(np.sum(d<0))
            pairwise.append({"metric":metric,"baseline":a,"n_pairs":25,"raw_p":p,"holm_adjusted_p":adjusted[i],"significant":adjusted[i]<.05,
              "median_difference":float(np.median(d)),"win":wins,"tie":ties,"loss":losses})
            effects.append({"metric":metric,"baseline":a,"rank_biserial_correlation":rbc,"positive_favors_CNSGA_II":True})
            wtl.append({"metric":metric,"baseline":a,"block_win":wins,"block_tie":ties,"block_loss":losses})
    write_csv(stats_dir/"friedman_test_results.csv",friedman); write_csv(stats_dir/"pairwise_wilcoxon_holm.csv",pairwise); write_csv(stats_dir/"effect_size_rank_biserial.csv",effects); write_csv(stats_dir/"win_tie_loss_summary.csv",wtl)
    write_csv(tables/"table_exp7_statistics.csv",pairwise)

    # Relative improvement and paper tables.
    relative=[]
    for scenario in SCENARIOS:
      ss=[r for r in summary if r["scenario"]==scenario]; main=next(r for r in ss if r["algorithm"]==MAIN)
      hv_sorted=sorted(ss,key=lambda r:r["HV_mean"],reverse=True); igd_sorted=sorted(ss,key=lambda r:r["IGD_mean"])
      relative.append({"scenario":scenario,"comparison":"scenario_second_best","HV_relative_improvement":(main["HV_mean"]-hv_sorted[1 if hv_sorted[0]["algorithm"]==MAIN else 0]["HV_mean"])/max(abs(hv_sorted[1 if hv_sorted[0]["algorithm"]==MAIN else 0]["HV_mean"]),1e-15),
        "IGD_relative_reduction":(igd_sorted[1 if igd_sorted[0]["algorithm"]==MAIN else 0]["IGD_mean"]-main["IGD_mean"])/max(abs(igd_sorted[1 if igd_sorted[0]["algorithm"]==MAIN else 0]["IGD_mean"]),1e-15)})
      for b in ALGORITHMS[1:]:
        br=next(r for r in ss if r["algorithm"]==b); relative.append({"scenario":scenario,"comparison":b,"HV_relative_improvement":(main["HV_mean"]-br["HV_mean"])/max(abs(br["HV_mean"]),1e-15),"IGD_relative_reduction":(br["IGD_mean"]-main["IGD_mean"])/max(abs(br["IGD_mean"]),1e-15)})
    write_csv(tables/"relative_improvement_summary.csv",relative)
    for metric in ("HV","IGD"):
        rows=[]
        for a in ALGORITHMS:
            row={"algorithm":a}
            for scenario in SCENARIOS:
                s=next(r for r in summary if r["scenario"]==scenario and r["algorithm"]==a); row[f"{scenario}_mean"]=s[f"{metric}_mean"];row[f"{scenario}_std"]=s[f"{metric}_std"]
                rank=s[f"{metric}_rank"];row[f"{scenario}_best_flag"]=rank==1;row[f"{scenario}_second_best_flag"]=rank==2
            rows.append(row)
        write_csv(tables/f"table_exp7_{metric.lower()}_mean_std.csv",rows)
    write_csv(tables/"table_exp7_efficiency.csv",summary)

    # Figures.
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.labelsize":10,"axes.titlesize":11,"legend.fontsize":8,"axes.facecolor":"white","figure.facecolor":"white"})
    x=np.arange(5); width=.15
    for metric,label in (("HV","Hypervolume (higher is better)"),("IGD","IGD (lower is better)")):
        fig,ax=plt.subplots(figsize=(7.2,4.3))
        for i,a in enumerate(ALGORITHMS):
            means=[next(r[f"{metric}_mean"] for r in summary if r["scenario"]==s and r["algorithm"]==a) for s in SCENARIOS]; stds=[next(r[f"{metric}_std"] for r in summary if r["scenario"]==s and r["algorithm"]==a) for s in SCENARIOS]
            ax.bar(x+(i-2)*width,means,width,yerr=stds,label=a,color=COLORS[a],edgecolor="black",linewidth=.35,hatch=HATCHES[a],error_kw={"elinewidth":.7,"capsize":1.5})
        ax.set_xticks(x,SCENARIOS);ax.set_ylabel(label);ax.grid(axis="y",color="#D0D0D0",linestyle="--",linewidth=.6,alpha=.8);ax.set_axisbelow(True);ax.legend(ncol=3,frameon=False,loc="best")
        save_figure(fig,figures/f"fig_exp7_{metric.lower()}_formal")
    for scenario in SCENARIOS:
        fig,ax=plt.subplots(figsize=(6.2,4.6))
        for a in ALGORITHMS:
            points=np.asarray([[r["Phi_T"],r["Phi_E"]] for sd in SEEDS for r in runs[(scenario,a,sd)]["front"]]);ax.scatter(points[:,0],points[:,1],s=16,alpha=.72,label=a,color=COLORS[a],marker=MARKERS[a],linewidths=.25,edgecolors="white")
        ref=np.asarray([[r["Phi_T"],r["Phi_E"]] for r in references[scenario]]);ref=ref[np.argsort(ref[:,0])];ax.plot(ref[:,0],ref[:,1],color="black",linewidth=.8,label="Pooled reference front",zorder=1)
        ax.set_xlabel(r"$\Phi_T$");ax.set_ylabel(r"$\Phi_E$");ax.grid(color="#D0D0D0",linestyle="--",linewidth=.6,alpha=.8);ax.legend(frameon=False,loc="best")
        save_figure(fig,figures/f"fig_exp7_pareto_{scenario}")
    for field,label,name in (("front_size","Official front size","front_size"),("requests_per_second","Requests per second","efficiency")):
        fig,ax=plt.subplots(figsize=(7.2,4.3))
        for a in ALGORITHMS:
            means=[next(r[f"{field}_mean"] for r in summary if r["scenario"]==s and r["algorithm"]==a) for s in SCENARIOS]
            ax.plot(x,means,label=a,color=COLORS[a],marker=MARKERS[a],linestyle=LINES[a],linewidth=1.3,markersize=4.5)
        ax.set_xticks(x,SCENARIOS);ax.set_ylabel(label);ax.grid(color="#D0D0D0",linestyle="--",linewidth=.6,alpha=.8);ax.legend(ncol=3,frameon=False)
        save_figure(fig,figures/f"fig_exp7_{name}")

    not_hv=[s for s in SCENARIOS if next(r for r in summary if r["scenario"]==s and r["algorithm"]==MAIN)["HV_rank"]!=1]
    not_igd=[s for s in SCENARIOS if next(r for r in summary if r["scenario"]==s and r["algorithm"]==MAIN)["IGD_rank"]!=1]
    interpretation=["# Exp-7 Formal结果解读","",f"CNSGA-II的HV总体平均排名为{next(r['HV_average_rank'] for r in rank_summary if r['algorithm']==MAIN):.3f}，IGD总体平均排名为{next(r['IGD_average_rank'] for r in rank_summary if r['algorithm']==MAIN):.3f}。",
      f"HV并非第一的场景：{', '.join(not_hv) if not_hv else '无'}；IGD并非第一的场景：{', '.join(not_igd) if not_igd else '无'}。",
      "统计显著性、Holm校正及rank-biserial效应量见statistics目录；应同时报告不利block，不得仅陈述优势。",
      f"125个run中front size=1为{sum(r['front_size']==1 for r in metrics)}个，front规模不能单独解释为质量优劣，需结合HV与IGD。",
      f"Formal pooled reference fronts在{sum(r['tradeoff_pair_exists'] for r in tradeoff)}/5个场景含多个互不支配点，说明这些场景存在观测到的双目标trade-off。",
      "可写结论应限定为冻结场景、预算和代理评价口径下的统计结果；不能声称全局最优，也不能把front更大等同于算法更优。"]
    (final/"formal_result_interpretation.md").write_text("\n\n".join(interpretation)+"\n",encoding="utf-8")
    (final/"README.md").write_text("# Exp-7 Formal最终分析\n\n全部结果来自125个已完成run的持久化paid evaluation缓存，objective reevaluation为0。S1–S5是不同优化问题，HV共享界和IGD pooled reference front均在scenario内由5算法×5种子构造。\n",encoding="utf-8")
    write_json(audit/"protocol_binding.json",{"protocol_clarification_hash":"927bdea228157a6f19a59385133f39623f6eeeb9e695e9c2a01b9f1628e07b35","effective_protocol_hash":"71b1cf230eb2e49d5d01e12ff7bbc6c3ce6ccfe432cbd2627b5a7d40fa7a2548","candidate_scope":"all paid canonical-unique feasible evaluations per run","scenario_specific_metric_reference":True})
    write_json(audit/"objective_reevaluation_audit.json",{"objective_reevaluation_count":0,"evaluator_called":False,"source_of_truth":"persisted checkpoint evaluator cache"})
    all_files=[p for p in final.rglob("*") if p.is_file()]; manifest={"formal_results_frozen":True,"files":[{"path":p.relative_to(formal).as_posix(),"sha256":sha(p)} for p in sorted(all_files)],"source_run_count":125,"total_requests":2500000,"oracle_matches":125}
    write_json(audit/"finalization_manifest.json",manifest); write_json(audit/"README.json",{"status":"passed","generated_at":datetime.now().astimezone().isoformat()})
    (audit/"README.md").write_text("# Exp-7 Formal最终冻结审计\n\n125/125 run完整，全部20,000 requests；EvaluationArchive oracle 125/125一致；未执行任何objective reevaluation。\n",encoding="utf-8")
    created=[p.relative_to(audit).as_posix() for p in audit.rglob("*") if p.is_file()];write_json(audit/"created_files.json",{"count":len(created)+1,"files":sorted(created+["created_files.json"])})
    print(json.dumps({"final":str(final),"audit":str(audit),"reference_sizes":{s:len(references[s]) for s in SCENARIOS},"rank_summary":rank_summary,"friedman":friedman,"pairwise":pairwise,"effects":effects,"not_hv":not_hv,"not_igd":not_igd,"front_one":sum(r['front_size']==1 for r in metrics),"tradeoff":tradeoff},ensure_ascii=False, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)))


if __name__=="__main__": main()
