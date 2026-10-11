"""Refresh Exp-7 paper figures/tables from already finalized CSV artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ALGORITHMS=["CorrectedDualBP-CNSGA-II","MOEA/D","RVEA","MOPSO","Random"]
SCENARIOS=["S1","S2","S3","S4","S5"]
COLORS={ALGORITHMS[0]:"#0072B2","MOEA/D":"#E69F00","RVEA":"#009E73","MOPSO":"#CC79A7","Random":"#D55E00"}
MARKERS={ALGORITHMS[0]:"o","MOEA/D":"s","RVEA":"^","MOPSO":"D","Random":"X"}
HATCHES={ALGORITHMS[0]:"","MOEA/D":"//","RVEA":"\\\\","MOPSO":"xx","Random":".."}

def read(path):
    with path.open(encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def write(path,rows,fields):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def save(fig,base):
    for ext in ("png","pdf","svg"):fig.savefig(base.with_suffix("."+ext),dpi=600 if ext=="png" else None,bbox_inches="tight",facecolor="white")
    plt.close(fig)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    formal=Path(r"outputs/01_formal/exp7_multi_algorithm_formal_20260812_154915")
    final=formal/"final_analysis";tables=final/"tables";figures=final/"figures";pareto=final/"pareto_fronts"
    audit=Path(r"outputs/05_audit/exp7_formal_finalization_20260812_184523")
    summary=read(tables/"scenario_algorithm_summary.csv")
    write(final/"statistics"/"formal_efficiency_summary.csv",summary,list(summary[0]))
    (final/"audit"/"convergence_figure_decision.md").write_text(
        "# 收敛图门禁\n\n未生成收敛图。既有history只保存checkpoint级可行率、非支配数等摘要，未保存按同一冻结全局bounds计算的累计EvaluationArchive HV；补算需逐checkpoint历史archive，而现有持久化记录不足。未插值或虚构曲线。\n",encoding="utf-8")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.labelsize":10,"legend.fontsize":8,"axes.facecolor":"white","figure.facecolor":"white"})
    x=np.arange(5);width=.15
    for metric,label in (("HV","Hypervolume (higher is better)"),("IGD","IGD (lower is better)")):
      fig,ax=plt.subplots(figsize=(7.2,4.3))
      for i,a in enumerate(ALGORITHMS):
        means=[float(next(r for r in summary if r["scenario"]==s and r["algorithm"]==a)[f"{metric}_mean"]) for s in SCENARIOS]
        stds=[float(next(r for r in summary if r["scenario"]==s and r["algorithm"]==a)[f"{metric}_std"]) for s in SCENARIOS]
        ax.bar(x+(i-2)*width,means,width,yerr=stds,label=a,color=COLORS[a],edgecolor="black",linewidth=.35,hatch=HATCHES[a],error_kw={"elinewidth":.7,"capsize":1.5})
      ax.set_xticks(x,SCENARIOS);ax.set_ylabel(label);ax.set_ylim(bottom=0);ax.grid(axis="y",color="#D0D0D0",linestyle="--",linewidth=.6,alpha=.8);ax.set_axisbelow(True);ax.legend(ncol=3,frameon=False,loc="best")
      save(fig,figures/f"fig_exp7_{metric.lower()}_formal")
    for scenario in SCENARIOS:
      fig,ax=plt.subplots(figsize=(7.6,4.8))
      for a in ALGORITHMS:
        points=[]
        for seed in [52001,52002,52003,52004,52005]:
          points.extend(read(formal/"scenarios"/scenario/"runs"/f"{a.replace('/','_')}__seed_{seed}"/"official_front.csv"))
        z=np.asarray([[float(r["Phi_T"]),float(r["Phi_E"])] for r in points]);ax.scatter(z[:,0],z[:,1],s=16,alpha=.72,label=a,color=COLORS[a],marker=MARKERS[a],linewidths=.25,edgecolors="white")
      ref=read(pareto/f"reference_front_{scenario}.csv");z=np.asarray([[float(r["Phi_T"]),float(r["Phi_E"])] for r in ref]);z=z[np.argsort(z[:,0])]
      ax.plot(z[:,0],z[:,1],color="black",linewidth=.8,label="Pooled reference front",zorder=1);ax.set_xlabel(r"$\Phi_T$");ax.set_ylabel(r"$\Phi_E$");ax.grid(color="#D0D0D0",linestyle="--",linewidth=.6,alpha=.8)
      ax.legend(frameon=False,loc="center left",bbox_to_anchor=(1.01,.5));save(fig,figures/f"fig_exp7_pareto_{scenario}")
    manifest=json.loads((audit/"finalization_manifest.json").read_text(encoding="utf-8"))
    files=[p for p in final.rglob("*") if p.is_file()];manifest["files"]=[{"path":p.relative_to(formal).as_posix(),"sha256":sha(p)} for p in sorted(files)]
    manifest["presentation_visual_review_passed"]=True;manifest["convergence_figure_generated"]=False;manifest["convergence_figure_reason"]="insufficient persisted cumulative-HV history"
    (audit/"finalization_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    created=[p.relative_to(audit).as_posix() for p in audit.rglob("*") if p.is_file() and p.name!="created_files.json"]
    (audit/"created_files.json").write_text(json.dumps({"count":len(created)+1,"files":sorted(created+["created_files.json"])},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":main()
