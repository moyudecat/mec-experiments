"""Frozen OE-RVEA execution and reporting chain for final Exp-4/5.

This module changes neither OE-RVEA nor the evaluator.  It binds them and
adds experiment-specific persistence, representative selection and reports.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import shutil
import time
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr, wilcoxon

from mec_dualbp.experiments.exp1_preview_protocol import canonical_json_hash
from mec_dualbp.experiments.exp456_formal_execution import _parameters
from mec_dualbp.experiments.exp456_formal_runner import CenterDeterministicEngine
from mec_dualbp.experiments.exp7_chain_runner import run_one
from mec_dualbp.optimization.dualbp_engine_registry import IMPROVED_MODE, create_corrected_dualbp_engine
from mec_dualbp.optimization.evaluation_archive import EvaluationArchive
from mec_dualbp.optimization.outer_contract import CanonicalSolution, IMPROVED_CORRECTED_DUALBP_HASH, interval_possibility
from mec_dualbp.physics.batch import evaluate_edge_comm_energy_batch_si, evaluate_edge_core_latency_batch_si
from mec_dualbp.uncertainty.sampling import lhs_user_disturbances
from mec_dualbp.utils.atomic_io import safe_atomic_write_json

OE_DESIGN_HASH = "6e9795b539095f841d51bb45db47be1535d66d3c530d81f96ed91ea1503428d4"
OE_IMPLEMENTATION_HASH = "95e9ee309294e7797c71b65b0a4a1fd239ac77a5d476e313d7e34f4fdec2fc57"
TIME_HASH = "75418503bc5399fcba8ef1db8b2b5b9cef14e8939fbd88532c54491c2eb61a9c"
ENERGY_HASH = "051e8ed1f9d7440a0655c7db4ff54b6b31925638f7518380b7fc4a2e356524e7"
EVALUATOR_HASH = IMPROVED_CORRECTED_DUALBP_HASH
SEEDS = (72001, 72002, 72003, 72004, 72005)
POPULATION, BUDGET = 100, 20_000
ALGORITHM = "OE-RVEA"
SCENARIOS = ("S2", "S3", "S4")
DEFAULT = {"Gamma": 1.0, "eta": .90, "Fmax": 20e9, "Smax": 4783138003.689591}
LEVELS = {
    "Gamma": (1., 2., 3., 4., 5.), "eta": (.75, .80, .85, .90, .95),
    "Fmax": (12e9, 16e9, 20e9, 24e9, 28e9),
    "Smax": tuple(4783138003.689591 * x for x in (.4, .6, .8, 1., 1.2)),
}


class _CenterFormalAdapterEngine(CenterDeterministicEngine):
    """Expose center physics through the common Formal prediction-field contract."""
    prediction_mode = "corr_pred"

    def evaluate(self, solution: CanonicalSolution) -> Mapping[str, Any]:
        value = dict(super().evaluate(solution))
        value["prediction_mode"] = "corr_pred"
        value["evaluator_mode"] = "deterministic_center_physics"
        return value


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); safe_atomic_write_json(path, value)


def _csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields or (list(rows[0]) if rows else [])); tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scenario(project: Path, name: str) -> dict[str, Any]:
    path = project / "configs/scenarios" / f"exp7_{name.lower()}_formal_v1.json"
    value = json.loads(path.read_text(encoding="utf-8")); value["corrected_dualbp_hash"] = EVALUATOR_HASH
    return value


def _protocol_bodies(project: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    common = {
        "status": "FROZEN", "algorithm": ALGORITHM, "outer_seeds": list(SEEDS),
        "population": POPULATION, "objective_requests_per_run": BUDGET, "workers": 1,
        "oe_design_hash": OE_DESIGN_HASH, "oe_implementation_hash": OE_IMPLEMENTATION_HASH,
        "evaluator_hash": EVALUATOR_HASH, "time_model_hash": TIME_HASH, "energy_model_hash": ENERGY_HASH,
        "canonical_repair": "mec_dualbp.optimization.outer_contract.canonical_outer_repair",
        "cache_hits_consume_request_budget": True, "epsilon": None,
    }
    exp4 = {**common, "experiment": "Exp-4-OE-RVEA-V2", "scenarios": list(SCENARIOS),
            "methods": ["Center-OE-RVEA", "Robust-OE-RVEA"], "planned_runs": 30,
            "matched_initialization": True,
            "representative_rule": "feasible Rank-1 -> pooled min-max -> equal-weight ideal distance -> QoS/max-objective/hash",
            "perturbation_points": 10003, "ordinary_lhs_points": 10000,
            "perturbation_components": ["ordinary LHS", "center", "positive anchor", "negative anchor"]}
    stress = {"status": "FROZEN", "experiment": "Exp-4-STRESS", "parent_exp4": "hash-bound",
              "Gamma_eval": [1, 2, 3, 4, 5], "reoptimization": False,
              "representatives": 30, "points_per_representative_gamma": 10003,
              "paired_perturbation_identity": True, "workers": 1}
    exp5 = {**common, "experiment": "Exp-5-OE-RVEA-V2", "scenario": "S3", "design": "OFAT",
            "default": DEFAULT, "levels": {k: list(v) for k, v in LEVELS.items()},
            "conceptual_cells": 100, "independent_runs": 85, "exact_default_reuse_records": 15}
    return exp4, stress, exp5


def freeze_protocol(project: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    exp4, stress, exp5 = _protocol_bodies(project)
    hashes = {"exp4_oe_rvea_v2_protocol_hash": canonical_json_hash(exp4),
              "exp4_stress_protocol_hash": canonical_json_hash(stress),
              "exp5_oe_rvea_v2_protocol_hash": canonical_json_hash(exp5)}
    _json(output / "exp4_v2_protocol.json", {**exp4, **{k: hashes[k] for k in hashes if k.startswith("exp4_oe")}})
    _json(output / "exp4_stress_protocol.json", {**stress, "exp4_stress_protocol_hash": hashes["exp4_stress_protocol_hash"]})
    _json(output / "exp5_v2_protocol.json", {**exp5, "exp5_oe_rvea_v2_protocol_hash": hashes["exp5_oe_rvea_v2_protocol_hash"]})
    _json(output / "oe_rvea_binding.json", {"design_hash": OE_DESIGN_HASH, "implementation_hash": OE_IMPLEMENTATION_HASH,
        "mechanism": "Exact Objective-Equivalent Solution Compression", "epsilon": None, "modified": False})
    _json(output / "evaluator_binding.json", {"corrected_dualbp_hash": EVALUATOR_HASH,
        "time_model_hash": TIME_HASH, "energy_model_hash": ENERGY_HASH, "modified": False})
    _json(output / "seed_independence.json", {"seeds": list(SEEDS), "historical_seeds": list(range(52001, 52006)),
        "intersection": [], "status": "passed"})
    (output / "metric_definitions.md").write_text(
        "# 指标定义\n\n扰动审计的用户级 QoS 裕量为 `tau_n - T_n`（s）；正值满足。优化代表解的 C7 裕量为 `possibility_n - eta`（无量纲）。搜索可行候选比例为可行目标请求数/总目标请求数。HV/IGD仅为同一参数水平内的前沿质量辅助指标，不跨水平排名。\n", encoding="utf-8")
    (output / "decision_statistic_definitions.md").write_text(
        "# 决策统计定义\n\n卸载率=mean(m)，缓存率=mean(c)，缓存利用率=sum(c*s)/Smax，CPU利用率=sum(f)/Fmax，正CPU均值仅对f>0取均值，本地比例=1-mean(m)。\n", encoding="utf-8")
    (output / "forbidden_changes.md").write_text(
        "# 禁止变更\n\n本轮不得修改OE-RVEA、模型、alpha、物理公式、场景、参数水平、种子或根据结果调参；不得运行Exp-8。\n", encoding="utf-8")
    (output / "README.md").write_text("# Exp-4/5 OE-RVEA V2 协议冻结\n\n本目录为运行前预注册协议，只绑定最终OE-RVEA与Improved Corrected DualBP。\n", encoding="utf-8")
    _json(output / "protocol_hashes.json", hashes)
    _json(output / "created_files.json", sorted(p.name for p in output.iterdir()) + ["created_files.json"])
    return hashes


def _cache(run: Path) -> Mapping[tuple[float, ...], Mapping[str, Any]]:
    return pickle.loads((run / "checkpoint.pkl").read_bytes())["evaluator"]["cache"]


def _front_records(run: Path) -> tuple[dict[str, Any], ...]:
    return EvaluationArchive.from_cache(_cache(run)).official_front()


def _representative(front: Sequence[Mapping[str, Any]], lo: np.ndarray, hi: np.ndarray) -> dict[str, Any]:
    values = np.asarray([[r["Phi_T"], r["Phi_E"]] for r in front], dtype=float)
    z = np.divide(values - lo, hi - lo, out=np.zeros_like(values), where=hi > lo)
    distance, worst = np.linalg.norm(z, axis=1), z.max(axis=1)
    index = min(range(len(front)), key=lambda i: (float(distance[i]), -float(front[i].get("qos_min_possibility") or 0),
                                                   float(worst[i]), str(front[i]["decision_hash"])))
    return dict(front[index]) | {"ideal_distance": float(distance[index])}


def _mechanism(run: Path) -> dict[str, Any]:
    adapter = pickle.loads((run / "checkpoint.pkl").read_bytes())["adapter"]
    if adapter.name != ALGORITHM or getattr(adapter.algorithm, "design_hash", None) != OE_DESIGN_HASH:
        raise RuntimeError("OE-RVEA identity mismatch")
    rows = list(adapter.objective_equivalence_audit)
    _csv(run / "objective_equivalence_audit.csv", rows)
    return {"oe_rvea_design_hash": OE_DESIGN_HASH, "oe_rvea_implementation_hash": OE_IMPLEMENTATION_HASH,
            "oe_audit_generations": len(rows), "oe_compression_triggered": any(int(r["compressed_candidate_count"]) > 0 for r in rows)}


def _mirror(run: Path, target: Path, protocol_hash: str) -> dict[str, Any]:
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    required = ("status.json", "checkpoint.pkl", "evaluation_archive_metadata.json", "final_front.csv", "history.csv")
    if status["status"] != "completed" or status["requests"] != BUDGET or any(not (run / x).exists() for x in required):
        raise RuntimeError("persistence validation failed")
    hashes = {p.relative_to(run).as_posix(): _sha(p) for p in sorted(run.rglob("*")) if p.is_file()}
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if {p.relative_to(target).as_posix(): _sha(p) for p in sorted(target.rglob("*")) if p.is_file()} != hashes:
            raise RuntimeError("recovery mirror drift")
    else:
        tmp = target.with_name(target.name + ".tmp"); shutil.copytree(run, tmp); os.replace(tmp, target)
    mirror_hashes = {p.relative_to(target).as_posix(): _sha(p) for p in sorted(target.rglob("*")) if p.is_file()}
    if mirror_hashes != hashes: raise RuntimeError("recovery mirror hash mismatch")
    return {"protocol_hash": protocol_hash, "source": run.as_posix(), "mirror": target.as_posix(), "hashes": hashes}


def _run(project: Path, root: Path, mirror: Path, scenario: dict[str, Any], seed: int,
         engine: Any, protocol_hash: str, identity: str) -> dict[str, Any]:
    run = root / "runs" / f"{ALGORITHM}__seed_{seed}"
    status_path = run / "status.json"
    if status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed" and int(existing.get("requests", -1)) == BUDGET:
            target = mirror / run.name
            source_hashes = {p.relative_to(run).as_posix(): _sha(p) for p in sorted(run.rglob("*")) if p.is_file()}
            target_hashes = ({p.relative_to(target).as_posix(): _sha(p) for p in sorted(target.rglob("*")) if p.is_file()}
                             if target.exists() else {})
            if source_hashes != target_hashes:
                raise RuntimeError("completed run/recovery mirror drift")
            return existing | {"completed_skip": True}
    result = run_one(root, ALGORITHM, seed, POPULATION, BUDGET, scenario, engine)
    result.update(_mechanism(run), evaluator_hash=EVALUATOR_HASH, effective_protocol_hash=protocol_hash,
                  experiment_identity=identity, actual_algorithm=ALGORITHM)
    _json(run / "status.json", result)
    ledger = _mirror(run, mirror / run.name, protocol_hash)
    _json(mirror / "manifests" / f"{identity}__{seed}.json", ledger)
    return result


def _solution(record: Mapping[str, Any], n: int) -> CanonicalSolution:
    key = np.asarray(record["canonical_key"], dtype=float)
    return CanonicalSolution(key[:n], key[n:2*n], key[2*n:3*n])


def _diagnostic(engine: Any, solution: CanonicalSolution) -> dict[str, Any]:
    result = dict(engine.evaluate(solution)); eta = float(engine.scenario["eta"])
    possibilities: list[float] = []
    if hasattr(engine, "_predict_corrected"):
        gamma=float(engine.scenario["Gamma"]); active=np.flatnonzero(solution.offload==1)
        lo=np.empty(len(engine.users)); hi=np.empty(len(engine.users))
        for i in np.flatnonzero(solution.offload==0):
            bounds=engine.local_bounds[int(i)]; lo[i],hi[i]=bounds.local_latency_lower_s,bounds.local_latency_upper_s
        if len(active):
            tx=[]
            for i in active: tx.append(engine._features(engine.users[int(i)],gamma,float(solution.edge_cpu[i]))[0])
            dummy=np.asarray([engine._features(engine.users[int(i)],gamma,float(solution.edge_cpu[i]))[1] for i in active])
            tl,tu,_,_=engine._predict_corrected(np.asarray(tx),dummy)
            for j,i in enumerate(active):
                user=engine.users[int(i)]; extra=(1-solution.cache[i])*user["s_n_bit"]/user["backhaul_rate_bit_per_s"]
                lo[i],hi[i]=tl[j]+extra,tu[j]+extra
        possibilities=[interval_possibility(float(lo[i]),float(hi[i]),float(engine.users[i]["tau_s"])) for i in range(len(lo))]
    if not possibilities: possibilities=[float(result["qos_min_possibility"])]
    margins=np.asarray(possibilities)-eta
    return {"c7_min_margin": float(margins.min()), "c7_mean_margin": float(margins.mean()),
            "c7_q05_margin": float(np.quantile(margins,.05)),
            "c7_violating_user_count": int(np.sum(margins < -1e-12)),
            "representative_feasible": bool(result["feasible"])}


@lru_cache(maxsize=600)
def _paired_disturbances(seed: int, user_index: int, gamma: float) -> np.ndarray:
    """Memoize identical registered perturbations; numerical bytes are unchanged."""
    child = int(np.random.SeedSequence([seed, user_index, int(gamma * 1000)]).generate_state(1)[0])
    value = lhs_user_disturbances(10_000, gamma, (True,) * 6, child, True, True).initialization_samples
    value.flags.writeable = False
    return value


def _empirical(scenario: Mapping[str, Any], solution: CanonicalSolution, seed: int, gamma: float) -> dict[str, Any]:
    count = 10_000; samples = count + 3; total_t = np.zeros(samples); total_e = np.zeros(samples)
    user_margin = []
    sample_hash = hashlib.sha256()
    for i, user in enumerate(scenario["users"]):
        z = _paired_disturbances(seed, i, gamma)
        sample_hash.update(z.tobytes())
        if solution.offload[i] == 0:
            c = .5*(user["C_cycle_lower"]+user["C_cycle_upper"]) + .5*(user["C_cycle_upper"]-user["C_cycle_lower"])*z[:,2]
            fl = .5*(user["F_L_cycle_per_s_lower"]+user["F_L_cycle_per_s_upper"]) + .5*(user["F_L_cycle_per_s_upper"]-user["F_L_cycle_per_s_lower"])*z[:,5]
            t, e = c/fl, user["kappa_SI"]*c*fl**2
        else:
            p = _parameters(user, float(solution.edge_cpu[i]))
            t = evaluate_edge_core_latency_batch_si(z, p, chunk_size=5000)
            t += (1-solution.cache[i])*user["s_n_bit"]/user["backhaul_rate_bit_per_s"]
            e = evaluate_edge_comm_energy_batch_si(z, p, chunk_size=5000)
        total_t += t; total_e += e; user_margin.append(float(user["tau_s"]) - t)
    margins = np.asarray(user_margin); sample_min = margins.min(axis=0)
    return {"Gamma_eval": gamma, "empirical_samples": samples, "perturbation_hash": sample_hash.hexdigest(),
            "empirical_feasible_rate": float(np.mean(sample_min >= 0)), "empirical_violation_rate": float(np.mean(sample_min < 0)),
            "minimum_qos_margin_s": float(sample_min.min()), "q05_qos_margin_s": float(np.quantile(sample_min,.05)),
            "median_qos_margin_s": float(np.median(sample_min)), "mean_qos_margin_s": float(sample_min.mean()),
            "c7_violating_user_count": int(np.sum(np.any(margins < 0, axis=1))),
            "maximum_latency_s": float(total_t.max()), "q95_latency_s": float(np.quantile(total_t,.95)),
            "maximum_energy_j": float(total_e.max()), "q95_energy_j": float(np.quantile(total_e,.95))}


def _describe(rows: Sequence[Mapping[str, Any]], group: Sequence[str], metrics: Sequence[str]) -> list[dict[str, Any]]:
    keys = sorted({tuple(r[k] for k in group) for r in rows}); out=[]
    for key in keys:
        selected=[r for r in rows if tuple(r[k] for k in group)==key]; row=dict(zip(group,key))
        for metric in metrics:
            a=np.asarray([float(r[metric]) for r in selected]); row.update({f"{metric}_mean":a.mean(),f"{metric}_std":a.std(ddof=1) if len(a)>1 else 0,
                f"{metric}_median":np.median(a),f"{metric}_iqr":np.quantile(a,.75)-np.quantile(a,.25)})
        out.append(row)
    return out


def _rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    d=x-y; d=d[d!=0]
    if not len(d): return 0.0
    ranks=rankdata(np.abs(d)); return float((ranks[d>0].sum()-ranks[d<0].sum())/ranks.sum())


def run_exp4(project: Path, formal: Path, stress: Path, audit: Path, mirror: Path,
             protocol_dir: Path) -> dict[str, Any]:
    started=time.monotonic(); formal.mkdir(parents=True,exist_ok=True); stress.mkdir(parents=True,exist_ok=True); audit.mkdir(parents=True,exist_ok=True)
    hashes=json.loads((protocol_dir/"protocol_hashes.json").read_text(encoding="utf-8")); ph=hashes["exp4_oe_rvea_v2_protocol_hash"]
    statuses=[]; reps=[]; plateau=[]
    for sname in SCENARIOS:
        scenario=_scenario(project,sname); fronts={}
        for method in ("center","robust"):
            engine=_CenterFormalAdapterEngine(scenario) if method=="center" else create_corrected_dualbp_engine(project,scenario,evaluator_mode=IMPROVED_MODE,evaluator_hash=EVALUATOR_HASH)
            for seed in SEEDS:
                root=formal/"methods"/method/"scenarios"/sname
                status=_run(project,root,mirror/method/sname,scenario,seed,engine,ph,f"Exp4-{sname}-{method}")
                statuses.append({"scenario":sname,"method":method,**status}); front=_front_records(root/"runs"/f"{ALGORITHM}__seed_{seed}"); fronts[(method,seed)]=front
                cache=_cache(root/"runs"/f"{ALGORITHM}__seed_{seed}")
                plateau.append({"scenario":sname,"method":method,"seed":seed,"raw_archive_size":len(cache),
                    "unique_decision_count":len({r["decision_hash"] for r in EvaluationArchive.from_cache(cache).records}),
                    "exact_unique_objective_pair_count":len({(r["Phi_T"],r["Phi_E"]) for r in EvaluationArchive.from_cache(cache).records})})
        allv=np.asarray([[r["Phi_T"],r["Phi_E"]] for f in fronts.values() for r in f]); lo,hi=allv.min(0),allv.max(0)
        for (method,seed),front in fronts.items():
            rep=_representative(front,lo,hi); solution=_solution(rep,len(scenario["users"]));
            engine=_CenterFormalAdapterEngine(scenario) if method=="center" else create_corrected_dualbp_engine(project,scenario,evaluator_mode=IMPROVED_MODE,evaluator_hash=EVALUATOR_HASH)
            row={"scenario":sname,"method":method,"seed":seed,"Phi_T":rep["Phi_T"],"Phi_E":rep["Phi_E"],"decision_hash":rep["decision_hash"],
                 "canonical_key_json":json.dumps(rep["canonical_key"]),**_diagnostic(engine,solution),**_empirical(scenario,solution,seed,float(scenario["Gamma"]))}
            reps.append(row)
    _csv(formal/"formal_manifest.csv",statuses); _csv(formal/"representative_solutions.csv",reps); _csv(formal/"empirical_perturbation_audit.csv",reps)
    _csv(formal/"objective_plateau_summary.csv",plateau)
    _csv(formal/"method_summary.csv",_describe(reps,["scenario","method"],["Phi_T","Phi_E","empirical_feasible_rate"]))
    _csv(formal/"qos_margin_summary.csv",_describe(reps,["scenario","method"],["minimum_qos_margin_s","q05_qos_margin_s","median_qos_margin_s","mean_qos_margin_s","c7_violating_user_count"]))
    _csv(formal/"tail_risk_summary.csv",_describe(reps,["scenario","method"],["maximum_latency_s","q95_latency_s","maximum_energy_j","q95_energy_j"]))
    decision=[]
    for r in reps:
        scenario=_scenario(project,r["scenario"]); sol=_solution({"canonical_key":json.loads(r["canonical_key_json"])},len(scenario["users"])); sizes=np.asarray([u["s_n_bit"] for u in scenario["users"]])
        pos=sol.edge_cpu[sol.edge_cpu>0]; decision.append({"scenario":r["scenario"],"method":r["method"],"seed":r["seed"],"offloading_ratio":sol.offload.mean(),"cache_ratio":sol.cache.mean(),
            "cache_used_bit":float(np.dot(sol.cache,sizes)),"cache_utilization":float(np.dot(sol.cache,sizes)/scenario["Smax_bit"]),"sum_edge_cpu":sol.edge_cpu.sum(),
            "cpu_utilization":sol.edge_cpu.sum()/scenario["F_max_cycle_per_s"],"mean_positive_edge_cpu":pos.mean() if len(pos) else 0,"local_execution_proportion":1-sol.offload.mean()})
    _csv(formal/"decision_statistics.csv",decision)
    paired=[]
    for metric in ("empirical_violation_rate","minimum_qos_margin_s","q05_qos_margin_s","maximum_latency_s","maximum_energy_j"):
        x=np.asarray([float(r[metric]) for r in reps if r["method"]=="center"]); y=np.asarray([float(r[metric]) for r in reps if r["method"]=="robust"])
        try: w=wilcoxon(x,y,zero_method="wilcox",alternative="two-sided") ; stat,p=float(w.statistic),float(w.pvalue)
        except ValueError: stat,p=0.,1.
        if not np.isfinite(stat) or not np.isfinite(p): stat,p=0.,1.
        paired.append({"metric":metric,"n_pairs":len(x),"wilcoxon_statistic":stat,"p_value":p,"rank_biserial_center_minus_robust":_rank_biserial(x,y),"ties":int(np.sum(x==y))})
    _csv(formal/"paired_statistics.csv",paired)
    stress_rows=[]
    for r in reps:
        scenario=_scenario(project,r["scenario"]); sol=_solution({"canonical_key":json.loads(r["canonical_key_json"])},len(scenario["users"]));
        for gamma in range(1,6): stress_rows.append({"scenario":r["scenario"],"method":r["method"],"seed":r["seed"],**_empirical(scenario,sol,int(r["seed"]),float(gamma))})
    _csv(stress/"stress_records.csv",stress_rows); metrics=("empirical_violation_rate","minimum_qos_margin_s","q05_qos_margin_s","maximum_latency_s","q95_latency_s","maximum_energy_j","q95_energy_j")
    _csv(stress/"stress_level_summary.csv",_describe(stress_rows,["Gamma_eval","method"],metrics))
    stress_pair=[]
    for gamma in range(1,6):
        for metric in metrics:
            x=np.asarray([float(r[metric]) for r in stress_rows if r["Gamma_eval"]==gamma and r["method"]=="center"]); y=np.asarray([float(r[metric]) for r in stress_rows if r["Gamma_eval"]==gamma and r["method"]=="robust"])
            stress_pair.append({"Gamma_eval":gamma,"metric":metric,"center_mean":x.mean(),"robust_mean":y.mean(),"paired_mean_difference":float((x-y).mean()),"paired_hash_mismatch":0})
    _csv(stress/"stress_paired_comparison.csv",stress_pair)
    initial_mismatch=sum(1 for s in SCENARIOS for seed in SEEDS if next(r for r in statuses if r["scenario"]==s and r["method"]=="center" and r["seed"]==seed)["initial_pool_hash"] != next(r for r in statuses if r["scenario"]==s and r["method"]=="robust" and r["seed"]==seed)["initial_pool_hash"])
    paired_hash_mismatch=0
    for s in SCENARIOS:
        for seed in SEEDS:
            for gamma in range(1,6):
                a=next(r for r in stress_rows if r["scenario"]==s and r["method"]=="center" and r["seed"]==seed and r["Gamma_eval"]==gamma)
                b=next(r for r in stress_rows if r["scenario"]==s and r["method"]=="robust" and r["seed"]==seed and r["Gamma_eval"]==gamma)
                paired_hash_mismatch += a["perturbation_hash"] != b["perturbation_hash"]
    oracle=sum(bool(json.loads((formal/"methods"/m/"scenarios"/s/"runs"/f"{ALGORITHM}__seed_{seed}"/"evaluation_archive_metadata.json").read_text())["deterministic_consistency_passed"]) for s in SCENARIOS for m in ("center","robust") for seed in SEEDS)
    failed=sum(r["status"]!="completed" for r in statuses); budget_bad=sum(r["requests"]!=BUDGET for r in statuses); nonfinite=sum(r["nonfinite_count"] for r in statuses)
    passed=len(statuses)==30 and failed==budget_bad==nonfinite==initial_mismatch==paired_hash_mismatch==0 and oracle==30 and len(reps)==30 and len(stress_rows)==150
    summary={"EXP4_CORRECTNESS_GATE":"PASS" if passed else "FAIL","completed":len(statuses)-failed,"planned":30,"failed":failed,"budget_violations":budget_bad,"oracle_pass":oracle,
             "initialization_mismatch":initial_mismatch,"paired_perturbation_hash_mismatch":paired_hash_mismatch,"representatives":len(reps),"stress_records":len(stress_rows),"wall_seconds":time.monotonic()-started}
    _json(formal/"formal_summary.json",summary); _json(stress/"summary.json",summary); _json(audit/"summary.json",summary)
    (audit/"README.md").write_text("# Exp-4 OE-RVEA 最终审计\n\n门禁仅评价工程正确性，不要求鲁棒方法性能领先。\n",encoding="utf-8")
    _csv(audit/"formal_manifest.csv",statuses); return summary


def run_exp5(project: Path, formal: Path, audit: Path, mirror: Path, protocol_dir: Path) -> dict[str, Any]:
    started=time.monotonic(); formal.mkdir(parents=True,exist_ok=True); audit.mkdir(parents=True,exist_ok=True)
    ph=json.loads((protocol_dir/"protocol_hashes.json").read_text())["exp5_oe_rvea_v2_protocol_hash"]
    executed={}; manifest=[]; representatives=[]; plateau=[]
    for factor,levels in LEVELS.items():
        for level in levels:
            condition=dict(DEFAULT); condition[factor]=float(level); identity=canonical_json_hash(condition)
            if identity not in executed:
                scenario=_scenario(project,"S3"); scenario.update(Gamma=condition["Gamma"],eta=condition["eta"],F_max_cycle_per_s=condition["Fmax"],Smax_bit=condition["Smax"])
                engine=create_corrected_dualbp_engine(project,scenario,evaluator_mode=IMPROVED_MODE,evaluator_hash=EVALUATOR_HASH); runroot=formal/"conditions"/identity/"scenarios"/"S3"
                statuses=[]; fronts=[]
                for seed in SEEDS:
                    status=_run(project,runroot,mirror/identity/"S3",scenario,seed,engine,ph,f"Exp5-{identity}"); statuses.append(status); fronts.append(_front_records(runroot/"runs"/f"{ALGORITHM}__seed_{seed}"))
                allv=np.asarray([[r["Phi_T"],r["Phi_E"]] for front in fronts for r in front]); lo,hi=allv.min(0),allv.max(0); reps=[]
                for seed,status,front in zip(SEEDS,statuses,fronts):
                    rep=_representative(front,lo,hi); sol=_solution(rep,len(scenario["users"])); sizes=np.asarray([u["s_n_bit"] for u in scenario["users"]]); pos=sol.edge_cpu[sol.edge_cpu>0]
                    diag=_diagnostic(engine,sol); row={"condition_hash":identity,"seed":seed,"Phi_T":rep["Phi_T"],"Phi_E":rep["Phi_E"],**diag,
                        "offloading_ratio":sol.offload.mean(),"cache_ratio":sol.cache.mean(),"cache_used_bit":float(np.dot(sol.cache,sizes)),"cache_utilization":float(np.dot(sol.cache,sizes)/condition["Smax"]),
                        "sum_edge_cpu":sol.edge_cpu.sum(),"cpu_utilization":sol.edge_cpu.sum()/condition["Fmax"],"mean_positive_edge_cpu":pos.mean() if len(pos) else 0,"local_execution_proportion":1-sol.offload.mean(),"search_feasible_candidate_ratio":status["feasible_rate"]}
                    reps.append(row); cache=_cache(runroot/"runs"/f"{ALGORITHM}__seed_{seed}"); records=EvaluationArchive.from_cache(cache).records
                    pairs={(r["Phi_T"],r["Phi_E"]) for r in records}
                    plateau.append({"condition_hash":identity,"seed":seed,"raw_archive_size":len(cache),
                        "unique_decision_count":len({r["decision_hash"] for r in records}),
                        "exact_unique_objective_pair_count":len(pairs),
                        "objective_unique_ratio":len(pairs)/len(records)})
                executed[identity]=(statuses,reps)
            statuses,reps=executed[identity]
            reused=sum(1 for r in manifest if r["condition_hash"]==identity)>0
            for status,rep in zip(statuses,reps): manifest.append({"factor":factor,"level":level,"condition_hash":identity,"exact_default_reuse":reused,**status}); representatives.append({"factor":factor,"level":level,**rep})
    _csv(formal/"formal_manifest.csv",manifest); _csv(formal/"representative_records.csv",representatives); _csv(formal/"objective_plateau_summary.csv",plateau)
    metrics=("Phi_T","Phi_E","c7_min_margin","c7_q05_margin","c7_mean_margin","c7_violating_user_count","representative_feasible","offloading_ratio","cache_ratio","cache_used_bit","cache_utilization","sum_edge_cpu","cpu_utilization","mean_positive_edge_cpu","local_execution_proportion","search_feasible_candidate_ratio")
    desc=_describe(representatives,["factor","level"],metrics); _csv(formal/"sensitivity_summary.csv",desc); _csv(formal/"sensitivity_mean_std.csv",desc); _csv(formal/"sensitivity_median_iqr.csv",desc)
    _csv(formal/"qos_margin_summary.csv",[{k:v for k,v in r.items() if k in ("factor","level") or k.startswith("c7_") or k.startswith("representative_feasible")} for r in desc])
    _csv(formal/"decision_response_summary.csv",[{k:v for k,v in r.items() if k in ("factor","level") or any(k.startswith(x) for x in ("offloading","cache_","sum_edge","cpu_","mean_positive","local_"))} for r in desc])
    _csv(formal/"search_feasible_candidate_ratio.csv",[{k:v for k,v in r.items() if k in ("factor","level") or k.startswith("search_feasible")} for r in desc])
    magnitudes=[]; trends=[]
    for factor in LEVELS:
        group=sorted([r for r in desc if r["factor"]==factor],key=lambda r:float(r["level"])); levels=np.asarray([float(r["level"]) for r in group])
        for metric in ("Phi_T","Phi_E"):
            vals=np.asarray([float(r[f"{metric}_mean"]) for r in group]); default_level=DEFAULT[factor]; default_val=float(next(r for r in group if math.isclose(float(r["level"]),default_level))[f"{metric}_mean"])
            magnitudes.append({"factor":factor,"metric":metric,"normalized_response_amplitude":float((vals.max()-vals.min())/abs(default_val))})
        for metric in ("Phi_T","Phi_E","c7_min_margin","offloading_ratio","cache_utilization","cpu_utilization"):
            rho,p=spearmanr(levels,[float(r[f"{metric}_mean"]) for r in group])
            defined=bool(np.isfinite(rho) and np.isfinite(p))
            trends.append({"factor":factor,"metric":metric,"spearman_rho":float(rho) if defined else None,
                "p_value":float(p) if defined else None,
                "interpretation":"monotonic trend diagnostic only" if defined else "undefined: constant response across levels"})
    _csv(formal/"sensitivity_magnitude.csv",magnitudes); _csv(formal/"trend_diagnostic.csv",trends)
    _csv(formal/"hv_igd_auxiliary_summary.csv",[{"factor":f,"level":x,"status":"NOT_COMPARABLE_ACROSS_LEVELS"} for f,v in LEVELS.items() for x in v])
    unique_status=[s for statuses,_ in executed.values() for s in statuses]; failed=sum(r["status"]!="completed" for r in unique_status); budget_bad=sum(r["requests"]!=BUDGET for r in unique_status); nonfinite=sum(r["nonfinite_count"] for r in unique_status)
    oracle=sum(bool(json.loads((formal/"conditions"/ident/"scenarios"/"S3"/"runs"/f"{ALGORITHM}__seed_{seed}"/"evaluation_archive_metadata.json").read_text())["deterministic_consistency_passed"]) for ident in executed for seed in SEEDS)
    reuse=sum(bool(r["exact_default_reuse"]) for r in manifest); passed=len(unique_status)==85 and len(manifest)==100 and reuse==15 and failed==budget_bad==nonfinite==0 and oracle==85
    summary={"EXP5_CORRECTNESS_GATE":"PASS" if passed else "FAIL","independent_completed":len(unique_status)-failed,"independent_planned":85,"conceptual_cells":len(manifest),"default_reuse_records":reuse,"failed":failed,"budget_violations":budget_bad,"oracle_pass":oracle,"wall_seconds":time.monotonic()-started}
    _json(formal/"formal_summary.json",summary); _json(audit/"summary.json",summary); (audit/"README.md").write_text("# Exp-5 OE-RVEA 最终审计\n",encoding="utf-8"); _csv(audit/"formal_manifest.csv",manifest); return summary


def sanitize_existing_statistical_reports(exp4: Path, exp5: Path) -> dict[str, int]:
    """Replace mathematically undefined all-tie/constant statistics explicitly."""
    changed = {"wilcoxon_all_tie": 0, "spearman_constant": 0}
    for path, fields in ((exp4 / "paired_statistics.csv", ("wilcoxon_statistic", "p_value")),
                         (exp5 / "trend_diagnostic.csv", ("spearman_rho", "p_value"))):
        with path.open(encoding="utf-8-sig", newline="") as stream: rows=list(csv.DictReader(stream))
        for row in rows:
            bad=any(str(row.get(f,"")).lower() in ("nan","inf","-inf") for f in fields)
            if not bad: continue
            if path.name == "paired_statistics.csv":
                row["wilcoxon_statistic"],row["p_value"]="0.0","1.0"; changed["wilcoxon_all_tie"]+=1
            else:
                row["spearman_rho"],row["p_value"]="",""; row["interpretation"]="undefined: constant response across levels"; changed["spearman_constant"]+=1
        _csv(path,rows,list(rows[0]) if rows else list(fields))
    _json(exp4 / "statistical_edge_case_audit.json", changed)
    _json(exp5 / "statistical_edge_case_audit.json", changed)
    return changed


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--project-root",default="."); p.add_argument("--protocol-dir",required=True); p.add_argument("--exp4-formal",required=True); p.add_argument("--exp4-stress",required=True); p.add_argument("--exp4-audit",required=True); p.add_argument("--exp4-mirror",required=True); p.add_argument("--exp5-formal",required=True); p.add_argument("--exp5-audit",required=True); p.add_argument("--exp5-mirror",required=True); a=p.parse_args(); project=Path(a.project_root).resolve(); started=time.monotonic()
    e4=run_exp4(project,Path(a.exp4_formal),Path(a.exp4_stress),Path(a.exp4_audit),Path(a.exp4_mirror),Path(a.protocol_dir))
    e5=None
    if e4["EXP4_CORRECTNESS_GATE"]=="PASS": e5=run_exp5(project,Path(a.exp5_formal),Path(a.exp5_audit),Path(a.exp5_mirror),Path(a.protocol_dir))
    print(json.dumps({"exp4":e4,"exp5":e5,"total_wall_seconds":time.monotonic()-started},ensure_ascii=False),flush=True)


if __name__ == "__main__": main()
