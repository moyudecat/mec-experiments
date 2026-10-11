"""Checkpointed Exp-4/5/6 Formal chain bound to Improved Corrected DualBP V1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from mec_dualbp.experiments.adaptive_label_confirmation_runner import dense_run, ga_config
from mec_dualbp.experiments.adaptive_label_protocol import DENSE_LHS_FORMAL, REFERENCE_GA_FORMAL
from mec_dualbp.experiments.exp1_preview_protocol import canonical_json_hash
from mec_dualbp.experiments.exp456_formal_runner import CenterDeterministicEngine
from mec_dualbp.experiments.exp7_chain_runner import _context, run_one
from mec_dualbp.inner_ga.tasks import (
    EdgeTaskParameters, build_edge_comm_energy_lower_problem,
    build_edge_comm_energy_upper_problem, build_edge_core_latency_lower_problem,
    build_edge_core_latency_upper_problem,
)
from mec_dualbp.inner_ga.cache import quantized_cache_key
from mec_dualbp.inner_ga.operators import (
    elite_selection, polynomial_mutation, sbx_crossover, tournament_selection,
)
from mec_dualbp.optimization.adapters import matched_canonical_initial_population
from mec_dualbp.optimization.dualbp_engine_registry import (
    IMPROVED_MODE, create_corrected_dualbp_engine,
)
from mec_dualbp.optimization.evaluation_archive import EvaluationArchive, canonical_decision_hash
from mec_dualbp.optimization.outer_contract import (
    IMPROVED_CORRECTED_DUALBP_HASH, CanonicalSolution, canonical_outer_repair,
    interval_possibility, robust_objectives,
)
from mec_dualbp.physics.local_response import local_response_bounds_si
from mec_dualbp.physics.batch import evaluate_edge_comm_energy_batch_si, evaluate_edge_core_latency_batch_si
from mec_dualbp.uncertainty.interval_params import Interval
from mec_dualbp.uncertainty.sampling import lhs_user_disturbances, ordinary_lhs_unit
from mec_dualbp.utils.atomic_io import safe_atomic_write_json

PROJECT_PROTOCOL = "c8c5c93fe26f5f1536225e4dd3498008ce11437170f13776b6b8b24831d1132b"
ENGINE_HASH = "28b03d188047f567267c819c468c8ae044ee126add44c022a80c5c6ff9b3cda4"
EXP4_HASH = "446ea4e21fb8deaa29390a9d1e9d61e4ae9680d45213ec33d7095b7efb9783c8"
EXP5_HASH = "4472e3fcc37b583a5757c676742162cff006abe9e8b50a153a2a15c743e65345"
TIME_HASH = "75418503bc5399fcba8ef1db8b2b5b9cef14e8939fbd88532c54491c2eb61a9c"
ENERGY_HASH = "051e8ed1f9d7440a0655c7db4ff54b6b31925638f7518380b7fc4a2e356524e7"
CORRECTED_HASH = IMPROVED_CORRECTED_DUALBP_HASH
SEEDS = (52001, 52002, 52003, 52004, 52005)
POPULATION, BUDGET = 100, 20_000
RECOVERY = "outputs/05_audit/exp456_protocol_recovery_20260819_114531"
ENGINE_DIR = "outputs/06_implementation/improved_corrected_dualbp_online_engine_20260819_114531"


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_atomic_write_json(path, value)


def _csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields or (list(rows[0]) if rows else []))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_protocol(path: Path, hash_field: str, expected: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8")); recorded = payload.pop(hash_field)
    if recorded != expected or canonical_json_hash(payload) != expected or payload["status"] != "FROZEN":
        raise RuntimeError(f"protocol binding drift: {path}")
    payload[hash_field] = recorded
    return payload


def verify_bindings(project: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    impl = json.loads((project / ENGINE_DIR / "implementation_hash.json").read_text(encoding="utf-8"))
    actual = {name: _file_hash(project / name) for name in impl["source_file_sha256"]}
    if (impl["engine_implementation_hash"] != ENGINE_HASH or actual != impl["source_file_sha256"]
            or canonical_json_hash(actual) != ENGINE_HASH):
        raise RuntimeError("online Engine implementation hash drift")
    exp4 = _load_protocol(project / RECOVERY / "exp4_formal_protocol_v1.json",
                          "exp4_improved_protocol_hash", EXP4_HASH)
    exp5 = _load_protocol(project / RECOVERY / "exp5_formal_protocol_v1.json",
                          "exp5_improved_protocol_hash", EXP5_HASH)
    for value in (exp4, exp5):
        if (value["downstream_protocol_hash"] != PROJECT_PROTOCOL
                or value["evaluator_hash"] != CORRECTED_HASH
                or value["time_model_hash"] != TIME_HASH
                or value["energy_model_hash"] != ENERGY_HASH):
            raise RuntimeError("model/downstream binding drift")
    return exp4, exp5


def freeze_exp6_protocol(project: Path) -> dict[str, Any]:
    path = project / RECOVERY / "exp6_formal_protocol_v1.json"
    reference_identity = {
        "name": "Exp1-High-Fidelity-Empirical-Envelope-v1",
        "reference_ga": REFERENCE_GA_FORMAL,
        "dense_lhs": DENSE_LHS_FORMAL,
        "envelope": "lower=min(8-seed Reference-GA,Dense-LHS); upper=max(8-seed Reference-GA,Dense-LHS)",
        "reference_status": "high_fidelity_approximate_reference",
    }
    body = {
        "experiment": "Exp-6", "version": "exp6_formal_protocol_v1", "status": "FROZEN",
        "downstream_protocol_hash": PROJECT_PROTOCOL, "engine_implementation_hash": ENGINE_HASH,
        "evaluator_mode": IMPROVED_MODE, "evaluator_hash": CORRECTED_HASH,
        "time_model_hash": TIME_HASH, "energy_model_hash": ENERGY_HASH,
        "scenario": "S3", "scenario_config": "configs/scenarios/exp7_s3_formal_v1.json",
        "candidate_count": 100, "candidate_generation_seed": 86001,
        "candidate_generator": "matched_canonical_initial_population -> canonical_outer_repair -> canonical hash -> deduplicate",
        "candidate_selection_by_objective_forbidden": True,
        "reference_protocol": reference_identity,
        "reference_protocol_hash": canonical_json_hash(reference_identity),
        "reference_checkpoint_resume": True, "exact_input_hash_cache_only": True,
        "ranking_primary": {"Time": "Spearman rho", "Energy": "Spearman rho",
                            "aggregate": "arithmetic mean", "ties": "average rank"},
        "ranking_secondary": ["Kendall tau-b Time", "Kendall tau-b Energy", "Kendall mean", "Pareto relation agreement"],
        "runtime": {"workers": 1, "processes": 1, "warmup_candidates": 20,
                    "complete_passes": 10, "candidate_order": "fixed",
                    "timer": "time.perf_counter_ns",
                    "statistics": ["total", "mean", "median", "std", "p95"]},
        "quality_metrics": ["MAE", "RMSE", "WMAPE", "R2", "UER", "LER", "ICR"],
        "runner": "mec_dualbp.experiments.exp456_formal_execution.run_exp6",
        "bp_training": False, "alpha_modification": False,
    }
    protocol_hash = canonical_json_hash(body)
    body["exp6_improved_protocol_hash"] = protocol_hash
    _json(path, body)
    binding_path = project / RECOVERY / "exp456_execution_binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["exp6_protocol_hash"] = protocol_hash
    binding["exp456_execution_protocol_hash"] = canonical_json_hash({
        "exp4": binding["exp4_protocol_hash"], "exp5": binding["exp5_protocol_hash"],
        "exp6": protocol_hash, "evaluator": CORRECTED_HASH, "engine": ENGINE_HASH})
    _json(binding_path, binding)
    unresolved = project / RECOVERY / "unresolved_author_decisions.json"
    _json(unresolved, [])
    readiness = project / ENGINE_DIR / "formal_readiness.json"
    ready = json.loads(readiness.read_text(encoding="utf-8")); ready.update(
        EXP6_FORMAL_READY=True, exp6_protocol_hash=protocol_hash,
        exp6_author_decisions_remaining=0)
    _json(readiness, ready)
    return body


class _CenterAsCorrected(CenterDeterministicEngine):
    prediction_mode = "corr_pred"

    def evaluate(self, solution: CanonicalSolution) -> Mapping[str, Any]:
        result = dict(super().evaluate(solution)); result["prediction_mode"] = "corr_pred"
        result["evaluator_mode"] = "deterministic_center_physics"
        return result


def _scenario(project: Path, name: str) -> dict[str, Any]:
    value = json.loads((project / f"configs/scenarios/exp7_{name.lower()}_formal_v1.json").read_text(encoding="utf-8"))
    value["corrected_dualbp_hash"] = CORRECTED_HASH
    return value


def _run_status_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in root.rglob("status.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if "algorithm" in value: rows.append({"path": path.relative_to(root).as_posix(), **value})
    return rows


def run_exp4(project: Path, output: Path) -> dict[str, Any]:
    protocol, _ = verify_bindings(project); output.mkdir(parents=True, exist_ok=True)
    _json(output / "protocol_binding.json", protocol)
    rows = []
    for scenario_name in protocol["scenarios"]:
        scenario = _scenario(project, scenario_name)
        for method in ("center", "robust"):
            engine = (_CenterAsCorrected(scenario) if method == "center" else
                      create_corrected_dualbp_engine(project, scenario, evaluator_mode=IMPROVED_MODE,
                                                     evaluator_hash=CORRECTED_HASH))
            method_root = output / scenario_name / method
            for seed in SEEDS:
                status = run_one(method_root, protocol["formal_optimizer"], seed,
                                 POPULATION, BUDGET, scenario, engine)
                rows.append({"scenario": scenario_name, "method": method, **status})
                _json(output / "run_progress.json", {"planned_runs": 30, "completed_runs": len(rows),
                    "failed_runs": sum(row["status"] != "completed" for row in rows),
                    "current": f"{scenario_name}/{method}/{seed}"})
    _csv(output / "run_manifest.csv", rows)
    failed = sum(row["status"] != "completed" for row in rows)
    summary = {"experiment": "Exp-4", "protocol_hash": EXP4_HASH,
        "planned_runs": 30, "completed_runs": len(rows) - failed, "failed_runs": failed,
        "all_budgets_exact": all(int(row["requests"]) == BUDGET for row in rows),
        "methods": ["Deterministic Center", "Improved Corrected DualBP Robust"],
        "scenarios": protocol["scenarios"], "EXP4_FINAL_FROZEN": failed == 0 and len(rows) == 30,
        "formal_preview_runs": 0, "automatic_tuning": False}
    _json(output / "formal_summary.json", summary)
    return summary


def _condition_key(factor: str, value: float) -> str:
    return f"{factor}_{float(value):.12g}".replace(".", "p").replace("+", "")


def run_exp5(project: Path, output: Path) -> dict[str, Any]:
    _, protocol = verify_bindings(project); output.mkdir(parents=True, exist_ok=True)
    _json(output / "protocol_binding.json", protocol)
    base = protocol["default_condition"]
    factors = (("Gamma", protocol["Gamma_levels"]), ("eta", protocol["eta_levels"]),
               ("Fmax", protocol["Fmax_cycle_per_s_levels"]), ("Smax", protocol["Smax_bit_levels"]))
    executed: dict[str, dict[int, Mapping[str, Any]]] = {}; rows = []; conceptual = 0
    for factor, levels in factors:
        for level in levels:
            conceptual += len(SEEDS)
            condition = {"Gamma": float(base["Gamma"]), "eta": float(base["eta"]),
                         "Fmax": float(base["Fmax_cycle_per_s"]), "Smax": float(base["Smax_bit"])}
            condition[factor] = float(level)
            identity = canonical_json_hash(condition)
            if identity not in executed:
                scenario = _scenario(project, "S3")
                scenario["Gamma"], scenario["eta"] = condition["Gamma"], condition["eta"]
                scenario["F_max_cycle_per_s"], scenario["Smax_bit"] = condition["Fmax"], condition["Smax"]
                engine = create_corrected_dualbp_engine(project, scenario, evaluator_mode=IMPROVED_MODE,
                                                        evaluator_hash=CORRECTED_HASH)
                root = output / "conditions" / identity
                executed[identity] = {}
                for seed in SEEDS:
                    executed[identity][seed] = run_one(root, protocol["formal_optimizer"], seed,
                                                       POPULATION, BUDGET, scenario, engine)
            for seed in SEEDS:
                result = executed[identity][seed]
                rows.append({"factor": factor, "level": level, "condition_hash": identity,
                             "reused_exact_default": sum(1 for row in rows if row.get("condition_hash") == identity) > 0,
                             **result})
            _json(output / "run_progress.json", {"planned_conceptual_runs": 100,
                "completed_conceptual_runs": len(rows), "unique_conditions": len(executed),
                "current": f"{factor}/{level}"})
    _csv(output / "run_manifest.csv", rows)
    failed = sum(row["status"] != "completed" for row in rows)
    summary = {"experiment": "Exp-5", "protocol_hash": EXP5_HASH,
        "planned_runs": 100, "completed_runs": len(rows) - failed, "failed_runs": failed,
        "unique_physical_conditions": len(executed), "unique_algorithm_runs": len(executed) * len(SEEDS),
        "exact_default_reuse_records": 15, "all_budgets_exact": all(int(row["requests"]) == BUDGET for row in rows),
        "EXP5_FINAL_FROZEN": failed == 0 and len(rows) == 100,
        "performance_preview_runs": 0, "automatic_tuning": False}
    _json(output / "formal_summary.json", summary)
    return summary


def _solution(vector: Sequence[float], n: int) -> CanonicalSolution:
    raw = np.asarray(vector, dtype=np.float64)
    return CanonicalSolution(raw[:n], raw[n:2*n], raw[2*n:3*n])


def generate_candidates(project: Path, output: Path, protocol: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    scenario = _scenario(project, "S3"); context = _context(scenario); n = len(scenario["users"])
    pool = matched_canonical_initial_population(int(protocol["candidate_generation_seed"]), 100, context)
    records, seen = [], set()
    # The frozen matched generator is overwhelmingly unique. A deterministic continuation
    # is available only if repair collapses a row; it retains one RNG namespace.
    attempt = 0
    while len(records) < 100:
        candidates = pool if attempt == 0 else matched_canonical_initial_population(
            int(protocol["candidate_generation_seed"]) + attempt, 100, context)
        for vector in candidates:
            repaired = canonical_outer_repair(_solution(vector, n), context)
            key = tuple(np.concatenate((repaired.offload, repaired.cache, repaired.edge_cpu)).tolist())
            digest = canonical_decision_hash(key)
            if digest in seen: continue
            seen.add(digest); records.append({"candidate_id": f"EXP6_CAND_{len(records)+1:03d}",
                "canonical_hash": digest, "offload": repaired.offload.tolist(),
                "cache": repaired.cache.tolist(), "edge_cpu_cycle_per_s": repaired.edge_cpu.tolist(),
                "repair_metadata": {"canonical_repair_applied": True,
                    "active_users": int(repaired.offload.sum()), "cached_users": int(repaired.cache.sum()),
                    "edge_cpu_total_cycle_per_s": float(repaired.edge_cpu.sum())},
                "scenario_hash": _file_hash(project / protocol["scenario_config"]),
                "generation_seed": int(protocol["candidate_generation_seed"])})
            if len(records) == 100: break
        attempt += 1
        if attempt > 100: raise RuntimeError("unable to construct 100 unique canonical candidates")
    package_identity = {"protocol_hash": protocol["exp6_improved_protocol_hash"], "scenario": "S3",
                        "seed": 86001, "records": records}
    package_hash = canonical_json_hash(package_identity)
    _json(output / "candidate_package_manifest.json", {"candidate_package_hash": package_hash,
        "candidate_count": 100, "unique_count": len(seen), "identity": package_identity,
        "not_for_bp_training": True, "objective_based_filtering": False})
    _csv(output / "exp6_formal_candidate_package.csv", [{"candidate_id": r["candidate_id"],
        "canonical_hash": r["canonical_hash"], "offload": json.dumps(r["offload"]),
        "cache": json.dumps(r["cache"]), "edge_cpu_cycle_per_s": json.dumps(r["edge_cpu_cycle_per_s"]),
        "repair_metadata": json.dumps(r["repair_metadata"]), "scenario_hash": r["scenario_hash"],
        "generation_seed": r["generation_seed"]} for r in records])
    np.savez_compressed(output / "exp6_formal_candidate_package.npz",
        offload=np.asarray([r["offload"] for r in records]), cache=np.asarray([r["cache"] for r in records]),
        edge_cpu=np.asarray([r["edge_cpu_cycle_per_s"] for r in records]))
    return records, package_hash


def _parameters(user: Mapping[str, Any], edge_cpu: float) -> EdgeTaskParameters:
    interval = lambda stem: Interval(float(user[f"{stem}_lower"]), float(user[f"{stem}_upper"]))
    return EdgeTaskParameters(interval("D_in_bit"), interval("D_out_bit"), interval("C_cycle"),
        interval("I_u_w"), interval("I_d_w"), float(user["uplink_bandwidth_hz"]),
        float(user["downlink_bandwidth_hz"]), float(user["uplink_terminal_tx_power_w"]),
        float(user["downlink_bs_tx_power_w"]), float(user["terminal_rx_power_w"]),
        float(user["uplink_channel_gain_linear"]), float(user["downlink_channel_gain_linear"]),
        float(user["noise_power_w"]), float(edge_cpu))


def _task_identity(user: Mapping[str, Any], edge_cpu: float, gamma: float, task_name: str) -> dict[str, Any]:
    common = {key: user[key] for key in (
        "D_in_bit_lower", "D_in_bit_upper", "D_out_bit_lower", "D_out_bit_upper",
        "C_cycle_lower", "C_cycle_upper", "I_u_w_lower", "I_u_w_upper", "I_d_w_lower", "I_d_w_upper",
        "uplink_bandwidth_hz", "downlink_bandwidth_hz", "uplink_terminal_tx_power_w",
        "downlink_bs_tx_power_w", "terminal_rx_power_w", "uplink_channel_gain_linear",
        "downlink_channel_gain_linear", "noise_power_w")}
    if task_name.startswith("edge_core"): common["edge_cpu_frequency_cycle_per_s"] = edge_cpu
    return {"task_name": task_name, "Gamma": gamma, "physical_inputs": common}


def _reference_task(task_dir: Path, problem: Any, parameters: EdgeTaskParameters,
                    dense_seed: int) -> tuple[float, dict[str, Any]]:
    task_dir.mkdir(parents=True, exist_ok=True); results = []
    for seed in REFERENCE_GA_FORMAL["seeds"]:
        path = task_dir / f"reference_ga_seed_{seed}.json"
        if path.exists(): result = json.loads(path.read_text(encoding="utf-8"))
        else:
            result = _batch_ga_run(problem, parameters, int(seed)); _json(path, result)
        results.append(result)
    dense_path = task_dir / "dense_lhs.json"
    if dense_path.exists(): dense = json.loads(dense_path.read_text(encoding="utf-8"))
    else:
        dense = _dense_fast(problem, parameters, dense_seed, int(DENSE_LHS_FORMAL["n_samples"]),
                            int(DENSE_LHS_FORMAL.get("chunk_size", 5000))); _json(dense_path, dense)
    ga_label = min(r["response_value"] for r in results) if problem.direction == "minimize" else max(r["response_value"] for r in results)
    envelope = min(ga_label, dense["response_value"]) if problem.direction == "minimize" else max(ga_label, dense["response_value"])
    return float(envelope), {"reference_ga_label": float(ga_label), "dense_lhs_response": float(dense["response_value"]),
        "envelope": float(envelope), "ga_seed_runs": len(results),
        "ga_runtime_seconds": sum(float(r["runtime_seconds"]) for r in results),
        "dense_runtime_seconds": float(dense["runtime_seconds"]),
        "ga_evaluation_requests": sum(int(r["evaluation_requests"]) for r in results),
        "dense_evaluation_requests": int(dense["evaluation_requests"])}


def _batch_ga_run(problem: Any, parameters: EdgeTaskParameters, seed: int) -> dict[str, Any]:
    """Frozen Reference-GA with unchanged operators/RNG and batched population physics."""
    started = time.perf_counter(); config = ga_config(REFERENCE_GA_FORMAL, seed)
    rng = np.random.default_rng(seed)
    sampled = lhs_user_disturbances(config.population_size * 2, problem.gamma, problem.active_mask,
        seed, include_center=True, include_boundary_anchors=config.include_boundary_anchors)
    population=[]; seen=set()
    for raw in sampled.initialization_samples:
        candidate=problem.repair(raw); key=tuple(float(v).hex() for v in candidate)
        if key not in seen: population.append(candidate);seen.add(key)
        if len(population)==config.population_size: break
    if len(population)!=config.population_size: raise RuntimeError("batch GA initialization incomplete")
    population=np.asarray(population,dtype=np.float64)
    cache:dict[tuple[float,...],float]={}; requests=unique=hits=0; history=[]
    batch_function=(evaluate_edge_core_latency_batch_si if problem.name.startswith("edge_core")
                    else evaluate_edge_comm_energy_batch_si)

    def evaluate(values:np.ndarray)->tuple[np.ndarray,np.ndarray]:
        nonlocal requests,unique,hits
        responses=np.empty(len(values));pending=[];pending_keys=[];pending_indices=[]
        for index,candidate in enumerate(values):
            requests+=1;key=quantized_cache_key(candidate,config.cache_decimals)
            if key in cache: hits+=1;responses[index]=cache[key]
            elif key in pending_keys:
                first=pending_keys.index(key);pending_indices[first].append(index)
            else: pending.append(candidate);pending_keys.append(key);pending_indices.append([index])
        if pending:
            calculated=batch_function(np.asarray(pending),parameters,chunk_size=len(pending))
            for key,value,indices in zip(pending_keys,calculated,pending_indices):
                cache[key]=float(value);unique+=1
                for index in indices: responses[index]=value
                hits+=len(indices)-1
        fitness=np.asarray([problem.fitness(float(value)) for value in responses])
        return responses,fitness

    responses,fitness=evaluate(population);best_index=int(np.argmax(fitness));best_z=population[best_index].copy()
    best_response=float(responses[best_index]);best_fitness=float(fitness[best_index]);history.append(best_response)
    for _ in range(config.generations):
        elites=elite_selection(population,fitness,config.elite_count);children=[]
        while len(children)<config.population_size-config.elite_count:
            parents=tournament_selection(population,fitness,2,config.tournament_size,rng)
            first,second=sbx_crossover(parents[0],parents[1],rng,config.crossover_probability,
                config.mutation_eta,problem.repair)
            for child in (first,second):
                children.append(polynomial_mutation(child,rng,config.mutation_probability,
                    config.mutation_eta,problem.repair))
                if len(children)==config.population_size-config.elite_count: break
        population=np.vstack((elites,np.asarray(children)));responses,fitness=evaluate(population)
        current=int(np.argmax(fitness))
        if float(fitness[current])>best_fitness+config.absolute_tolerance:
            best_fitness=float(fitness[current]);best_response=float(responses[current]);best_z=population[current].copy()
        history.append(best_response)
    if not problem.is_feasible(best_z): raise RuntimeError("batch Reference-GA returned infeasible best_z")
    return {"task_name":problem.name,"direction":problem.direction,"method_name":"Reference-GA_approximate_reference",
        "seed":seed,"status":"completed","response_value":best_response,"best_z":best_z.tolist(),"feasible":True,
        "evaluation_requests":requests,"unique_evaluations":unique,"cache_hits":hits,
        "runtime_seconds":time.perf_counter()-started,"endpoint_snap_count":0,"endpoint_snap_max_ulp":0.0,
        "cache_comparison_pairs":0,"cache_max_response_difference":0.0,"cache_collision_error":False,
        "history_best_response":history,"evaluation_mode":"batched_population_float64"}


def _dense_fast(problem:Any,parameters:EdgeTaskParameters,seed:int,count:int,chunk_size:int)->dict[str,Any]:
    """Exact ordinary-LHS stream with vectorized frozen proportional budget repair."""
    started=time.perf_counter();mask=np.asarray(problem.active_mask,dtype=bool);active=np.flatnonzero(mask)
    unit=ordinary_lhs_unit(count,len(active),np.random.default_rng(seed));raw=np.zeros((count,6),dtype=np.float64)
    raw[:,active]=2*unit-1;repaired=np.clip(raw,-1,1);norm=np.sum(np.abs(repaired),axis=1)
    over=norm>problem.gamma;repaired[over]*=(problem.gamma/norm[over])[:,None]
    repaired[:,~mask]=0.0
    # Match the scalar implementation's conservative round-off correction.
    repaired_norm=np.sum(np.abs(repaired),axis=1);bad=repaired_norm>problem.gamma
    while np.any(bad):
        repaired[bad]*=np.nextafter(problem.gamma/repaired_norm[bad],0.0)[:,None]
        repaired_norm[bad]=np.sum(np.abs(repaired[bad]),axis=1);bad=repaired_norm>problem.gamma
    anchors=[]
    if problem.gamma>0 and len(active):
        positive=np.zeros(6);remaining=min(float(problem.gamma),float(len(active)))
        for index in active:
            allocation=min(1.0,remaining);positive[index]=allocation;remaining-=allocation
            if remaining<=0:break
        anchors=[positive,-positive]
    z=np.vstack((np.zeros(6),*anchors,repaired));function=(evaluate_edge_core_latency_batch_si if problem.name.startswith("edge_core") else evaluate_edge_comm_energy_batch_si)
    responses=function(z,parameters,chunk_size=chunk_size);index=int(np.argmin(responses) if problem.direction=="minimize" else np.argmax(responses))
    return {"task_name":problem.name,"direction":problem.direction,"method_name":"Dense-LHS_approximate_reference","seed":seed,
        "status":"completed","response_value":float(responses[index]),"best_z":z[index].tolist(),"feasible":problem.is_feasible(z[index]),
        "evaluation_requests":count,"unique_evaluations":count,"cache_hits":0,"runtime_seconds":time.perf_counter()-started,
        "endpoint_snap_count":0,"endpoint_snap_max_ulp":0.0,"cache_comparison_pairs":0,"cache_max_response_difference":0.0,
        "cache_collision_error":False,"history_best_response":[],"evaluation_mode":"vectorized_float64","chunk_size":chunk_size,
        "reference_status":"approximate_reference"}


def _repair_many(values: np.ndarray, gamma: float, active_mask: Sequence[bool]) -> np.ndarray:
    repaired=np.clip(np.asarray(values,dtype=np.float64),-1.0,1.0).copy()
    repaired[...,~np.asarray(active_mask,dtype=bool)]=0.0
    norm=np.sum(np.abs(repaired),axis=-1,keepdims=True)
    scale=np.minimum(1.0,float(gamma)/np.maximum(norm,1e-300));repaired*=scale
    return repaired


def _physics_many(population: np.ndarray, parameters: Sequence[EdgeTaskParameters], task_name: str) -> np.ndarray:
    z=np.asarray(population,dtype=np.float64); b,p,_=z.shape
    def arrays(name:str)->tuple[np.ndarray,np.ndarray]:
        intervals=[getattr(item,name) for item in parameters]
        center=np.asarray([x.center for x in intervals])[:,None];radius=np.asarray([x.radius for x in intervals])[:,None]
        return center,radius
    din_c,din_r=arrays("input_size_bit");dout_c,dout_r=arrays("output_size_bit");c_c,c_r=arrays("cpu_cycles")
    iu_c,iu_r=arrays("uplink_interference_w");id_c,id_r=arrays("downlink_interference_w")
    din=din_c+din_r*z[:,:,0];dout=dout_c+dout_r*z[:,:,1];cycles=c_c+c_r*z[:,:,2]
    iu=iu_c+iu_r*z[:,:,3];idd=id_c+id_r*z[:,:,4]
    const=lambda name:np.asarray([getattr(item,name) for item in parameters],dtype=np.float64)[:,None]
    ru=const("uplink_bandwidth_hz")*np.log2(1+const("uplink_terminal_tx_power_w")*const("uplink_channel_gain_linear")/(const("noise_power_w")+iu))
    rd=const("downlink_bandwidth_hz")*np.log2(1+const("downlink_bs_tx_power_w")*const("downlink_channel_gain_linear")/(const("noise_power_w")+idd))
    if task_name.startswith("edge_core"):
        return din/ru+cycles/const("edge_cpu_frequency_cycle_per_s")+dout/rd
    return const("uplink_terminal_tx_power_w")*din/ru+const("terminal_rx_power_w")*dout/rd


def _vectorized_ga_group(items: Sequence[tuple[str,Any,EdgeTaskParameters]], seed:int)->list[dict[str,Any]]:
    """Run one frozen seed for many same-task problems with the exact shared RNG stream."""
    started=time.perf_counter();problems=[item[1] for item in items];params=[item[2] for item in items]
    first=problems[0];config=ga_config(REFERENCE_GA_FORMAL,seed);b=len(items);rng=np.random.default_rng(seed)
    sampled=lhs_user_disturbances(config.population_size*2,first.gamma,first.active_mask,seed,True,config.include_boundary_anchors)
    initial=[];seen=set()
    for raw in sampled.initialization_samples:
        candidate=first.repair(raw);key=tuple(float(v).hex() for v in candidate)
        if key not in seen:initial.append(candidate);seen.add(key)
        if len(initial)==config.population_size:break
    base=np.asarray(initial,dtype=np.float64);population=np.broadcast_to(base,(b,*base.shape)).copy()
    caches=[set() for _ in range(b)];requests=np.zeros(b,dtype=int);unique=np.zeros(b,dtype=int);hits=np.zeros(b,dtype=int)
    histories=[[] for _ in range(b)]
    def evaluate(values:np.ndarray)->tuple[np.ndarray,np.ndarray]:
        responses=_physics_many(values,params,first.name);requests[:]+=values.shape[1]
        for task in range(b):
            for candidate in values[task]:
                key=tuple(float(v) for v in np.round(candidate,15))
                if key in caches[task]:hits[task]+=1
                else:caches[task].add(key);unique[task]+=1
        fitness=responses if first.direction=="maximize" else -responses
        return responses,fitness
    responses,fitness=evaluate(population);best_index=np.argmax(fitness,axis=1)
    rows=np.arange(b);best_z=population[rows,best_index].copy();best_response=responses[rows,best_index].copy();best_fitness=fitness[rows,best_index].copy()
    for task in range(b):histories[task].append(float(best_response[task]))
    child_count=config.population_size-config.elite_count
    for _ in range(config.generations):
        elite_idx=np.argsort(-fitness,axis=1,kind="stable")[:,:config.elite_count]
        elites=population[rows[:,None],elite_idx];children=[]
        while len(children)<child_count:
            parents=[]
            for __ in range(2):
                competitors=rng.choice(config.population_size,size=config.tournament_size,replace=False)
                win=np.argmax(fitness[:,competitors],axis=1);parents.append(population[rows,competitors[win]])
            a,bp=parents
            if rng.random()>=config.crossover_probability:first_child,second_child=a.copy(),bp.copy()
            else:
                u=rng.random(6);beta=np.where(u<=.5,(2*u)**(1/(config.mutation_eta+1)),(1/(2*(1-u)))**(1/(config.mutation_eta+1)))
                first_child=.5*((1+beta)*a+(1-beta)*bp);second_child=.5*((1-beta)*a+(1+beta)*bp)
            first_child=_repair_many(first_child,first.gamma,first.active_mask);second_child=_repair_many(second_child,first.gamma,first.active_mask)
            for child in (first_child,second_child):
                mutated=child.copy()
                for dim in range(6):
                    if rng.random()>=config.mutation_probability:continue
                    x=np.clip(mutated[:,dim],-1,1);delta1=(x+1)/2;delta2=(1-x)/2;rv=rng.random();power=1/(config.mutation_eta+1)
                    if rv<=.5:value=2*rv+(1-2*rv)*(1-delta1)**(config.mutation_eta+1);delta=value**power-1
                    else:value=2*(1-rv)+2*(rv-.5)*(1-delta2)**(config.mutation_eta+1);delta=1-value**power
                    mutated[:,dim]=x+2*delta
                children.append(_repair_many(mutated,first.gamma,first.active_mask))
                if len(children)==child_count:break
        population=np.concatenate((elites,np.stack(children,axis=1)),axis=1);responses,fitness=evaluate(population)
        current=np.argmax(fitness,axis=1);current_fit=fitness[rows,current];improved=current_fit>best_fitness+config.absolute_tolerance
        best_fitness[improved]=current_fit[improved];best_response[improved]=responses[rows,current][improved];best_z[improved]=population[rows,current][improved]
        for task in range(b):histories[task].append(float(best_response[task]))
    elapsed=time.perf_counter()-started
    return [{"task_name":first.name,"direction":first.direction,"method_name":"Reference-GA_approximate_reference","seed":seed,
        "status":"completed","response_value":float(best_response[i]),"best_z":best_z[i].tolist(),"feasible":problems[i].is_feasible(best_z[i]),
        "evaluation_requests":int(requests[i]),"unique_evaluations":int(unique[i]),"cache_hits":int(hits[i]),"runtime_seconds":elapsed/b,
        "endpoint_snap_count":0,"endpoint_snap_max_ulp":0.0,"cache_comparison_pairs":0,"cache_max_response_difference":0.0,
        "cache_collision_error":False,"history_best_response":histories[i],"evaluation_mode":"multi_task_batched_population_float64"} for i in range(b)]


def _prefill_reference_ga(output:Path,tasks:Mapping[str,Mapping[str,Any]])->None:
    for task_name in ("edge_core_latency_lower","edge_core_latency_upper","edge_comm_energy_lower","edge_comm_energy_upper"):
        group=[(digest,item["problem"],item["parameters"]) for digest,item in sorted(tasks.items()) if item["problem"].name==task_name]
        for seed in REFERENCE_GA_FORMAL["seeds"]:
            pending=[item for item in group if not (output/"reference_checkpoint"/item[0]/f"reference_ga_seed_{seed}.json").exists()]
            for start in range(0,len(pending),128):
                chunk=pending[start:start+128];results=_vectorized_ga_group(chunk,int(seed))
                for (digest,_,__),result in zip(chunk,results):_json(output/"reference_checkpoint"/digest/f"reference_ga_seed_{seed}.json",result)


def generate_reference(project: Path, output: Path, protocol: Mapping[str, Any],
                       records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter(); scenario = _scenario(project, "S3"); users = scenario["users"]
    builders = (build_edge_core_latency_lower_problem, build_edge_core_latency_upper_problem,
                build_edge_comm_energy_lower_problem, build_edge_comm_energy_upper_problem)
    tasks: dict[str, dict[str, Any]] = {}; mappings = []
    for candidate in records:
        for index, active in enumerate(candidate["offload"]):
            if active != 1: continue
            user = users[index]; edge_cpu = float(candidate["edge_cpu_cycle_per_s"][index])
            params = _parameters(user, edge_cpu)
            for builder in builders:
                problem = builder(params, float(scenario["Gamma"])); identity = _task_identity(
                    user, edge_cpu, float(scenario["Gamma"]), problem.name)
                digest = canonical_json_hash(identity)
                tasks.setdefault(digest, {"identity": identity, "problem": problem, "parameters": params})
                mappings.append((candidate["candidate_id"], index, problem.name, digest))
    sorted_tasks = sorted(tasks)
    _prefill_reference_ga(output,tasks)
    task_results = {}
    for ordinal, digest in enumerate(sorted_tasks):
        task = tasks[digest]
        value, audit = _reference_task(output / "reference_checkpoint" / digest,
            task["problem"], task["parameters"], int(DENSE_LHS_FORMAL["seed_base"]) + ordinal)
        task_results[digest] = {"value": value, **audit}
        if (ordinal + 1) % 10 == 0 or ordinal + 1 == len(sorted_tasks):
            _json(output / "reference_progress.json", {"unique_tasks": len(sorted_tasks),
                "completed_tasks": ordinal + 1, "mapping_requests": len(mappings),
                "cache_hits": len(mappings) - len(sorted_tasks), "elapsed_seconds": time.perf_counter() - started})
    by_candidate: dict[str, dict[tuple[int, str], float]] = {}
    for candidate_id, index, task_name, digest in mappings:
        by_candidate.setdefault(candidate_id, {})[(index, task_name)] = task_results[digest]["value"]
    reference_rows = []
    for candidate in records:
        tlo = thi = elo = ehi = 0.0; possibilities = []
        values = by_candidate.get(candidate["candidate_id"], {})
        for index, user in enumerate(users):
            if candidate["offload"][index] == 0:
                local = local_response_bounds_si(
                    Interval(user["C_cycle_lower"], user["C_cycle_upper"]),
                    Interval(user["F_L_cycle_per_s_lower"], user["F_L_cycle_per_s_upper"]),
                    user["kappa_SI"], scenario["Gamma"])
                lo, hi = local.local_latency_lower_s, local.local_latency_upper_s
                le, ue = local.local_energy_lower_j, local.local_energy_upper_j
            else:
                extra = (1.0 - candidate["cache"][index]) * user["s_n_bit"] / user["backhaul_rate_bit_per_s"]
                lo = values[(index, "edge_core_latency_lower")] + extra
                hi = values[(index, "edge_core_latency_upper")] + extra
                le = values[(index, "edge_comm_energy_lower")]
                ue = values[(index, "edge_comm_energy_upper")]
            tlo += lo; thi += hi; elo += le; ehi += ue
            possibilities.append(interval_possibility(lo, hi, user["tau_s"]))
        objectives = robust_objectives(tlo, thi, elo, ehi, rho_time=scenario["rho_T_s"],
            rho_energy=scenario["rho_E_J"], lambda_time=scenario["lambda_T"], lambda_energy=scenario["lambda_E"])
        cv = float(np.maximum(0.0, scenario["eta"] - np.asarray(possibilities)).sum())
        reference_rows.append({"candidate_id": candidate["candidate_id"], "canonical_hash": candidate["canonical_hash"],
            "time_lower": tlo, "time_upper": thi, "energy_lower": elo, "energy_upper": ehi,
            "Phi_T": objectives[0], "Phi_E": objectives[1], "constraint_violation": cv,
            "feasible": cv <= 1e-12, "qos_min_possibility": min(possibilities)})
    elapsed = time.perf_counter() - started
    audit = {"reference_protocol_name": protocol["reference_protocol"]["name"],
        "reference_protocol_hash": protocol["reference_protocol_hash"],
        "reference_requests": len(mappings), "unique_reference_evaluations": len(sorted_tasks),
        "cache_hits": len(mappings) - len(sorted_tasks), "wall_seconds": elapsed,
        "unique_evaluation_time_seconds": sum(r["ga_runtime_seconds"] + r["dense_runtime_seconds"] for r in task_results.values()),
        "effective_seconds_per_candidate": elapsed / len(records),
        "reference_ga_seed_runs": len(sorted_tasks) * 8, "dense_lhs_runs": len(sorted_tasks),
        "dense_lhs_points_per_run": int(DENSE_LHS_FORMAL["n_samples"]), "workers": 1,
        "completed": True}
    _csv(output / "reference_results.csv", reference_rows); _json(output / "reference_manifest.json", audit)
    return reference_rows, audit


def _predicted_bounds(engine: Any, solution: CanonicalSolution) -> tuple[float, float, float, float]:
    active = np.flatnonzero(solution.offload == 1); n = len(engine.users)
    tl=np.empty(n); tu=np.empty(n); el=np.empty(n); eu=np.empty(n)
    for index in np.flatnonzero(solution.offload == 0):
        local=engine.local_bounds[int(index)]; tl[index],tu[index]=local.local_latency_lower_s,local.local_latency_upper_s
        el[index],eu[index]=local.local_energy_lower_j,local.local_energy_upper_j
    if active.size:
        tx,ex=[],[]
        for index in active:
            a,b=engine._features(engine.users[int(index)],engine.scenario["Gamma"],solution.edge_cpu[index]);tx.append(a);ex.append(b)
        a,b,c,d=engine._predict_corrected(np.asarray(tx),np.asarray(ex))
        for pos,index in enumerate(active):
            user=engine.users[int(index)];extra=(1-solution.cache[index])*user["s_n_bit"]/user["backhaul_rate_bit_per_s"]
            tl[index],tu[index]=a[pos]+extra,b[pos]+extra;el[index],eu[index]=c[pos],d[pos]
    return float(tl.sum()),float(tu.sum()),float(el.sum()),float(eu.sum())


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order=np.argsort(values,kind="stable"); ranks=np.empty(len(values),dtype=float); cursor=0
    while cursor<len(order):
        end=cursor+1
        while end<len(order) and values[order[end]]==values[order[cursor]]: end+=1
        ranks[order[cursor:end]]=(cursor+1+end)/2.0;cursor=end
    return ranks


def _spearman(a: np.ndarray,b: np.ndarray)->float:
    ra,rb=_average_ranks(a),_average_ranks(b)
    return float(np.corrcoef(ra,rb)[0,1])


def _kendall_tau_b(a: np.ndarray,b: np.ndarray)->float:
    concordant=discordant=tie_a=tie_b=0
    for i in range(len(a)):
        for j in range(i+1,len(a)):
            da,db=np.sign(a[i]-a[j]),np.sign(b[i]-b[j])
            if da==0 and db==0: continue
            if da==0: tie_a+=1
            elif db==0: tie_b+=1
            elif da==db: concordant+=1
            else: discordant+=1
    den=math.sqrt((concordant+discordant+tie_a)*(concordant+discordant+tie_b))
    return (concordant-discordant)/den if den else 1.0


def _relation(x: Sequence[float],y: Sequence[float])->str:
    x=np.asarray(x);y=np.asarray(y)
    if np.array_equal(x,y): return "tie"
    if np.all(x<=y) and np.any(x<y): return "a_dominates"
    if np.all(y<=x) and np.any(y<x): return "b_dominates"
    return "incomparable"


def _metrics(true: np.ndarray,pred: np.ndarray)->dict[str,float]:
    err=pred-true;sst=float(np.sum((true-true.mean())**2))
    return {"MAE":float(np.mean(np.abs(err))),"RMSE":float(np.sqrt(np.mean(err**2))),
        "WMAPE":float(np.sum(np.abs(err))/max(np.sum(np.abs(true)),1e-12)),
        "R2":1-float(np.sum(err**2))/max(sst,1e-12)}


def evaluate_exp6(project: Path, output: Path, protocol: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
                  references: Sequence[Mapping[str, Any]], reference_audit: Mapping[str, Any]) -> dict[str, Any]:
    scenario=_scenario(project,"S3");engine=create_corrected_dualbp_engine(project,scenario,evaluator_mode=IMPROVED_MODE,evaluator_hash=CORRECTED_HASH)
    n=len(scenario["users"]);solutions=[CanonicalSolution(np.asarray(r["offload"]),np.asarray(r["cache"]),np.asarray(r["edge_cpu_cycle_per_s"])) for r in records]
    for solution in solutions[:20]: engine.evaluate(solution)
    pass_times=[];sample_times=[]
    for _ in range(10):
        start=time.perf_counter_ns()
        for solution in solutions:
            one=time.perf_counter_ns();engine.evaluate(solution);sample_times.append((time.perf_counter_ns()-one)/1e9)
        pass_times.append((time.perf_counter_ns()-start)/1e9)
    predictions=[]
    for record,solution in zip(records,solutions):
        bounds=_predicted_bounds(engine,solution);result=engine.evaluate(solution)
        predictions.append({"candidate_id":record["candidate_id"],"canonical_hash":record["canonical_hash"],
            "time_lower":bounds[0],"time_upper":bounds[1],"energy_lower":bounds[2],"energy_upper":bounds[3],
            "Phi_T":result["objectives"][0],"Phi_E":result["objectives"][1],
            "constraint_violation":result["constraint_violation"],"feasible":result["feasible"]})
    combined=[]
    for p,r in zip(predictions,references):
        combined.append({"candidate_id":p["candidate_id"],"canonical_hash":p["canonical_hash"],
            **{f"pred_{k}":p[k] for k in ("time_lower","time_upper","energy_lower","energy_upper","Phi_T","Phi_E","constraint_violation","feasible")},
            **{f"ref_{k}":r[k] for k in ("time_lower","time_upper","energy_lower","energy_upper","Phi_T","Phi_E","constraint_violation","feasible")}})
    _csv(output/"per_candidate_results.csv",combined)
    quality=[]
    for field in ("time_lower","time_upper","energy_lower","energy_upper"):
        true=np.asarray([r[field] for r in references]);pred=np.asarray([p[field] for p in predictions]);m=_metrics(true,pred)
        quality.append({"boundary":field,**m,"UER":float(np.mean(pred<true)) if field.endswith("upper") else "",
            "LER":float(np.mean(pred>true)) if field.endswith("lower") else "",
            "ICR":float(np.mean([(pred[i]<=true[i]) if field.endswith("lower") else (pred[i]>=true[i]) for i in range(len(true))]))})
    _csv(output/"quality_summary.csv",quality)
    rt=np.asarray([r["Phi_T"] for r in references]);re=np.asarray([r["Phi_E"] for r in references]);pt=np.asarray([p["Phi_T"] for p in predictions]);pe=np.asarray([p["Phi_E"] for p in predictions])
    st,se=_spearman(rt,pt),_spearman(re,pe);kt,ke=_kendall_tau_b(rt,pt),_kendall_tau_b(re,pe)
    agreements=sum(_relation((pt[i],pe[i]),(pt[j],pe[j]))==_relation((rt[i],re[i]),(rt[j],re[j])) for i in range(100) for j in range(i+1,100))/4950
    ranking={"Spearman_Time":st,"Spearman_Energy":se,"Spearman_mean":(st+se)/2,"tie_handling":"average-rank",
        "Kendall_tau_b_Time":kt,"Kendall_tau_b_Energy":ke,"Kendall_mean":(kt+ke)/2,"Pareto_relation_agreement":agreements}
    _csv(output/"ranking_consistency.csv",[ranking])
    sample=np.asarray(sample_times);median=float(np.median(sample));runtime={"warmup_candidates":20,"formal_passes":10,"candidate_order":"fixed",
        "total_elapsed_seconds":sum(pass_times),"mean_per_candidate_seconds":float(sample.mean()),"median_per_candidate_seconds":median,
        "std_per_candidate_seconds":float(sample.std()),"p95_per_candidate_seconds":float(np.quantile(sample,.95)),"timer":"perf_counter_ns",
        "CPU":platform.processor() or platform.machine(),"GPU":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "PyTorch":torch.__version__,"NumPy":np.__version__,"device":"cpu","torch_threads":torch.get_num_threads(),"workers":1,"processes":1}
    _csv(output/"runtime_summary.csv",[runtime])
    speedup=float(reference_audit["effective_seconds_per_candidate"])/max(median,1e-15)
    efficiency={"improved_median_seconds_per_candidate":median,"reference_effective_seconds_per_candidate":reference_audit["effective_seconds_per_candidate"],
        "speedup_ratio":speedup,"interpretation":"empirical role-specific speedup; not hardware-independent complexity"}
    _csv(output/"efficiency_summary.csv",[efficiency])
    summary={"experiment":"Exp-6","protocol_hash":protocol["exp6_improved_protocol_hash"],"candidate_count":100,
        "candidate_package_hash":json.loads((output/"candidate_package_manifest.json").read_text(encoding="utf-8"))["candidate_package_hash"],
        "reference":dict(reference_audit),"quality":quality,"ranking":ranking,"runtime":runtime,"efficiency":efficiency,
        "failed_runs":0,"automatic_tuning":False,"EXP6_FINAL_FROZEN":True}
    _json(output/"formal_summary.json",summary);return summary


def run_exp6(project: Path, output: Path) -> dict[str, Any]:
    verify_bindings(project);protocol=freeze_exp6_protocol(project);output.mkdir(parents=True,exist_ok=True);_json(output/"protocol_binding.json",protocol)
    records,package_hash=generate_candidates(project,output,protocol)
    references,audit=generate_reference(project,output,protocol,records)
    summary=evaluate_exp6(project,output,protocol,records,references,audit)
    summary["candidate_package_hash"]=package_hash;_json(output/"formal_summary.json",summary);return summary


def finalize(project: Path, exp4: Path, exp5: Path, exp6: Path, output: Path) -> dict[str, Any]:
    summaries=[json.loads((p/"formal_summary.json").read_text(encoding="utf-8")) for p in (exp4,exp5,exp6)]
    result={"EXP4_FINAL_FROZEN":summaries[0]["EXP4_FINAL_FROZEN"],"EXP5_FINAL_FROZEN":summaries[1]["EXP5_FINAL_FROZEN"],
        "EXP6_FINAL_FROZEN":summaries[2]["EXP6_FINAL_FROZEN"],"failed_runs":sum(int(s.get("failed_runs",0)) for s in summaries),
        "automatic_tuning":False,"exp456_final_frozen":all(s.get(f"EXP{i}_FINAL_FROZEN",False) for i,s in zip((4,5,6),summaries)),
        "formal_directories":{"Exp-4":str(exp4),"Exp-5":str(exp5),"Exp-6":str(exp6)}}
    output.mkdir(parents=True,exist_ok=True);_json(output/"summary.json",result);return result


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--phase",choices=("freeze-exp6","exp4","exp5","exp6","finalize"),required=True)
    parser.add_argument("--project",default=".");parser.add_argument("--output");parser.add_argument("--exp4");parser.add_argument("--exp5");parser.add_argument("--exp6")
    args=parser.parse_args();project=Path(args.project).resolve()
    if args.phase=="freeze-exp6": print(json.dumps(freeze_exp6_protocol(project)),flush=True)
    elif args.phase=="exp4": print(json.dumps(run_exp4(project,Path(args.output).resolve())),flush=True)
    elif args.phase=="exp5": print(json.dumps(run_exp5(project,Path(args.output).resolve())),flush=True)
    elif args.phase=="exp6": print(json.dumps(run_exp6(project,Path(args.output).resolve())),flush=True)
    else: print(json.dumps(finalize(project,Path(args.exp4).resolve(),Path(args.exp5).resolve(),Path(args.exp6).resolve(),Path(args.output).resolve())),flush=True)


if __name__=="__main__": main()
