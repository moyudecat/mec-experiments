"""Formal postprocessing and final freeze for Exp-4/5/6."""
from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mec_dualbp.experiments.exp456_formal_execution import _csv, _json, _parameters
from mec_dualbp.optimization.evaluation_archive import EvaluationArchive
from mec_dualbp.optimization.outer_contract import CanonicalSolution
from mec_dualbp.physics.batch import (
    evaluate_edge_comm_energy_batch_si,
    evaluate_edge_core_latency_batch_si,
)
from mec_dualbp.uncertainty.sampling import lhs_user_disturbances


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _run_cache(path: Path) -> Mapping[tuple[float, ...], Mapping[str, Any]]:
    with path.open("rb") as stream:
        return pickle.load(stream)["evaluator"]["cache"]


def _representative(cache: Mapping[tuple[float, ...], Mapping[str, Any]], lower: np.ndarray,
                    span: np.ndarray) -> tuple[float, ...]:
    front = EvaluationArchive.from_cache(cache).official_front()
    if not front:
        raise RuntimeError("representative requires a feasible official front")
    values = np.asarray([[row["Phi_T"], row["Phi_E"]] for row in front])
    normalized = (values - lower) / np.where(span > 0, span, 1)
    distance = np.linalg.norm(normalized, axis=1)
    order = sorted(range(len(front)), key=lambda i: (
        distance[i], -float(front[i].get("qos_min_possibility") or 0),
        float(normalized[i].max()), front[i]["decision_hash"],
    ))
    return tuple(front[order[0]]["canonical_key"])


def _empirical(scenario: Mapping[str, Any], key: Sequence[float], seed: int,
               count: int = 10_000) -> dict[str, Any]:
    users = scenario["users"]
    n = len(users)
    solution = CanonicalSolution(np.asarray(key[:n]), np.asarray(key[n:2*n]), np.asarray(key[2*n:]))
    total_t = np.zeros(count + 3)
    total_e = np.zeros(count + 3)
    deadline_ok = []
    for index, user in enumerate(users):
        child_seed = int(np.random.SeedSequence([seed, index]).generate_state(1)[0])
        z = lhs_user_disturbances(
            count, scenario["Gamma"], (True,) * 6, child_seed, True, True,
        ).initialization_samples
        if solution.offload[index] == 0:
            c = .5 * (user["C_cycle_lower"] + user["C_cycle_upper"]) + \
                .5 * (user["C_cycle_upper"] - user["C_cycle_lower"]) * z[:, 2]
            f = .5 * (user["F_L_cycle_per_s_lower"] + user["F_L_cycle_per_s_upper"]) + \
                .5 * (user["F_L_cycle_per_s_upper"] - user["F_L_cycle_per_s_lower"]) * z[:, 5]
            t = c / f
            e = user["kappa_SI"] * c * f ** 2
        else:
            parameters = _parameters(user, float(solution.edge_cpu[index]))
            t = evaluate_edge_core_latency_batch_si(z, parameters, chunk_size=5000)
            t += (1 - solution.cache[index]) * user["s_n_bit"] / user["backhaul_rate_bit_per_s"]
            e = evaluate_edge_comm_energy_batch_si(z, parameters, chunk_size=5000)
        total_t += t
        total_e += e
        deadline_ok.append(t <= user["tau_s"])
    empirical_feasible = np.all(np.asarray(deadline_ok), axis=0)
    return {
        "empirical_samples": len(total_t),
        "empirical_feasible_rate": float(empirical_feasible.mean()),
        "time_mean_s": float(total_t.mean()), "time_max_s": float(total_t.max()),
        "energy_mean_J": float(total_e.mean()), "energy_max_J": float(total_e.max()),
    }


def postprocess_exp4(project: Path, formal: Path) -> dict[str, Any]:
    audits = []
    for scenario_name in ("S2", "S3", "S4"):
        scenario = json.loads((project / f"configs/scenarios/exp7_{scenario_name.lower()}_formal_v1.json").read_text(encoding="utf-8"))
        caches: dict[tuple[str, int], Mapping[tuple[float, ...], Mapping[str, Any]]] = {}
        all_values = []
        for method in ("center", "robust"):
            for seed in (52001, 52002, 52003, 52004, 52005):
                run = formal / scenario_name / method / "runs" / f"CorrectedDualBP-CNSGA-II__seed_{seed}"
                cache = _run_cache(run / "checkpoint.pkl")
                caches[(method, seed)] = cache
                all_values.extend([[r["Phi_T"], r["Phi_E"]] for r in EvaluationArchive.from_cache(cache).feasible_records()])
        array = np.asarray(all_values)
        lower, span = array.min(axis=0), array.max(axis=0) - array.min(axis=0)
        for (method, seed), cache in caches.items():
            key = _representative(cache, lower, span)
            audits.append({"scenario": scenario_name, "method": method, "seed": seed,
                           **_empirical(scenario, key, seed)})
    _csv(formal / "empirical_perturbation_audit.csv", audits)
    method_summary = []
    for scenario in ("S2", "S3", "S4"):
        for method in ("center", "robust"):
            chosen = [r for r in audits if r["scenario"] == scenario and r["method"] == method]
            method_summary.append({
                "scenario": scenario, "method": method,
                "empirical_feasible_rate_mean": float(np.mean([r["empirical_feasible_rate"] for r in chosen])),
                "time_max_mean_s": float(np.mean([r["time_max_s"] for r in chosen])),
                "energy_max_mean_J": float(np.mean([r["energy_max_J"] for r in chosen])),
            })
    _csv(formal / "method_summary.csv", method_summary)
    center = np.mean([r["empirical_feasible_rate"] for r in audits if r["method"] == "center"])
    robust = np.mean([r["empirical_feasible_rate"] for r in audits if r["method"] == "robust"])
    result = json.loads((formal / "formal_summary.json").read_text(encoding="utf-8"))
    result.update(empirical_perturbation_completed=True, empirical_samples_per_representative=10003,
                  center_empirical_feasible_rate=float(center), robust_empirical_feasible_rate=float(robust),
                  core_conclusion=f"Robust empirical feasibility={robust:.4f}; Center={center:.4f}.",
                  EXP4_FINAL_FROZEN=True)
    _json(formal / "formal_summary.json", result)
    return result


def _front_representative(path: Path) -> tuple[float, float]:
    rows = _read_csv(path)
    values = np.asarray([[float(r["objective_1"]), float(r["objective_2"])] for r in rows])
    if len(values) == 0:
        return np.nan, np.nan
    lower = values.min(axis=0)
    normalized = (values - lower) / np.where(values.max(axis=0) > lower, values.max(axis=0) - lower, 1)
    index = int(np.argmin(np.linalg.norm(normalized, axis=1)))
    return float(values[index, 0]), float(values[index, 1])


def postprocess_exp5(formal: Path) -> dict[str, Any]:
    rows = _read_csv(formal / "run_manifest.csv")
    summary = []
    for factor in ("Gamma", "eta", "Fmax", "Smax"):
        levels = sorted({float(r["level"]) for r in rows if r["factor"] == factor})
        for level in levels:
            selected = [r for r in rows if r["factor"] == factor and float(r["level"]) == level]
            objectives = []
            for row in selected:
                front = formal / "conditions" / row["condition_hash"] / "runs" / \
                    f"CorrectedDualBP-CNSGA-II__seed_{row['seed']}" / "final_front.csv"
                objectives.append(_front_representative(front))
            summary.append({
                "factor": factor, "level": level,
                "Phi_T_mean": float(np.mean([x[0] for x in objectives])),
                "Phi_E_mean": float(np.mean([x[1] for x in objectives])),
                "feasible_rate_mean": float(np.mean([float(r["feasible_rate"]) for r in selected])),
            })
    _csv(formal / "sensitivity_summary.csv", summary)
    conclusions = {}
    for factor in ("Gamma", "eta", "Fmax", "Smax"):
        group = [r for r in summary if r["factor"] == factor]
        first, last = group[0], group[-1]
        conclusions[factor] = (
            f"{factor}: Phi_T {first['Phi_T_mean']:.6g}->{last['Phi_T_mean']:.6g}; "
            f"Phi_E {first['Phi_E_mean']:.6g}->{last['Phi_E_mean']:.6g}; "
            f"feasibility {first['feasible_rate_mean']:.4f}->{last['feasible_rate_mean']:.4f}."
        )
    result = json.loads((formal / "formal_summary.json").read_text(encoding="utf-8"))
    result.update(sensitivity_conclusions=conclusions, sensitivity_summary_completed=True,
                  EXP5_FINAL_FROZEN=True)
    _json(formal / "formal_summary.json", result)
    return result


def correct_exp6_runtime(formal: Path, session_seconds: Sequence[float]) -> dict[str, Any]:
    total = float(sum(session_seconds))
    manifest = json.loads((formal / "reference_manifest.json").read_text(encoding="utf-8"))
    manifest.update(reference_session_wall_seconds=list(session_seconds), wall_seconds=total,
                    effective_seconds_per_candidate=total / 100)
    _json(formal / "reference_manifest.json", manifest)
    summary = json.loads((formal / "formal_summary.json").read_text(encoding="utf-8"))
    summary["reference"] = manifest
    median = float(summary["runtime"]["median_per_candidate_seconds"])
    summary["efficiency"]["reference_effective_seconds_per_candidate"] = total / 100
    summary["efficiency"]["speedup_ratio"] = (total / 100) / median
    _csv(formal / "efficiency_summary.csv", [summary["efficiency"]])
    _json(formal / "formal_summary.json", summary)
    return summary


def final_freeze(exp4: Path, exp5: Path, exp6: Path, output: Path) -> dict[str, Any]:
    a, b, c = [json.loads((p / "formal_summary.json").read_text(encoding="utf-8")) for p in (exp4, exp5, exp6)]
    result = {
        "EXP4_FINAL_FROZEN": bool(a["EXP4_FINAL_FROZEN"] and a["empirical_perturbation_completed"]),
        "EXP5_FINAL_FROZEN": bool(b["EXP5_FINAL_FROZEN"] and b["sensitivity_summary_completed"]),
        "EXP6_FINAL_FROZEN": bool(c["EXP6_FINAL_FROZEN"]),
        "failed_runs": a["failed_runs"] + b["failed_runs"] + c["failed_runs"],
        "automatic_tuning": False, "bp_training": 0, "exp7_runs": 0, "exp8_runs": 0,
        "formal_directories": {"Exp-4": str(exp4), "Exp-5": str(exp5), "Exp-6": str(exp6)},
    }
    result["exp456_final_frozen"] = all(result[k] for k in ("EXP4_FINAL_FROZEN", "EXP5_FINAL_FROZEN", "EXP6_FINAL_FROZEN"))
    output.mkdir(parents=True, exist_ok=True)
    _json(output / "summary.json", result)
    return result
