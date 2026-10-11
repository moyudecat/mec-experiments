"""Minimal gated Smoke/Preview/Formal runner for frozen Exp-7 algorithms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mec_dualbp.datasets.protocol import ENERGY_FEATURES, TIME_FEATURES
from mec_dualbp.experiments.dualbp_alpha_calibration import infer, load_model_and_scaler
from mec_dualbp.optimization.adapters import (
    FORMAL_ALGORITHMS, RVEA_4D_FINITE_SPAN_DESIGN_HASH,
    V2_ALGORITHM, V2_DESIGN_HASH, OuterAlgorithmConfig, create_adapter,
    matched_canonical_initial_population,
)
from mec_dualbp.optimization.metrics import (
    global_normalize, hypervolume_2d_min, igd, pooled_reference_front,
)
from mec_dualbp.optimization.evaluation_archive import EvaluationArchive
from mec_dualbp.optimization.outer_contract import (
    CORRECTED_DUALBP_HASH, CanonicalOuterEvaluator, CanonicalSolution,
    LEGACY_OBJECTIVE_MODE, OBJECTIVE_NAMES, ROBUST_4D_OBJECTIVE_MODE,
    RepairContext, interval_possibility, objective_payload,
)
from mec_dualbp.physics.local_response import local_response_bounds_si
from mec_dualbp.training.calibration import apply_directional_correction
from mec_dualbp.uncertainty.interval_params import Interval
from mec_dualbp.utils.atomic_io import safe_atomic_write_bytes, safe_atomic_write_json


TRAINING_DIR = "outputs/01_formal/dualbp_training_formal_20260811_003126"
CORRECTED_DIR = "outputs/01_formal/dualbp_alpha_calibration_formal_20260811_134010"
OUTER_SEEDS = (52001, 52002, 52003, 52004, 52005)


def _atomic_json(path: Path, value: object) -> None:
    safe_atomic_write_json(path, value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields or (list(rows[0]) if rows else []))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CorrectedDualBPEngine:
    def __init__(self, project: Path, scenario: Mapping[str, Any]) -> None:
        torch.set_num_threads(1)
        corrected_dir = project / CORRECTED_DIR
        recorded = json.loads((corrected_dir / "corrected_dualbp_v1_hash.json").read_text(encoding="utf-8"))["sha256"]
        if recorded != CORRECTED_DUALBP_HASH or scenario["corrected_dualbp_hash"] != CORRECTED_DUALBP_HASH:
            raise RuntimeError("Corrected DualBP hash drift")
        formal = project / TRAINING_DIR
        self.time_model, self.time_scaler, self.time_config = load_model_and_scaler("time_bp", formal)
        self.energy_model, self.energy_scaler, self.energy_config = load_model_and_scaler("energy_bp", formal)
        alpha = json.loads((corrected_dir / "alpha_values.json").read_text(encoding="utf-8"))
        self.alpha = (
            float(alpha["alpha_T_minus"]), float(alpha["alpha_T_plus"]),
            float(alpha["alpha_E_minus"]), float(alpha["alpha_E_plus"]),
        )
        self.scenario = scenario
        self.users = list(scenario["users"])
        self.local_bounds = [
            local_response_bounds_si(
                Interval(float(user["C_cycle_lower"]), float(user["C_cycle_upper"])),
                Interval(float(user["F_L_cycle_per_s_lower"]), float(user["F_L_cycle_per_s_upper"])),
                float(user["kappa_SI"]), float(scenario["Gamma"]),
            )
            for user in self.users
        ]

    @staticmethod
    def _features(user: Mapping[str, Any], gamma: float, edge_cpu: float) -> tuple[list[float], list[float]]:
        values = {
            "edge_cpu_frequency_cycle_per_s": edge_cpu,
            "D_in_bit_lower": user["D_in_bit_lower"], "D_in_bit_upper": user["D_in_bit_upper"],
            "D_out_bit_lower": user["D_out_bit_lower"], "D_out_bit_upper": user["D_out_bit_upper"],
            "C_cycle_lower": user["C_cycle_lower"], "C_cycle_upper": user["C_cycle_upper"],
            "I_u_w_lower": user["I_u_w_lower"], "I_u_w_upper": user["I_u_w_upper"],
            "I_d_w_lower": user["I_d_w_lower"], "I_d_w_upper": user["I_d_w_upper"],
            "uplink_bandwidth_hz": user["uplink_bandwidth_hz"],
            "downlink_bandwidth_hz": user["downlink_bandwidth_hz"],
            "uplink_terminal_tx_power_w": user["uplink_terminal_tx_power_w"],
            "downlink_bs_tx_power_w": user["downlink_bs_tx_power_w"],
            "terminal_rx_power_w": user["terminal_rx_power_w"],
            "uplink_channel_gain_linear": user["uplink_channel_gain_linear"],
            "downlink_channel_gain_linear": user["downlink_channel_gain_linear"],
            "noise_power_w": user["noise_power_w"], "Gamma": gamma,
        }
        return ([float(values[name]) for name in TIME_FEATURES],
                [float(values[name]) for name in ENERGY_FEATURES])

    def _predict_corrected(self, time_features: np.ndarray,
                           energy_features: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                                  np.ndarray, np.ndarray]:
        """Legacy prediction hook kept byte-for-byte equivalent in numerical semantics.

        The hook exists solely so the separately registered Improved engine can reuse the
        canonical system aggregation without changing historical Exp-7 loading paths.
        """
        raw_tl, raw_tu = infer(self.time_model, self.time_scaler, time_features)
        raw_el, raw_eu = infer(self.energy_model, self.energy_scaler, energy_features)
        corr_tl, corr_tu = apply_directional_correction(raw_tl, raw_tu, self.alpha[0], self.alpha[1])
        corr_el, corr_eu = apply_directional_correction(raw_el, raw_eu, self.alpha[2], self.alpha[3])
        return corr_tl, corr_tu, corr_el, corr_eu

    def evaluate(self, solution: CanonicalSolution) -> dict[str, Any]:
        gamma = float(self.scenario["Gamma"])
        active = np.flatnonzero(solution.offload == 1)
        time_lo = np.empty(len(self.users), dtype=np.float64)
        time_hi = np.empty(len(self.users), dtype=np.float64)
        energy_lo = np.empty(len(self.users), dtype=np.float64)
        energy_hi = np.empty(len(self.users), dtype=np.float64)
        for index in np.flatnonzero(solution.offload == 0):
            local = self.local_bounds[int(index)]
            time_lo[index], time_hi[index] = local.local_latency_lower_s, local.local_latency_upper_s
            energy_lo[index], energy_hi[index] = local.local_energy_lower_j, local.local_energy_upper_j
        if active.size:
            tx, ex = [], []
            for index in active:
                time_features, energy_features = self._features(self.users[int(index)], gamma, float(solution.edge_cpu[index]))
                tx.append(time_features); ex.append(energy_features)
            corr_tl, corr_tu, corr_el, corr_eu = self._predict_corrected(
                np.asarray(tx, dtype=np.float64), np.asarray(ex, dtype=np.float64))
            for position, index in enumerate(active):
                user = self.users[int(index)]
                extra = (1.0 - solution.cache[index]) * float(user["s_n_bit"]) / float(user["backhaul_rate_bit_per_s"])
                time_lo[index], time_hi[index] = corr_tl[position] + extra, corr_tu[position] + extra
                energy_lo[index], energy_hi[index] = corr_el[position], corr_eu[position]
        arrays = (time_lo, time_hi, energy_lo, energy_hi)
        if not all(np.isfinite(array).all() for array in arrays):
            raise FloatingPointError("nonfinite system response")
        possibilities = np.asarray([
            interval_possibility(float(time_lo[i]), float(time_hi[i]), float(self.users[i]["tau_s"]))
            for i in range(len(self.users))
        ])
        cv = float(np.maximum(0.0, float(self.scenario["eta"]) - possibilities).sum())
        payload = objective_payload(
            float(time_lo.sum()), float(time_hi.sum()),
            float(energy_lo.sum()), float(energy_hi.sum()),
            objective_mode=getattr(self, "objective_mode", LEGACY_OBJECTIVE_MODE),
            rho_time=float(self.scenario["rho_T_s"]), rho_energy=float(self.scenario["rho_E_J"]),
            lambda_time=float(self.scenario["lambda_T"]), lambda_energy=float(self.scenario["lambda_E"]),
        )
        return {
            "prediction_mode": "corr_pred",
            **payload,
            "constraint_violation": cv,
            "feasible": cv <= 1e-12,
            "qos_min_possibility": float(possibilities.min()),
            "c7_min_margin": float(possibilities.min() - float(self.scenario["eta"])),
            "c7_violating_user_count": int(np.sum(possibilities < float(self.scenario["eta"]) - 1e-12)),
        }


def _context(scenario: Mapping[str, Any]) -> RepairContext:
    sizes = np.asarray([float(user["s_n_bit"]) for user in scenario["users"]])
    benefit_density = np.asarray(scenario["cache_benefit_per_bit"], dtype=np.float64)
    return RepairContext(
        float(scenario["f_min_cycle_per_s"]), float(scenario["f_max_cycle_per_s"]),
        float(scenario["F_max_cycle_per_s"]), sizes, float(scenario["Smax_bit"]), benefit_density,
    )


def _checkpoint(path: Path, adapter: Any, evaluator: CanonicalOuterEvaluator) -> None:
    payload = {
        "adapter": adapter,
        "algorithm_identity": adapter.name,
        "v2_design_hash": V2_DESIGN_HASH if adapter.name == V2_ALGORITHM else None,
        "elite_fraction": 0.10 if adapter.name == V2_ALGORITHM else None,
        "population": adapter.config.population_size,
        "objective_count": adapter.objective_count,
        "rvea_4d_design_hash": (
            RVEA_4D_FINITE_SPAN_DESIGN_HASH
            if adapter.objective_count > 2 and adapter.name in {"RVEA", "OE-RVEA"}
            else None),
        "evaluator": {
            "requests": evaluator.requests,
            "unique_evaluations": evaluator.unique_evaluations,
            "cache_hits": evaluator.cache_hits,
            "cache": evaluator._cache,
        },
    }
    safe_atomic_write_bytes(path, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def _restore(path: Path, engine: CorrectedDualBPEngine, expected_algorithm: str,
             expected_population: int, expected_objective_count: int = 2) -> tuple[Any, CanonicalOuterEvaluator]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    recorded_algorithm = payload.get("algorithm_identity", getattr(payload.get("adapter"), "name", None))
    if recorded_algorithm != expected_algorithm:
        raise RuntimeError("checkpoint algorithm identity mismatch")
    recorded_population = int(payload.get(
        "population", getattr(payload["adapter"].config, "population_size", -1)))
    if recorded_population != expected_population:
        raise RuntimeError("checkpoint population mismatch")
    recorded_objective_count = int(payload.get(
        "objective_count", getattr(payload["adapter"], "objective_count", 2)))
    if recorded_objective_count != expected_objective_count:
        raise RuntimeError("checkpoint objective-count mismatch")
    if (expected_objective_count > 2 and expected_algorithm in {"RVEA", "OE-RVEA"}
            and payload.get("rvea_4d_design_hash") != RVEA_4D_FINITE_SPAN_DESIGN_HASH):
        raise RuntimeError("checkpoint 4D RVEA adaptation design mismatch")
    if expected_algorithm == V2_ALGORITHM and payload.get("v2_design_hash") != V2_DESIGN_HASH:
        raise RuntimeError("checkpoint V2 design hash mismatch")
    if expected_algorithm != V2_ALGORITHM and payload.get("v2_design_hash") is not None:
        raise RuntimeError("V2 checkpoint cannot be loaded as V1/baseline")
    engine_hash = getattr(engine, "corrected_dualbp_hash", CORRECTED_DUALBP_HASH)
    evaluator = CanonicalOuterEvaluator(
        engine.evaluate, corrected_dualbp_hash=engine_hash,
        prediction_mode=getattr(engine, "prediction_mode", "corr_pred"))
    state = payload["evaluator"]
    evaluator.requests, evaluator.unique_evaluations, evaluator.cache_hits = (
        int(state["requests"]), int(state["unique_evaluations"]), int(state["cache_hits"])
    )
    evaluator._cache = dict(state["cache"])
    return payload["adapter"], evaluator


def run_one(output: Path, algorithm: str, seed: int, population: int, budget: int,
            scenario: Mapping[str, Any], engine: CorrectedDualBPEngine, *,
            objective_mode: str = LEGACY_OBJECTIVE_MODE) -> dict[str, Any]:
    if objective_mode not in OBJECTIVE_NAMES:
        raise ValueError(f"unsupported objective mode: {objective_mode}")
    objective_names = tuple(OBJECTIVE_NAMES[objective_mode])
    objective_count = len(objective_names)
    engine.objective_mode = objective_mode
    run_dir = output / "runs" / f"{algorithm.replace('/', '_')}__seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path, checkpoint_path = run_dir / "status.json", run_dir / "checkpoint.pkl"
    if status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        existing_mode = existing.get("objective_mode", LEGACY_OBJECTIVE_MODE)
        if (existing.get("status") == "completed"
                and int(existing.get("requests", -1)) == budget
                and existing_mode == objective_mode):
            return existing | {"completed_skip": True}
    started = time.monotonic()
    context = _context(scenario)
    config = OuterAlgorithmConfig(population, budget, 1, objective_count)
    initial_pool = matched_canonical_initial_population(seed, population, context)
    initial_hash = hashlib.sha256(initial_pool.tobytes()).hexdigest()
    if checkpoint_path.exists():
        adapter, evaluator = _restore(
            checkpoint_path, engine, algorithm, population, objective_count)
        recovered = True
    else:
        adapter = create_adapter(algorithm, len(scenario["users"]), context, config, seed, initial_pool)
        engine_hash = getattr(engine, "corrected_dualbp_hash", CORRECTED_DUALBP_HASH)
        evaluator = CanonicalOuterEvaluator(
            engine.evaluate, corrected_dualbp_hash=engine_hash,
            prediction_mode=getattr(engine, "prediction_mode", "corr_pred"))
        recovered = False
    feasible_count = 0
    numerical_failures = 0
    illegal_count = 0
    checkpoint_stride = max(population, 500)
    next_checkpoint = ((evaluator.requests // checkpoint_stride) + 1) * checkpoint_stride
    _atomic_json(status_path, {"status": "running", "requests": evaluator.requests, "budget": budget, "recovered": recovered})
    try:
        while evaluator.requests < budget:
            pending = adapter.ask()
            remaining = budget - evaluator.requests
            paid_pending = pending[:remaining]
            for solution in paid_pending:
                if (not np.isin(solution.offload, (0, 1)).all() or not np.isin(solution.cache, (0, 1)).all()
                        or np.any(solution.cache > solution.offload) or not np.isfinite(solution.edge_cpu).all()):
                    illegal_count += 1
            if len(paid_pending) == len(pending):
                results = adapter.evaluate(evaluator)
            else:
                # Deterministic exact-budget tail: pay only for the first q
                # candidates in native ask order.  A partial pymoo/population
                # batch is not told back to the search algorithm; no free or
                # fabricated objective values enter search state.
                results = [dict(evaluator.evaluate_outer_solution_canonical(x, context)) for x in paid_pending]
            feasible_count += sum(bool(row["feasible"]) for row in results)
            numerical_failures += sum(not np.isfinite(np.asarray(row["objectives"], dtype=float)).all() for row in results)
            if len(paid_pending) == len(pending):
                adapter.tell(results)
            else:
                # The run budget is now exhausted.  Discard the unpaid suffix
                # and clear transient ask state without changing population,
                # archive, pbest, leaders, or any evaluated search result.
                adapter.pending = ()
                if hasattr(adapter, "_infills"):
                    adapter._infills = None
            if evaluator.requests >= next_checkpoint:
                _checkpoint(checkpoint_path, adapter, evaluator)
                _atomic_json(status_path, {"status": "running", "requests": evaluator.requests, "budget": budget, "recovered": recovered})
                next_checkpoint += checkpoint_stride
        _checkpoint(checkpoint_path, adapter, evaluator)
        evaluation_archive = EvaluationArchive.from_cache(evaluator._cache)
        if evaluation_archive.objective_names != objective_names:
            raise RuntimeError("evaluation archive objective schema mismatch")
        front_records = evaluation_archive.official_front()
        front = np.asarray(
            [row["objectives"] for row in front_records], dtype=np.float64
        ).reshape(-1, objective_count)
        front_rows = []
        for row in front_records:
            front_rows.append({
                "scenario": scenario["scenario_id"], "gamma": float(scenario["Gamma"]),
                "seed": seed, "decision_hash": row["decision_hash"],
                **{name: row[name] for name in objective_names},
                **{field: row.get(field) for field in (
                    "T_lower", "T_upper", "E_lower", "E_upper",
                    "c_T", "r_T", "c_E", "r_E")},
                "feasible": row["feasible"],
                "constraint_violation": row["constraint_violation"],
                "QoS": row["qos_min_possibility"],
                "c7_min_margin": row.get("c7_min_margin"),
                "c7_violating_user_count": row.get("c7_violating_user_count"),
            })
        front_fields = list(front_rows[0]) if front_rows else [
            "scenario", "gamma", "seed", "decision_hash", *objective_names]
        _write_csv(run_dir / "final_front.csv", front_rows, front_fields)
        if front_rows:
            lower, upper = front.min(axis=0), front.max(axis=0)
            normalized = np.divide(
                front - lower, upper - lower,
                out=np.zeros_like(front), where=upper > lower)
            distance = np.linalg.norm(normalized, axis=1)
            worst = normalized.max(axis=1)
            representative_index = min(
                range(len(front_rows)),
                key=lambda index: (
                    float(distance[index]),
                    -float(front_rows[index]["QoS"] or 0.0),
                    float(worst[index]),
                    str(front_rows[index]["decision_hash"]),
                ))
            _atomic_json(run_dir / "representative_solution.json", {
                **front_rows[representative_index],
                "selection_rule": "within-run min-max equal-weight ideal distance; QoS/worst/hash ties",
                "ideal_distance": float(distance[representative_index]),
            })
        _atomic_json(run_dir / "evaluation_archive_metadata.json",
                     evaluation_archive.metadata(paid_requests=evaluator.requests, cache_hits=evaluator.cache_hits))
        history = [asdict(row) for row in adapter.get_history()]
        _write_csv(run_dir / "history.csv", history, list(history[0]) if history else ["evaluation_requests"])
        elapsed = time.monotonic() - started
        final = {
            "status": "completed", "scenario": scenario["scenario_id"], "algorithm": algorithm,
            "seed": seed, "population": population, "budget": budget,
            "gamma": float(scenario["Gamma"]), "objective_mode": objective_mode,
            "objective_names": list(objective_names), "objective_count": objective_count,
            "requests": evaluator.requests, "unique_evaluations": evaluator.unique_evaluations,
            "cache_hits": evaluator.cache_hits, "feasible_evaluations": feasible_count,
            "feasible_rate": feasible_count / budget, "illegal_canonical_count": illegal_count,
            "nonfinite_count": numerical_failures, "front_size": len(front),
            "official_front_source": "EvaluationArchive",
            "history_records": len(history), "initial_pool_hash": initial_hash,
            "wall_seconds": elapsed, "requests_per_second": budget / max(elapsed, 1e-12),
            "checkpoint_exists": checkpoint_path.exists(), "recovered": recovered,
            "completed_skip": False,
        }
        if algorithm == "MOPSO":
            final["mopso_archive_size"] = int(len(adapter.archive_f))
            final["mopso_archive_all_paid"] = evaluator.requests >= len(adapter.archive_f)
        if algorithm == V2_ALGORITHM:
            final["v2_design_hash"] = V2_DESIGN_HASH
            final["elite_fraction"] = 0.10
            final["ca_selection_records"] = len(adapter.ca_selection_audit)
            final["ca_only_retained_total"] = sum(
                int(row["ca_only_retained_count"]) for row in adapter.ca_selection_audit)
            _write_csv(run_dir / "ca_selection_audit.csv", list(adapter.ca_selection_audit))
        if algorithm in {"RVEA", "OE-RVEA"} and objective_count > 2:
            adaptation_rows = list(adapter.reference_vector_adaptation_audit)
            final["rvea_4d_design_hash"] = RVEA_4D_FINITE_SPAN_DESIGN_HASH
            final["reference_vector_adaptation_records"] = len(adaptation_rows)
            final["reference_vector_adaptation_all_finite"] = all(
                bool(row["finite_after"]) for row in adaptation_rows)
            _write_csv(run_dir / "reference_vector_adaptation_audit.csv", adaptation_rows)
        _atomic_json(run_dir / "run_manifest.json", {
            "scenario": scenario["scenario_id"], "gamma": float(scenario["Gamma"]),
            "seed": seed, "algorithm": algorithm, "population": population,
            "evaluation_budget": budget, "workers": 1,
            "objective_mode": objective_mode, "objective_names": list(objective_names),
            "all_objectives_minimize": True,
            "rvea_4d_design_hash": (
                RVEA_4D_FINITE_SPAN_DESIGN_HASH
                if algorithm in {"RVEA", "OE-RVEA"} and objective_count > 2 else None),
            "checkpoint": "checkpoint.pkl", "final_front": "final_front.csv",
            "representative_solution": "representative_solution.json",
            "runtime_seconds": elapsed, "evaluation_count": evaluator.requests,
        })
        _atomic_json(status_path, final)
        return final
    except Exception as error:
        _atomic_json(status_path, {
            "status": "failed", "algorithm": algorithm, "seed": seed,
            "requests": evaluator.requests, "budget": budget, "error": str(error),
            "traceback": traceback.format_exc(),
        })
        raise


def run_phase(project: Path, scenario_path: Path, output: Path, *, phase: str) -> dict[str, Any]:
    started = time.monotonic()
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    if scenario.get("scenario_id") != "S3" or len(scenario.get("users", [])) != 20:
        raise RuntimeError("phase requires frozen S3")
    output.mkdir(parents=True, exist_ok=True)
    if phase == "smoke":
        population, budget, seeds = 10, 100, (52001,)
    elif phase == "preview":
        population, budget, seeds = 50, 2000, (52001, 52002)
    else:
        raise ValueError("run_phase supports smoke or preview")
    config = {
        "phase": phase, "scenario": "S3", "algorithms": list(FORMAL_ALGORITHMS),
        "seeds": list(seeds), "population": population, "budget_per_run": budget,
        "workers": 1, "scenario_hash": _hash_file(scenario_path),
        "corrected_dualbp_hash": CORRECTED_DUALBP_HASH,
    }
    _atomic_json(output / "config.json", config)
    engine = CorrectedDualBPEngine(project, scenario)
    results = []
    for seed in seeds:
        for algorithm in FORMAL_ALGORITHMS:
            results.append(run_one(output, algorithm, seed, population, budget, scenario, engine))
            _atomic_json(output / "run_progress.json", {
                "phase": phase, "completed_runs": len(results),
                "total_runs": len(seeds) * len(FORMAL_ALGORITHMS),
                "requests_completed": sum(int(row["requests"]) for row in results),
            })
    requested = [int(row["requests"]) for row in results]
    nonfinite = sum(int(row["nonfinite_count"]) for row in results)
    illegal = sum(int(row["illegal_canonical_count"]) for row in results)
    all_completed = len(results) == len(seeds) * len(FORMAL_ALGORITHMS) and all(row["status"] == "completed" for row in results)
    budget_ok = bool(requested) and min(requested) == max(requested) == budget
    mopso_ok = all(row.get("mopso_archive_all_paid", True) for row in results)
    history_ok = all(int(row["history_records"]) > 0 for row in results)
    fronts_exported = all((output / "runs" / f"{row['algorithm'].replace('/', '_')}__seed_{row['seed']}" / "final_front.csv").exists() for row in results)
    all_infeasible = sum(int(row["feasible_evaluations"]) for row in results) == 0
    common = {
        "common_evaluator": "CanonicalOuterEvaluator.evaluate_outer_solution_canonical",
        "common_repair": "canonical_outer_repair",
        "passed": True,
    }
    _atomic_json(output / "evaluator_repair_integrity.json", common)
    if phase == "preview":
        fronts = []
        for row in results:
            path = output / "runs" / f"{row['algorithm'].replace('/', '_')}__seed_{row['seed']}" / "final_front.csv"
            with path.open(encoding="utf-8-sig") as stream:
                values = [[float(item["objective_1"]), float(item["objective_2"])] for item in csv.DictReader(stream)]
            if values:
                fronts.append(values)
        metric_pipeline = {"passed": False, "reason": "no feasible fronts"}
        if fronts:
            reference = pooled_reference_front(fronts)
            lo, hi = reference.min(axis=0), reference.max(axis=0)
            hi = np.where(hi > lo, hi, lo + 1.0)
            norm_ref = global_normalize(reference, lo, hi)
            ref_point = np.maximum(norm_ref.max(axis=0) + 0.1, 1.1)
            metric_pipeline = {
                "passed": True, "reference_points": len(reference),
                "reference_hv": hypervolume_2d_min(norm_ref, ref_point),
                "reference_igd": igd(norm_ref, norm_ref),
            }
        _atomic_json(output / "hv_igd_pipeline_check.json", metric_pipeline)
    else:
        metric_pipeline = {"passed": True, "not_required": True}
    passed = (all_completed and budget_ok and nonfinite == 0 and illegal == 0 and mopso_ok
              and history_ok and fronts_exported and common["passed"] and metric_pipeline["passed"]
              and (phase != "preview" or not all_infeasible))
    elapsed = time.monotonic() - started
    summary = {
        f"exp7_{phase}_decision": "passed" if passed else "failed",
        "status": "completed_and_frozen" if passed else "failed",
        "runs_completed": len(results), "runs_expected": len(seeds) * len(FORMAL_ALGORITHMS),
        "requests_per_run": requested, "maximum_budget_difference": max(requested) - min(requested),
        "total_requests": sum(requested), "nonfinite_count": nonfinite,
        "illegal_canonical_count": illegal, "all_infeasible": all_infeasible,
        "mopso_archive_passed": mopso_ok, "history_complete": history_ok,
        "fronts_exported": fronts_exported, "checkpoint_passed": all(row["checkpoint_exists"] for row in results),
        "wall_seconds": elapsed, "overall_requests_per_second": sum(requested) / max(elapsed, 1e-12),
        "corrected_dualbp_hash": CORRECTED_DUALBP_HASH,
        "results": results,
    }
    _atomic_json(output / "summary.json", summary)
    _write_csv(output / "run_summary.csv", results)
    (output / "README.md").write_text(f"# Exp-7 {phase}\n\n本目录仅执行预注册工程门禁，不作算法排名或论文结论。\n", encoding="utf-8")
    return summary


def formal_preflight(project: Path, output: Path) -> dict[str, Any]:
    required = [project / "configs" / "scenarios" / f"exp7_{name.lower()}_formal_v1.json" for name in ("S1", "S2", "S3", "S4", "S5")]
    missing = [path.relative_to(project).as_posix() for path in required if not path.exists()]
    result = {
        "formal_preflight_passed": not missing,
        "missing_scenario_configs": missing,
        "required_runs": 125,
        "total_budget": 2_500_000,
        "formal_started": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "formal_preflight.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=("smoke", "preview"), required=True)
    args = parser.parse_args()
    summary = run_phase(Path(args.project_root).resolve(), Path(args.scenario).resolve(), Path(args.output).resolve(), phase=args.phase)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
