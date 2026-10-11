"""Shared Exp-4/5/6 runner binding; importing this module never starts a Formal run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mec_dualbp.experiments.exp7_chain_runner import _context, run_one
from mec_dualbp.optimization.dualbp_engine_registry import create_corrected_dualbp_engine
from mec_dualbp.optimization.outer_contract import (
    CanonicalOuterEvaluator, CanonicalSolution, IMPROVED_CORRECTED_DUALBP_HASH,
    LEGACY_OBJECTIVE_MODE, interval_possibility, objective_payload,
)
from mec_dualbp.physics.channel import downlink_rate_bit_per_s, uplink_rate_bit_per_s
from mec_dualbp.physics.energy import edge_comm_energy_point_j
from mec_dualbp.physics.latency import cache_miss_latency_s, edge_core_latency_point_s
from mec_dualbp.physics.local_response import local_energy_point_j, local_latency_point_s


class CenterDeterministicEngine:
    """Exact z=0 physical baseline used only by the frozen Exp-4 comparator."""

    prediction_mode = "deterministic_center"
    corrected_dualbp_hash = IMPROVED_CORRECTED_DUALBP_HASH

    def __init__(self, scenario: Mapping[str, Any]) -> None:
        self.scenario = scenario
        self.users = list(scenario["users"])

    @staticmethod
    def _center(user: Mapping[str, Any], stem: str) -> float:
        return .5 * (float(user[f"{stem}_lower"]) + float(user[f"{stem}_upper"]))

    def evaluate(self, solution: CanonicalSolution) -> Mapping[str, Any]:
        times, energies, possibilities = [], [], []
        for index, user in enumerate(self.users):
            if solution.offload[index] == 0:
                latency = local_latency_point_s(
                    self._center(user, "C_cycle"), self._center(user, "F_L_cycle_per_s"))
                energy = local_energy_point_j(
                    self._center(user, "C_cycle"), self._center(user, "F_L_cycle_per_s"),
                    float(user["kappa_SI"]))
            else:
                i_u, i_d = self._center(user, "I_u_w"), self._center(user, "I_d_w")
                ru = uplink_rate_bit_per_s(
                    user["uplink_bandwidth_hz"], user["uplink_terminal_tx_power_w"],
                    user["uplink_channel_gain_linear"], user["noise_power_w"], i_u)
                rd = downlink_rate_bit_per_s(
                    user["downlink_bandwidth_hz"], user["downlink_bs_tx_power_w"],
                    user["downlink_channel_gain_linear"], user["noise_power_w"], i_d)
                d_in, d_out = self._center(user, "D_in_bit"), self._center(user, "D_out_bit")
                latency = edge_core_latency_point_s(
                    d_in, ru, self._center(user, "C_cycle"), solution.edge_cpu[index], d_out, rd)
                latency += (1.0 - solution.cache[index]) * cache_miss_latency_s(
                    user["s_n_bit"], user["backhaul_rate_bit_per_s"])
                energy = edge_comm_energy_point_j(
                    user["uplink_terminal_tx_power_w"], d_in, ru,
                    user["terminal_rx_power_w"], d_out, rd)
            times.append(latency); energies.append(energy)
            possibilities.append(interval_possibility(latency, latency, float(user["tau_s"])))
        cv = float(np.maximum(
            0.0, float(self.scenario["eta"]) - np.asarray(possibilities, dtype=np.float64)).sum())
        total_time, total_energy = float(sum(times)), float(sum(energies))
        payload = objective_payload(
            total_time, total_time, total_energy, total_energy,
            objective_mode=LEGACY_OBJECTIVE_MODE,
            rho_time=float(self.scenario["rho_T_s"]), rho_energy=float(self.scenario["rho_E_J"]),
            lambda_time=float(self.scenario["lambda_T"]), lambda_energy=float(self.scenario["lambda_E"]))
        eta = float(self.scenario["eta"])
        minimum_possibility = float(min(possibilities))
        return {"prediction_mode": self.prediction_mode, **payload,
                "constraint_violation": cv, "feasible": cv <= 1e-12,
                "qos_min_possibility": minimum_possibility,
                "c7_min_margin": minimum_possibility - eta,
                "c7_violating_user_count": int(np.sum(np.asarray(possibilities) < eta - 1e-12)),
                "evaluator_mode": "deterministic_center_physics"}


def build_outer_stack(project: Path, scenario: Mapping[str, Any], protocol: Mapping[str, Any]):
    """Build the versioned engine and unchanged canonical repair/evaluation stack."""
    mode = str(protocol["evaluator_mode"])
    engine = (CenterDeterministicEngine(scenario) if mode == "deterministic_center_physics"
              else create_corrected_dualbp_engine(
                  project, scenario, evaluator_mode=mode,
                  evaluator_hash=str(protocol["evaluator_hash"])))
    evaluator = CanonicalOuterEvaluator(
        engine.evaluate, corrected_dualbp_hash=str(protocol["evaluator_hash"]),
        prediction_mode=getattr(engine, "prediction_mode", "corr_pred"),
    )
    return engine, evaluator, _context(scenario)


def evaluate_one_canonical(project: Path, scenario: Mapping[str, Any],
                           protocol: Mapping[str, Any], solution: CanonicalSolution) -> Mapping[str, Any]:
    """Non-performance engineering entry used by tests and the readiness Smoke."""
    _, evaluator, context = build_outer_stack(project, scenario, protocol)
    return evaluator.evaluate_outer_solution_canonical(solution, context)


def run_frozen_formal(project: Path, scenario_path: Path, protocol_path: Path,
                      output: Path, *, seed: int, method: str = "robust") -> Mapping[str, Any]:
    """Future explicit Formal entry; the recovery task does not call it."""
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN":
        raise RuntimeError("Formal execution requires a FROZEN protocol")
    if seed not in protocol["outer_seeds"]:
        raise ValueError("seed is outside the frozen protocol")
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    scenario["corrected_dualbp_hash"] = protocol["evaluator_hash"]
    selected = dict(protocol)
    if method == "center" and protocol.get("experiment") == "Exp-4":
        selected["evaluator_mode"] = "deterministic_center_physics"
    elif method != "robust":
        raise ValueError("method must be robust, or center for Exp-4")
    engine, _, _ = build_outer_stack(project, scenario, selected)
    return run_one(
        output, str(protocol["formal_optimizer"]), seed,
        int(protocol["population"]), int(protocol["request_budget_per_run"]),
        scenario, engine,
    )
