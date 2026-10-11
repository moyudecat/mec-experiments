"""Deterministic, self-contained Exp-7 S3 scenario construction."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mec_dualbp.datasets.protocol import _sample_from_seed
from mec_dualbp.experiments.exp1_formal_protocol import SAMPLE_TYPES
from mec_dualbp.experiments.exp1_preview_protocol import canonical_json_hash
from mec_dualbp.physics.local_response import local_response_bounds_si
from mec_dualbp.uncertainty.interval_params import Interval


SCENARIO_SEED = 73003
N_USERS = 20
GAMMA = 3.0
ETA = 0.90
BACKHAUL_RATE = 100.0e6
KAPPA = 1.0e-28
S_N_MB_RANGE = (20.0, 100.0)
MB_TO_BIT = 8.0e6
CORRECTED_DUALBP_HASH = "a15b8fb591df123b475f521ef35e4037b6ca201f4172caf804d3af571afe13e2"
PARENT_PROTOCOL_HASH = "5cd1d55d4b4aafcbab11c4986b9d151f409c03b80e5e30c2129e3aed2cb1426c"


def _sha_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)


SCENARIOS = {
    "S1": {"seed": 73001, "N": 10, "Gamma": 1.0, "eta": 0.80, "F_max": 24.0e9, "storage_ratio": 1.2,
           "rho_T": 8.809164041, "rho_E": 0.4650848491},
    "S2": {"seed": 73002, "N": 20, "Gamma": 2.0, "eta": 0.85, "F_max": 20.0e9, "storage_ratio": 1.0,
           "rho_T": 17.618328082, "rho_E": 0.9301696982},
    "S3": {"seed": 73003, "N": 20, "Gamma": 3.0, "eta": 0.90, "F_max": 20.0e9, "storage_ratio": 1.0,
           "rho_T": 17.618328082, "rho_E": 0.9301696982},
    "S4": {"seed": 73004, "N": 30, "Gamma": 3.0, "eta": 0.90, "F_max": 16.0e9, "storage_ratio": 0.8,
           "rho_T": 26.427492123, "rho_E": 1.3952545473},
    "S5": {"seed": 73005, "N": 40, "Gamma": 4.0, "eta": 0.95, "F_max": 14.0e9, "storage_ratio": 0.7,
           "rho_T": 35.236656164, "rho_E": 1.8603393964},
}


def sample_type_allocation(scenario_id: str = "S3", n_users: int = N_USERS,
                           seed: int = SCENARIO_SEED) -> tuple[dict[str, int], dict[str, Any]]:
    token = f"EXP7|{scenario_id}|{seed}|sample_type_balance"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    base, remainder = divmod(n_users, len(SAMPLE_TYPES))
    ranked = sorted(SAMPLE_TYPES, key=lambda name: hashlib.sha256(f"{token}|{name}".encode()).hexdigest())
    selected = tuple(ranked[:remainder])
    counts = {name: base + int(name in selected) for name in SAMPLE_TYPES}
    return counts, {
        "token": token,
        "sha256": digest,
        "allocation": "balanced quotient/remainder; remainder classes ranked by SHA256",
        "extra_count_types": list(selected),
    }


def _storage_lhs_bit(seed: int, n_users: int) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    unit = (np.arange(n_users, dtype=np.float64) + rng.random(n_users)) / n_users
    values = (S_N_MB_RANGE[0] + unit * (S_N_MB_RANGE[1] - S_N_MB_RANGE[0])) * MB_TO_BIT
    permutation = rng.permutation(n_users)
    assigned = values[permutation]
    return assigned, {
        "method": "deterministic 1-D stratified LHS",
        "rng": f"numpy.default_rng({seed})",
        "engineering_range_MB": list(S_N_MB_RANGE),
        "MB_definition": "1 MB = 1e6 byte = 8e6 bit",
        "strata": n_users,
        "assignment_permutation": permutation.tolist(),
        "vector_sha256": canonical_json_hash(assigned.tolist()),
    }


def build_scenario_payload(scenario_id: str) -> dict[str, Any]:
    if scenario_id not in SCENARIOS:
        raise ValueError("unknown Exp-7 scenario")
    spec = SCENARIOS[scenario_id]
    seed, n_users = int(spec["seed"]), int(spec["N"])
    gamma, eta = float(spec["Gamma"]), float(spec["eta"])
    counts, balance = sample_type_allocation(scenario_id, n_users, seed)
    sample_types = [name for name in SAMPLE_TYPES for _ in range(counts[name])]
    if len(sample_types) != n_users:
        raise ArithmeticError("scenario allocation size mismatch")
    storage, storage_rule = _storage_lhs_bit(seed, n_users)
    users: list[dict[str, Any]] = []
    for ordinal, (sample_type, s_n) in enumerate(zip(sample_types, storage), 1):
        user_id = f"{scenario_id}_USER_{ordinal:02d}"
        seed_token = f"EXP7|{scenario_id}|{seed}|{user_id}|{sample_type}"
        user_seed = _sha_seed(seed_token)
        sample = _sample_from_seed(
            sample_id=user_id, role="Scenario", gamma=int(gamma), sample_type=sample_type,
            seed_name=f"EXP7_{scenario_id}_SCENARIO_{ordinal:02d}", seed_value=user_seed,
            formal_eligible=False,
        )
        p = sample.parameters
        local = local_response_bounds_si(sample.parameters.cpu_cycles, sample.local_cpu_frequency, KAPPA, gamma)
        tau = local.local_latency_upper_s
        users.append({
            "user_id": user_id,
            "sample_type": sample_type,
            "generation_seed_token": seed_token,
            "generation_seed_sha256": hashlib.sha256(seed_token.encode("utf-8")).hexdigest(),
            "generation_seed_value": user_seed,
            "sample_content_hash": sample.content_hash,
            "D_in_bit_lower": p.input_size_bit.lower,
            "D_in_bit_upper": p.input_size_bit.upper,
            "D_out_bit_lower": p.output_size_bit.lower,
            "D_out_bit_upper": p.output_size_bit.upper,
            "C_cycle_lower": p.cpu_cycles.lower,
            "C_cycle_upper": p.cpu_cycles.upper,
            "I_u_w_lower": p.uplink_interference_w.lower,
            "I_u_w_upper": p.uplink_interference_w.upper,
            "I_d_w_lower": p.downlink_interference_w.lower,
            "I_d_w_upper": p.downlink_interference_w.upper,
            "F_L_cycle_per_s_lower": sample.local_cpu_frequency.lower,
            "F_L_cycle_per_s_upper": sample.local_cpu_frequency.upper,
            "uplink_bandwidth_hz": p.uplink_bandwidth_hz,
            "downlink_bandwidth_hz": p.downlink_bandwidth_hz,
            "uplink_terminal_tx_power_w": p.uplink_terminal_tx_power_w,
            "downlink_bs_tx_power_w": p.downlink_bs_tx_power_w,
            "terminal_rx_power_w": p.terminal_rx_power_w,
            "uplink_channel_gain_linear": p.uplink_channel_gain_linear,
            "downlink_channel_gain_linear": p.downlink_channel_gain_linear,
            "noise_power_w": p.noise_power_w,
            "s_n_bit": float(s_n),
            "backhaul_rate_bit_per_s": BACKHAUL_RATE,
            "kappa_SI": KAPPA,
            "tau_s": tau,
        })
    base_s0 = 0.5 * sum(float(user["s_n_bit"]) for user in users)
    smax = float(spec["storage_ratio"]) * base_s0
    payload = {
        "scenario_schema_version": "exp7_scenario_v1",
        "scenario_id": scenario_id,
        "scenario_seed": seed,
        "scenario_seed_namespace": f"EXP7_{scenario_id}_SCENARIO_*",
        "outer_seed_namespace": "EXP7_OUTER_*",
        "N": n_users,
        "Gamma": gamma,
        "eta": eta,
        "S0_bit": base_s0,
        "S0_semantics": "scenario cache-capacity baseline; not initial occupied cache",
        "Smax_bit": smax,
        "Smax_rule": f"{spec['storage_ratio']} * (0.5 * sum(s_n_bit))",
        "R_bh_bit_per_s": BACKHAUL_RATE,
        "kappa_SI": KAPPA,
        "tau_rule": "T_local_upper from local_response_bounds_si",
        "f_min_cycle_per_s": 1.5e9,
        "f_max_cycle_per_s": 5.0e9,
        "F_max_cycle_per_s": float(spec["F_max"]),
        "rho_T_s": float(spec["rho_T"]),
        "rho_E_J": float(spec["rho_E"]),
        "lambda_T": 1.0,
        "lambda_E": 1.0,
        "sample_type_allocation": counts,
        "sample_type_balance_rule": balance,
        "s_n_generation_rule": storage_rule,
        "sampling_domain_binding": {
            "module": "mec_dualbp.datasets.protocol._sample_from_seed",
            "logic": "frozen DualBP sample-type generator",
        },
        "cache_benefit_semantics": "avoided backhaul delay divided by s_n",
        "cache_benefit_per_bit": [1.0 / BACKHAUL_RATE] * n_users,
        "cache_tie_break": "user_id ascending",
        "corrected_dualbp_hash": CORRECTED_DUALBP_HASH,
        "exp7_parent_protocol_hash": PARENT_PROTOCOL_HASH,
        "users": users,
    }
    validate_scenario(payload)
    return payload


def build_s3_payload() -> dict[str, Any]:
    return build_scenario_payload("S3")


def validate_scenario(payload: Mapping[str, Any]) -> dict[str, Any]:
    users = list(payload.get("users", []))
    failures: list[str] = []
    expected_n = int(payload.get("N", 0))
    if expected_n <= 0 or len(users) != expected_n:
        failures.append("N/users")
    ids = [str(user.get("user_id")) for user in users]
    if len(set(ids)) != expected_n:
        failures.append("user_ids")
    required = (
        "D_in_bit_lower", "D_in_bit_upper", "D_out_bit_lower", "D_out_bit_upper",
        "C_cycle_lower", "C_cycle_upper", "I_u_w_lower", "I_u_w_upper",
        "I_d_w_lower", "I_d_w_upper", "F_L_cycle_per_s_lower", "F_L_cycle_per_s_upper",
        "uplink_bandwidth_hz", "downlink_bandwidth_hz", "uplink_terminal_tx_power_w",
        "downlink_bs_tx_power_w", "terminal_rx_power_w", "uplink_channel_gain_linear",
        "downlink_channel_gain_linear", "noise_power_w", "s_n_bit",
        "backhaul_rate_bit_per_s", "kappa_SI", "tau_s",
    )
    for user in users:
        values = [user.get(name) for name in required]
        if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in values):
            failures.append(f"missing/nonfinite:{user.get('user_id')}")
        if any(float(user[name]) <= 0 for name in required):
            failures.append(f"nonpositive:{user.get('user_id')}")
    total = sum(float(user["s_n_bit"]) for user in users) if users else math.nan
    spec = SCENARIOS[str(payload.get("scenario_id"))]
    expected_smax = float(spec["storage_ratio"]) * 0.5 * total
    if not math.isclose(float(payload.get("Smax_bit", math.nan)), expected_smax, rel_tol=0, abs_tol=1e-6):
        failures.append("Smax_rule")
    if any(float(user["backhaul_rate_bit_per_s"]) != BACKHAUL_RATE for user in users):
        failures.append("R_bh")
    if any(float(user["kappa_SI"]) != KAPPA for user in users):
        failures.append("kappa")
    for user in users:
        expected = local_response_bounds_si(
            Interval(float(user["C_cycle_lower"]), float(user["C_cycle_upper"])),
            Interval(float(user["F_L_cycle_per_s_lower"]), float(user["F_L_cycle_per_s_upper"])),
            KAPPA, float(payload["Gamma"]),
        ).local_latency_upper_s
        if not math.isclose(float(user["tau_s"]), expected, rel_tol=1e-14):
            failures.append(f"tau:{user['user_id']}")
    if payload.get("corrected_dualbp_hash") != CORRECTED_DUALBP_HASH:
        failures.append("corrected_dualbp_hash")
    return {"passed": not failures, "failures": failures, "N": len(users)}


def write_s3_scenario(project: Path, config_path: Path, audit_dir: Path) -> dict[str, Any]:
    payload = build_s3_payload()
    audit = validate_scenario(payload)
    if not audit["passed"]:
        raise RuntimeError(f"S3 closure failed: {audit['failures']}")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    # Canonical bytes avoid platform newline translation changing the scenario hash.
    config_path.write_bytes(encoded.encode("utf-8"))
    scenario_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    reloaded = json.loads(config_path.read_text(encoding="utf-8"))
    reload_hash = hashlib.sha256((json.dumps(reloaded, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")).hexdigest()
    hash_payload = {"sha256": scenario_hash, "reload_sha256": reload_hash, "stable": scenario_hash == reload_hash}
    (config_path.with_name(config_path.stem + "_hash.json")).write_text(json.dumps(hash_payload, indent=2) + "\n", encoding="utf-8")
    fields = ["user_id", "sample_type", "generation_seed_value", "s_n_bit", "tau_s"]
    with (audit_dir / "s3_users.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(payload["users"])
    closure = {
        **audit,
        "s3_scenario_status": "frozen",
        "scenario_hash": scenario_hash,
        "reload_hash_stable": scenario_hash == reload_hash,
        "C1_C7_parameters_complete": True,
        "corrected_dualbp_inputs_constructible": True,
        "corrected_dualbp_hash": CORRECTED_DUALBP_HASH,
        "config_path": config_path.relative_to(project).as_posix(),
    }
    (audit_dir / "s3_scenario_closure_audit.json").write_text(json.dumps(closure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (audit_dir / "README.md").write_text("# Exp-7 S3 最终冻结\n\nS3已按作者批准规则确定性生成并通过必要闭合门禁。\n", encoding="utf-8")
    (audit_dir / "summary.json").write_text(json.dumps(closure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return closure


def write_all_scenarios(project: Path, config_dir: Path) -> dict[str, str]:
    config_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for scenario_id in SCENARIOS:
        payload = build_scenario_payload(scenario_id)
        path = config_dir / f"exp7_{scenario_id.lower()}_formal_v1.json"
        encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        path.write_bytes(encoded)
        value = hashlib.sha256(encoded).hexdigest()
        path.with_name(path.stem + "_hash.json").write_text(json.dumps({"sha256": value}, indent=2) + "\n", encoding="utf-8")
        hashes[scenario_id] = value
    return hashes
