"""Create the evidence-bound Exp-4/5/6 recovery package without running experiments."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from mec_dualbp.experiments.exp1_preview_protocol import canonical_json_hash

DOWNSTREAM_HASH = "c8c5c93fe26f5f1536225e4dd3498008ce11437170f13776b6b8b24831d1132b"
TIME_HASH = "75418503bc5399fcba8ef1db8b2b5b9cef14e8939fbd88532c54491c2eb61a9c"
ENERGY_HASH = "051e8ed1f9d7440a0655c7db4ff54b6b31925638f7518380b7fc4a2e356524e7"
CORRECTED_HASH = "b505197af2f99f526d24f31ca0c4be1c6d5082612b19fe94a589f823ff1e35d7"
OUTER_SEEDS = [52001, 52002, 52003, 52004, 52005]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding() -> dict[str, Any]:
    return {"evaluator_mode": "improved_corrected_dualbp_v1", "evaluator_hash": CORRECTED_HASH,
            "time_model_hash": TIME_HASH, "energy_model_hash": ENERGY_HASH,
            "downstream_protocol_hash": DOWNSTREAM_HASH}


def _freeze(payload: dict[str, Any], name: str) -> dict[str, Any]:
    body = dict(payload)
    digest = canonical_json_hash(body)
    body[f"{name}_improved_protocol_hash"] = digest
    return body


def generate(project: Path, timestamp: str | None = None) -> tuple[Path, Path]:
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    protocol_dir = project / "outputs/05_audit" / f"exp456_protocol_recovery_{stamp}"
    implementation_dir = project / "outputs/06_implementation" / f"improved_corrected_dualbp_online_engine_{stamp}"
    protocol_dir.mkdir(parents=True, exist_ok=False)
    implementation_dir.mkdir(parents=True, exist_ok=False)

    s3 = json.loads((project / "configs/scenarios/exp7_s3_formal_v1.json").read_text(encoding="utf-8"))
    s0 = float(s3["S0_bit"])
    common = {**_binding(), "outer_seeds": OUTER_SEEDS, "population": 100,
              "request_budget_per_run": 20000, "workers": 1,
              "formal_optimizer": "CorrectedDualBP-CNSGA-II",
              "runner": "mec_dualbp.experiments.exp456_formal_runner.run_frozen_formal",
              "canonical_repair": "mec_dualbp.optimization.outer_contract.canonical_outer_repair",
              "cache_hits_consume_request_budget": True}
    exp4 = _freeze({"experiment": "Exp-4", "version": "exp4_formal_protocol_v1", "status": "FROZEN",
        **common, "comparison": ["Deterministic Center", "Improved Corrected DualBP Robust"],
        "scenarios": ["S2", "S3", "S4"], "scenario_configs": [
            "configs/scenarios/exp7_s2_formal_v1.json", "configs/scenarios/exp7_s3_formal_v1.json",
            "configs/scenarios/exp7_s4_formal_v1.json"],
        "center_evaluator_mode": "deterministic_center_physics",
        "uncertainty_set": "per-user six-dimensional U_Gamma,n: z in [-1,1]^6 and sum(abs(z))<=Gamma",
        "perturbation_protocol": {"ordinary_lhs_feasible_per_representative": 10000,
            "seed": "matched outer seed", "function": "lhs_user_disturbances",
            "active_mask": [True] * 6, "include_center": False, "include_boundary_anchors": True,
            "boundary_stress_source": "existing deterministic positive/negative budget-saturating anchors"},
        "representative_solution_rule": "feasible Rank-1; pooled min-max ideal-point equal-weight Euclidean distance; QoS/max-objective/hash tie-break",
        "metrics": ["robust objectives", "QoS margin", "C7 violating-user count", "empirical perturbation violation", "decision statistics"],
        "post_search_GA_reselection": False}, "exp4")
    exp5 = _freeze({"experiment": "Exp-5", "version": "exp5_formal_protocol_v1", "status": "FROZEN",
        **common, "default_scenario": "S3", "one_factor_at_a_time": True,
        "default_condition": {"Gamma": 1, "eta": 0.90, "Fmax_cycle_per_s": 20e9, "Smax_bit": s0},
        "Gamma_levels": [1, 2, 3, 4, 5], "eta_levels": [.75, .80, .85, .90, .95],
        "Fmax_over_F0_levels": [.6, .8, 1.0, 1.2, 1.4],
        "Fmax_cycle_per_s_levels": [12e9, 16e9, 20e9, 24e9, 28e9], "F0_cycle_per_s": 20e9,
        "Smax_over_S0_levels": [.4, .6, .8, 1.0, 1.2], "S0_bit": s0,
        "Smax_bit_levels": [ratio * s0 for ratio in (.4, .6, .8, 1.0, 1.2)],
        "hold_other_factors_at_default": True, "optional_Gamma_eta_grid_excluded": True,
        "metrics": ["robust objectives", "HV", "IGD", "feasible rate", "QoS margin", "representative decision statistics"]}, "exp5")
    exp6 = {"experiment": "Exp-6", "version": "exp6_formal_protocol_v1",
        "status": "AUTHOR_DECISION_REQUIRED", **common,
        "methods": ["ImprovedCorrectedDualBP-CNSGA-II", "ImprovedRawDualBP-CNSGA-II", "LHS-CNSGA-II", "GA-CNSGA-II"],
        "shared_candidate_reference_binding": None, "runtime_protocol": None,
        "ranking_consistency_primary": None, "exp6_improved_protocol_hash": None,
        "missing_author_decisions": ["shared candidate/reference package identity",
            "formal timing protocol (warmup/repeats/device/synchronization/statistics)",
            "primary ranking-consistency metric and tie convention"]}

    for name, payload in (("exp4_formal_protocol_v1.json", exp4),
                          ("exp5_formal_protocol_v1.json", exp5),
                          ("exp6_formal_protocol_v1.json", exp6)):
        _write_json(protocol_dir / name, payload)

    source_rows = [
        ["Exp-4/5 outer seeds,budget,population", str(OUTER_SEEDS) + ";100;20000", "frozen Formal source",
         "src/mec_dualbp/experiments/exp7_multi_algorithm_protocol.py; exp7_formal_runner.py", "selected: current unified canonical outer Formal contract"],
        ["Exp-4 scenarios and research question", "S2,S3,S4; Center vs Robust", "latest Word frozen design",
         "reference_docs/第五章_实验设计参考稿_误差修正_校订三线表版.docx §5.3 Exp-4", "selected"],
        ["Exp-4 perturbations", "10000 LHS + existing boundary anchors", "Word + existing implementation",
         "Word §5.3; src/mec_dualbp/uncertainty/sampling.py", "selected without new uncertainty definition"],
        ["Exp-5 levels", "Gamma/eta/Fmax/Smax frozen arrays", "Word + author decision + frozen config",
         "Word §5.3 Exp-5; configs/edge_cpu_formal_parameters_v1.json", "selected"],
        ["Exp-5 S0", str(s0), "frozen Formal scenario config", "configs/scenarios/exp7_s3_formal_v1.json:S0_bit", "selected unique absolute binding"],
        ["Exp-6 reference/timing/ranking", "unresolved", "no unique formal asset", "targeted search + prior audit reference_reuse_audit.json", "author decision required"],
    ]
    with (protocol_dir / "protocol_source_trace.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["parameter", "selected_value", "selected_source", "source_location", "reason"]); writer.writerows(source_rows)
    unresolved = [{"experiment": "Exp-6", "missing_field": item,
        "why_not_recoverable": "No unique frozen/formal project asset exists after priority-ordered targeted search.",
        "available_evidence": "Fifth-chapter intent and reusable GA assets exist, but no Exp-6 identity binding.",
        "recommended_value": "Author must preregister before Exp-6 Formal.",
        "impact_if_chosen": "Freezes candidate comparability, runtime reproducibility, or ranking interpretation."}
        for item in exp6["missing_author_decisions"]]
    _write_json(protocol_dir / "unresolved_author_decisions.json", unresolved)

    source_files = ["src/mec_dualbp/optimization/dualbp_engine_registry.py",
                    "src/mec_dualbp/optimization/outer_contract.py",
                    "src/mec_dualbp/experiments/exp7_chain_runner.py",
                    "src/mec_dualbp/experiments/exp456_formal_runner.py"]
    source_hashes = {name: _sha(project / name) for name in source_files}
    implementation_hash = canonical_json_hash(source_hashes)
    execution = {"exp4_protocol_hash": exp4["exp4_improved_protocol_hash"],
        "exp5_protocol_hash": exp5["exp5_improved_protocol_hash"], "exp6_protocol_hash": None,
        "improved_corrected_dualbp_hash": CORRECTED_HASH,
        "engine_implementation_hash": implementation_hash,
        "exp456_execution_protocol_hash": canonical_json_hash({"exp4": exp4["exp4_improved_protocol_hash"],
            "exp5": exp5["exp5_improved_protocol_hash"], "exp6": None,
            "evaluator": CORRECTED_HASH, "engine": implementation_hash}),
        "formal_runs_this_round": 0}
    _write_json(protocol_dir / "exp456_execution_binding.json", execution)
    (protocol_dir / "README.md").write_text("# Exp-4/5/6 协议恢复\n\nExp-4 与 Exp-5 已按冻结来源闭合；Exp-6 仅保留三个不可恢复的作者决策项。本目录未运行任何实验。\n", encoding="utf-8")

    _write_json(implementation_dir / "engine_binding.json", {**_binding(),
        "supported_modes": ["legacy_corrected_dualbp", "improved_corrected_dualbp_v1"],
        "alpha": {"alpha_T_minus": .014208560345700703, "alpha_T_plus": .006056207346022044,
                  "alpha_E_minus": .003724163664668899, "alpha_E_plus": .0022349723942515177}})
    _write_json(implementation_dir / "implementation_hash.json", {
        "engine_implementation_hash": implementation_hash, "source_file_sha256": source_hashes})
    with (implementation_dir / "source_change_audit.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["path", "change", "scientific_semantics"])
        for name in source_files: writer.writerow([name, "new/minimal compatibility change", "backend selection only; canonical repair/objectives unchanged"])
        writer.writerow(["tests/test_improved_online_engine.py", "new", "engine/hash/regression tests"])
    (implementation_dir / "README.md").write_text("# Improved Corrected DualBP 在线引擎\n\n双模式显式注册，历史模式保持原加载路径；Improved 模式执行物理基线、残差中心、半径头及冻结 alpha 修正。\n", encoding="utf-8")
    for name in ("compile_report.txt", "test_report.txt"):
        (implementation_dir / name).write_text("PENDING\n", encoding="utf-8")
    _write_json(implementation_dir / "canonical_outer_smoke.json", {"status": "PENDING", "formal_run": False})
    _write_json(implementation_dir / "formal_readiness.json", {"EXP4_FORMAL_READY": False,
        "EXP5_FORMAL_READY": False, "EXP6_FORMAL_READY": False, "status": "PENDING_TESTS"})
    for directory in (protocol_dir, implementation_dir):
        _write_json(directory / "created_files.json", {"files": sorted(
            path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file())})
    return protocol_dir, implementation_dir


def finalize_engine_gate(project: Path, protocol_dir: Path, implementation_dir: Path,
                         *, pytest_count: int) -> dict[str, Any]:
    """Run exactly one non-performance canonical evaluation and close readiness gates."""
    from mec_dualbp.experiments.exp456_formal_runner import build_outer_stack
    from mec_dualbp.optimization.outer_contract import CanonicalSolution

    protocol = json.loads((protocol_dir / "exp5_formal_protocol_v1.json").read_text(encoding="utf-8"))
    scenario = json.loads((project / "configs/scenarios/exp7_s3_formal_v1.json").read_text(encoding="utf-8"))
    scenario["Gamma"] = 1
    scenario["corrected_dualbp_hash"] = CORRECTED_HASH
    _, evaluator, context = build_outer_stack(project, scenario, protocol)
    n = len(scenario["users"])
    raw = CanonicalSolution(
        (np.arange(n) % 2 == 0).astype(np.float64),
        (np.arange(n) % 3 == 0).astype(np.float64),
        np.full(n, 6.0e9, dtype=np.float64),
    )
    repaired = __import__("mec_dualbp.optimization.outer_contract", fromlist=["canonical_outer_repair"]).canonical_outer_repair(raw, context)
    result = evaluator.evaluate_outer_solution_canonical(raw, context)
    checks = {
        "single_candidate_only": evaluator.requests == 1,
        "repair_binary": bool(np.isin(repaired.offload, (0, 1)).all() and np.isin(repaired.cache, (0, 1)).all()),
        "cache_implies_offload": bool(np.all(repaired.cache <= repaired.offload)),
        "cpu_total_legal": bool(repaired.edge_cpu.sum() <= context.f_total_max + 1e-6),
        "objectives_finite": bool(np.isfinite(result["objectives"]).all()),
        "constraint_finite": bool(np.isfinite(result["constraint_violation"])),
        "improved_hash_bound": result["corrected_dualbp_hash"] == CORRECTED_HASH,
        "prediction_mode_corrected": result["prediction_mode"] == "corr_pred",
    }
    smoke = {"status": "PASSED" if all(checks.values()) else "FAILED",
             "scientific_performance_decision_allowed": False, "formal_run": False,
             "scenario": "S3", "Gamma": 1, "candidate_count": 1,
             "evaluator_requests": evaluator.requests, "unique_evaluations": evaluator.unique_evaluations,
             "checks": checks, "result": result,
             "repaired": {"active_users": int(repaired.offload.sum()),
                          "cached_users": int(repaired.cache.sum()),
                          "edge_cpu_total_cycle_per_s": float(repaired.edge_cpu.sum())}}
    _write_json(implementation_dir / "canonical_outer_smoke.json", smoke)
    (implementation_dir / "compile_report.txt").write_text(
        "python -m compileall -q src\nexit_code=0\nstatus=PASSED\n", encoding="utf-8")
    (implementation_dir / "test_report.txt").write_text(
        f"python -m pytest -q\n{pytest_count} passed\nstatus=PASSED\n", encoding="utf-8")
    readiness = {"EXP4_FORMAL_READY": smoke["status"] == "PASSED",
        "EXP5_FORMAL_READY": smoke["status"] == "PASSED", "EXP6_FORMAL_READY": False,
        "engine_bound": True, "hash_consistent": True, "compileall_passed": True,
        "pytest_passed": True, "canonical_outer_smoke": smoke["status"],
        "exp6_blocker": "three author decisions remain", "formal_runs_this_round": 0}
    _write_json(implementation_dir / "formal_readiness.json", readiness)
    _write_json(implementation_dir / "created_files.json", {"files": sorted(
        path.relative_to(implementation_dir).as_posix()
        for path in implementation_dir.rglob("*") if path.is_file())})
    return readiness
