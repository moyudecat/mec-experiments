"""Execute the hash-frozen final Exp-8 OE-RVEA protocol.

This module is orchestration and reporting only.  It reuses the registered
OE-RVEA adapter, Improved Corrected DualBP evaluator, canonical repair,
EvaluationArchive, and Exp-4/5 persistence chain without changing them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from mec_dualbp.experiments.exp1_preview_protocol import canonical_json_hash
from mec_dualbp.experiments.exp45_oe_rvea_v2 import (
    ALGORITHM, BUDGET, ENERGY_HASH, EVALUATOR_HASH, OE_DESIGN_HASH,
    OE_IMPLEMENTATION_HASH, POPULATION, TIME_HASH, _cache, _diagnostic,
    _front_records, _json, _mechanism, _mirror, _representative, _run,
    _solution,
)
from mec_dualbp.optimization.dualbp_engine_registry import IMPROVED_MODE, create_corrected_dualbp_engine
from mec_dualbp.optimization.evaluation_archive import EvaluationArchive
from mec_dualbp.utils.atomic_io import safe_atomic_write_json


EXPECTED_OVERALL = "effaad44c47b4ffdf481817f741d48d76e105a8b5e44bf31896149681c624117"
EXPECTED_A = "7f6b99cc2d337dc45152f22635941fe42b685cb991616e56b992efe1cbe5f6ff"
EXPECTED_B = "10f63c8167a94fba69943f28ca25f81ca58723a4daaec178d9816954982448ed"
SEEDS = (82001, 82002, 82003, 82004, 82005)
SCENARIO_FILES = {
    "G1": "exp8_g1_generalization_v1.json",
    "G2": "exp8_g2_generalization_v1.json",
    "G3": "exp8_g3_generalization_v1.json",
    "N10": "exp8_scale_n10_v1.json",
    "N20": "exp8_g2_generalization_v1.json",
    "N30": "exp8_scale_n30_v1.json",
    "N40": "exp8_scale_n40_v1.json",
}
EXPECTED_SCENARIO_HASHES = {
    "G1": "539bb5175fb70ab9a901789394ecf61f48322f8ed0a5fc6eb8501f3056046ca9",
    "G2": "110a5c7306183bdbf6f7abdb0fd790a4a42b53bc44025c10951c2493927bc925",
    "G3": "acb3e78bf98521c3d2ee1d13b856e93e1cbc011d903abb22e54f62f404104820",
    "N10": "aa44a74dddafc594c889f637b8d5aa1b2436e9b7bf1e775ed76f39580148f1fe",
    "N20": "110a5c7306183bdbf6f7abdb0fd790a4a42b53bc44025c10951c2493927bc925",
    "N30": "403be395cccb8d733558731381d18103e47c60f1c9f988f08b415ad6726480e4",
    "N40": "6d918ac1b2a428a7a4bc54d712b91f7f2e3352966ee5b686959a9d777108f4cc",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields or (list(rows[0]) if rows else []))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _protocol_hash(path: Path, field: str) -> tuple[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = str(payload.pop(field))
    return recorded, canonical_json_hash(payload)


def preflight(project: Path, protocol_dir: Path) -> dict[str, Any]:
    a_recorded, a_actual = _protocol_hash(protocol_dir / "exp8a_generalization_protocol.json", "exp8a_protocol_hash")
    b_recorded, b_actual = _protocol_hash(protocol_dir / "exp8b_scalability_protocol.json", "exp8b_protocol_hash")
    binding = json.loads((protocol_dir / "protocol_hashes.json").read_text(encoding="utf-8"))
    overall_recorded = str(binding.pop("exp8_overall_protocol_hash"))
    overall_actual = canonical_json_hash(binding)
    scenario_hashes = {key: _sha(project / "configs" / "scenarios" / value) for key, value in SCENARIO_FILES.items()}
    scenarios = {key: json.loads((project / "configs" / "scenarios" / value).read_text(encoding="utf-8"))
                 for key, value in SCENARIO_FILES.items()}
    resource_failures: list[str] = []
    for key in ("N10", "N20", "N30", "N40"):
        row = scenarios[key]; n = int(row["N"])
        checks = (
            math.isclose(float(row["F_max_cycle_per_s"]), n / 20 * 2.0e10, abs_tol=1e-6),
            math.isclose(float(row["Smax_bit"]), n / 20 * 4783138003.689591, abs_tol=1e-6),
            math.isclose(float(row["rho_T_s"]), n * .8809164041, abs_tol=1e-12),
            math.isclose(float(row["rho_E_J"]), n * .04650848491, abs_tol=1e-12),
            float(row["Gamma"]) == 1.0, float(row["eta"]) == .90,
            float(row["f_min_cycle_per_s"]) == 1.5e9, float(row["f_max_cycle_per_s"]) == 5.0e9,
        )
        if not all(checks): resource_failures.append(key)
    checks = {
        "exp8a_protocol_hash": a_recorded == a_actual == EXPECTED_A,
        "exp8b_protocol_hash": b_recorded == b_actual == EXPECTED_B,
        "overall_protocol_hash": overall_recorded == overall_actual == EXPECTED_OVERALL,
        "oe_design_hash": binding["oe_rvea_design_hash"] == OE_DESIGN_HASH,
        "oe_implementation_hash": binding["oe_rvea_implementation_hash"] == OE_IMPLEMENTATION_HASH,
        "evaluator_hash": binding["improved_corrected_dualbp_hash"] == EVALUATOR_HASH,
        "time_model_hash": binding["improved_time_model_hash"] == TIME_HASH,
        "energy_model_hash": binding["improved_energy_model_hash"] == ENERGY_HASH,
        "scenario_hashes": scenario_hashes == EXPECTED_SCENARIO_HASHES,
        "G2_N20_exact": scenario_hashes["G2"] == scenario_hashes["N20"],
        "outer_seeds": list(SEEDS) == [82001, 82002, 82003, 82004, 82005],
        "population": POPULATION == 100,
        "budget": BUDGET == 20_000,
        "workers": 1,
        "resource_scaling": not resource_failures,
    }
    result = {"passed": all(checks.values()), "checks": checks, "resource_failures": resource_failures,
              "scenario_hashes": scenario_hashes, "performance_preview": 0}
    if not result["passed"]:
        raise RuntimeError(f"Exp-8 preflight failed: {result}")
    return result


def _mechanism_metrics(run: Path) -> dict[str, Any]:
    adapter = pickle.loads((run / "checkpoint.pkl").read_bytes())["adapter"]
    rows = list(adapter.objective_equivalence_audit)
    ratios = [float(row["compression_ratio"]) for row in rows]
    return {
        "mean_compression_ratio": float(np.mean(ratios)) if ratios else 0.0,
        "compression_triggered": any(int(row["compressed_candidate_count"]) > 0 for row in rows),
        "compression_fallback_triggered": any(bool(row["compression_fallback_triggered"]) for row in rows),
        "oe_audit_generations": len(rows),
    }


def _postprocess_cell(project: Path, root: Path, scenario: dict[str, Any], statuses: Sequence[Mapping[str, Any]],
                      section: str, cell: str, scenario_hash: str, reused: bool = False) -> list[dict[str, Any]]:
    fronts = [_front_records(root / "runs" / f"{ALGORITHM}__seed_{seed}") for seed in SEEDS]
    if any(not front for front in fronts):
        raise RuntimeError(f"{cell} contains empty feasible front")
    pooled = np.asarray([[r["Phi_T"], r["Phi_E"]] for front in fronts for r in front], dtype=float)
    lo, hi = pooled.min(axis=0), pooled.max(axis=0)
    engine = create_corrected_dualbp_engine(project, scenario, evaluator_mode=IMPROVED_MODE, evaluator_hash=EVALUATOR_HASH)
    rows: list[dict[str, Any]] = []
    for seed, status, front in zip(SEEDS, statuses, fronts):
        run = root / "runs" / f"{ALGORITHM}__seed_{seed}"
        rep = _representative(front, lo, hi)
        solution = _solution(rep, len(scenario["users"]))
        diag = _diagnostic(engine, solution)
        archive = EvaluationArchive.from_cache(_cache(run))
        pairs = {(r["Phi_T"], r["Phi_E"]) for r in archive.records}
        sizes = np.asarray([float(user["s_n_bit"]) for user in scenario["users"]])
        mech = _mechanism_metrics(run)
        metadata = json.loads((run / "evaluation_archive_metadata.json").read_text(encoding="utf-8"))
        rows.append({
            "section": section, "cell": cell, "scenario_id": scenario["scenario_id"], "N": int(scenario["N"]),
            "seed": seed, "scenario_hash": scenario_hash, "artifact_reused": reused,
            "Phi_T": float(rep["Phi_T"]), "Phi_E": float(rep["Phi_E"]),
            "final_feasible": bool(diag["representative_feasible"]),
            "c7_violating_user_count": int(diag["c7_violating_user_count"]),
            "c7_min_margin": float(diag["c7_min_margin"]),
            "offloading_ratio": float(solution.offload.mean()),
            "cache_utilization": float(np.dot(solution.cache, sizes) / float(scenario["Smax_bit"])),
            "cpu_utilization": float(solution.edge_cpu.sum() / float(scenario["F_max_cycle_per_s"])),
            "objective_unique_ratio": len(pairs) / max(len(archive.records), 1),
            **mech,
            "runtime_seconds": float(status["wall_seconds"]),
            "runtime_per_request_seconds": float(status["wall_seconds"]) / BUDGET,
            "unique_evaluated_decisions": int(status["unique_evaluations"]),
            "total_request_count": int(status["requests"]),
            "cache_hit_rate": int(status["cache_hits"]) / BUDGET,
            "failed": status["status"] != "completed",
            "budget_violation": int(status["requests"]) != BUDGET,
            "nonfinite_count": int(status["nonfinite_count"]),
            "oracle_pass": bool(metadata["deterministic_consistency_passed"]),
            "checkpoint_exists": (run / "checkpoint.pkl").exists(),
            "final_front_exists": (run / "final_front.csv").exists(),
            "evaluator_hash": status.get("evaluator_hash"),
            "oe_design_hash": status.get("oe_rvea_design_hash"),
            "oe_implementation_hash": status.get("oe_rvea_implementation_hash"),
            "effective_protocol_hash": status.get("effective_protocol_hash"),
        })
    return rows


def _run_cell(project: Path, root: Path, mirror: Path, scenario: dict[str, Any], protocol_hash: str,
              identity_prefix: str, section: str, cell: str, scenario_hash: str) -> list[dict[str, Any]]:
    engine = create_corrected_dualbp_engine(project, scenario, evaluator_mode=IMPROVED_MODE, evaluator_hash=EVALUATOR_HASH)
    statuses = []
    for seed in SEEDS:
        started = time.monotonic()
        status = _run(project, root, mirror, scenario, seed, engine, protocol_hash, f"{identity_prefix}-{seed}")
        statuses.append(status)
        print(json.dumps({"completed": f"{section}/{cell}/{seed}", "requests": status["requests"],
                          "wall_seconds": status["wall_seconds"], "step_seconds": time.monotonic()-started}), flush=True)
    return _postprocess_cell(project, root, scenario, statuses, section, cell, scenario_hash)


def _describe(rows: Sequence[Mapping[str, Any]], group: str, metrics: Sequence[str]) -> list[dict[str, Any]]:
    result = []
    for value in sorted({row[group] for row in rows}, key=lambda x: (isinstance(x, str), x)):
        subset = [row for row in rows if row[group] == value]
        out: dict[str, Any] = {group: value, "runs": len(subset)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in subset], dtype=float)
            out.update({f"{metric}_mean": float(values.mean()),
                        f"{metric}_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                        f"{metric}_median": float(np.median(values)),
                        f"{metric}_IQR": float(np.quantile(values, .75)-np.quantile(values, .25))})
        out["feasibility_rate"] = float(np.mean([bool(row["final_feasible"]) for row in subset]))
        out["compression_triggered_run_count"] = sum(bool(row["compression_triggered"]) for row in subset)
        out["fallback_run_count"] = sum(bool(row["compression_fallback_triggered"]) for row in subset)
        result.append(out)
    return result


def _all_files_hashes(root: Path) -> list[dict[str, Any]]:
    return [{"path": path.relative_to(root).as_posix(), "sha256": _sha(path), "bytes": path.stat().st_size}
            for path in sorted(root.rglob("*")) if path.is_file()]


def execute(project: Path, protocol_dir: Path, formal: Path, audit: Path, recovery: Path) -> dict[str, Any]:
    started = time.monotonic()
    pre = preflight(project, protocol_dir)
    formal.mkdir(parents=True, exist_ok=False); audit.mkdir(parents=True, exist_ok=False); recovery.mkdir(parents=True, exist_ok=False)
    for name in ("exp8a", "exp8b", "shared_reuse", "per_run", "summary"):
        (formal / name).mkdir()
    _json(formal / "summary" / "preflight.json", pre)

    configs = project / "configs" / "scenarios"
    a_rows: list[dict[str, Any]] = []
    scenarios: dict[str, dict[str, Any]] = {}
    for gid in ("G1", "G2", "G3"):
        scenario = json.loads((configs / SCENARIO_FILES[gid]).read_text(encoding="utf-8")); scenarios[gid] = scenario
        a_rows.extend(_run_cell(project, formal / "exp8a" / gid, recovery / "exp8a" / gid, scenario,
                                EXPECTED_A, f"Exp8A-{gid}", "Exp8A", gid, EXPECTED_SCENARIO_HASHES[gid]))
    if not (len(a_rows) == 15 and all(r["final_feasible"] and r["oracle_pass"] and not r["failed"]
                                      and not r["budget_violation"] and r["nonfinite_count"] == 0 for r in a_rows)):
        raise RuntimeError("Exp-8A correctness gate failed")

    b_rows: list[dict[str, Any]] = []
    for cell in ("N10", "N30", "N40"):
        scenario = json.loads((configs / SCENARIO_FILES[cell]).read_text(encoding="utf-8")); scenarios[cell] = scenario
        b_rows.extend(_run_cell(project, formal / "exp8b" / cell, recovery / "exp8b" / cell, scenario,
                                EXPECTED_B, f"Exp8B-{cell}", "Exp8B", cell, EXPECTED_SCENARIO_HASHES[cell]))
    reused = [dict(row, section="Exp8B", cell="N20", artifact_reused=True) for row in a_rows if row["cell"] == "G2"]
    b_rows.extend(reused)

    reuse_audit = []
    for row in reused:
        checks = {
            "scenario_hash": row["scenario_hash"] == EXPECTED_SCENARIO_HASHES["N20"],
            "seed": int(row["seed"]) in SEEDS,
            "algorithm_hashes": row["oe_design_hash"] == OE_DESIGN_HASH and row["oe_implementation_hash"] == OE_IMPLEMENTATION_HASH,
            "evaluator_hash": row["evaluator_hash"] == EVALUATOR_HASH,
            "population": POPULATION == 100, "budget": row["total_request_count"] == BUDGET,
        }
        reuse_audit.append({"source": f"exp8a/G2/OE-RVEA__seed_{row['seed']}", "target": f"exp8b/N20/OE-RVEA__seed_{row['seed']}",
                            "seed": row["seed"], **checks, "reuse_legal": all(checks.values())})
    _json(formal / "shared_reuse" / "artifact_reuse_manifest.json", {"rows": reuse_audit, "legal_count": sum(r["reuse_legal"] for r in reuse_audit)})

    metrics = ("Phi_T", "Phi_E", "c7_violating_user_count", "c7_min_margin", "offloading_ratio",
               "cache_utilization", "cpu_utilization", "objective_unique_ratio", "mean_compression_ratio",
               "runtime_seconds", "runtime_per_request_seconds", "unique_evaluated_decisions", "cache_hit_rate")
    a_summary = _describe(a_rows, "cell", metrics)
    b_summary = _describe(b_rows, "N", metrics)
    by_n = {int(row["N"]): row for row in b_summary}
    for n, row in by_n.items():
        scenario = scenarios["G2"] if n == 20 else scenarios[f"N{n}"]
        row.update({"decision_dimension": 3*n, "Fmax_cycle_per_s": scenario["F_max_cycle_per_s"],
                    "Smax_bit": scenario["Smax_bit"], "rho_T_s": scenario["rho_T_s"], "rho_E_J": scenario["rho_E_J"]})

    trend_rows = []
    ns = np.asarray([10., 20., 30., 40.])
    for metric in ("runtime_seconds", "runtime_per_request_seconds", "objective_unique_ratio",
                   "mean_compression_ratio", "Phi_T", "Phi_E"):
        values = np.asarray([by_n[int(n)][f"{metric}_mean"] for n in ns])
        rho, p = spearmanr(ns, values)
        trend_rows.append({"x": "N", "y": metric, "spearman_rho": float(rho), "p_value_descriptive_only": float(p),
                           "n_scales": 4, "interpretation_scope": "descriptive_only"})
    n10_runtime = by_n[10]["runtime_seconds_mean"]
    runtime_scale = [{"N": n, "mean_runtime_seconds": by_n[n]["runtime_seconds_mean"],
                      "runtime_ratio_vs_N10": by_n[n]["runtime_seconds_mean"] / n10_runtime,
                      "mean_runtime_per_request_seconds": by_n[n]["runtime_per_request_seconds_mean"]}
                     for n in (10, 20, 30, 40)]

    independent_rows = a_rows + [row for row in b_rows if not row["artifact_reused"]]
    conceptual_rows = a_rows + b_rows
    completion = []
    for row in conceptual_rows:
        hash_ok = (row["oe_design_hash"] == OE_DESIGN_HASH and row["oe_implementation_hash"] == OE_IMPLEMENTATION_HASH
                   and row["evaluator_hash"] == EVALUATOR_HASH and row["scenario_hash"] == EXPECTED_SCENARIO_HASHES[row["cell"]])
        completion.append({"section": row["section"], "cell": row["cell"], "seed": row["seed"],
                           "artifact_reused": row["artifact_reused"], "completed": not row["failed"],
                           "budget_ok": not row["budget_violation"], "hash_ok": hash_ok,
                           "oracle_pass": row["oracle_pass"], "feasible_representative": row["final_feasible"],
                           "checkpoint_exists": row["checkpoint_exists"], "front_exists": row["final_front_exists"]})

    hash_mismatch = sum(not row["hash_ok"] for row in completion)
    failed = sum(bool(row["failed"]) for row in independent_rows)
    budget_bad = sum(bool(row["budget_violation"]) for row in independent_rows)
    nonfinite = sum(int(row["nonfinite_count"]) for row in independent_rows)
    oracle_pass = sum(bool(row["oracle_pass"]) for row in conceptual_rows)
    feasible_cells = all(any(row["final_feasible"] for row in conceptual_rows if row["cell"] == cell)
                         for cell in ("G1", "G2", "G3", "N10", "N20", "N30", "N40"))
    gate_pass = (len(independent_rows) == 30 and len(conceptual_rows) == 35 and len(reuse_audit) == 5
                 and all(row["reuse_legal"] for row in reuse_audit) and failed == budget_bad == hash_mismatch == nonfinite == 0
                 and oracle_pass == 35 and feasible_cells)

    _csv(formal / "per_run" / "exp8_independent_runs.csv", independent_rows)
    _csv(formal / "per_run" / "exp8_conceptual_cells.csv", conceptual_rows)
    _csv(formal / "summary" / "exp8a_generalization_summary.csv", a_summary)
    _csv(formal / "summary" / "exp8b_scalability_summary.csv", list(by_n.values()))
    _csv(formal / "summary" / "trend_diagnostic.csv", trend_rows)
    _csv(formal / "summary" / "runtime_scaling_summary.csv", runtime_scale)
    _json(formal / "summary" / "status.json", {"EXP8_CORRECTNESS_GATE": "PASS" if gate_pass else "FAIL",
          "EXP8_FINAL_FROZEN": gate_pass, "independent_completed": len(independent_rows)-failed,
          "conceptual_resolved": len(conceptual_rows), "reuse_legal": sum(r["reuse_legal"] for r in reuse_audit),
          "failed": failed, "budget_violations": budget_bad, "hash_mismatch": hash_mismatch,
          "nonfinite": nonfinite, "oracle_pass": oracle_pass, "wall_seconds": time.monotonic()-started})

    oracle_rows = [{"section": r["section"], "cell": r["cell"], "seed": r["seed"], "artifact_reused": r["artifact_reused"],
                    "oracle_pass": r["oracle_pass"]} for r in conceptual_rows]
    feasibility_rows = [{"section": r["section"], "cell": r["cell"], "seed": r["seed"],
                         "final_feasible": r["final_feasible"], "c7_violating_user_count": r["c7_violating_user_count"],
                         "c7_min_margin": r["c7_min_margin"]} for r in conceptual_rows]
    compression_rows = _describe(conceptual_rows, "cell", ("objective_unique_ratio", "mean_compression_ratio"))
    _csv(audit / "formal_completion_audit.csv", completion)
    _csv(audit / "artifact_reuse_audit.csv", reuse_audit)
    _csv(audit / "evaluation_archive_oracle.csv", oracle_rows)
    _csv(audit / "exp8a_generalization_summary.csv", a_summary)
    _csv(audit / "exp8a_seed_level_results.csv", a_rows)
    _csv(audit / "exp8b_scalability_summary.csv", list(by_n.values()))
    _csv(audit / "exp8b_seed_level_results.csv", b_rows)
    _csv(audit / "runtime_scaling_summary.csv", runtime_scale)
    _csv(audit / "compression_summary.csv", compression_rows)
    _csv(audit / "feasibility_audit.csv", feasibility_rows)
    _csv(audit / "trend_diagnostic.csv", trend_rows)
    _json(audit / "protocol_binding.json", {"overall": EXPECTED_OVERALL, "exp8a": EXPECTED_A, "exp8b": EXPECTED_B,
          "oe_design": OE_DESIGN_HASH, "oe_implementation": OE_IMPLEMENTATION_HASH,
          "evaluator": EVALUATOR_HASH, "time_model": TIME_HASH, "energy_model": ENERGY_HASH})
    _json(audit / "scenario_binding.json", EXPECTED_SCENARIO_HASHES)
    _json(audit / "scale_binding.json", {str(n): {k: by_n[n][k] for k in ("decision_dimension", "Fmax_cycle_per_s", "Smax_bit", "rho_T_s", "rho_E_J")} for n in by_n})
    gate = {"EXP8_CORRECTNESS_GATE": "PASS" if gate_pass else "FAIL", "EXP8_FINAL_FROZEN": gate_pass,
            "independent_completed": len(independent_rows)-failed, "independent_planned": 30,
            "conceptual_resolved": len(conceptual_rows), "conceptual_planned": 35,
            "reuse_legal": sum(r["reuse_legal"] for r in reuse_audit), "reuse_planned": 5,
            "failed": failed, "budget_violations": budget_bad, "hash_mismatch": hash_mismatch,
            "NaN_Inf": nonfinite, "oracle_pass": oracle_pass, "oracle_planned": 35,
            "all_cells_have_feasible_representative": feasible_cells,
            "scientific_parameters_modified": False, "algorithm_tuning": False,
            "wall_seconds": time.monotonic()-started}
    _json(audit / "final_gate.json", gate)
    readme = ("# Exp-8 OE-RVEA 正式结果冻结\n\n本目录冻结 Exp-8A 独立场景泛化与 Exp-8B 固定请求预算规模扩展的正式结果。"
              f"总门禁：`{gate['EXP8_CORRECTNESS_GATE']}`；独立运行 {gate['independent_completed']}/30，概念单元 {gate['conceptual_resolved']}/35，合法复用 {gate['reuse_legal']}/5。\n")
    (audit / "README.md").write_text(readme, encoding="utf-8")
    manifest = {"formal_files": _all_files_hashes(formal), "recovery_files": _all_files_hashes(recovery)}
    _json(audit / "final_result_manifest.json", manifest)
    created = sorted([p.relative_to(project).as_posix() for base in (formal, recovery, audit)
                      for p in base.rglob("*") if p.is_file()])
    _json(audit / "created_files.json", {"count": len(created)+1, "files": created + [(audit/"created_files.json").relative_to(project).as_posix()]})
    if not gate_pass:
        raise RuntimeError(f"Exp-8 final correctness gate failed: {gate}")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--protocol-dir", required=True)
    parser.add_argument("--formal-dir", required=True)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--recovery-dir", required=True)
    args = parser.parse_args()
    result = execute(Path(args.project_root).resolve(), Path(args.protocol_dir).resolve(), Path(args.formal_dir).resolve(),
                     Path(args.audit_dir).resolve(), Path(args.recovery_dir).resolve())
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
