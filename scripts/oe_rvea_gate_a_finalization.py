"""Offline finalization for the preregistered OE-RVEA Gate-A."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata, wilcoxon

from mec_dualbp.experiments.oe_rvea_gate_a_runner import (
    ALGORITHMS, BUDGET, DESIGN_HASH, IMPLEMENTATION_HASH, POPULATION,
    SCENARIOS, SEEDS, protocol_hash,
)
from mec_dualbp.optimization.evaluation_archive import EvaluationArchive
from mec_dualbp.optimization.metrics import global_normalize, hypervolume_2d_min
from mec_dualbp.optimization.outer_contract import IMPROVED_CORRECTED_DUALBP_HASH


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _objective_front(rows: list[dict[str, Any]]) -> np.ndarray:
    unique = {(float(row["Phi_T"]), float(row["Phi_E"])) for row in rows}
    ordered = sorted(unique)
    selected: list[tuple[float, float]] = []
    best_energy = np.inf
    for point in ordered:
        if point[1] < best_energy:
            selected.append(point)
            best_energy = point[1]
    return np.asarray(selected, dtype=np.float64).reshape(-1, 2)


def _paired_test(difference: np.ndarray) -> tuple[float, float]:
    nonzero = difference[difference != 0]
    if len(nonzero) == 0:
        return 1.0, 0.0
    test = wilcoxon(difference, zero_method="wilcox", alternative="two-sided")
    ranks = rankdata(np.abs(nonzero))
    rbc = float(np.sum(ranks * np.sign(nonzero)) / np.sum(ranks))
    return float(test.pvalue), rbc


def finalize(formal: Path, mirror: Path, audit: Path) -> dict[str, Any]:
    audit.mkdir(parents=True, exist_ok=True)
    summary = json.loads((formal / "summary.json").read_text(encoding="utf-8"))
    if not (summary["completed_runs"] == summary["persisted_completed_runs"] ==
            summary["mirror_valid_completed_runs"] == 50 and summary["failed_runs"] == 0):
        raise RuntimeError("Gate-A completion/persistence gate failed")

    manifests = {}
    for path in (mirror / "manifests").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        key = tuple(row["identity"][name] for name in ("scenario", "algorithm", "seed"))
        manifests[key] = row

    completion: list[dict[str, Any]] = []
    persistence: list[dict[str, Any]] = []
    feasibility: list[dict[str, Any]] = []
    compression: list[dict[str, Any]] = []
    runs: dict[tuple[str, str, int], dict[str, Any]] = {}
    for scenario in SCENARIOS:
        for algorithm in ALGORITHMS:
            for seed in SEEDS:
                run_dir = formal / "scenarios" / scenario / "runs" / f"{algorithm}__seed_{seed}"
                status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
                payload = pickle.loads((run_dir / "checkpoint.pkl").read_bytes())
                archive = EvaluationArchive.from_cache(payload["evaluator"]["cache"])
                front = archive.official_front()
                required = all((run_dir / name).is_file() for name in (
                    "status.json", "checkpoint.pkl", "evaluation_archive_metadata.json", "final_front.csv", "history.csv"))
                design_ok = (algorithm == "RVEA" and status.get("oe_rvea_design_hash") is None) or (
                    algorithm == "OE-RVEA" and status.get("oe_rvea_design_hash") == DESIGN_HASH)
                implementation_ok = (algorithm == "RVEA" and status.get("oe_rvea_implementation_hash") is None) or (
                    algorithm == "OE-RVEA" and status.get("oe_rvea_implementation_hash") == IMPLEMENTATION_HASH)
                completion.append({
                    "scenario": scenario, "algorithm": algorithm, "seed": seed,
                    "status": status["status"], "requests": status["requests"],
                    "required_artifacts": required,
                    "protocol_match": status.get("effective_protocol_hash") == protocol_hash(),
                    "evaluator_match": status.get("evaluator_hash") == IMPROVED_CORRECTED_DUALBP_HASH,
                    "design_match": design_ok, "implementation_match": implementation_ok,
                    "nonfinite_count": status["nonfinite_count"], "front_size": len(front),
                })
                manifest = manifests[(scenario, algorithm, seed)]
                mirror_valid = (manifest["source_artifact_hashes"] == manifest["mirror_artifact_hashes"]
                                and all(_sha(run_dir / path) == digest
                                        for path, digest in manifest["source_artifact_hashes"].items()))
                persistence.append({"scenario": scenario, "algorithm": algorithm, "seed": seed,
                                    "persisted": True, "mirror_valid": mirror_valid,
                                    "artifact_count": len(manifest["source_artifact_hashes"])})
                has_feasible = len(archive.feasible_records()) > 0
                feasibility.append({"scenario": scenario, "algorithm": algorithm, "seed": seed,
                                    "has_feasible_solution": has_feasible,
                                    "feasible_evaluations": status["feasible_evaluations"],
                                    "feasible_rate": status["feasible_rate"]})
                if algorithm == "OE-RVEA":
                    compression.append({
                        "scenario": scenario, "seed": seed,
                        "compression_triggered": status["oe_compression_triggered"],
                        "mean_compression_ratio": status["oe_mean_compression_ratio"],
                        "median_compression_ratio": status["oe_median_compression_ratio"],
                        "max_compression_ratio": status["oe_max_compression_ratio"],
                        "fallback_generations": status["oe_fallback_generations"],
                        "fallback_generation_rate": status["oe_fallback_generation_rate"],
                        "mean_survivor_unique_objective_count": status["oe_mean_survivor_unique_objective_count"],
                    })
                runs[(scenario, algorithm, seed)] = {"status": status, "archive": archive, "front": front}

    _csv(audit / "formal_completion_audit.csv", completion)
    _csv(audit / "persistence_audit.csv", persistence)
    _csv(audit / "feasibility_audit.csv", feasibility)
    _csv(audit / "compression_by_run.csv", compression)
    if not all(row["status"] == "completed" and row["requests"] == BUDGET and
               row["required_artifacts"] and row["protocol_match"] and row["evaluator_match"] and
               row["design_match"] and row["implementation_match"] and row["nonfinite_count"] == 0
               for row in completion):
        raise RuntimeError("Gate-A run artifact/hash/budget gate failed")
    if not all(row["mirror_valid"] for row in persistence):
        raise RuntimeError("Gate-A mirror integrity gate failed")
    for scenario in SCENARIOS:
        for seed in SEEDS:
            hashes = {runs[(scenario, algorithm, seed)]["status"]["initial_pool_hash"] for algorithm in ALGORITHMS}
            if len(hashes) != 1:
                raise RuntimeError("Gate-A matched initialization mismatch")

    metrics: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        feasible_rows = [row for algorithm in ALGORITHMS for seed in SEEDS
                         for row in runs[(scenario, algorithm, seed)]["archive"].feasible_records()]
        values = np.asarray([[row["Phi_T"], row["Phi_E"]] for row in feasible_rows], dtype=np.float64)
        lower, upper = values.min(axis=0), values.max(axis=0)
        safe_upper = np.where(upper > lower, upper, lower + 1.0)
        pooled = _objective_front([row for algorithm in ALGORITHMS for seed in SEEDS
                                  for row in runs[(scenario, algorithm, seed)]["front"]])
        norm_reference = global_normalize(pooled, lower, safe_upper)
        reference_point = global_normalize([upper + 0.1 * (upper - lower)], lower, safe_upper)[0]
        reference_point = np.where(upper > lower, reference_point, 0.1 * np.maximum(np.abs(upper), 1.0))
        for algorithm in ALGORITHMS:
            for seed in SEEDS:
                item = runs[(scenario, algorithm, seed)]
                front_values = np.asarray([[row["Phi_T"], row["Phi_E"]] for row in item["front"]])
                normalized = global_normalize(front_values, lower, safe_upper)
                distance = np.linalg.norm(norm_reference[:, None, :] - normalized[None, :, :], axis=2)
                metrics.append({
                    "scenario": scenario, "algorithm": algorithm, "seed": seed,
                    "HV": hypervolume_2d_min(normalized, reference_point),
                    "IGD": float(distance.min(axis=1).mean()), "front_size": len(front_values),
                })

    matched: list[dict[str, Any]] = []
    differences: dict[str, list[float]] = {"HV": [], "IGD": []}
    for scenario in SCENARIOS:
        for seed in SEEDS:
            oe = next(row for row in metrics if row["scenario"] == scenario and row["seed"] == seed and row["algorithm"] == "OE-RVEA")
            baseline = next(row for row in metrics if row["scenario"] == scenario and row["seed"] == seed and row["algorithm"] == "RVEA")
            hv_diff = float(oe["HV"] - baseline["HV"])
            igd_diff = float(baseline["IGD"] - oe["IGD"])
            differences["HV"].append(hv_diff); differences["IGD"].append(igd_diff)
            matched.append({
                "scenario": scenario, "seed": seed,
                "oe_hv": oe["HV"], "rvea_hv": baseline["HV"], "hv_difference_positive_favors_oe": hv_diff,
                "oe_igd": oe["IGD"], "rvea_igd": baseline["IGD"], "igd_difference_positive_favors_oe": igd_diff,
                "oe_front_size": oe["front_size"], "rvea_front_size": baseline["front_size"],
            })
    _csv(audit / "matched_case_metrics.csv", matched)

    hv_summary = []
    igd_summary = []
    for algorithm in ALGORITHMS:
        subset = [row for row in metrics if row["algorithm"] == algorithm]
        for name, target in (("HV", hv_summary), ("IGD", igd_summary)):
            vector = np.asarray([row[name] for row in subset])
            target.append({"algorithm": algorithm, "mean": float(vector.mean()),
                           "std": float(vector.std(ddof=1)), "median": float(np.median(vector))})
    _csv(audit / "hv_summary.csv", hv_summary); _csv(audit / "igd_summary.csv", igd_summary)

    wtl: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    for metric in ("HV", "IGD"):
        difference = np.asarray(differences[metric])
        p_value, rbc = _paired_test(difference)
        wtl.append({"metric": metric, "win": int(np.sum(difference > 0)),
                    "tie": int(np.sum(difference == 0)), "loss": int(np.sum(difference < 0))})
        tests.append({"metric": metric, "test": "paired Wilcoxon two-sided",
                      "p_value": p_value, "rank_biserial": rbc,
                      "positive_effect_favors": "OE-RVEA"})
    _csv(audit / "pairwise_wtl.csv", wtl); _csv(audit / "wilcoxon_effect_size.csv", tests)

    compression_summary = [{
        "oe_runs": 25,
        "compression_triggered_runs": sum(bool(row["compression_triggered"]) for row in compression),
        "mean_of_run_mean_compression_ratio": float(np.mean([row["mean_compression_ratio"] for row in compression])),
        "median_of_run_mean_compression_ratio": float(np.median([row["mean_compression_ratio"] for row in compression])),
        "max_run_compression_ratio": float(max(row["max_compression_ratio"] for row in compression)),
        "fallback_triggered_runs": sum(int(row["fallback_generations"]) > 0 for row in compression),
        "total_fallback_generations": sum(int(row["fallback_generations"]) for row in compression),
    }]
    _csv(audit / "compression_mechanism_summary.csv", compression_summary)

    hv_wtl, igd_wtl = wtl
    hv_test, igd_test = tests
    oe_hv = next(row["mean"] for row in hv_summary if row["algorithm"] == "OE-RVEA")
    rvea_hv = next(row["mean"] for row in hv_summary if row["algorithm"] == "RVEA")
    oe_igd = next(row["mean"] for row in igd_summary if row["algorithm"] == "OE-RVEA")
    rvea_igd = next(row["mean"] for row in igd_summary if row["algorithm"] == "RVEA")
    no_feasibility_regression = all(row["has_feasible_solution"] for row in feasibility)
    gates = {
        "A_HV_wins_gt_losses": hv_wtl["win"] > hv_wtl["loss"],
        "B_IGD_wins_gt_losses": igd_wtl["win"] > igd_wtl["loss"],
        "C_mean_HV_OE_gt_RVEA": oe_hv > rvea_hv,
        "D_mean_IGD_OE_lt_RVEA": oe_igd < rvea_igd,
        "E_at_least_one_significant_favorable": (
            (hv_test["p_value"] < 0.05 and hv_test["rank_biserial"] > 0) or
            (igd_test["p_value"] < 0.05 and igd_test["rank_biserial"] > 0)),
        "F_other_metric_not_significantly_favorable_to_RVEA": not (
            (hv_test["p_value"] < 0.05 and hv_test["rank_biserial"] < 0) or
            (igd_test["p_value"] < 0.05 and igd_test["rank_biserial"] < 0)),
        "G_no_feasibility_regression": no_feasibility_regression,
    }
    decision = "PASS" if all(gates.values()) else "FAIL"
    result = {
        **gates, "OE_RVEA_GATE_A": decision,
        "future_full_comparison_ready": decision == "PASS",
        "automatic_tuning": False, "gate_a_runs": 50,
        "formal_elapsed_seconds": summary["elapsed_seconds"],
    }
    _json(audit / "gate_a_result.json", result)
    _json(audit / "design_binding.json", {"design_hash": DESIGN_HASH, "matched": True})
    _json(audit / "implementation_binding.json", {"implementation_hash": IMPLEMENTATION_HASH, "matched": True})
    _json(audit / "evaluator_binding.json", {"evaluator_hash": IMPROVED_CORRECTED_DUALBP_HASH, "matched": True})
    manifest = {
        "formal": str(formal), "audit": str(audit), "completed": 50, "persisted": 50,
        "failed": 0, "budget_violations": 0, "matched_pairs": 25,
        "means": {"OE_HV": oe_hv, "RVEA_HV": rvea_hv, "OE_IGD": oe_igd, "RVEA_IGD": rvea_igd},
        "wtl": {row["metric"]: [row["win"], row["tie"], row["loss"]] for row in wtl},
        "tests": {row["metric"]: {"p": row["p_value"], "rank_biserial": row["rank_biserial"]} for row in tests},
        "compression": compression_summary[0], "gate": result,
        "scientific_parameters_modified": False,
    }
    _json(audit / "final_result_manifest.json", manifest)
    (audit / "README.md").write_text(
        "# OE-RVEA Gate-A 最终审计\n\n本目录仅使用S1–S5、seeds 62001–62005的50个独立Gate-A运行；未混入旧Exp-7。\n",
        encoding="utf-8")
    files = sorted(path.name for path in audit.iterdir() if path.is_file())
    _json(audit / "created_files.json", {"count": len(files) + 1, "files": files + ["created_files.json"]})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", required=True); parser.add_argument("--mirror", required=True)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args()
    result = finalize(Path(args.formal).resolve(), Path(args.mirror).resolve(), Path(args.audit).resolve())
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
