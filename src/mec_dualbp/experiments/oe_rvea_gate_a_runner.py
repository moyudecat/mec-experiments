"""Frozen serial OE-RVEA V1 versus Standard RVEA Gate-A scheduler."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from mec_dualbp.experiments.exp7_chain_runner import _write_csv, run_one
from mec_dualbp.experiments.exp7_formal_runner import SCENARIO_HASHES
from mec_dualbp.experiments.exp7_persistence_guard import Exp7PersistenceGuard
from mec_dualbp.optimization.adapters import (
    OE_RVEA_ALGORITHM, OE_RVEA_V1_DESIGN_HASH,
)
from mec_dualbp.optimization.dualbp_engine_registry import (
    IMPROVED_MODE, create_corrected_dualbp_engine,
)
from mec_dualbp.optimization.outer_contract import IMPROVED_CORRECTED_DUALBP_HASH
from mec_dualbp.utils.atomic_io import safe_atomic_write_json


DESIGN_HASH = "6e9795b539095f841d51bb45db47be1535d66d3c530d81f96ed91ea1503428d4"
IMPLEMENTATION_HASH = "95e9ee309294e7797c71b65b0a4a1fd239ac77a5d476e313d7e34f4fdec2fc57"
ADAPTER_SOURCE_HASH = "eef5201454745ad0f339f54d36c616799c2f843a14ab3f623cebc4338bf22473"
PYMOO_RVEA_SOURCE_HASH = "68f21a5ce49063d90b1e3a12163a5fb21be396ea37d9ead6a8e261eb8a5d2de7"
ALGORITHMS = (OE_RVEA_ALGORITHM, "RVEA")
SCENARIOS = ("S1", "S2", "S3", "S4", "S5")
SEEDS = (62001, 62002, 62003, 62004, 62005)
POPULATION = 100
BUDGET = 20_000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_payload() -> dict[str, Any]:
    return {
        "identity": "OE-RVEA-V1-vs-Standard-RVEA-independent-Gate-A",
        "design_hash": DESIGN_HASH,
        "implementation_hash": IMPLEMENTATION_HASH,
        "evaluator_hash": IMPROVED_CORRECTED_DUALBP_HASH,
        "algorithms": list(ALGORITHMS), "scenarios": list(SCENARIOS),
        "scenario_hashes": SCENARIO_HASHES, "seeds": list(SEEDS),
        "population": POPULATION, "requests_per_run": BUDGET,
        "workers": 1, "equivalence": "exact-only", "epsilon": None,
    }


def protocol_hash() -> str:
    raw = json.dumps(protocol_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def preflight(project: Path) -> dict[str, Any]:
    design_dir = project / "outputs/05_audit/oe_rvea_v1_design_freeze_20260820_021857"
    implementation_dir = project / "outputs/06_implementation/oe_rvea_v1_implementation_20260820_021857"
    design = json.loads((design_dir / "design_hash.json").read_text(encoding="utf-8"))
    implementation = json.loads((implementation_dir / "implementation_hash.json").read_text(encoding="utf-8"))
    gate = json.loads((design_dir / "gate_a_protocol.json").read_text(encoding="utf-8"))
    scenario_paths = [project / "configs/scenarios" / f"exp7_{name.lower()}_formal_v1.json" for name in SCENARIOS]
    actual_scenarios = {_name: _sha(path) for _name, path in zip(SCENARIOS, scenario_paths)}
    import pymoo.algorithms.moo.rvea as pymoo_rvea
    adapter_path = project / "src/mec_dualbp/optimization/adapters.py"
    pymoo_path = Path(pymoo_rvea.__file__).resolve()
    checks = {
        "design_hash": design["oe_rvea_v1_design_hash"] == DESIGN_HASH == OE_RVEA_V1_DESIGN_HASH,
        "implementation_hash": implementation["oe_rvea_v1_implementation_hash"] == IMPLEMENTATION_HASH,
        "adapter_source_hash": _sha(adapter_path) == ADAPTER_SOURCE_HASH,
        "pymoo_rvea_source_hash": _sha(pymoo_path) == PYMOO_RVEA_SOURCE_HASH,
        "gate_status": gate["status"] == "preregistered_not_run" and gate["runs_executed_this_round"] == 0,
        "algorithms": tuple(gate["algorithms"]) == ALGORITHMS,
        "scenarios": tuple(gate["scenarios"]) == SCENARIOS and actual_scenarios == SCENARIO_HASHES,
        "seeds": tuple(gate["outer_seeds"]) == SEEDS,
        "population_budget": gate["population"] == POPULATION and gate["requests_per_run"] == BUDGET,
        "evaluator_hash": implementation_dir.exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Gate-A preflight mismatch: {checks}")
    return {
        "passed": True, "checks": checks, "protocol_hash": protocol_hash(),
        **protocol_payload(), "total_runs": 50, "matched_cases": 25,
        "performance_preview_runs": 0, "algorithm_tuning": False,
    }


def _checkpoint_binding_ok(checkpoint: Path, algorithm: str) -> None:
    if not checkpoint.exists():
        return
    payload = pickle.loads(checkpoint.read_bytes())
    adapter = payload["adapter"]
    if adapter.name != algorithm:
        raise RuntimeError("Gate-A checkpoint algorithm mismatch")
    if algorithm == OE_RVEA_ALGORITHM and getattr(adapter.algorithm, "design_hash", None) != DESIGN_HASH:
        raise RuntimeError("Gate-A checkpoint OE-RVEA design mismatch")


def _mechanism_record(run_dir: Path, algorithm: str) -> dict[str, Any]:
    payload = pickle.loads((run_dir / "checkpoint.pkl").read_bytes())
    adapter = payload["adapter"]
    if algorithm == OE_RVEA_ALGORITHM:
        if getattr(adapter.algorithm, "design_hash", None) != DESIGN_HASH:
            raise RuntimeError("completed OE-RVEA checkpoint design mismatch")
        rows = list(adapter.objective_equivalence_audit)
        _write_csv(run_dir / "objective_equivalence_audit.csv", rows)
        ratios = [float(row["compression_ratio"]) for row in rows]
        fallbacks = sum(bool(row["compression_fallback_triggered"]) for row in rows)
        return {
            "oe_compression_enabled": True,
            "oe_audit_generations": len(rows),
            "oe_mean_compression_ratio": sum(ratios) / len(ratios) if ratios else 0.0,
            "oe_median_compression_ratio": float(__import__("numpy").median(ratios)) if ratios else 0.0,
            "oe_max_compression_ratio": max(ratios) if ratios else 0.0,
            "oe_fallback_generations": fallbacks,
            "oe_fallback_generation_rate": fallbacks / len(rows) if rows else 0.0,
            "oe_mean_survivor_unique_objective_count": (
                sum(int(row["survivor_unique_objective_count"]) for row in rows) / len(rows) if rows else 0.0),
            "oe_compression_triggered": any(int(row["compressed_candidate_count"]) > 0 for row in rows),
        }
    if type(adapter.algorithm).__module__ != "pymoo.algorithms.moo.rvea" or type(adapter.algorithm).__name__ != "RVEA":
        raise RuntimeError("Standard RVEA regression identity mismatch")
    return {"oe_compression_enabled": False}


def run_gate_a(project: Path, output: Path, mirror: Path) -> dict[str, Any]:
    started = time.monotonic()
    checked = preflight(project)
    output.mkdir(parents=True, exist_ok=True)
    active = {
        "status": "active", "protocol_hash": protocol_hash(),
        "evaluator_hash": IMPROVED_CORRECTED_DUALBP_HASH,
        "process_start": datetime.now().astimezone().isoformat(), "expected_runs": 50,
    }
    guard = Exp7PersistenceGuard(output, mirror, protocol_hash(), IMPROVED_CORRECTED_DUALBP_HASH)
    guard.initialize(active)
    safe_atomic_write_json(output / "formal_preflight.json", checked)
    stress = output / "preflight_status_stress.json"
    for index in range(25):
        safe_atomic_write_json(stress, {"index": index})
    if json.loads(stress.read_text(encoding="utf-8"))["index"] != 24:
        raise RuntimeError("Gate-A atomic persistence preflight failed")

    completed = len(guard.manifests())
    failed = 0
    recovered_count = 0
    log = output / "run_progress.log"
    for scenario_name in SCENARIOS:
        scenario_path = project / "configs/scenarios" / f"exp7_{scenario_name.lower()}_formal_v1.json"
        frozen = json.loads(scenario_path.read_text(encoding="utf-8"))
        scenario = dict(frozen)
        scenario["corrected_dualbp_hash"] = IMPROVED_CORRECTED_DUALBP_HASH
        engine = create_corrected_dualbp_engine(
            project, scenario, evaluator_mode=IMPROVED_MODE,
            evaluator_hash=IMPROVED_CORRECTED_DUALBP_HASH)
        for seed in SEEDS:
            matched_hashes: list[str] = []
            for algorithm in ALGORITHMS:
                run_root = output / "scenarios" / scenario_name
                run_dir = run_root / "runs" / f"{algorithm}__seed_{seed}"
                persisted, recovered = guard.watchdog(completed)
                completed = persisted
                recovered_count += recovered
                try:
                    _checkpoint_binding_ok(run_dir / "checkpoint.pkl", algorithm)
                    result = run_one(run_root, algorithm, seed, POPULATION, BUDGET, scenario, engine)
                    mechanism = _mechanism_record(run_dir, algorithm)
                    result.update({
                        **mechanism, "evaluator_mode": IMPROVED_MODE,
                        "evaluator_hash": IMPROVED_CORRECTED_DUALBP_HASH,
                        "effective_protocol_hash": protocol_hash(),
                        "scenario_file_hash": SCENARIO_HASHES[scenario_name],
                        "oe_rvea_design_hash": DESIGN_HASH if algorithm == OE_RVEA_ALGORITHM else None,
                        "oe_rvea_implementation_hash": IMPLEMENTATION_HASH if algorithm == OE_RVEA_ALGORITHM else None,
                    })
                    safe_atomic_write_json(run_dir / "status.json", result)
                    matched_hashes.append(str(result["initial_pool_hash"]))
                    guard.mirror_run(run_dir)
                    completed, recovered = guard.watchdog(completed)
                    recovered_count += recovered
                    with log.open("a", encoding="utf-8", newline="\n") as stream:
                        stream.write(f"completed {scenario_name} {algorithm} {seed} requests={result['requests']}\n")
                except Exception as error:
                    failed += 1
                    with log.open("a", encoding="utf-8", newline="\n") as stream:
                        stream.write(f"failed {scenario_name} {algorithm} {seed}: {error!r}\n")
                safe_atomic_write_json(output / "run_progress.json", {
                    "status": "running", "completed_runs": completed, "failed_runs": failed,
                    "processed_runs": completed + failed, "total_runs": 50,
                    "elapsed_seconds": time.monotonic() - started,
                })
                guard.write_progress(failed=failed, running=None,
                                     last_completed={"scenario": scenario_name, "algorithm": algorithm, "seed": seed},
                                     recovered=recovered_count)
            if len(matched_hashes) == 2 and matched_hashes[0] != matched_hashes[1]:
                raise RuntimeError("matched canonical initial population hash mismatch")
    summary = {
        "status": "completed" if completed == 50 and failed == 0 else "completed_with_failures",
        "completed_runs": completed, "persisted_completed_runs": len(guard.manifests()),
        "mirror_valid_completed_runs": len(guard.manifests()), "failed_runs": failed,
        "total_runs": 50, "elapsed_seconds": time.monotonic() - started,
        "recovered_from_mirror_count": recovered_count, "protocol_hash": protocol_hash(),
    }
    safe_atomic_write_json(output / "summary.json", summary)
    safe_atomic_write_json(output / "run_progress.json", summary)
    marker = {**active, "status": "completed", "completed_at": datetime.now().astimezone().isoformat()}
    safe_atomic_write_json(output / "FORMAL_ACTIVE.json", marker)
    safe_atomic_write_json(mirror / "FORMAL_ACTIVE.json", marker)
    guard.write_progress(failed=failed, running=None, last_completed=None, recovered=recovered_count)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--recovery-mirror", required=True)
    args = parser.parse_args()
    result = run_gate_a(Path(args.project_root).resolve(), Path(args.output).resolve(),
                        Path(args.recovery_mirror).resolve())
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
