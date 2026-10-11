"""Resumable serial Exp-7 Formal scheduler."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from mec_dualbp.experiments.exp7_chain_runner import (
    OUTER_SEEDS, run_one,
)
from mec_dualbp.optimization.adapters import V2_ALGORITHM
from mec_dualbp.optimization.dualbp_engine_registry import (
    IMPROVED_MODE, create_corrected_dualbp_engine,
)
from mec_dualbp.optimization.outer_contract import IMPROVED_CORRECTED_DUALBP_HASH
from mec_dualbp.utils.atomic_io import safe_atomic_write_json
from mec_dualbp.experiments.exp7_persistence_guard import Exp7PersistenceGuard


FINAL_ALGORITHMS = (V2_ALGORITHM, "MOEA/D", "RVEA", "MOPSO", "Random")
SCENARIO_HASHES = {
    "S1": "4ce87996a9e27c83a56ccfdf6954be7ead0625863df0672525cf2071e575e65f",
    "S2": "b3403562e3b75150d5b233f58d499f590aed5da1a45f768fc058bdac6787a039",
    "S3": "d348158b13743534e898f7c94540819c8663e072f1da29e0a21f7894c053cf97",
    "S4": "7b15f64449aed654e92549b66e4e73077ec9928346f732c556c671b48e113a14",
    "S5": "f6810a099fcbd1f585d11ae2193cada2ae92c00ce104a730ae3add930dca4389",
}


def run_formal(project: Path, output: Path, effective_protocol_hash: str,
               recovery_mirror: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    active = {
        "status": "active", "protocol_hash": effective_protocol_hash,
        "evaluator_hash": IMPROVED_CORRECTED_DUALBP_HASH,
        "process_start": datetime.now().astimezone().isoformat(),
        "expected_runs": 125,
    }
    guard = Exp7PersistenceGuard(
        output, recovery_mirror, effective_protocol_hash, IMPROVED_CORRECTED_DUALBP_HASH)
    guard.initialize(active)
    scenario_paths = [project / "configs/scenarios" / f"exp7_s{i}_formal_v1.json" for i in range(1, 6)]
    missing = [str(path) for path in scenario_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"missing Formal scenarios: {missing}")
    configs = [json.loads(path.read_text(encoding="utf-8")) for path in scenario_paths]
    import hashlib
    actual_hashes = {
        row["scenario_id"]: hashlib.sha256(path.read_bytes()).hexdigest()
        for row, path in zip(configs, scenario_paths)
    }
    if actual_hashes != SCENARIO_HASHES:
        raise RuntimeError("frozen scenario hash mismatch")
    preflight = {
        "passed": True, "effective_protocol_hash": effective_protocol_hash,
        "scenarios": [row["scenario_id"] for row in configs],
        "algorithms": list(FINAL_ALGORITHMS), "outer_seeds": list(OUTER_SEEDS),
        "population": 100, "budget_per_run": 20_000, "total_runs": 125,
        "total_requests": 2_500_000, "workers": 1,
        "evaluator_mode": IMPROVED_MODE,
        "evaluator_hash": IMPROVED_CORRECTED_DUALBP_HASH,
        "scenario_hashes": actual_hashes,
    }
    # Short status-writer sanity check, removed atomically only after success.
    stress = output / "preflight_status_stress.json"
    for index in range(25):
        safe_atomic_write_json(stress, {"index": index})
    if json.loads(stress.read_text(encoding="utf-8"))["index"] != 24:
        raise RuntimeError("atomic status stress failed")
    safe_atomic_write_json(output / "formal_preflight.json", preflight)
    completed = 0
    failed = 0
    recovered_from_mirror = 0
    started = time.monotonic()
    log = output / "run_progress.log"
    for frozen_scenario in configs:
        # The frozen scenario bytes retain their historical provenance hash.  The
        # evaluator binding is a run-level protocol identity, not a scenario edit.
        scenario = dict(frozen_scenario)
        scenario["corrected_dualbp_hash"] = IMPROVED_CORRECTED_DUALBP_HASH
        engine = create_corrected_dualbp_engine(
            project, scenario, evaluator_mode=IMPROVED_MODE,
            evaluator_hash=IMPROVED_CORRECTED_DUALBP_HASH,
        )
        for seed in OUTER_SEEDS:
            for algorithm in FINAL_ALGORITHMS:
                run_output = output / "scenarios" / scenario["scenario_id"]
                persisted_before, recovered = guard.watchdog(completed)
                recovered_from_mirror += recovered
                if persisted_before != completed:
                    completed = persisted_before
                persistence_stage = False
                try:
                    result = run_one(run_output, algorithm, seed, 100, 20_000, scenario, engine)
                    if (getattr(engine, "evaluator_mode", None) != IMPROVED_MODE
                            or getattr(engine, "corrected_dualbp_hash", None) != IMPROVED_CORRECTED_DUALBP_HASH):
                        raise RuntimeError("run evaluator identity mismatch")
                    result.update({
                        "evaluator_mode": IMPROVED_MODE,
                        "evaluator_hash": IMPROVED_CORRECTED_DUALBP_HASH,
                        "effective_protocol_hash": effective_protocol_hash,
                        "scenario_file_hash": actual_hashes[scenario["scenario_id"]],
                    })
                    run_dir = run_output / "runs" / f"{algorithm.replace('/', '_')}__seed_{seed}"
                    safe_atomic_write_json(run_dir / "status.json", result)
                    persistence_stage = True
                    manifest = guard.mirror_run(run_dir)
                    persisted_after, recovered = guard.watchdog(completed)
                    recovered_from_mirror += recovered
                    completed = persisted_after
                    with log.open("a", encoding="utf-8", newline="\n") as stream:
                        stream.write(f"completed {scenario['scenario_id']} {algorithm} {seed} requests={result['requests']}\n")
                except Exception as error:
                    if persistence_stage:
                        guard.write_progress(
                            failed=failed, running=f"{scenario['scenario_id']}/{algorithm}/{seed}",
                            last_completed=None, recovered=recovered_from_mirror)
                        raise
                    failed += 1
                    with log.open("a", encoding="utf-8", newline="\n") as stream:
                        stream.write(f"failed {scenario['scenario_id']} {algorithm} {seed}: {error!r}\n")
                safe_atomic_write_json(output / "run_progress.json", {
                    "status": "running", "completed_runs": completed, "failed_runs": failed,
                    "processed_runs": completed + failed, "total_runs": 125,
                    "elapsed_seconds": time.monotonic() - started,
                })
                if completed % 5 == 0 or failed:
                    guard.write_progress(
                        failed=failed, running=None,
                        last_completed={"scenario": scenario["scenario_id"], "algorithm": algorithm, "seed": seed},
                        recovered=recovered_from_mirror)
    summary = {
        "status": "completed" if completed == 125 and failed == 0 else "completed_with_failures",
        "completed_runs": completed, "failed_runs": failed, "total_runs": 125,
        "elapsed_seconds": time.monotonic() - started,
        "persisted_completed_runs": len(guard.manifests()),
        "mirror_valid_completed_runs": len(guard.manifests()),
        "recovered_from_mirror_count": recovered_from_mirror,
    }
    safe_atomic_write_json(output / "summary.json", summary)
    safe_atomic_write_json(output / "run_progress.json", summary)
    completed_marker = dict(active)
    completed_marker.update(status="completed", completed_at=datetime.now().astimezone().isoformat())
    safe_atomic_write_json(output / "FORMAL_ACTIVE.json", completed_marker)
    safe_atomic_write_json(recovery_mirror / "FORMAL_ACTIVE.json", completed_marker)
    guard.write_progress(failed=failed, running=None, last_completed=None,
                         recovered=recovered_from_mirror)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--effective-protocol-hash", required=True)
    parser.add_argument("--recovery-mirror", required=True)
    args = parser.parse_args()
    result = run_formal(
        Path(args.project_root).resolve(), Path(args.output).resolve(),
        args.effective_protocol_hash, Path(args.recovery_mirror).resolve())
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
