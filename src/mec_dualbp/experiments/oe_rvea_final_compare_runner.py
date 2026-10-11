"""Serial scheduler for the 100 additional runs of the final six-algorithm comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from mec_dualbp.experiments.exp7_chain_runner import run_one
from mec_dualbp.experiments.exp7_formal_runner import SCENARIO_HASHES
from mec_dualbp.experiments.exp7_persistence_guard import Exp7PersistenceGuard
from mec_dualbp.experiments.oe_rvea_gate_a_runner import (
    BUDGET, DESIGN_HASH, IMPLEMENTATION_HASH, POPULATION, SCENARIOS, SEEDS,
)
from mec_dualbp.optimization.adapters import V2_ALGORITHM, V2_DESIGN_HASH
from mec_dualbp.optimization.dualbp_engine_registry import IMPROVED_MODE, create_corrected_dualbp_engine
from mec_dualbp.optimization.outer_contract import IMPROVED_CORRECTED_DUALBP_HASH
from mec_dualbp.utils.atomic_io import safe_atomic_write_json


ALGORITHMS = (V2_ALGORITHM, "MOEA/D", "MOPSO", "Random")
ALL_ALGORITHMS = ("OE-RVEA", "RVEA") + ALGORITHMS
CA_IMPLEMENTATION_HASH = "984f7a1873ea37b29af2121c377e9485113223bf07eae21e4308978b1864b3b2"
GATE_FORMAL = "outputs/01_formal/oe_rvea_gate_a_formal_20260820_083049"
GATE_AUDIT = "outputs/05_audit/oe_rvea_gate_a_finalization_20260820_091825"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def final_protocol_payload() -> dict[str, Any]:
    return {
        "identity": "OE-RVEA-final-six-algorithm-comparison",
        "oe_design_hash": DESIGN_HASH, "oe_implementation_hash": IMPLEMENTATION_HASH,
        "evaluator_hash": IMPROVED_CORRECTED_DUALBP_HASH,
        "ca_design_hash": V2_DESIGN_HASH, "ca_implementation_hash": CA_IMPLEMENTATION_HASH,
        "algorithms": list(ALL_ALGORITHMS), "scenario_hashes": SCENARIO_HASHES,
        "seeds": list(SEEDS), "population": POPULATION, "requests_per_run": BUDGET,
        "matched_initialization": True,
        "evaluation_archive": "all paid -> canonical decision dedup -> feasible -> exact objective nondominated",
        "metrics": "per-scenario six-algorithm pooled min-max HV/IGD",
        "statistics": "Friedman; paired Wilcoxon; Holm; rank-biserial; exact ties",
        "final_gate": "preregistered A-G", "workers": 1,
        "baseline_configs": {
            "MOPSO": {"w": 0.5, "c1": 1.5, "c2": 1.5, "archive_max_size": 100},
            "MOEA/D": "frozen adapter operators", "Random": "frozen adapter",
        },
    }


def final_protocol_hash() -> str:
    raw = json.dumps(final_protocol_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def preflight(project: Path) -> dict[str, Any]:
    gate_formal = project / GATE_FORMAL
    gate_result = json.loads((project / GATE_AUDIT / "gate_a_result.json").read_text(encoding="utf-8"))
    ca_binding = json.loads((project / "outputs/06_implementation/exp7_cnsga_v2_implementation_20260813_002739/v2_implementation_hash.json").read_text(encoding="utf-8"))
    gate_statuses = list(gate_formal.glob("scenarios/*/runs/*/status.json"))
    gate_rows = [json.loads(path.read_text(encoding="utf-8")) for path in gate_statuses]
    scenario_paths = [project / "configs/scenarios" / f"exp7_{name.lower()}_formal_v1.json" for name in SCENARIOS]
    checks = {
        "gate_pass": gate_result["OE_RVEA_GATE_A"] == "PASS" and gate_result["future_full_comparison_ready"] is True,
        "gate_50_immutable_reuse": len(gate_rows) == 50 and all(row["status"] == "completed" and row["requests"] == BUDGET for row in gate_rows),
        "oe_hashes": all(row.get("oe_rvea_design_hash") == DESIGN_HASH and row.get("oe_rvea_implementation_hash") == IMPLEMENTATION_HASH
                         for row in gate_rows if row["algorithm"] == "OE-RVEA"),
        "evaluator": all(row.get("evaluator_hash") == IMPROVED_CORRECTED_DUALBP_HASH for row in gate_rows),
        "ca_hashes": ca_binding["v2_design_hash"] == V2_DESIGN_HASH and ca_binding["v2_implementation_hash"] == CA_IMPLEMENTATION_HASH,
        "scenario_hashes": {name: _sha(path) for name, path in zip(SCENARIOS, scenario_paths)} == SCENARIO_HASHES,
        "additional_matrix": len(ALGORITHMS) * len(SCENARIOS) * len(SEEDS) == 100,
    }
    if not all(checks.values()):
        raise RuntimeError(f"final comparison preflight mismatch: {checks}")
    return {"passed": True, "checks": checks, "final_protocol_hash": final_protocol_hash(),
            **final_protocol_payload(), "gate_a_reused_runs": 50, "new_runs": 100, "final_runs": 150}


def _gate_initial_hash(project: Path, scenario: str, seed: int) -> str:
    rows = []
    for algorithm in ("OE-RVEA", "RVEA"):
        path = project / GATE_FORMAL / "scenarios" / scenario / "runs" / f"{algorithm}__seed_{seed}" / "status.json"
        rows.append(json.loads(path.read_text(encoding="utf-8"))["initial_pool_hash"])
    if len(set(rows)) != 1:
        raise RuntimeError("Gate-A initial population pair drift")
    return rows[0]


def _checkpoint_ok(path: Path, algorithm: str) -> None:
    if not path.exists():
        return
    adapter = pickle.loads(path.read_bytes())["adapter"]
    if adapter.name != algorithm:
        raise RuntimeError("additional Formal checkpoint identity mismatch")
    if algorithm == V2_ALGORITHM and getattr(adapter.algorithm.survival, "design_hash", None) != V2_DESIGN_HASH:
        raise RuntimeError("CA-CNSGA-II V2 checkpoint design mismatch")


def run_additional(project: Path, output: Path, mirror: Path) -> dict[str, Any]:
    started = time.monotonic(); checked = preflight(project); output.mkdir(parents=True, exist_ok=True)
    active = {"status": "active", "protocol_hash": final_protocol_hash(),
              "evaluator_hash": IMPROVED_CORRECTED_DUALBP_HASH,
              "process_start": datetime.now().astimezone().isoformat(), "expected_runs": 100}
    guard = Exp7PersistenceGuard(output, mirror, final_protocol_hash(), IMPROVED_CORRECTED_DUALBP_HASH)
    guard.initialize(active); safe_atomic_write_json(output / "formal_preflight.json", checked)
    stress = output / "preflight_status_stress.json"
    for index in range(25): safe_atomic_write_json(stress, {"index": index})
    if json.loads(stress.read_text(encoding="utf-8"))["index"] != 24: raise RuntimeError("atomic persistence preflight failed")
    completed, failed, recovered_count = len(guard.manifests()), 0, 0
    log = output / "run_progress.log"
    for scenario_name in SCENARIOS:
        frozen = json.loads((project / "configs/scenarios" / f"exp7_{scenario_name.lower()}_formal_v1.json").read_text(encoding="utf-8"))
        scenario = dict(frozen); scenario["corrected_dualbp_hash"] = IMPROVED_CORRECTED_DUALBP_HASH
        engine = create_corrected_dualbp_engine(project, scenario, evaluator_mode=IMPROVED_MODE,
                                                evaluator_hash=IMPROVED_CORRECTED_DUALBP_HASH)
        for seed in SEEDS:
            expected_initial = _gate_initial_hash(project, scenario_name, seed)
            for algorithm in ALGORITHMS:
                run_root = output / "scenarios" / scenario_name
                run_dir = run_root / "runs" / f"{algorithm.replace('/', '_')}__seed_{seed}"
                completed, recovered = guard.watchdog(completed); recovered_count += recovered
                try:
                    _checkpoint_ok(run_dir / "checkpoint.pkl", algorithm)
                    result = run_one(run_root, algorithm, seed, POPULATION, BUDGET, scenario, engine)
                    if result["initial_pool_hash"] != expected_initial: raise RuntimeError("Gate-A matched initialization mismatch")
                    result.update({"evaluator_mode": IMPROVED_MODE, "evaluator_hash": IMPROVED_CORRECTED_DUALBP_HASH,
                                   "effective_protocol_hash": final_protocol_hash(),
                                   "scenario_file_hash": SCENARIO_HASHES[scenario_name],
                                   "gate_a_initial_pool_hash": expected_initial,
                                   "ca_v2_design_hash": V2_DESIGN_HASH if algorithm == V2_ALGORITHM else None,
                                   "ca_v2_implementation_hash": CA_IMPLEMENTATION_HASH if algorithm == V2_ALGORITHM else None})
                    safe_atomic_write_json(run_dir / "status.json", result); guard.mirror_run(run_dir)
                    completed, recovered = guard.watchdog(completed); recovered_count += recovered
                    with log.open("a", encoding="utf-8", newline="\n") as stream:
                        stream.write(f"completed {scenario_name} {algorithm} {seed} requests={result['requests']}\n")
                except Exception as error:
                    failed += 1
                    with log.open("a", encoding="utf-8", newline="\n") as stream:
                        stream.write(f"failed {scenario_name} {algorithm} {seed}: {error!r}\n")
                safe_atomic_write_json(output / "run_progress.json", {"status": "running", "completed_runs": completed,
                    "failed_runs": failed, "processed_runs": completed + failed, "total_runs": 100,
                    "elapsed_seconds": time.monotonic() - started})
                guard.write_progress(failed=failed, running=None,
                    last_completed={"scenario": scenario_name, "algorithm": algorithm, "seed": seed}, recovered=recovered_count)
    summary = {"status": "completed" if completed == 100 and failed == 0 else "completed_with_failures",
               "completed_runs": completed, "persisted_completed_runs": len(guard.manifests()),
               "mirror_valid_completed_runs": len(guard.manifests()), "failed_runs": failed, "total_runs": 100,
               "elapsed_seconds": time.monotonic() - started, "recovered_from_mirror_count": recovered_count,
               "final_protocol_hash": final_protocol_hash()}
    safe_atomic_write_json(output / "summary.json", summary); safe_atomic_write_json(output / "run_progress.json", summary)
    marker = {**active, "status": "completed", "completed_at": datetime.now().astimezone().isoformat()}
    safe_atomic_write_json(output / "FORMAL_ACTIVE.json", marker); safe_atomic_write_json(mirror / "FORMAL_ACTIVE.json", marker)
    guard.write_progress(failed=failed, running=None, last_completed=None, recovered=recovered_count)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True); parser.add_argument("--recovery-mirror", required=True)
    args = parser.parse_args(); print(json.dumps(run_additional(Path(args.project_root).resolve(), Path(args.output).resolve(), Path(args.recovery_mirror).resolve()), ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
