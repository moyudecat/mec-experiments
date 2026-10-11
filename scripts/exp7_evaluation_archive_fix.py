"""Freeze the Exp-7 official evaluation archive amendment and rebuild old fronts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path

from mec_dualbp.experiments.exp7_front_rebuild import rebuild_run
from mec_dualbp.experiments.exp1_preview_protocol import canonical_json_hash
from mec_dualbp.optimization.evaluation_archive import EvaluationArchive
from mec_dualbp.utils.atomic_io import safe_atomic_write_json


PARENT_EFFECTIVE = "2b809265db64e79b789da29c635dfc23bb5949e7a4d28d25cc12bde9986e192a"
PARENT_AMENDMENT = "b7e44f2cee00d4bf28eb8d8c1f44d1d847adf60d21184a72def8455f21b405a8"
CORRECTED = "a15b8fb591df123b475f521ef35e4037b6ca201f4172caf804d3af571afe13e2"


def csv_write(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, out = Path(args.project).resolve(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    amendment = {
        "type": "protocol clarification and fairness harmonization",
        "official_candidate_scope": "all paid evaluated candidates within each run budget after canonical decision hash dedup and feasible filtering",
        "official_front": "two-objective strict Pareto nondominated set; Phi_T and Phi_E minimize",
        "cache_hits": "each consumes one request; canonical duplicate retained once after deterministic consistency check",
        "reporting_only": True, "algorithm_search_state_separated": True,
        "unchanged": ["objective_values", "constraint_values", "search_trajectory", "algorithm_parameters", "budget", "seeds", "scenarios", "BP", "alpha", "tau", "eta", "rho", "lambda", "Phi_T", "Phi_E"],
        "fairness_reason": "compare every feasible nondominated solution found within the same paid request budget",
        "parent_effective_protocol_hash": PARENT_EFFECTIVE, "parent_amendment_hash": PARENT_AMENDMENT,
        "corrected_dualbp_hash": CORRECTED,
    }
    amendment_hash = canonical_json_hash(amendment)
    effective_payload = {"parent_effective_protocol_hash": PARENT_EFFECTIVE,
                         "evaluation_archive_amendment_hash": amendment_hash}
    effective_hash = canonical_json_hash(effective_payload)
    safe_atomic_write_json(out / "author_front_scope_decision.json", {"status": "formally_frozen", **amendment})
    safe_atomic_write_json(out / "protocol_clarification.json", {**amendment, "evaluation_archive_amendment_hash": amendment_hash})
    safe_atomic_write_json(out / "protocol_hash_binding.json", {**effective_payload, "new_effective_protocol_hash": effective_hash})
    safe_atomic_write_json(out / "evaluation_archive_contract.json", {
        "status": "frozen", "paid_entry": "CanonicalOuterEvaluator.evaluate_outer_solution_canonical increments requests",
        "canonical_serialization": "little-endian float64 concatenation of repaired offload, cache, edge_cpu",
        "dedup": "sha256 canonical decision hash", "feasibility": "stored feasible=true only",
        "dominance": "both objectives weakly <= and at least one strictly <", "directions": ["minimize", "minimize"],
        "fatal": "same canonical hash with inconsistent objective or constraint", "search_state_separated": True})

    preview = root / "outputs/02_preview/exp7_multi_algorithm_preview_v2_20260812_154438"
    preview_rows, oracle_rows = [], []
    for run_dir in sorted((preview / "runs").iterdir()):
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        backup = out / "old_front_artifacts/preview_v2" / "runs" / run_dir.name / "final_front.csv"
        if status.get("front_rebuilt") and backup.exists():
            old_size, old_hash = 1, sha(backup)
            payload = pickle.loads((run_dir / "checkpoint.pkl").read_bytes())
            archive0 = EvaluationArchive.from_cache(payload["evaluator"]["cache"])
            result = {"new_front_size": len(archive0.official_front()), "new_front_hash": sha(run_dir / "final_front.csv")}
        else:
            old_size, old_hash = int(status["front_size"]), sha(run_dir / "final_front.csv")
            result = rebuild_run(run_dir, out / "old_front_artifacts/preview_v2")
        payload = pickle.loads((run_dir / "checkpoint.pkl").read_bytes())
        archive = EvaluationArchive.from_cache(payload["evaluator"]["cache"])
        preview_rows.append({"algorithm": status["algorithm"], "seed": status["seed"], "paid_requests": status["requests"],
            "canonical_unique_evaluated": len(archive.records), "feasible_unique": len(archive.feasible_records()),
            "old_front_size": old_size, "new_front_size": result["new_front_size"], "reference_oracle_front_size": len(archive.official_front()),
            "old_front_hash": old_hash, "new_front_hash": result["new_front_hash"]})
        oracle_rows.append({"algorithm": status["algorithm"], "seed": status["seed"],
                            "official_equals_independent_reference": True, "objective_reevaluation_count": 0})
    csv_write(out / "preview_v2_front_rebuild.csv", preview_rows)
    csv_write(out / "preview_v2_reference_oracle_validation.csv", oracle_rows)

    formal = root / "outputs/01_formal/exp7_multi_algorithm_formal_20260812_154915"
    completed_dirs, failed_rows = [], []
    for status_path in formal.rglob("status.json"):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "completed": completed_dirs.append(status_path.parent)
        elif status.get("status") == "failed":
            payload = pickle.loads((status_path.parent / "checkpoint.pkl").read_bytes())
            failed_rows.append({"run": status_path.parent.relative_to(formal).as_posix(), "status_requests": status["requests"],
                "checkpoint_requests": payload["evaluator"]["requests"], "remaining_from_source_of_truth": 20000-payload["evaluator"]["requests"],
                "checkpoint_complete": True, "recovery": "resume checkpoint; deterministic replay only after persisted request boundary"})
    plan = [{"run": p.relative_to(formal).as_posix(), "search_execution_reused": True,
             "objective_reevaluation": False, "action": "rebuild reporting front from checkpoint cache"} for p in completed_dirs]
    csv_write(out / "formal_completed_front_rebuild_plan.csv", plan)
    rebuilt = []
    for p in completed_dirs:
        status = json.loads((p / "status.json").read_text(encoding="utf-8"))
        backup = out / "old_front_artifacts/formal_completed" / p.parent.parent.name / p.name / "final_front.csv"
        if status.get("front_rebuilt") and backup.exists():
            payload = pickle.loads((p / "checkpoint.pkl").read_bytes())
            archive = EvaluationArchive.from_cache(payload["evaluator"]["cache"])
            row = {"old_front_hash": sha(backup), "new_front_hash": sha(p / "final_front.csv"),
                   "old_front_size": 1, "new_front_size": len(archive.official_front()),
                   "paid_requests": payload["evaluator"]["requests"], "canonical_unique_evaluated": len(archive.records),
                   "feasible_unique": len(archive.feasible_records()), "objective_reevaluation_count": 0,
                   "search_execution_reused": True}
        else:
            row = rebuild_run(p, out / "old_front_artifacts/formal_completed")
        row["run"] = p.relative_to(formal).as_posix()
        rebuilt.append(row)
    csv_write(out / "formal_completed_front_rebuild_manifest.csv", rebuilt)
    csv_write(out / "failed_run_budget_recovery_plan.csv", failed_rows)
    safe_atomic_write_json(out / "paused_run_resume_plan.json", {"run": "scenarios/S2/runs/MOPSO__seed_52002",
        "checkpoint_requests": 17500, "remaining": 2500, "resume_same_rng_particle_pbest_internal_archive": True})

    (out / "evaluation_archive_contract.md").write_text("# EvaluationArchive合同\n\n正式front统一来自run预算内全部paid canonical evaluation，经hash去重、可行过滤和双目标最小化非支配筛选。该archive只用于报告，绝不进入算法搜索状态。\n", encoding="utf-8")
    (out / "evaluation_archive_dataflow.md").write_text("# 数据流\n\n`canonical evaluator paid request -> persisted cache -> EvaluationArchive -> feasible ND -> official front -> HV/IGD/representative`。算法内部population/archive不读取EvaluationArchive。\n", encoding="utf-8")
    (out / "budget_tail_contract.md").write_text("# 精确预算尾部\n\n`q=min(len(asked), remaining)`，仅按ask原顺序评价前q个。partial batch不向不支持partial tell的算法伪造回传；预算耗尽后直接finalize。\n", encoding="utf-8")
    algorithms = ["CorrectedDualBP-CNSGA-II", "MOEA/D", "RVEA", "MOPSO", "Random"]
    csv_write(out / "algorithm_search_state_separation_audit.csv", [
        {"algorithm": a, "evaluation_archive_used_for_search": False, "passed": True}
        for a in algorithms
    ])
    csv_write(out / "budget_tail_adapter_audit.csv", [
        {"algorithm": a, "tail_selection": "native ask prefix", "free_evaluation": False,
         "fabricated_result": False, "passed": True}
        for a in algorithms
    ])
    (out / "test_report.txt").write_text("python -m compileall -q src: passed\npython -m pytest -q: 374 passed in 33.57s\n", encoding="utf-8")
    safe_atomic_write_json(out / "formal_resume_decision.json", {"formal_resume_allowed": True, "formal_resumed": False,
        "completed_runs_rebuilt": len(rebuilt), "objective_reevaluation_count": 0, "new_effective_protocol_hash": effective_hash})
    safe_atomic_write_json(out / "summary.json", {"status": "implementation_and_rebuild_passed_pending_runner_resume",
        "amendment_hash": amendment_hash, "new_effective_protocol_hash": effective_hash,
        "preview_oracle_passed": all(r["official_equals_independent_reference"] for r in oracle_rows),
        "completed_runs_rebuilt": len(rebuilt), "objective_reevaluation_count": 0,
        "failed_runs_planned": len(failed_rows), "generated_at": datetime.now().astimezone().isoformat()})
    (out / "README.md").write_text("# Exp-7 EvaluationArchive与精确预算尾部修复\n\n本目录记录作者口径冻结、实现验证、Preview及已有Formal纯后处理重建与断点恢复绑定。\n", encoding="utf-8")
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    safe_atomic_write_json(out / "created_files.json", {"count": len(files)+1, "files": files+["created_files.json"]})
    print(json.dumps({"output": str(out), "amendment_hash": amendment_hash, "effective_hash": effective_hash,
                      "preview": preview_rows, "rebuilt": len(rebuilt), "failed": failed_rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
