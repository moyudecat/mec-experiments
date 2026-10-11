"""Rebuild Exp-7 reporting fronts from persisted paid-evaluation caches."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from mec_dualbp.optimization.evaluation_archive import EvaluationArchive
from mec_dualbp.utils.atomic_io import safe_atomic_write_json


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def representative(front: tuple[dict[str, Any], ...]) -> dict[str, Any] | None:
    if not front:
        return None
    values = np.asarray([[r["Phi_T"], r["Phi_E"]] for r in front], dtype=np.float64)
    lo, hi = values.min(axis=0), values.max(axis=0)
    normalized = np.divide(values - lo, hi - lo, out=np.zeros_like(values), where=hi > lo)
    distance = np.linalg.norm(normalized, axis=1)
    worst = normalized.max(axis=1)
    margin = [float(r["qos_min_possibility"] or 0.0) for r in front]
    index = min(range(len(front)), key=lambda i: (float(distance[i]), -margin[i], float(worst[i]), front[i]["decision_hash"]))
    return {**front[index], "normalized_ideal_distance": float(distance[index]),
            "maximum_normalized_objective": float(worst[index])}


def rebuild_run(run_dir: Path, backup_root: Path | None = None) -> dict[str, Any]:
    status_path, checkpoint_path = run_dir / "status.json", run_dir / "checkpoint.pkl"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    payload = pickle.loads(checkpoint_path.read_bytes())
    archive = EvaluationArchive.from_cache(payload["evaluator"]["cache"])
    front = archive.official_front()
    front_path = run_dir / "final_front.csv"
    old_hash, old_size = _sha(front_path), int(status.get("front_size", 0))
    if backup_root is not None and front_path.exists():
        target = backup_root / run_dir.parent.parent.name / run_dir.name / "final_front.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(front_path, target)
    rows = [{"decision_hash": r["decision_hash"], "objective_1": r["Phi_T"], "objective_2": r["Phi_E"],
             "qos_min_possibility": r["qos_min_possibility"]} for r in front]
    _csv(front_path, rows, ["decision_hash", "objective_1", "objective_2", "qos_min_possibility"])
    metadata = archive.metadata(paid_requests=payload["evaluator"]["requests"], cache_hits=payload["evaluator"]["cache_hits"])
    safe_atomic_write_json(run_dir / "evaluation_archive_metadata.json", metadata)
    safe_atomic_write_json(run_dir / "representative_solution.json", representative(front))
    status.update({"front_size": len(front), "official_front_source": "EvaluationArchive",
                   "front_rebuilt": True, "search_execution_reused": True,
                   "objective_reevaluation": False})
    safe_atomic_write_json(status_path, status)
    return {"run": run_dir.as_posix(), "old_front_hash": old_hash, "new_front_hash": _sha(front_path),
            "old_front_size": old_size, "new_front_size": len(front),
            "paid_requests": payload["evaluator"]["requests"],
            "canonical_unique_evaluated": len(archive.records), "feasible_unique": len(archive.feasible_records()),
            "objective_reevaluation_count": 0, "search_execution_reused": True}
