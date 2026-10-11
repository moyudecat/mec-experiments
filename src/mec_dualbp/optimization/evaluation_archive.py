"""Algorithm-independent official archive for paid Exp-7 evaluations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def canonical_decision_hash(key: tuple[float, ...]) -> str:
    """Hash the frozen canonical float64 serialization (m, c, f)."""
    return hashlib.sha256(np.asarray(key, dtype="<f8").tobytes()).hexdigest()


def _nondominated_indices(objectives: np.ndarray) -> np.ndarray:
    """Exact minimization filter (no epsilon), preserving legacy 2D ordering."""
    if objectives.ndim != 2 or objectives.shape[1] < 2:
        raise ValueError("objective matrix must have at least two columns")
    if objectives.shape[1] != 2:
        keep = np.ones(len(objectives), dtype=bool)
        for index, point in enumerate(objectives):
            keep[index] = not np.any(
                np.all(objectives <= point, axis=1)
                & np.any(objectives < point, axis=1))
        return np.flatnonzero(keep)
    order = np.lexsort((objectives[:, 1], objectives[:, 0]))
    selected: list[int] = []
    best_prior_e = np.inf
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        t_value = objectives[order[cursor], 0]
        while end < len(order) and objectives[order[end], 0] == t_value:
            end += 1
        group = order[cursor:end]
        group_min_e = float(objectives[group, 1].min())
        if group_min_e < best_prior_e:
            selected.extend(int(i) for i in group if objectives[i, 1] == group_min_e)
            best_prior_e = group_min_e
        cursor = end
    return np.asarray(sorted(selected), dtype=int)


@dataclass(frozen=True)
class EvaluationArchive:
    """Read-only view over canonical-unique paid evaluation records."""

    records: tuple[dict[str, Any], ...]
    objective_names: tuple[str, ...] = ("Phi_T", "Phi_E")

    @classmethod
    def from_cache(cls, cache: Mapping[tuple[float, ...], Mapping[str, Any]]) -> "EvaluationArchive":
        records: list[dict[str, Any]] = []
        seen: dict[str, dict[str, Any]] = {}
        archive_names: tuple[str, ...] | None = None
        for raw_key, raw_result in cache.items():
            key = tuple(float(value) for value in raw_key)
            digest = canonical_decision_hash(key)
            result = dict(raw_result)
            objective = np.asarray(result.get("objectives"), dtype=np.float64)
            cv = float(result.get("constraint_violation", np.nan))
            raw_names = result.get("objective_names")
            names = tuple(raw_names) if raw_names is not None else (
                ("Phi_T", "Phi_E") if objective.shape == (2,) else ())
            if (objective.ndim != 1 or objective.size < 2 or len(names) != objective.size
                    or len(set(names)) != len(names) or not all(isinstance(name, str) for name in names)
                    or not np.isfinite(objective).all() or not np.isfinite(cv)):
                raise ValueError("evaluation archive requires named finite objective and constraint records")
            if archive_names is None:
                archive_names = names
            elif names != archive_names:
                raise ValueError("evaluation archive objective schema mismatch")
            record = {
                "decision_hash": digest, "canonical_key": key,
                "objectives": tuple(float(value) for value in objective),
                **{name: float(value) for name, value in zip(names, objective)},
                "constraint_violation": cv,
                "feasible": bool(result.get("feasible", cv <= 1e-12)),
                "qos_min_possibility": result.get("qos_min_possibility"),
            }
            for field in ("T_lower", "T_upper", "E_lower", "E_upper",
                          "c_T", "r_T", "c_E", "r_E", "Phi_T", "Phi_E",
                          "c7_violating_user_count", "c7_min_margin"):
                if field in result:
                    record[field] = result[field]
            previous = seen.get(digest)
            if previous is not None:
                comparable = (previous["canonical_key"] == record["canonical_key"]
                              and previous["objectives"] == record["objectives"]
                              and previous["constraint_violation"] == record["constraint_violation"]
                              and previous["feasible"] == record["feasible"])
                if not comparable:
                    raise RuntimeError("fatal deterministic consistency error for canonical decision")
                continue
            seen[digest] = record
            records.append(record)
        return cls(tuple(records), archive_names or ("Phi_T", "Phi_E"))

    def feasible_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(row for row in self.records if row["feasible"] is True)

    def official_front(self) -> tuple[dict[str, Any], ...]:
        feasible = self.feasible_records()
        if not feasible:
            return ()
        objectives = np.asarray([row["objectives"] for row in feasible], dtype=np.float64)
        selected = _nondominated_indices(objectives)
        return tuple(feasible[int(index)] for index in selected)

    def metadata(self, *, paid_requests: int, cache_hits: int) -> dict[str, Any]:
        front = self.official_front()
        return {
            "candidate_scope": "all_paid_evaluated_canonical_unique_within_run_budget",
            "paid_requests": int(paid_requests), "cache_hits": int(cache_hits),
            "canonical_unique_evaluated": len(self.records),
            "feasible_unique": len(self.feasible_records()), "official_front_size": len(front),
            "objective_names": list(self.objective_names),
            "objective_count": len(self.objective_names),
            "deterministic_consistency_passed": True,
            "search_state_separated": True,
        }
