"""Exp-1任务manifest、统一Schema和边界配对测试。"""

import csv
import json

import pytest

from mec_dualbp.experiments.exp1_inner_ga_label_budget_smoke import (
    build_exp1_smoke_samples,
    build_exp1_task_manifest,
)
from mec_dualbp.experiments.exp1_smoke_protocol import (
    METHOD_RESULT_CORE_FIELDS,
    validate_boundary_pairs,
    validate_method_result_schema,
    validate_task_manifest,
)


def _method_row(method: str) -> dict[str, object]:
    return {
        "sample_id": "S",
        "gamma": 1,
        "task_name": "edge_core_latency_upper",
        "direction": "maximize",
        "method_name": method,
        "response_value": 1.0,
        "best_z_json": "[0,0,0,0,0,0]",
        "runtime_seconds": 0.1,
        "evaluation_requests": 2,
        "unique_evaluations": 1,
        "cache_hits": 1,
        "endpoint_snap_count": 0,
        "endpoint_snap_max_ulp": 0,
        "cache_collision_error": False,
        "status": "completed",
    }


def test_48_task_manifest_is_complete_and_unique() -> None:
    samples = build_exp1_smoke_samples()
    manifest = build_exp1_task_manifest(samples)
    assert len(samples) == 12 and len(manifest) == 48
    assert sum(sample.gamma == 1 for sample in samples) == 6
    assert sum(sample.gamma == 3 for sample in samples) == 6
    assert len({row["task_id"] for row in manifest}) == 48
    validate_task_manifest(manifest)


def test_standard_high_lhs_share_core_schema() -> None:
    for method in (
        "standard_candidate_smoke",
        "high_budget_candidate_smoke",
        "dense_feasible_lhs_smoke",
    ):
        row = _method_row(method)
        validate_method_result_schema(row)
        assert all(field in row for field in METHOD_RESULT_CORE_FIELDS)


def test_missing_schema_field_is_rejected() -> None:
    row = _method_row("standard_candidate_smoke")
    del row["response_value"]
    with pytest.raises(ValueError):
        validate_method_result_schema(row)


def test_boundary_pair_check_rejects_inversion() -> None:
    valid = [{"sample_id": "S", "response_family": "latency", "lower_response": 1.0, "upper_response": 2.0}]
    validate_boundary_pairs(valid)
    invalid = [{"sample_id": "S", "response_family": "latency", "lower_response": 3.0, "upper_response": 2.0}]
    with pytest.raises(ValueError):
        validate_boundary_pairs(invalid)


def test_summary_and_csv_row_counts_can_be_reconciled(tmp_path) -> None:
    rows = build_exp1_task_manifest(build_exp1_smoke_samples())
    csv_path = tmp_path / "manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"task_instances": len(rows)}), encoding="utf-8")
    with csv_path.open(encoding="utf-8") as stream:
        csv_count = len(list(csv.DictReader(stream)))
    assert json.loads(summary_path.read_text(encoding="utf-8"))["task_instances"] == csv_count == 48

