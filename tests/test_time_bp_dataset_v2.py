from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from mec_dualbp.datasets.protocol import ENERGY_FEATURES
from mec_dualbp.datasets.time_bp_v2 import (
    F_FIELD, audit_manifest, build_manifest, canonical_hash, file_sha256,
    load_parameters, read_csv, role_frequency_values, seed_from_namespace,
)


ROOT = Path(__file__).resolve().parents[1]
PARAM_PATH = ROOT / "configs/edge_cpu_formal_parameters_v1.json"
BASE_PATH = ROOT / "outputs/05_audit/bp_dataset_protocol_20260810_122228/formal_600_manifest.csv"
LABEL_PATH = ROOT / "outputs/01_formal/bp_dataset_labels_formal_20260810_132254/formal_labels.csv"
PARAMS = load_parameters(PARAM_PATH)
BASE = read_csv(BASE_PATH)
LABELS = read_csv(LABEL_PATH)
PARAM_HASH = file_sha256(PARAM_PATH)
V2_HASH = canonical_hash({"test": "frozen-v2"})
ROWS = build_manifest(BASE, LABELS, PARAMS, parameter_config_hash=PARAM_HASH, v2_config_hash=V2_HASH)
AUDIT = audit_manifest(ROWS, BASE, LABELS, PARAMS)


def test_author_frozen_edge_cpu_values() -> None:
    assert PARAMS["f_min_E_cycle_per_s"] == 1.5e9
    assert PARAMS["f_max_E_cycle_per_s"] == 5.0e9
    assert PARAMS["F0_cycle_per_s"] == 20.0e9
    assert PARAMS["default_F_max_cycle_per_s"] == 20.0e9
    assert PARAMS["source_type"] == "author_decision"


def test_active_domain_and_m_zero_semantics() -> None:
    assert PARAMS["active_user_domain_only"] is True
    assert PARAMS["local_user_f_value"] == 0.0
    assert PARAMS["time_bp_called_when_m_equals_zero"] is False
    assert all(row["active_edge_user_only"] and row["m_zero_excluded"] for row in ROWS)


def test_stable_namespace_seed_is_reproducible() -> None:
    assert seed_from_namespace("TIME_V2_TRAIN_LOW") == seed_from_namespace("TIME_V2_TRAIN_LOW")
    assert seed_from_namespace("TIME_V2_TRAIN_LOW") != seed_from_namespace("TIME_V2_TEST_LOW")


def test_role_stratified_values_are_bitwise_reproducible() -> None:
    first = role_frequency_values("Train", 420, 1.5e9, 5e9)
    second = role_frequency_values("Train", 420, 1.5e9, 5e9)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_v2_has_1800_rows_and_three_per_base() -> None:
    assert len(ROWS) == 1800
    counts = Counter(row["base_sample_id"] for row in ROWS)
    assert len(counts) == 600 and set(counts.values()) == {3}


def test_v2_role_counts_are_frozen() -> None:
    assert Counter(row["role"] for row in ROWS) == {
        "Train": 1260, "Validation": 180, "Calibration": 180, "Test": 180}


def test_v2_level_counts_are_balanced() -> None:
    assert Counter(row["f_level_role"] for row in ROWS) == {
        "existing_upper_anchor": 600, "new_low_stratum": 600, "new_mid_high_stratum": 600}


def test_low_and_mid_high_strata_are_legal() -> None:
    low = [float(row[F_FIELD]) for row in ROWS if row["f_level_role"] == "new_low_stratum"]
    high = [float(row[F_FIELD]) for row in ROWS if row["f_level_role"] == "new_mid_high_stratum"]
    assert all(1.5e9 <= value < 3.25e9 for value in low)
    assert all(3.25e9 <= value < 5.0e9 for value in high)


def test_each_role_has_exact_lower_and_upper_anchor() -> None:
    for role in ("Train", "Validation", "Calibration", "Test"):
        values = [float(row[F_FIELD]) for row in ROWS if row["role"] == role]
        assert min(values) == 1.5e9 and max(values) == 5.0e9


def test_no_role_or_base_leakage() -> None:
    groups = defaultdict(list)
    for row in ROWS:
        groups[row["base_sample_id"]].append(row)
    assert all(len({row["role"] for row in group}) == 1 for group in groups.values())
    assert AUDIT["role_leakage"] == AUDIT["base_group_leakage"] == 0


def test_no_duplicate_f_inside_base_or_out_of_domain() -> None:
    assert AUDIT["duplicate_f_inside_base"] == 0
    assert AUDIT["f_below_min"] == AUDIT["f_above_max"] == 0


def test_existing_5ghz_labels_are_reused_and_hash_bound() -> None:
    existing = [row for row in ROWS if row["f_level_role"] == "existing_upper_anchor"]
    assert len(existing) == 600
    assert all(float(row[F_FIELD]) == PARAMS["f_max_E_cycle_per_s"] for row in existing)
    assert all(row["label_source"] == "existing_v1" and row["execution_status"] == "reused" for row in existing)
    assert AUDIT["existing_label_hash_mismatch"] == 0


def test_new_rows_are_queued_and_define_2400_future_tasks() -> None:
    queued = [row for row in ROWS if row["execution_status"] == "queued"]
    assert len(queued) == 1200
    assert all(row["time_lower"] == row["time_upper"] == "" for row in queued)
    assert AUDIT["future_time_tasks"] == 2400


def test_energy_dataset_is_not_expanded_and_has_no_f() -> None:
    assert len(LABELS) == 600
    assert F_FIELD not in ENERGY_FEATURES


def test_adaptive_protocol_hash_is_unchanged() -> None:
    expected = "041cbe449ab59ee6a43156ae90030921f70f06c0a21a265c40457a614ad9b561"
    assert {row["adaptive_protocol_hash"] for row in ROWS} == {expected}


def test_manifest_full_audit_passes() -> None:
    assert AUDIT["passed"] is True
    assert AUDIT["non_f_feature_changed_inside_base"] == 0
    assert AUDIT["base_missing"] == AUDIT["base_extra"] == 0
