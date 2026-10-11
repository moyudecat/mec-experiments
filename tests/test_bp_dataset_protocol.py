from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from mec_dualbp.datasets.protocol import (
    ENERGY_FEATURES, FORMAL_ROLE_COUNTS, LABEL_FIELDS, REPLICATES_PER_STRATUM,
    ROLE_SEED_BASES, TIME_FEATURES, audit_role_balance, build_label_record,
    center_radius, generate_formal_dataset_samples, generate_smoke_samples,
    pairwise_duplicate_audit, sample_manifest_row,
)
from mec_dualbp.experiments.adaptive_label_protocol import PROTOCOL_NAME


FORMAL = generate_formal_dataset_samples()
SMOKE = generate_smoke_samples()


def _fake_tasks() -> dict[str, dict[str, object]]:
    return {
        "edge_core_latency_lower": {"adaptive_label": 1.0, "selected_label_route": "b4_stable", "b4_relative_response_std": 0.001, "gate_triggered": False, "gate_trigger_reasons": []},
        "edge_core_latency_upper": {"adaptive_label": 3.0, "selected_label_route": "b4_stable", "b4_relative_response_std": 0.002, "gate_triggered": False, "gate_trigger_reasons": []},
        "edge_comm_energy_lower": {"adaptive_label": 2.0, "selected_label_route": "b4_stable", "b4_relative_response_std": 0.003, "gate_triggered": False, "gate_trigger_reasons": []},
        "edge_comm_energy_upper": {"adaptive_label": 4.0, "selected_label_route": "b4_stable", "b4_relative_response_std": 0.004, "gate_triggered": False, "gate_trigger_reasons": []},
    }


def test_formal_role_counts_are_420_60_60_60() -> None:
    assert Counter(sample.role for sample in FORMAL) == FORMAL_ROLE_COUNTS


@pytest.mark.parametrize("role", ["Train", "Validation", "Calibration", "Test"])
def test_each_role_is_gamma_type_balanced(role: str) -> None:
    counts = Counter((sample.gamma, sample.sample_type) for sample in FORMAL if sample.role == role)
    assert len(counts) == 30
    assert set(counts.values()) == {REPLICATES_PER_STRATUM[role]}


def test_role_balance_audit_passes() -> None:
    assert audit_role_balance(FORMAL)["passed"] is True


def test_role_seed_namespaces_are_mutually_exclusive() -> None:
    assert len(set(ROLE_SEED_BASES.values())) == 4
    assert all(sample.generation_seed.startswith(f"BP_{sample.role.upper()}_") for sample in FORMAL)
    assert len({sample.generation_seed for sample in FORMAL}) == 600


def test_formal_ids_and_content_are_unique() -> None:
    assert len({sample.sample_id for sample in FORMAL}) == 600
    assert len({sample.content_hash for sample in FORMAL}) == 600


def test_duplicate_and_near_duplicate_audit() -> None:
    duplicate, near, summary = pairwise_duplicate_audit(FORMAL, {})
    assert duplicate == []
    assert near == []
    assert summary["Current"] == {"exact_duplicate_pairs": 0, "near_duplicate_pairs": 0}


def test_time_features_exclude_cache() -> None:
    assert all("cache" not in field.lower() for field in TIME_FEATURES)


def test_energy_features_exclude_cpu_cycles_edge_cpu_and_cache() -> None:
    assert not any(field.startswith("C_cycle") for field in ENERGY_FEATURES)
    assert "edge_cpu_frequency_cycle_per_s" not in ENERGY_FEATURES
    assert all("cache" not in field.lower() for field in ENERGY_FEATURES)


def test_gamma_is_in_both_bp_inputs() -> None:
    assert "Gamma" in TIME_FEATURES
    assert "Gamma" in ENERGY_FEATURES


def test_terminal_receive_and_bs_transmit_power_have_distinct_energy_roles() -> None:
    assert "terminal_rx_power_w" in ENERGY_FEATURES
    assert "downlink_bs_tx_power_w" in ENERGY_FEATURES
    assert "terminal_rx_power_w" not in TIME_FEATURES


def test_center_radius_conversion() -> None:
    assert center_radius(1.0, 3.0) == (2.0, 1.0)


def test_center_radius_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError):
        center_radius(2.0, 1.0)


def test_label_record_has_legal_bounds_and_all_schema_fields() -> None:
    row = build_label_record(FORMAL[0], _fake_tasks(), dataset_config_hash="dataset", label_protocol_config_hash="label")
    assert set(LABEL_FIELDS).issubset(row)
    assert row["time_lower"] <= row["time_upper"]
    assert row["energy_lower"] <= row["energy_upper"]
    assert row["time_center"] >= row["time_radius"] >= 0
    assert row["energy_center"] >= row["energy_radius"] >= 0


def test_alpha_fields_are_null_before_calibration() -> None:
    row = build_label_record(FORMAL[0], _fake_tasks(), dataset_config_hash="dataset", label_protocol_config_hash="label")
    assert all(row[name] is None for name in ("alpha_T_lower", "alpha_T_upper", "alpha_E_lower", "alpha_E_upper"))
    assert row["alpha_calibration_status"] == "not_started"


def test_alpha_status_is_metadata_not_nullable_coefficient() -> None:
    row = build_label_record(FORMAL[0], _fake_tasks(), dataset_config_hash="dataset", label_protocol_config_hash="label")
    nullable = ("alpha_T_lower", "alpha_T_upper", "alpha_E_lower", "alpha_E_upper")
    assert all(row[name] is None for name in nullable)
    assert row["alpha_calibration_status"] is not None


def test_raw_and_corrected_predictions_are_null_before_training() -> None:
    row = build_label_record(FORMAL[0], _fake_tasks(), dataset_config_hash="dataset", label_protocol_config_hash="label")
    assert all(value is None for name, value in row.items() if name.startswith(("raw_pred_", "corr_pred_")))


def test_smoke_samples_cannot_enter_formal_roles() -> None:
    rows = [sample_manifest_row(sample, "dataset") for sample in SMOKE]
    assert all(row["not_for_formal_bp_dataset"] is True for row in rows)
    assert all(row["not_for_bp_training"] and row["not_for_bp_validation"] and row["not_for_bp_calibration"] and row["not_for_bp_test"] for row in rows)


def test_formal_manifest_starts_without_labels() -> None:
    rows = [sample_manifest_row(sample, "dataset") for sample in FORMAL]
    assert all(row["label_status"] == "not_generated" and row["execution_status"] == "queued" for row in rows)


def test_adaptive_protocol_hash_binding_is_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    design_dirs = sorted((root / "outputs/05_audit").glob("bp_dataset_protocol_*/label_protocol_binding.json"))
    assert design_dirs
    binding = json.loads(design_dirs[-1].read_text(encoding="utf-8"))
    confirmation = json.loads((root / "outputs/01_formal/adaptive_b4_reference_confirmation_20260810_000412/config_hash.json").read_text(encoding="utf-8"))
    assert binding["label_protocol_name"] == PROTOCOL_NAME
    assert binding["confirmation_config_hash"] == confirmation["config_hash"]
    assert binding["binding_consistent"] is True


def test_formal_generation_force_escalation_is_not_configured() -> None:
    root = Path(__file__).resolve().parents[1]
    config_paths = sorted((root / "outputs/05_audit").glob("bp_dataset_protocol_*/dataset_config.json"))
    assert config_paths
    config = json.loads(config_paths[-1].read_text(encoding="utf-8"))
    assert config.get("force_escalation", False) is False


def test_no_third_qos_network_or_certified_fields_in_schema() -> None:
    assert not any("qos" in field.lower() or "certified" in field.lower() for field in LABEL_FIELDS)
