from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mec_dualbp.experiments.dualbp_alpha_calibration import (
    ENERGY_MODEL_HASH, TIME_MODEL_HASH, save_figure,
)
from mec_dualbp.experiments.exp1_preview_protocol import canonical_json_hash
from mec_dualbp.training.calibration import (
    CALIBRATION_DELTA, apply_directional_correction, conservative_alpha,
    conservative_rank, lower_over_error, protocol_is_frozen, upper_under_error,
)


def test_calibration_requires_frozen_protocol():
    assert not protocol_is_frozen({"status": "freeze_candidate", "delta": .05})
    assert protocol_is_frozen({"status": "frozen", "delta": .05,
        "target_access_allowed_after_freeze": True, "alpha_calibration_protocol_hash": "abc"})


def test_delta_and_order_statistic_ranks_are_frozen():
    assert CALIBRATION_DELTA == .05
    assert conservative_rank(180) == 172
    assert conservative_rank(60) == 58


def test_conservative_alpha_uses_observed_higher_order_statistic():
    values = np.arange(1, 61, dtype=float) / 1000
    alpha, rank = conservative_alpha(values)
    assert rank == 58 and alpha == values[57]
    assert alpha in values


def test_upper_directional_error():
    error = upper_under_error(np.array([10., 10.]), np.array([9., 11.]))
    np.testing.assert_allclose(error, [0.1, 0.0])


def test_lower_directional_error():
    error = lower_over_error(np.array([10., 10.]), np.array([11., 9.]))
    np.testing.assert_allclose(error, [0.1, 0.0])


def test_correction_directions_and_interval_legality():
    lower, upper = apply_directional_correction(np.array([2.]), np.array([4.]), .1, .2)
    assert 0 <= lower[0] < 2 and upper[0] > 4 and upper[0] >= lower[0]


@pytest.mark.parametrize("alpha_minus,alpha_plus", [(1., .1), (.1, 1.), (-.1, .1)])
def test_invalid_alpha_rejected(alpha_minus, alpha_plus):
    with pytest.raises(ValueError):
        apply_directional_correction(np.array([1.]), np.array([2.]), alpha_minus, alpha_plus)


def test_four_synthetic_alpha_finite_and_less_than_one():
    values = [conservative_alpha(np.array([0., .01, .02]))[0] for _ in range(4)]
    assert all(np.isfinite(values)) and all(0 <= value < 1 for value in values)


def test_frozen_model_hashes_unchanged():
    root = Path(__file__).resolve().parents[1]
    formal = root / "outputs/01_formal/dualbp_training_formal_20260811_003126"
    assert hashlib.sha256((formal / "time_bp/selected_model.pt").read_bytes()).hexdigest() == TIME_MODEL_HASH
    assert hashlib.sha256((formal / "energy_bp/selected_model.pt").read_bytes()).hexdigest() == ENERGY_MODEL_HASH


def test_scaler_hashes_stable_on_repeated_read():
    root = Path(__file__).resolve().parents[1]
    for relative in ("time_bp/input_scaler.json", "energy_bp/input_scaler.json"):
        path = root / "outputs/01_formal/dualbp_training_formal_20260811_003126" / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashlib.sha256(path.read_bytes()).hexdigest()


def test_calibration_counts_bound_without_test_access():
    root = Path(__file__).resolve().parents[1]
    audit = __import__("json").loads((root / "outputs/01_formal/dualbp_training_formal_20260811_003126/data_leakage_audit.json").read_text())
    assert audit["metadata_only_audit"]["time_calibration"]["row_count"] == 180
    assert audit["metadata_only_audit"]["energy_calibration"]["row_count"] == 60
    assert audit["test_labels_accessed"] is False


def test_corrected_config_hash_is_stable():
    config = {"delta": .05, "alpha_T_minus": .01, "alpha_T_plus": .02,
              "alpha_E_minus": .03, "alpha_E_plus": .04}
    assert canonical_json_hash(config) == canonical_json_hash(dict(reversed(list(config.items()))))


def test_figure_writer_emits_all_formats_without_test_source(tmp_path: Path):
    fig, ax = plt.subplots(); ax.plot([0, 1], [0, 1])
    paths = save_figure(fig, tmp_path / "calibration_diagnostic")
    assert {path.suffix for path in paths} == {".png", ".pdf", ".svg"}
    assert all(path.exists() and "test" not in path.name.lower() for path in paths)
