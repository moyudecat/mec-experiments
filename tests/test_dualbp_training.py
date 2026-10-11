from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from mec_dualbp.datasets.protocol import ENERGY_FEATURES, TIME_FEATURES
from mec_dualbp.models.dualbp import (
    ARCHITECTURES, LAMBDA_SAFETY_CANDIDATES, TRAINING_SEEDS,
    ConstrainedCenterRadiusMLP, dualbp_loss,
)
from mec_dualbp.training.dualbp import (
    TrainOnlyStandardScaler, assert_loader_role, f_stratum, regression_metrics,
    select_configuration, state_dict_sha256, validate_bounds,
)


def test_frozen_feature_dimensions_and_gamma():
    assert len(TIME_FEATURES) == 19 and len(ENERGY_FEATURES) == 17
    assert "Gamma" in TIME_FEATURES and "Gamma" in ENERGY_FEATURES
    assert "edge_cpu_frequency_cycle_per_s" in TIME_FEATURES
    assert "edge_cpu_frequency_cycle_per_s" not in ENERGY_FEATURES
    assert "C_cycle_lower" not in ENERGY_FEATURES and "cache_state" not in ENERGY_FEATURES


def test_scaler_train_only_and_constant_feature():
    values = np.array([[1.0, 5.0], [3.0, 5.0]])
    scaler = TrainOnlyStandardScaler.fit(values, role="Train")
    assert scaler.constant_feature == (False, True)
    assert scaler.scale[1] == 1.0 and np.isfinite(scaler.transform(values)).all()
    with pytest.raises(ValueError):
        TrainOnlyStandardScaler.fit(values, role="Validation")


@pytest.mark.parametrize("role", ["Calibration", "Test"])
def test_sealed_roles_cannot_enter_loader(role):
    with pytest.raises(ValueError):
        assert_loader_role(role, weight_update=False)
    with pytest.raises(ValueError):
        assert_loader_role(role, weight_update=True)


def test_train_and_validation_loader_roles():
    assert_loader_role("Train", weight_update=True)
    assert_loader_role("Train", weight_update=False)
    assert_loader_role("Validation", weight_update=False)
    with pytest.raises(ValueError):
        assert_loader_role("Validation", weight_update=True)


def test_center_radius_conversion_and_invalid_labels():
    center, radius, scale = validate_bounds(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    np.testing.assert_allclose(center, [2.0, 3.0]); np.testing.assert_allclose(radius, [1.0, 1.0])
    assert scale == 2.5
    with pytest.raises(ValueError):
        validate_bounds(np.array([2.0]), np.array([1.0]))


def test_constrained_head_always_produces_legal_interval():
    torch.manual_seed(1)
    model = ConstrainedCenterRadiusMLP(19, ARCHITECTURES["M1"], 2.0)
    center, radius, lower, upper = model(torch.randn(32, 19))
    assert torch.all(center > 0) and torch.all(radius >= 0) and torch.all(radius <= center)
    assert torch.all(lower >= 0) and torch.all(upper >= lower)


def _loss(lower_hat, upper_hat, lower_true=1.0, upper_true=3.0):
    lower_hat = torch.tensor([lower_hat]); upper_hat = torch.tensor([upper_hat])
    center, radius = (lower_hat + upper_hat) / 2, (upper_hat - lower_hat) / 2
    return dualbp_loss(center, radius, lower_hat, upper_hat, torch.tensor([lower_true]),
                       torch.tensor([upper_true]), 2.0, 1.0).safety.item()


def test_safety_upper_direction():
    assert _loss(1.0, 2.5) > 0
    assert _loss(1.0, 3.5) == 0


def test_safety_lower_direction():
    assert _loss(1.5, 3.0) > 0
    assert _loss(0.5, 3.0) == 0


def test_preregistered_grid_is_exact():
    assert ARCHITECTURES == {"M1": (64, 32), "M2": (64, 64, 32), "M3": (128, 64, 32)}
    assert LAMBDA_SAFETY_CANDIDATES == (0.25, 0.5, 1.0)
    assert TRAINING_SEEDS == (31001, 31002, 31003, 31004, 31005)


def test_selection_score_formula():
    true_l, true_u = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    pred_l, pred_u = np.array([1.1, 1.8]), np.array([2.8, 4.1])
    metrics = regression_metrics(true_l, true_u, pred_l, pred_u, 2.5)
    expected = (.35 * metrics["lower_wmape"] + .35 * metrics["upper_wmape"]
                + .15 * metrics["upper_under_relative_magnitude"]
                + .15 * metrics["lower_over_relative_magnitude"])
    assert metrics["validation_selection_score"] == pytest.approx(expected)


def test_configuration_selection_uses_five_seed_mean():
    rows = []
    for architecture in ARCHITECTURES:
        for lam in LAMBDA_SAFETY_CANDIDATES:
            for seed in TRAINING_SEEDS:
                score = 0.01 if (architecture, lam) == ("M2", .5) else 0.02
                rows.append({"architecture": architecture, "lambda_safety": lam, "training_seed": seed,
                             "validation_selection_score": score, "input_dim": 19})
    assert select_configuration(rows) == ("M2", .5)


@pytest.mark.parametrize("value,expected", [(1.5e9, "Low"), (3.249e9, "Low"),
                                              (3.25e9, "Mid-high"), (4.999e9, "Mid-high"),
                                              (5e9, "Upper-anchor")])
def test_f_stratum_boundaries(value, expected):
    assert f_stratum(value) == expected


def test_checkpoint_readable_and_hash_reproducible(tmp_path: Path):
    model = ConstrainedCenterRadiusMLP(17, ARCHITECTURES["M1"], 1.0)
    path = tmp_path / "model.pt"; torch.save({"model_state_dict": model.state_dict()}, path)
    first = state_dict_sha256(path); second = state_dict_sha256(path)
    assert first == second and len(first) == 64
    assert "model_state_dict" in torch.load(path, weights_only=False)


def test_dataset_bindings_exist_and_hashes_match():
    root = Path(__file__).resolve().parents[1]
    time_manifest = root / "outputs/01_formal/time_bp_dataset_v2_labels_formal_20260810_194719/frozen_v2_manifest_copy.csv"
    energy_manifest = root / "outputs/01_formal/bp_dataset_labels_formal_20260810_132254/frozen_manifest_copy.csv"
    assert hashlib.sha256(time_manifest.read_bytes()).hexdigest() == "946eebc32e69573fb5294e75eab7f5bea7cfcd765fa99d5c73d7e1c4dddbebe0"
    assert hashlib.sha256(energy_manifest.read_bytes()).hexdigest() == "e48ffc62f10ce4f46c4986154c9e31fbbe82bbb4a1b38e9a2bcf8455c23d6f40"
