from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mec_dualbp.experiments.exp7_chain_runner import CorrectedDualBPEngine
from mec_dualbp.experiments.improved_dualbp_v1_blind_test import _infer_improved
from mec_dualbp.models.improved_dualbp import energy_physical_baseline_j
from mec_dualbp.optimization.dualbp_engine_registry import (
    IMPROVED_MODE, LEGACY_MODE, ImprovedCorrectedDualBPEngine,
    create_corrected_dualbp_engine,
)
from mec_dualbp.optimization.outer_contract import (
    CORRECTED_DUALBP_HASH, IMPROVED_CORRECTED_DUALBP_HASH,
    IMPROVED_ENERGY_MODEL_HASH, IMPROVED_TIME_MODEL_HASH, CanonicalSolution,
)
from mec_dualbp.training.calibration import apply_directional_correction

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "configs/scenarios/exp7_s3_formal_v1.json"


def _scenario(*, improved: bool) -> dict:
    value = json.loads(SCENARIO.read_text(encoding="utf-8"))
    value["corrected_dualbp_hash"] = (
        IMPROVED_CORRECTED_DUALBP_HASH if improved else CORRECTED_DUALBP_HASH)
    return value


def _solution(n: int, *, active: bool) -> CanonicalSolution:
    m = np.ones(n) if active else np.zeros(n)
    return CanonicalSolution(m, np.zeros(n), np.full(n, 3.0e9) * m)


@pytest.fixture(scope="module")
def improved_engine() -> ImprovedCorrectedDualBPEngine:
    return create_corrected_dualbp_engine(
        ROOT, _scenario(improved=True), evaluator_mode=IMPROVED_MODE,
        evaluator_hash=IMPROVED_CORRECTED_DUALBP_HASH)


def test_legacy_mode_hash_load_and_output_regression() -> None:
    scenario = _scenario(improved=False)
    historical = CorrectedDualBPEngine(ROOT, scenario)
    registered = create_corrected_dualbp_engine(
        ROOT, scenario, evaluator_mode=LEGACY_MODE, evaluator_hash=CORRECTED_DUALBP_HASH)
    solution = _solution(len(scenario["users"]), active=False)
    expected = historical.evaluate(solution)
    observed = registered.evaluate(solution)
    assert {key: observed[key] for key in expected} == expected
    assert observed["corrected_dualbp_hash"] == CORRECTED_DUALBP_HASH


def test_improved_mode_loads_exact_frozen_hashes(improved_engine) -> None:
    assert improved_engine.corrected_dualbp_hash == IMPROVED_CORRECTED_DUALBP_HASH
    assert improved_engine.time_model_hash == IMPROVED_TIME_MODEL_HASH
    assert improved_engine.energy_model_hash == IMPROVED_ENERGY_MODEL_HASH


def test_mode_hash_substitution_fails_fast() -> None:
    with pytest.raises(RuntimeError, match="mode/hash mismatch"):
        create_corrected_dualbp_engine(
            ROOT, _scenario(improved=True), evaluator_mode=IMPROVED_MODE,
            evaluator_hash=CORRECTED_DUALBP_HASH)


def test_improved_raw_inference_matches_frozen_blind_test_call(improved_engine) -> None:
    user = improved_engine.users[0]
    tx, ex = improved_engine._features(user, improved_engine.scenario["Gamma"], 3.0e9)
    txm, exm = np.asarray([tx]), np.asarray([ex])
    raw_tl, raw_tu = _infer_improved(
        improved_engine.time_model, improved_engine.time_scaler, txm, "Time")
    raw_el, raw_eu = _infer_improved(
        improved_engine.energy_model, improved_engine.energy_scaler, exm, "Energy")
    expected = (*apply_directional_correction(raw_tl, raw_tu, *improved_engine.alpha[:2]),
                *apply_directional_correction(raw_el, raw_eu, *improved_engine.alpha[2:]))
    observed = improved_engine._predict_corrected(txm, exm)
    for left, right in zip(expected, observed):
        np.testing.assert_array_equal(left, right)


def test_terminal_energy_baseline_uses_terminal_receive_power(improved_engine) -> None:
    user = improved_engine.users[0]
    _, features = improved_engine._features(user, improved_engine.scenario["Gamma"], 3.0e9)
    fields = __import__("mec_dualbp.datasets.protocol", fromlist=["ENERGY_FEATURES"]).ENERGY_FEATURES
    index = fields.index("terminal_rx_power_w")
    lower = np.asarray([features], dtype=np.float64)
    higher = lower.copy(); higher[0, index] *= 2.0
    assert energy_physical_baseline_j(higher)[0] > energy_physical_baseline_j(lower)[0]


def test_improved_outputs_are_ordered_finite_deterministic(improved_engine) -> None:
    solution = _solution(len(improved_engine.users), active=True)
    first = improved_engine.evaluate(solution)
    second = improved_engine.evaluate(solution)
    np.testing.assert_array_equal(first["objectives"], second["objectives"])
    assert np.isfinite(first["objectives"]).all()
    assert np.isfinite(first["constraint_violation"])
    assert first["prediction_mode"] == "corr_pred"
    assert first["evaluator_mode"] == IMPROVED_MODE


def test_online_engine_freezes_parameters_and_never_enters_training(improved_engine) -> None:
    assert not improved_engine.time_model.training
    assert not improved_engine.energy_model.training
    assert all(not parameter.requires_grad for parameter in improved_engine.time_model.parameters())
    assert all(not parameter.requires_grad for parameter in improved_engine.energy_model.parameters())
