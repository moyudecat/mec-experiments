"""Versioned online Corrected DualBP engines with strict frozen-asset gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from mec_dualbp.experiments.exp1_formal_protocol import file_sha256
from mec_dualbp.experiments.exp1_preview_protocol import canonical_json_hash
from mec_dualbp.experiments.exp7_chain_runner import CorrectedDualBPEngine
from mec_dualbp.models.improved_dualbp import (
    ImprovedResidualDualBP, energy_physical_baseline_j, time_physical_baseline_s,
)
from mec_dualbp.optimization.outer_contract import (
    CORRECTED_DUALBP_HASH, ENERGY_MODEL_HASH, IMPROVED_CORRECTED_DUALBP_HASH,
    IMPROVED_ENERGY_MODEL_HASH, IMPROVED_TIME_MODEL_HASH, TIME_MODEL_HASH,
    LEGACY_OBJECTIVE_MODE, OBJECTIVE_NAMES,
)
from mec_dualbp.physics.local_response import local_response_bounds_si
from mec_dualbp.training.calibration import apply_directional_correction
from mec_dualbp.training.dualbp import TrainOnlyStandardScaler
from mec_dualbp.uncertainty.interval_params import Interval

LEGACY_MODE = "legacy_corrected_dualbp"
IMPROVED_MODE = "improved_corrected_dualbp_v1"
IMPROVED_TRAINING_DIR = "outputs/01_formal/improved_dualbp_v1_5seed_training_20260818_235906"
IMPROVED_FREEZE_FILE = (
    "outputs/01_formal/improved_dualbp_v1_blind_test_20260819_002144/"
    "final_freeze/improved_corrected_dualbp_hash.json"
)


def _model_content_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _scaler(payload: Mapping[str, Any]) -> TrainOnlyStandardScaler:
    return TrainOnlyStandardScaler(tuple(payload["mean"]), tuple(payload["scale"]),
                                   tuple(payload["constant_feature"]), payload["fitted_role"])


class ImprovedCorrectedDualBPEngine(CorrectedDualBPEngine):
    """Canonical outer engine using the frozen physics-residual Time/Energy models."""

    evaluator_mode = IMPROVED_MODE
    corrected_dualbp_hash = IMPROVED_CORRECTED_DUALBP_HASH
    time_model_hash = IMPROVED_TIME_MODEL_HASH
    energy_model_hash = IMPROVED_ENERGY_MODEL_HASH

    def __init__(self, project: Path, scenario: Mapping[str, Any]) -> None:
        torch.set_num_threads(1)
        freeze_path = project / IMPROVED_FREEZE_FILE
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        identity = freeze["identity"]
        if (freeze.get("improved_corrected_dualbp_hash") != self.corrected_dualbp_hash
                or canonical_json_hash(identity) != self.corrected_dualbp_hash):
            raise RuntimeError("Improved Corrected DualBP identity hash drift")
        if (identity.get("time_model_hash") != self.time_model_hash
                or identity.get("energy_model_hash") != self.energy_model_hash):
            raise RuntimeError("Improved model identity drift")
        if scenario.get("corrected_dualbp_hash") != self.corrected_dualbp_hash:
            raise RuntimeError("scenario is not bound to Improved Corrected DualBP V1")

        training = project / IMPROVED_TRAINING_DIR
        candidates = json.loads((training / "formal_candidate_models.json").read_text(encoding="utf-8"))
        self.time_model, self.time_scaler = self._load_model(training, candidates["Time"], "Time")
        self.energy_model, self.energy_scaler = self._load_model(training, candidates["Energy"], "Energy")
        alpha = identity["alpha"]
        self.alpha = (
            float(alpha["Time"]["alpha_minus"]), float(alpha["Time"]["alpha_plus"]),
            float(alpha["Energy"]["alpha_minus"]), float(alpha["Energy"]["alpha_plus"]),
        )
        self.scenario = scenario
        self.users = list(scenario["users"])
        self.local_bounds = [
            local_response_bounds_si(
                Interval(float(user["C_cycle_lower"]), float(user["C_cycle_upper"])),
                Interval(float(user["F_L_cycle_per_s_lower"]), float(user["F_L_cycle_per_s_upper"])),
                float(user["kappa_SI"]), float(scenario["Gamma"]),
            ) for user in self.users
        ]

    @staticmethod
    def _load_model(training: Path, candidate: Mapping[str, Any], network: str
                    ) -> tuple[ImprovedResidualDualBP, TrainOnlyStandardScaler]:
        expected_model = IMPROVED_TIME_MODEL_HASH if network == "Time" else IMPROVED_ENERGY_MODEL_HASH
        if candidate.get("model_hash") != expected_model:
            raise RuntimeError(f"{network} manifest model hash drift")
        checkpoint_path = training / str(candidate["checkpoint_path"])
        if file_sha256(checkpoint_path) != candidate.get("checkpoint_sha256"):
            raise RuntimeError(f"{network} checkpoint file hash drift")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = ImprovedResidualDualBP(checkpoint["input_dim"], checkpoint["delta_center_scale"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if _model_content_hash(model) != expected_model:
            raise RuntimeError(f"{network} checkpoint model-content hash drift")
        return model, _scaler(checkpoint["input_scaler"])

    def _predict_corrected(self, time_features: np.ndarray,
                           energy_features: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                                  np.ndarray, np.ndarray]:
        time_baseline = time_physical_baseline_s(time_features)
        energy_baseline = energy_physical_baseline_j(energy_features)
        with torch.inference_mode():
            _, _, raw_tl, raw_tu = self.time_model(
                torch.as_tensor(self.time_scaler.transform(time_features).astype(np.float32)),
                torch.as_tensor(time_baseline, dtype=torch.float32),
            )
            _, _, raw_el, raw_eu = self.energy_model(
                torch.as_tensor(self.energy_scaler.transform(energy_features).astype(np.float32)),
                torch.as_tensor(energy_baseline, dtype=torch.float32),
            )
        corr_tl, corr_tu = apply_directional_correction(
            raw_tl.numpy().astype(np.float64), raw_tu.numpy().astype(np.float64),
            self.alpha[0], self.alpha[1])
        corr_el, corr_eu = apply_directional_correction(
            raw_el.numpy().astype(np.float64), raw_eu.numpy().astype(np.float64),
            self.alpha[2], self.alpha[3])
        return corr_tl, corr_tu, corr_el, corr_eu

    def evaluate(self, solution):
        result = super().evaluate(solution)
        result.update({
            "evaluator_mode": self.evaluator_mode,
            "corrected_dualbp_hash": self.corrected_dualbp_hash,
            "time_model_hash": self.time_model_hash,
            "energy_model_hash": self.energy_model_hash,
        })
        return result


class LegacyRegisteredEngine:
    """Metadata adapter around the untouched historical online engine."""

    evaluator_mode = LEGACY_MODE
    corrected_dualbp_hash = CORRECTED_DUALBP_HASH
    time_model_hash = TIME_MODEL_HASH
    energy_model_hash = ENERGY_MODEL_HASH

    def __init__(self, project: Path, scenario: Mapping[str, Any]) -> None:
        self._engine = CorrectedDualBPEngine(project, scenario)

    def evaluate(self, solution):
        result = dict(self._engine.evaluate(solution))
        result.update({
            "evaluator_mode": self.evaluator_mode,
            "corrected_dualbp_hash": self.corrected_dualbp_hash,
            "time_model_hash": self.time_model_hash,
            "energy_model_hash": self.energy_model_hash,
        })
        return result


ENGINE_REGISTRY = {
    LEGACY_MODE: (CORRECTED_DUALBP_HASH, LegacyRegisteredEngine),
    IMPROVED_MODE: (IMPROVED_CORRECTED_DUALBP_HASH, ImprovedCorrectedDualBPEngine),
}


def create_corrected_dualbp_engine(project: Path, scenario: Mapping[str, Any], *,
                                   evaluator_mode: str, evaluator_hash: str,
                                   objective_mode: str = LEGACY_OBJECTIVE_MODE):
    """Create a frozen engine; reject unknown modes or mode/hash substitution."""
    try:
        registered_hash, engine_type = ENGINE_REGISTRY[evaluator_mode]
    except KeyError as error:
        raise ValueError(f"unknown evaluator mode: {evaluator_mode}") from error
    if evaluator_hash != registered_hash:
        raise RuntimeError("evaluator mode/hash mismatch")
    if objective_mode not in OBJECTIVE_NAMES:
        raise ValueError(f"unsupported objective mode: {objective_mode}")
    engine = engine_type(project, scenario)
    engine.objective_mode = objective_mode
    return engine
