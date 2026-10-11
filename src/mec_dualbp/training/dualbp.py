"""冻结 DualBP 训练协议的通用数据、指标和选模实现。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from mec_dualbp.models.dualbp import ARCHITECTURES, ConstrainedCenterRadiusMLP, dualbp_loss, parameter_count

SCALER_EPSILON = 1e-12


def assert_loader_role(role: str, *, weight_update: bool) -> None:
    allowed = {"Train"} if weight_update else {"Train", "Validation"}
    if role not in allowed:
        raise ValueError(f"role {role} is sealed for this training loader")


@dataclass(frozen=True, slots=True)
class TrainOnlyStandardScaler:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    constant_feature: tuple[bool, ...]
    fitted_role: str = "Train"

    @classmethod
    def fit(cls, values: np.ndarray, *, role: str = "Train", epsilon: float = SCALER_EPSILON) -> "TrainOnlyStandardScaler":
        if role != "Train":
            raise ValueError("scaler may only be fitted on Train")
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] == 0 or not np.isfinite(array).all():
            raise ValueError("finite two-dimensional Train features required")
        mean = array.mean(axis=0, dtype=np.float64)
        std = array.std(axis=0, dtype=np.float64)
        constant = std < epsilon
        scale = np.where(constant, 1.0, std)
        return cls(tuple(map(float, mean)), tuple(map(float, scale)), tuple(map(bool, constant)))

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        result = (array - np.asarray(self.mean)) / np.asarray(self.scale)
        if not np.isfinite(result).all():
            raise ValueError("nonfinite scaled input")
        return result

    def as_dict(self, fields: Sequence[str]) -> dict[str, Any]:
        return {"semantics": "x_scaled=(x-mean_train)/scale_train", "fitted_role": self.fitted_role,
                "epsilon": SCALER_EPSILON, "fields": list(fields), "mean": list(self.mean),
                "scale": list(self.scale), "constant_feature": list(self.constant_feature)}


def validate_bounds(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    lo, hi = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    if lo.shape != hi.shape or lo.ndim != 1 or not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("finite one-dimensional bounds required")
    center, radius = (lo + hi) * 0.5, (hi - lo) * 0.5
    if np.any(lo < 0) or np.any(hi < lo) or np.any(center < radius) or np.any(radius < 0):
        raise ValueError("invalid interval label")
    scale = float(np.median(center))
    if not math.isfinite(scale) or scale <= SCALER_EPSILON:
        scale = float(np.mean(np.abs(center)))
    if not math.isfinite(scale) or scale <= SCALER_EPSILON:
        raise ValueError("invalid center scale")
    return center, radius, scale


def f_stratum(value: float) -> str:
    f = float(value)
    if 1.5e9 <= f < 3.25e9:
        return "Low"
    if 3.25e9 <= f < 5.0e9:
        return "Mid-high"
    if f == 5.0e9:
        return "Upper-anchor"
    raise ValueError("f outside frozen Time-BP domain")


def regression_metrics(lower_true: np.ndarray, upper_true: np.ndarray,
                       lower_hat: np.ndarray, upper_hat: np.ndarray,
                       center_scale: float) -> dict[str, float]:
    lt, ut, lp, up = map(lambda x: np.asarray(x, dtype=np.float64),
                         (lower_true, upper_true, lower_hat, upper_hat))
    if not all(x.shape == lt.shape for x in (ut, lp, up)) or lt.size == 0:
        raise ValueError("aligned nonempty arrays required")
    def side(prefix: str, true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
        error = pred - true
        denominator = max(float(np.sum(np.abs(true))), 1e-12)
        ss_total = float(np.sum((true - true.mean()) ** 2))
        return {f"{prefix}_mae": float(np.mean(np.abs(error))),
                f"{prefix}_rmse": float(np.sqrt(np.mean(error ** 2))),
                f"{prefix}_r2": 1.0 - float(np.sum(error ** 2)) / max(ss_total, 1e-12),
                f"{prefix}_wmape": float(np.sum(np.abs(error))) / denominator}
    result = {**side("lower", lt, lp), **side("upper", ut, up)}
    result.update({"center_mae": float(np.mean(np.abs((lp + up - lt - ut) * 0.5))),
                   "radius_mae": float(np.mean(np.abs((up - lp - ut + lt) * 0.5))),
                   "uer": float(np.mean(up < ut)), "ler": float(np.mean(lp > lt)),
                   "upper_under_relative_magnitude": float(np.mean(np.maximum(ut - up, 0))) / (center_scale + 1e-12),
                   "lower_over_relative_magnitude": float(np.mean(np.maximum(lp - lt, 0))) / (center_scale + 1e-12)})
    result["validation_selection_score"] = (0.35 * result["lower_wmape"] + 0.35 * result["upper_wmape"]
        + 0.15 * result["upper_under_relative_magnitude"] + 0.15 * result["lower_over_relative_magnitude"])
    return result


def select_configuration(rows: Sequence[Mapping[str, Any]]) -> tuple[str, float]:
    grouped: dict[tuple[str, float], list[float]] = {}
    for row in rows:
        grouped.setdefault((str(row["architecture"]), float(row["lambda_safety"])), []).append(
            float(row["validation_selection_score"]))
    if set(grouped) != {(a, l) for a in ARCHITECTURES for l in (0.25, 0.5, 1.0)}:
        raise ValueError("incomplete preregistered grid")
    summaries = []
    for (architecture, lam), scores in grouped.items():
        summaries.append((float(np.mean(scores)), float(np.std(scores)),
                          parameter_count(int(rows[0]["input_dim"]), ARCHITECTURES[architecture]), architecture, lam))
    summaries.sort(key=lambda x: x[0])
    best_mean = summaries[0][0]
    tied = [row for row in summaries if row[0] - best_mean < 1e-4]
    tied.sort(key=lambda x: (x[1], x[2], x[3], x[4]))
    return tied[0][3], tied[0][4]


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass


def train_one(
    x_train: np.ndarray, lower_train: np.ndarray, upper_train: np.ndarray,
    x_validation: np.ndarray, lower_validation: np.ndarray, upper_validation: np.ndarray,
    *, architecture: str, lambda_safety: float, seed: int, center_scale: float,
    batch_size: int, max_epochs: int = 1500, scheduler_patience: int = 25,
    early_stopping_patience: int = 100, min_delta: float = 1e-5,
) -> tuple[ConstrainedCenterRadiusMLP, dict[str, Any], list[dict[str, float]]]:
    seed_everything(seed)
    model = ConstrainedCenterRadiusMLP(x_train.shape[1], ARCHITECTURES[architecture], center_scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                           patience=scheduler_patience, min_lr=1e-5)
    xt = torch.as_tensor(x_train, dtype=torch.float32)
    lot = torch.as_tensor(lower_train, dtype=torch.float32)
    upt = torch.as_tensor(upper_train, dtype=torch.float32)
    xv = torch.as_tensor(x_validation, dtype=torch.float32)
    lov = torch.as_tensor(lower_validation, dtype=torch.float32)
    upv = torch.as_tensor(upper_validation, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    best_score, best_epoch, stale = math.inf, 0, 0
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train(); order = torch.randperm(len(xt), generator=generator); total = 0.0
        for start in range(0, len(xt), batch_size):
            indices = order[start:start + batch_size]
            center, radius, lower, upper = model(xt[indices])
            parts = dualbp_loss(center, radius, lower, upper, lot[indices], upt[indices],
                                center_scale, lambda_safety)
            optimizer.zero_grad(set_to_none=True); parts.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            total += float(parts.total.detach()) * len(indices)
        model.eval()
        with torch.no_grad():
            _, _, lower, upper = model(xv)
        metrics = regression_metrics(lower_validation, upper_validation,
                                     lower.numpy(), upper.numpy(), center_scale)
        score = metrics["validation_selection_score"]; scheduler.step(score)
        history.append({"epoch": float(epoch), "train_loss": total / len(xt),
                        "validation_selection_score": score,
                        "learning_rate": float(optimizer.param_groups[0]["lr"])})
        if score < best_score - min_delta:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= early_stopping_patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        center, radius, lower, upper = model(xv)
    metrics = regression_metrics(lower_validation, upper_validation, lower.numpy(), upper.numpy(), center_scale)
    metrics.update({"architecture": architecture, "lambda_safety": lambda_safety, "training_seed": seed,
                    "input_dim": x_train.shape[1], "best_epoch": best_epoch, "epochs_ran": len(history),
                    "parameter_count": parameter_count(x_train.shape[1], ARCHITECTURES[architecture]),
                    "nonfinite_predictions": int(not np.isfinite(lower.numpy()).all() or not np.isfinite(upper.numpy()).all()),
                    "invalid_intervals": int(np.sum((lower.numpy() < 0) | (upper.numpy() < lower.numpy())))})
    return model, metrics, history


def state_dict_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
