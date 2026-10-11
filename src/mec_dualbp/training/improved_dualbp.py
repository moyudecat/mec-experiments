"""Train/Validation-only trainer for frozen Improved DualBP V1."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from mec_dualbp.models.improved_dualbp import (
    ImprovedResidualDualBP, improved_dualbp_loss, parameter_count,
)
from mec_dualbp.training.dualbp import regression_metrics, seed_everything


def train_only_rms(values: np.ndarray, *, role: str) -> float:
    if role != "Train":
        raise ValueError("output scale may only be fitted on Train")
    x = np.asarray(values, dtype=np.float64)
    scale = float(np.sqrt(np.mean(x * x, dtype=np.float64)))
    if not math.isfinite(scale) or scale <= 1e-12:
        raise ValueError("finite positive Train-only RMS scale required")
    return scale


def train_improved_one(x_train: np.ndarray, baseline_train: np.ndarray,
    lower_train: np.ndarray, upper_train: np.ndarray, x_validation: np.ndarray,
    baseline_validation: np.ndarray, lower_validation: np.ndarray, upper_validation: np.ndarray,
    *, seed: int, delta_center_scale: float, radius_scale: float,
    response_center_scale: float, batch_size: int, max_epochs: int = 1500,
    scheduler_patience: int = 25, early_stopping_patience: int = 100,
    min_delta: float = 1e-5) -> tuple[ImprovedResidualDualBP, dict[str, Any], list[dict[str, float]]]:
    seed_everything(seed)
    model = ImprovedResidualDualBP(x_train.shape[1], delta_center_scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5,
        patience=scheduler_patience, min_lr=1e-5)
    xt, bt = torch.as_tensor(x_train, dtype=torch.float32), torch.as_tensor(baseline_train, dtype=torch.float32)
    lt, ut = torch.as_tensor(lower_train, dtype=torch.float32), torch.as_tensor(upper_train, dtype=torch.float32)
    xv, bv = torch.as_tensor(x_validation, dtype=torch.float32), torch.as_tensor(baseline_validation, dtype=torch.float32)
    lv, uv = torch.as_tensor(lower_validation, dtype=torch.float32), torch.as_tensor(upper_validation, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    best_score, best_epoch, stale, best_state = math.inf, 0, 0, None
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train(); order = torch.randperm(len(xt), generator=generator); total = 0.0
        for start in range(0, len(xt), batch_size):
            idx = order[start:start + batch_size]
            center, radius, lower, upper = model(xt[idx], bt[idx])
            parts = improved_dualbp_loss(center, radius, lower, upper, bt[idx], lt[idx], ut[idx],
                delta_center_scale=delta_center_scale, radius_scale=radius_scale,
                response_center_scale=response_center_scale)
            optimizer.zero_grad(set_to_none=True); parts.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            total += float(parts.total.detach()) * len(idx)
        model.eval()
        with torch.no_grad(): _, _, lower, upper = model(xv, bv)
        current = regression_metrics(lower_validation, upper_validation,
            lower.numpy(), upper.numpy(), response_center_scale)
        score = current["validation_selection_score"]; scheduler.step(score)
        history.append({"epoch": float(epoch), "train_loss": total / len(xt),
            "validation_selection_score": score, "learning_rate": float(optimizer.param_groups[0]["lr"])})
        if score < best_score - min_delta:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= early_stopping_patience: break
    if best_state is None: raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): center, radius, lower, upper = model(xv, bv)
    metrics = regression_metrics(lower_validation, upper_validation, lower.numpy(), upper.numpy(), response_center_scale)
    metrics.update({"training_seed": seed, "best_epoch": best_epoch, "epochs_ran": len(history),
        "parameter_count": parameter_count(x_train.shape[1]),
        "nonfinite_predictions": int(not all(torch.isfinite(x).all() for x in (center, radius, lower, upper))),
        "invalid_intervals": int(torch.sum((lower < 0) | (upper < lower)).item()),
        "center_floor_activation_count": int(torch.sum(center <= 1.000001e-12).item())})
    return model, metrics, history
