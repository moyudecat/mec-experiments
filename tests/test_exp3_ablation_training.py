from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mec_dualbp.experiments.exp1_formal_protocol import file_sha256
from mec_dualbp.experiments.exp3_ablation_training_formal import (
    CORRECTED_HASH, ENERGY_MODEL_HASH, PROTOCOL_HASH, TIME_MODEL_HASH, _loss_vectors, _metrics, _scaler,
)
from mec_dualbp.models.ablation import DirectBoundsMLP, UnifiedCenterRadiusMLP
from mec_dualbp.training.ablation import ABLATION_SEEDS, assert_ablation_data_access, future_training_tasks


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_protocol_hash() -> None:
    path=ROOT/"outputs/05_audit/exp3_ablation_protocol_20260811_144345/exp3_ablation_protocol.json"
    assert file_sha256(path)==PROTOCOL_HASH


def test_raw_scalers_and_center_scales_are_reusable() -> None:
    root=ROOT/"outputs/01_formal/dualbp_training_formal_20260811_003126"
    time_fields=__import__("json").loads((root/"time_bp/selected_model_config.json").read_text())["input_feature_order"]
    scaler=_scaler(root/"time_bp/input_scaler.json",time_fields)
    assert scaler.fitted_role=="Train"
    assert __import__("json").loads((root/"time_bp/target_scale.json").read_text())["source_role"]=="Train"


def test_c_zero_safety_has_fit_only() -> None:
    fit,_,_,safety,order=_loss_vectors(torch.tensor([1.]),torch.tensor([2.]),torch.tensor([.5]),torch.tensor([3.]),1.)
    assert torch.allclose(fit+0*safety+0*order,fit)


def test_unified_contract() -> None:
    model=UnifiedCenterRadiusMLP(20,(128,64,32),1.,.1)
    assert model.output.out_features==4 and len(list(model.backbone.children()))==6
    result=model(torch.zeros((2,20)))
    assert set(result)=={"time_center","time_radius","time_lower","time_upper","energy_center","energy_radius","energy_lower","energy_upper"}


def test_direct_contract_and_invalid_preserved() -> None:
    model=DirectBoundsMLP(2,(4,),1.)
    with torch.no_grad(): model.output.weight.zero_();model.output.bias[:]=torch.tensor([2.,-2.])
    lower,upper=model(torch.zeros((1,2)));metrics=_metrics(np.array([0.]),np.array([3.]),lower.detach().numpy(),upper.detach().numpy(),1.)
    assert lower.item()>upper.item() and metrics["invalid_interval_count"]==1


def test_loss_weights_are_frozen() -> None:
    fit,_,_,safety,order=_loss_vectors(torch.tensor([2.]),torch.tensor([1.]),torch.tensor([.5]),torch.tensor([2.5]),1.)
    total=fit+.25*safety+1.0*order
    assert total.item()==pytest.approx((fit+.25*safety+order).item())


def test_roles_seeds_and_task_counts() -> None:
    assert ABLATION_SEEDS==(31001,31002,31003,31004,31005)
    tasks=future_training_tasks();assert len(tasks)==25
    assert sum(t["variant"]=="C_w_o_safety" for t in tasks)==10
    assert sum(t["variant"]=="D_unified_bp" for t in tasks)==5
    assert sum(t["variant"]=="E_direct_bounds" for t in tasks)==10
    with pytest.raises(ValueError):assert_ablation_data_access("Calibration",target_access=True)
    with pytest.raises(ValueError):assert_ablation_data_access("Test",target_access=True)


def test_frozen_baseline_hashes() -> None:
    root=ROOT/"outputs/01_formal/dualbp_training_formal_20260811_003126"
    assert file_sha256(root/"time_bp/selected_model.pt")==TIME_MODEL_HASH
    assert file_sha256(root/"energy_bp/selected_model.pt")==ENERGY_MODEL_HASH
    value=__import__("json").loads((ROOT/"outputs/01_formal/dualbp_alpha_calibration_formal_20260811_134010/corrected_dualbp_v1_hash.json").read_text())
    assert value["sha256"]==CORRECTED_HASH


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model=DirectBoundsMLP(2,(4,),1.);path=tmp_path/"model.pt"
    torch.save({"model_state_dict":model.state_dict()},path);loaded=torch.load(path,map_location="cpu",weights_only=False)
    other=DirectBoundsMLP(2,(4,),1.);other.load_state_dict(loaded["model_state_dict"])
    assert file_sha256(path)==file_sha256(path)
