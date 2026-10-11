from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mec_dualbp.experiments.exp3_ablation_protocol import (
    CORRECTED_HASH, ENERGY_MODEL_HASH, TIME_MODEL_HASH, build_protocol,
)
from mec_dualbp.models.ablation import DirectBoundsMLP, UnifiedCenterRadiusMLP, interval_loss
from mec_dualbp.training.ablation import (
    ABLATION_SEEDS, assert_ablation_data_access, bind_unified_energy_by_base,
    energy_f_invariance_diagnostic, future_training_tasks, stable_ordered_union,
)


def test_stable_union_contract() -> None:
    union = stable_ordered_union(("f", "C", "Gamma"), ("C", "P_r", "Gamma"))
    assert union == ("f", "C", "Gamma", "P_r")
    assert len(union) == len(set(union))


def test_unified_has_one_shared_trunk_and_valid_intervals() -> None:
    model = UnifiedCenterRadiusMLP(4, (128, 64, 32), 1.0, 0.1)
    assert hasattr(model, "backbone") and model.output.out_features == 4
    out = model(torch.zeros((3, 4)))
    assert torch.all(out["time_upper"] >= out["time_lower"])
    assert torch.all(out["energy_upper"] >= out["energy_lower"])


def test_energy_base_binding_and_repeat_weight() -> None:
    time = [{"base_sample_id": "b1", "role": "Train", "f": f} for f in (1, 2, 3)]
    bound = bind_unified_energy_by_base(time, {"b1": {"role": "Train", "energy_lower": 1.0, "energy_upper": 2.0}})
    assert all(row["energy_repeat_weight"] == pytest.approx(1/3) for row in bound)
    assert sum(row["energy_repeat_weight"] for row in bound) == pytest.approx(1.0)


def test_unified_role_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        bind_unified_energy_by_base([{"base_sample_id": "b", "role": "Train"}],
                                    {"b": {"role": "Test", "energy_lower": 1, "energy_upper": 2}})


def test_energy_f_invariance_diagnostic() -> None:
    rows = [{"base_sample_id": "b", "energy_lower_pred": value, "energy_upper_pred": value + 2}
            for value in (1.0, 1.0, 1.0)]
    result = energy_f_invariance_diagnostic(rows)
    assert result["relative_range_max"] == 0.0


def test_direct_bounds_nonnegative_without_posthoc_order() -> None:
    model = DirectBoundsMLP(2, (4,), 1.0)
    with torch.no_grad():
        model.output.weight.zero_(); model.output.bias[:] = torch.tensor([2.0, -2.0])
    lower, upper = model(torch.zeros((1, 2)))
    assert lower.item() >= 0 and upper.item() >= 0
    assert lower.item() > upper.item()  # 原样保留非法顺序


def test_direct_bounds_order_penalty_and_weights() -> None:
    parts = interval_loss(torch.tensor([2.0]), torch.tensor([1.0]), torch.tensor([0.5]), torch.tensor([2.5]),
                          1.0, lambda_safety=0.25, lambda_order=1.0)
    assert parts.order.item() == pytest.approx(1.0)
    assert parts.total.item() == pytest.approx((parts.fit + 0.25 * parts.safety + parts.order).item())


def test_without_safety_allows_zero_lambda() -> None:
    parts = interval_loss(torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([1.0]), torch.tensor([2.0]),
                          1.0, lambda_safety=0.0)
    assert parts.total.item() == pytest.approx(parts.fit.item())


def test_role_access_guards() -> None:
    assert_ablation_data_access("Train", target_access=True)
    assert_ablation_data_access("Validation", target_access=True)
    assert_ablation_data_access("Calibration", target_access=False)
    assert_ablation_data_access("Test", target_access=False)
    with pytest.raises(ValueError): assert_ablation_data_access("Calibration", target_access=True)
    with pytest.raises(ValueError): assert_ablation_data_access("Test", target_access=True)


def test_fixed_seeds_and_25_tasks() -> None:
    assert ABLATION_SEEDS == (31001, 31002, 31003, 31004, 31005)
    tasks = future_training_tasks()
    assert len(tasks) == 25
    assert {row["variant"] for row in tasks} == {"C_w_o_safety", "D_unified_bp", "E_direct_bounds"}
    assert all(not row["calibration_targets_accessed"] and not row["test_targets_accessed"] for row in tasks)


def test_protocol_generation_and_fairness(tmp_path: Path) -> None:
    project_root = Path.cwd()
    out = tmp_path / "protocol"
    summary = build_protocol(project_root, out)
    variants = json.loads((out / "variant_definitions.json").read_text(encoding="utf-8"))
    schema = json.loads((out / "unified_feature_schema.json").read_text(encoding="utf-8"))
    fairness = json.loads((out / "fairness_audit.json").read_text(encoding="utf-8"))
    baseline = json.loads((out / "baseline_artifact_binding.json").read_text(encoding="utf-8"))
    assert summary["exp3_protocol_decision"] == "passed"
    assert variants["A"]["alpha_correction"] and not variants["B"]["alpha_correction"]
    assert variants["C"]["lambda_safety"] == 0
    assert variants["D"]["single_shared_trunk"]
    assert variants["E"]["lambda_order"] == 1 and not variants["E"]["post_hoc_repair"]
    assert schema["union_dimension"] == 20
    assert schema["union_features"][0] == "edge_cpu_frequency_cycle_per_s"
    assert "terminal_rx_power_w" in schema["union_features"]
    assert fairness["passed"]
    assert baseline["time_model_hash_actual"] == TIME_MODEL_HASH
    assert baseline["energy_model_hash_actual"] == ENERGY_MODEL_HASH
    assert baseline["corrected_dualbp_hash_actual"] == CORRECTED_HASH
    assert not summary["ablation_test_evaluation_allowed"]
