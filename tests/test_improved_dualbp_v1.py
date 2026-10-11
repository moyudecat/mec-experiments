from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from mec_dualbp.datasets.protocol import ENERGY_FEATURES, TIME_FEATURES
from mec_dualbp.models.dualbp import parameter_count as original_parameter_count
from mec_dualbp.models.improved_dualbp import (
    ENERGY_DELTA_CENTER_SCALE, ENERGY_RADIUS_SCALE, HIDDEN_LAYERS,
    TIME_DELTA_CENTER_SCALE, TIME_RADIUS_SCALE, ImprovedResidualDualBP,
    energy_physical_baseline_j, improved_dualbp_loss, parameter_count,
    time_physical_baseline_s,
)
from mec_dualbp.physics.channel import downlink_rate_bit_per_s, uplink_rate_bit_per_s
from mec_dualbp.physics.energy import edge_comm_energy_point_j
from mec_dualbp.physics.latency import edge_core_latency_point_s
from mec_dualbp.training.improved_dualbp import train_only_rms


def _row(fields, values):
    return np.asarray([[values[name] for name in fields]], dtype=np.float64)


COMMON = {
    "D_in_bit_lower": 1e6, "D_in_bit_upper": 3e6, "D_out_bit_lower": 4e5,
    "D_out_bit_upper": 8e5, "I_u_w_lower": 1e-10, "I_u_w_upper": 3e-10,
    "I_d_w_lower": 2e-10, "I_d_w_upper": 6e-10, "uplink_bandwidth_hz": 1.2e6,
    "downlink_bandwidth_hz": 2e6, "uplink_terminal_tx_power_w": .22,
    "downlink_bs_tx_power_w": 1.1, "terminal_rx_power_w": .09,
    "uplink_channel_gain_linear": 1.1e-6, "downlink_channel_gain_linear": 1.8e-6,
    "noise_power_w": 1e-10, "Gamma": 3.0,
}


def test_time_baseline_matches_frozen_point_formula():
    v = {**COMMON, "edge_cpu_frequency_cycle_per_s": 3e9,
         "C_cycle_lower": 1e9, "C_cycle_upper": 3e9}
    got = time_physical_baseline_s(_row(TIME_FEATURES, v))[0]
    ru = uplink_rate_bit_per_s(1.2e6, .22, 1.1e-6, 1e-10, 2e-10)
    rd = downlink_rate_bit_per_s(2e6, 1.1, 1.8e-6, 1e-10, 4e-10)
    assert got == pytest.approx(edge_core_latency_point_s(2e6, ru, 2e9, 3e9, 6e5, rd))


def test_energy_baseline_uses_pd_for_rate_and_pr_for_energy():
    got = energy_physical_baseline_j(_row(ENERGY_FEATURES, COMMON))[0]
    ru = uplink_rate_bit_per_s(1.2e6, .22, 1.1e-6, 1e-10, 2e-10)
    rd = downlink_rate_bit_per_s(2e6, 1.1, 1.8e-6, 1e-10, 4e-10)
    assert got == pytest.approx(edge_comm_energy_point_j(.22, 2e6, ru, .09, 6e5, rd))


def test_baseline_module_has_no_ga_or_lhs_dependency():
    source = inspect.getsource(__import__("mec_dualbp.models.improved_dualbp", fromlist=["*"]))
    assert "inner_ga" not in source and "Dense-LHS" not in source and "dense_lhs" not in source


def test_residual_reconstruction_and_legal_radius_head():
    model = ImprovedResidualDualBP(19, TIME_DELTA_CENTER_SCALE)
    with torch.no_grad():
        for p in model.parameters(): p.zero_()
    baseline = torch.tensor([1.25, 2.5])
    center, radius, lower, upper = model(torch.zeros(2, 19), baseline)
    torch.testing.assert_close(center, baseline)
    torch.testing.assert_close(radius, center * .5)
    assert torch.all(radius >= 0) and torch.all(radius <= center)
    assert torch.all(lower >= 0) and torch.all(upper >= lower)


def test_radius_scale_is_live_in_normalized_loss():
    c, r, lo, up, b = [torch.tensor([x]) for x in (2., .8, 1.2, 2.8, 1.5)]
    truth_lo, truth_up = torch.tensor([1.0]), torch.tensor([3.0])
    first = improved_dualbp_loss(c, r, lo, up, b, truth_lo, truth_up,
        delta_center_scale=.5, radius_scale=.2, response_center_scale=2.).radius
    second = improved_dualbp_loss(c, r, lo, up, b, truth_lo, truth_up,
        delta_center_scale=.5, radius_scale=.4, response_center_scale=2.).radius
    assert first.item() != second.item()


def test_frozen_scales_and_parameter_count_identity():
    assert TIME_DELTA_CENTER_SCALE == 0.001984608838275729
    assert TIME_RADIUS_SCALE == 0.4359502024619836
    assert ENERGY_DELTA_CENTER_SCALE == 0.0004869135714533095
    assert ENERGY_RADIUS_SCALE == 0.015859765350198552
    assert HIDDEN_LAYERS == (128, 64, 32)
    assert parameter_count(19) == original_parameter_count(19, HIDDEN_LAYERS) == 12962
    assert parameter_count(17) == original_parameter_count(17, HIDDEN_LAYERS) == 12706


def test_output_scale_is_train_only():
    assert train_only_rms(np.array([3.0, 4.0]), role="Train") == pytest.approx(np.sqrt(12.5))
    for role in ("Validation", "Calibration", "Test"):
        with pytest.raises(ValueError):
            train_only_rms(np.array([1.0]), role=role)


def test_improved_has_independent_identity_from_original():
    from mec_dualbp.models.dualbp import ConstrainedCenterRadiusMLP
    assert ImprovedResidualDualBP is not ConstrainedCenterRadiusMLP
    assert set(ImprovedResidualDualBP(19, TIME_DELTA_CENTER_SCALE).state_dict()) != \
        set(ConstrainedCenterRadiusMLP(19, HIDDEN_LAYERS, .88).state_dict())
