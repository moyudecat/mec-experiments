"""上下行Shannon速率测试。"""

import math

import pytest

from mec_dualbp.physics.channel import downlink_rate_bit_per_s, uplink_rate_bit_per_s


def _uplink(interference: float = 1e-10, bandwidth: float = 1e6, power: float = 0.2, gain: float = 1e-6) -> float:
    return uplink_rate_bit_per_s(bandwidth, power, gain, 1e-10, interference)


def test_uplink_matches_log2_formula() -> None:
    expected = 1e6 * math.log2(1.0 + 0.2 * 1e-6 / (1e-10 + 2e-10))
    assert _uplink(interference=2e-10) == pytest.approx(expected)


def test_downlink_matches_log2_formula_and_uses_bs_power() -> None:
    expected = 2e6 * math.log2(1.0 + 1.5 * 2e-6 / (2e-10 + 3e-10))
    actual = downlink_rate_bit_per_s(2e6, 1.5, 2e-6, 2e-10, 3e-10)
    assert actual == pytest.approx(expected)


def test_rate_decreases_with_interference() -> None:
    assert _uplink(1e-11) > _uplink(1e-9)


def test_rate_is_nondecreasing_in_bandwidth_power_and_gain() -> None:
    baseline = _uplink()
    assert _uplink(bandwidth=2e6) > baseline
    assert _uplink(power=0.4) > baseline
    assert _uplink(gain=2e-6) > baseline


@pytest.mark.parametrize(
    "args",
    [
        (0.0, 0.2, 1e-6, 1e-10, 1e-10),
        (1e6, -0.1, 1e-6, 1e-10, 1e-10),
        (1e6, 0.2, -1e-6, 1e-10, 1e-10),
        (1e6, 0.2, 1e-6, -1e-10, 1e-10),
        (1e6, 0.2, 1e-6, 0.0, 0.0),
    ],
)
def test_invalid_domains_raise(args: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        uplink_rate_bit_per_s(*args)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_channel_inputs_raise(bad: float) -> None:
    with pytest.raises(ValueError):
        uplink_rate_bit_per_s(1e6, 0.2, 1e-6, 1e-10, bad)


def test_zero_signal_is_rejected_because_rate_must_be_positive() -> None:
    with pytest.raises(ValueError):
        uplink_rate_bit_per_s(1e6, 0.0, 1e-6, 1e-10, 1e-10)
    with pytest.raises(ValueError):
        downlink_rate_bit_per_s(1e6, 1.0, 0.0, 1e-10, 1e-10)

