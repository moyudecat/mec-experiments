"""区间中心、半径和扰动恢复测试。"""

import math

import pytest

from mec_dualbp.uncertainty.interval_params import (
    Interval,
    interval_center_radius,
    recover_interval_value,
)


def test_interval_center_radius_and_properties() -> None:
    interval = Interval(2.0, 6.0)
    assert interval.lower == 2.0
    assert interval.upper == 6.0
    assert interval.center == 4.0
    assert interval.radius == 2.0
    assert interval_center_radius(2.0, 6.0) == (4.0, 2.0)


@pytest.mark.parametrize("z,expected", [(-1.0, 2.0), (0.0, 4.0), (1.0, 6.0), (0.25, 4.5)])
def test_recovery_stays_inside_interval(z: float, expected: float) -> None:
    assert recover_interval_value(2.0, 6.0, z) == expected
    assert Interval(2.0, 6.0).recover(z) == expected


def test_degenerate_interval_is_allowed() -> None:
    interval = Interval(3.0, 3.0)
    assert interval.center == 3.0
    assert interval.radius == 0.0
    assert interval.recover(-1.0) == interval.recover(1.0) == 3.0


def test_invalid_endpoint_order_is_not_swapped() -> None:
    with pytest.raises(ValueError):
        Interval(2.0, 1.0)
    with pytest.raises(ValueError):
        interval_center_radius(2.0, 1.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_interval_inputs_are_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        Interval(bad, 2.0)
    with pytest.raises(ValueError):
        recover_interval_value(1.0, 2.0, bad)


@pytest.mark.parametrize("z", [-1.0001, 1.0001])
def test_out_of_range_disturbance_is_rejected(z: float) -> None:
    with pytest.raises(ValueError):
        recover_interval_value(1.0, 2.0, z)


def test_interval_is_immutable() -> None:
    interval = Interval(1.0, 2.0)
    with pytest.raises((AttributeError, TypeError)):
        interval.lower = 0.0  # type: ignore[misc]

