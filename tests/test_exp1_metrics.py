"""Exp-1方向感知漏界指标测试。"""

import pytest

from mec_dualbp.experiments.exp1_smoke_protocol import (
    direction_aware_miss_gap,
    relative_miss_gap,
    symmetric_relative_difference,
)


def test_upper_miss_gap_direction() -> None:
    assert direction_aware_miss_gap("maximize", 8.0, 10.0) == 2.0
    assert direction_aware_miss_gap("maximize", 12.0, 10.0) == 0.0


def test_lower_miss_gap_direction() -> None:
    assert direction_aware_miss_gap("minimize", 8.0, 6.0) == 2.0
    assert direction_aware_miss_gap("minimize", 4.0, 6.0) == 0.0


def test_relative_gap_uses_preconfigured_stable_denominator() -> None:
    assert relative_miss_gap(2e-12, 0.0, 1e-9) == pytest.approx(2e-3)
    assert relative_miss_gap(2.0, 4.0, 1e-9) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        relative_miss_gap(1.0, 0.0, 0.0)


def test_symmetric_relative_difference_is_order_independent() -> None:
    assert symmetric_relative_difference(8.0, 10.0, 1e-12) == pytest.approx(0.2)
    assert symmetric_relative_difference(10.0, 8.0, 1e-12) == pytest.approx(0.2)

