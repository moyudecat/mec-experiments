"""移动终端通信能耗测试。"""

import math

import pytest

from mec_dualbp.physics.energy import edge_comm_energy_point_j


def test_edge_comm_energy_formula() -> None:
    actual = edge_comm_energy_point_j(0.4, 2e6, 1e6, 0.1, 4e6, 2e6)
    assert actual == pytest.approx(1.0)


def test_downlink_term_uses_terminal_receive_power() -> None:
    low = edge_comm_energy_point_j(0.2, 1e6, 1e6, 0.05, 2e6, 1e6)
    high = edge_comm_energy_point_j(0.2, 1e6, 1e6, 0.15, 2e6, 1e6)
    assert high - low == pytest.approx(0.2)


def test_zero_data_and_power_are_valid() -> None:
    assert edge_comm_energy_point_j(0.0, 0.0, 1.0, 0.0, 0.0, 1.0) == 0.0


@pytest.mark.parametrize(
    "args",
    [
        (-1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        (1.0, -1.0, 1.0, 1.0, 1.0, 1.0),
        (1.0, 1.0, 0.0, 1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0, -1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0, 1.0, 1.0, math.nan),
    ],
)
def test_invalid_energy_inputs_raise(args: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        edge_comm_energy_point_j(*args)

