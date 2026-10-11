"""无全局随机状态的完全可复现性测试。"""

import random

from mec_dualbp.physics.channel import uplink_rate_bit_per_s
from mec_dualbp.physics.local_response import local_response_bounds_si
from mec_dualbp.uncertainty.interval_params import Interval


def test_identical_inputs_produce_identical_outputs() -> None:
    args = (1e6, 0.2, 1e-6, 1e-10, 2e-10)
    assert uplink_rate_bit_per_s(*args) == uplink_rate_bit_per_s(*args)
    cycles, frequency = Interval(2e9, 6e9), Interval(1e9, 3e9)
    first = local_response_bounds_si(cycles, frequency, 1e-28, 1.5)
    for _ in range(20):
        assert local_response_bounds_si(cycles, frequency, 1e-28, 1.5) == first


def test_solver_does_not_consume_global_random_state() -> None:
    random.seed(9127)
    expected = [random.random() for _ in range(5)]
    random.seed(9127)
    local_response_bounds_si(Interval(1.0, 4.0), Interval(2.0, 5.0), 0.25, 1.1)
    actual = [random.random() for _ in range(5)]
    assert actual == expected
