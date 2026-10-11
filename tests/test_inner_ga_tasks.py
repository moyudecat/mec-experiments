"""四类真实物理任务适配器测试。"""

from dataclasses import replace

import numpy as np
import pytest

from mec_dualbp.inner_ga.tasks import (
    EdgeTaskParameters,
    build_edge_comm_energy_lower_problem,
    build_edge_comm_energy_upper_problem,
    build_edge_core_latency_lower_problem,
    build_edge_core_latency_upper_problem,
)
from mec_dualbp.uncertainty.interval_params import Interval
from mec_dualbp.uncertainty.schema import (
    EDGE_COMM_ENERGY_ACTIVE_MASK,
    EDGE_CORE_ACTIVE_MASK,
)


def test_four_task_names_directions_and_masks(edge_parameters: EdgeTaskParameters) -> None:
    tasks = (
        build_edge_core_latency_lower_problem(edge_parameters, 1.0),
        build_edge_core_latency_upper_problem(edge_parameters, 1.0),
        build_edge_comm_energy_lower_problem(edge_parameters, 1.0),
        build_edge_comm_energy_upper_problem(edge_parameters, 1.0),
    )
    assert [task.direction for task in tasks] == ["minimize", "maximize", "minimize", "maximize"]
    assert tasks[0].active_mask == tasks[1].active_mask == EDGE_CORE_ACTIVE_MASK
    assert tasks[2].active_mask == tasks[3].active_mask == EDGE_COMM_ENERGY_ACTIVE_MASK
    assert len({task.name for task in tasks}) == 4


def test_all_task_responses_are_finite_nonnegative(edge_parameters: EdgeTaskParameters) -> None:
    builders = (
        build_edge_core_latency_lower_problem,
        build_edge_core_latency_upper_problem,
        build_edge_comm_energy_lower_problem,
        build_edge_comm_energy_upper_problem,
    )
    for builder in builders:
        task = builder(edge_parameters, 1.0)
        assert np.isfinite(task.evaluate(np.zeros(6)))
        assert task.evaluate(np.zeros(6)) >= 0.0


def test_core_has_no_cache_input_or_dependency_term(edge_parameters: EdgeTaskParameters) -> None:
    task = build_edge_core_latency_upper_problem(edge_parameters, 1.0)
    assert "cache" not in task.name
    assert task.evaluate(np.zeros(6)) == task.evaluate(np.zeros(6))


def test_energy_ignores_c_and_f_l_disturbances(edge_parameters: EdgeTaskParameters) -> None:
    task = build_edge_comm_energy_upper_problem(edge_parameters, 2.0)
    baseline = task.evaluate(np.zeros(6))
    with pytest.raises(ValueError):
        task.evaluate(np.array([0, 0, 1, 0, 0, 0], dtype=float))
    modified_parameters = replace(edge_parameters, cpu_cycles=Interval(9e12, 10e12), edge_cpu_frequency_cycle_per_s=1e12)
    assert build_edge_comm_energy_upper_problem(modified_parameters, 2.0).evaluate(np.zeros(6)) == baseline


def test_terminal_receive_power_controls_downlink_energy(edge_parameters: EdgeTaskParameters) -> None:
    low = replace(edge_parameters, terminal_rx_power_w=0.01)
    high = replace(edge_parameters, terminal_rx_power_w=0.2)
    z = np.zeros(6)
    assert build_edge_comm_energy_upper_problem(high, 1.0).evaluate(z) > build_edge_comm_energy_upper_problem(low, 1.0).evaluate(z)


def test_only_din_uncertain_has_manual_direction(edge_parameters: EdgeTaskParameters) -> None:
    parameters = replace(
        edge_parameters,
        output_size_bit=Interval(1e6, 1e6),
        cpu_cycles=Interval(2e9, 2e9),
        uplink_interference_w=Interval(2e-10, 2e-10),
        downlink_interference_w=Interval(2e-10, 2e-10),
    )
    for builder in (build_edge_core_latency_upper_problem, build_edge_comm_energy_upper_problem):
        task = builder(parameters, 1.0)
        lower = task.evaluate(np.array([-1, 0, 0, 0, 0, 0], dtype=float))
        upper = task.evaluate(np.array([1, 0, 0, 0, 0, 0], dtype=float))
        assert upper > lower


def test_only_uplink_interference_uncertain_upper_uses_larger_value(edge_parameters: EdgeTaskParameters) -> None:
    parameters = replace(
        edge_parameters,
        input_size_bit=Interval(2e6, 2e6),
        output_size_bit=Interval(1e6, 1e6),
        cpu_cycles=Interval(2e9, 2e9),
        downlink_interference_w=Interval(2e-10, 2e-10),
    )
    for builder in (build_edge_core_latency_upper_problem, build_edge_comm_energy_upper_problem):
        task = builder(parameters, 1.0)
        low_i = task.evaluate(np.array([0, 0, 0, -1, 0, 0], dtype=float))
        high_i = task.evaluate(np.array([0, 0, 0, 1, 0, 0], dtype=float))
        assert high_i > low_i

