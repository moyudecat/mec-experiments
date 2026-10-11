"""边缘时延点响应测试。"""

import inspect
import math

import pytest

from mec_dualbp.physics.latency import (
    cache_miss_latency_s,
    edge_core_latency_point_s,
    edge_total_latency_point_s,
)


def test_cache_miss_latency_formula() -> None:
    assert cache_miss_latency_s(8.0e6, 4.0e6) == pytest.approx(2.0)


def test_edge_core_latency_formula_and_interface_isolation() -> None:
    value = edge_core_latency_point_s(2e6, 1e6, 3e9, 6e9, 1e6, 2e6)
    assert value == pytest.approx(3.0)
    parameters = inspect.signature(edge_core_latency_point_s).parameters
    assert "cache_decision" not in parameters
    assert "cache_miss_latency_s" not in parameters


def test_edge_total_cache_hit_and_miss() -> None:
    assert edge_total_latency_point_s(1.25, 1, 0.75) == pytest.approx(1.25)
    assert edge_total_latency_point_s(1.25, 0, 0.75) == pytest.approx(2.0)


@pytest.mark.parametrize("cache_decision", [-1, 2, 0.0, 1.0, True, False])
def test_edge_total_rejects_nonbinary_cache_decisions(cache_decision: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        edge_total_latency_point_s(1.0, cache_decision, 2.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "call",
    [
        lambda: cache_miss_latency_s(-1.0, 1.0),
        lambda: cache_miss_latency_s(1.0, 0.0),
        lambda: edge_core_latency_point_s(1.0, 0.0, 1.0, 1.0, 1.0, 1.0),
        lambda: edge_core_latency_point_s(1.0, 1.0, 1.0, math.inf, 1.0, 1.0),
        lambda: edge_total_latency_point_s(math.nan, 0, 1.0),
    ],
)
def test_invalid_latency_inputs_raise(call: object) -> None:
    with pytest.raises(ValueError):
        call()  # type: ignore[operator]

