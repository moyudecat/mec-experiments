"""六维比例预算修复测试。"""

import numpy as np
import pytest

from mec_dualbp.uncertainty.budget_set import (
    is_user_budget_feasible,
    proportional_budget_repair,
    validate_user_disturbance,
)


def test_feasible_vector_is_preserved_without_aliasing() -> None:
    source = np.array([0.1, -0.2, 0.0, 0.3, 0.0, 0.0])
    original = source.copy()
    repaired = proportional_budget_repair(source, 1.0)
    assert np.array_equal(repaired, source)
    assert repaired is not source
    assert np.array_equal(source, original)


def test_over_budget_vector_is_radially_scaled() -> None:
    repaired = proportional_budget_repair([2.0, -2.0, 1.0, 0.0, 0.0, 0.0], 1.5)
    assert np.sum(np.abs(repaired)) == pytest.approx(1.5)
    assert is_user_budget_feasible(repaired.tolist(), 1.5)


def test_gamma_zero_returns_all_zero() -> None:
    assert np.array_equal(proportional_budget_repair(np.ones(6), 0.0), np.zeros(6))


def test_inactive_dimensions_are_zeroed_before_budget_scaling() -> None:
    mask = (True, False, True, False, False, False)
    repaired = proportional_budget_repair([1, 99, -1, 99, 99, 99], 1.0, mask)
    assert np.array_equal(repaired[[1, 3, 4, 5]], np.zeros(4))
    assert np.sum(np.abs(repaired)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "z,gamma,mask",
    [
        ([0, 0, 0, 0, 0], 1.0, None),
        ([0, 0, 0, 0, 0, float("nan")], 1.0, None),
        ([0, 0, 0, 0, 0, float("inf")], 1.0, None),
        ([0, 0, 0, 0, 0, 0], -1.0, None),
        ([0, 0, 0, 0, 0, 0], 7.0, None),
        ([0, 0, 0, 0, 0, 0], 1.0, (True, False)),
    ],
)
def test_invalid_repair_inputs_raise(z: object, gamma: float, mask: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        proportional_budget_repair(z, gamma, mask)  # type: ignore[arg-type]


def test_repair_is_deterministic_and_idempotent() -> None:
    source = np.array([2.0, -0.4, 0.8, 1.5, -3.0, 0.2])
    first = proportional_budget_repair(source, 2.25)
    assert np.array_equal(first, proportional_budget_repair(source, 2.25))
    assert np.array_equal(first, proportional_budget_repair(first, 2.25))
    assert validate_user_disturbance(first.tolist(), 2.25) == tuple(first)

