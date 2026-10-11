"""用户级六维Gamma预算集合测试。"""

import math

import pytest

from mec_dualbp.uncertainty.budget_set import (
    is_user_budget_feasible,
    validate_gamma,
    validate_user_disturbance,
)


@pytest.mark.parametrize("gamma", [0.0, 0.1, 1.0, 2.0, 6.0])
def test_validate_gamma_accepts_closed_domain(gamma: float) -> None:
    assert validate_gamma(gamma) == gamma


@pytest.mark.parametrize("gamma", [-0.1, 6.1, math.nan, math.inf])
def test_validate_gamma_rejects_invalid_values(gamma: float) -> None:
    with pytest.raises(ValueError):
        validate_gamma(gamma)


def test_gamma_zero_only_accepts_nominal_vector() -> None:
    assert is_user_budget_feasible([0.0] * 6, 0.0)
    assert not is_user_budget_feasible([1e-6, 0, 0, 0, 0, 0], 0.0)


def test_gamma_six_accepts_full_corner() -> None:
    corner = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    assert validate_user_disturbance(corner, 6.0) == tuple(corner)


def test_user_budget_is_checked_per_user_not_globally() -> None:
    user_a = [0.5, 0.5, 0, 0, 0, 0]
    user_b = [-0.5, -0.5, 0, 0, 0, 0]
    assert is_user_budget_feasible(user_a, 1.0)
    assert is_user_budget_feasible(user_b, 1.0)
    assert sum(map(abs, user_a + user_b)) > 1.0


def test_invalid_dimension_and_nonfinite_component_raise() -> None:
    with pytest.raises(ValueError):
        validate_user_disturbance([0.0] * 5, 1.0)
    with pytest.raises(ValueError):
        validate_user_disturbance([0, 0, 0, 0, 0, math.nan], 1.0)


def test_clear_component_or_budget_violation_is_infeasible() -> None:
    assert not is_user_budget_feasible([1.01, 0, 0, 0, 0, 0], 6.0)
    assert not is_user_budget_feasible([0.5, 0.5, 0.5, 0, 0, 0], 1.0)
    with pytest.raises(ValueError):
        validate_user_disturbance([0.5, 0.5, 0.5, 0, 0, 0], 1.0)


def test_atol_is_small_and_does_not_clip() -> None:
    assert is_user_budget_feasible([1.0 + 5e-13, 0, 0, 0, 0, 0], 1.0)
    with pytest.raises(ValueError):
        validate_user_disturbance([1.0 + 5e-13, 0, 0, 0, 0, 0], 1.0)
    with pytest.raises(ValueError):
        is_user_budget_feasible([0.0] * 6, 1.0, atol=1e-3)

