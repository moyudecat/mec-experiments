"""本地点响应与三段确定性求界测试。"""

import csv
import math
import os
import random
from pathlib import Path

import pytest

from mec_dualbp.physics.local_response import (
    LocalResponseBounds,
    local_energy_point_j,
    local_latency_point_s,
    local_response_bounds_si,
)
from mec_dualbp.uncertainty.interval_params import Interval

GRID_STEPS = 200
GRID_RELATIVE_TOLERANCE = 1e-4
RANDOM_CASES = 50
RANDOM_SEED = 20260803


def _grid_bounds(
    cycles: Interval, frequency: Interval, kappa: float, gamma: float
) -> tuple[float, float, float, float]:
    latency_values: list[float] = []
    energy_values: list[float] = []
    for i in range(GRID_STEPS + 1):
        z_c = -1.0 + 2.0 * i / GRID_STEPS
        for j in range(GRID_STEPS + 1):
            z_f = -1.0 + 2.0 * j / GRID_STEPS
            if abs(z_c) + abs(z_f) <= gamma + 1e-14:
                c_value = cycles.center + cycles.radius * z_c
                f_value = frequency.center + frequency.radius * z_f
                latency_values.append(local_latency_point_s(c_value, f_value))
                energy_values.append(local_energy_point_j(c_value, f_value, kappa))
    assert latency_values and energy_values
    return min(latency_values), max(latency_values), min(energy_values), max(energy_values)


def _max_differences(
    analytic: LocalResponseBounds, grid: tuple[float, float, float, float]
) -> tuple[float, float]:
    expected = (
        analytic.local_latency_lower_s,
        analytic.local_latency_upper_s,
        analytic.local_energy_lower_j,
        analytic.local_energy_upper_j,
    )
    differences = [abs(a - g) for a, g in zip(expected, grid)]
    relatives = [difference / max(abs(a), 1e-300) for difference, a in zip(differences, expected)]
    return max(differences), max(relatives)


def test_local_point_formulas_and_monotonicity() -> None:
    assert local_latency_point_s(4e9, 2e9) == pytest.approx(2.0)
    assert local_energy_point_j(4e9, 2e9, 1e-28) == pytest.approx(1.6)
    assert local_latency_point_s(5e9, 2e9) > local_latency_point_s(4e9, 2e9)
    assert local_latency_point_s(4e9, 3e9) < local_latency_point_s(4e9, 2e9)
    assert local_energy_point_j(5e9, 2e9, 1e-28) > local_energy_point_j(4e9, 2e9, 1e-28)
    assert local_energy_point_j(4e9, 3e9, 1e-28) > local_energy_point_j(4e9, 2e9, 1e-28)
    assert local_energy_point_j(4e9, 2e9, 2e-28) > local_energy_point_j(4e9, 2e9, 1e-28)


def test_gamma_zero_degenerates_to_center_point() -> None:
    cycles, frequency = Interval(2e9, 6e9), Interval(1e9, 3e9)
    result = local_response_bounds_si(cycles, frequency, 1e-28, 0.0)
    assert result.method == "center_point"
    assert result.candidate_count == 1
    assert result.local_latency_lower_s == result.local_latency_upper_s == 2.0
    assert result.local_energy_lower_j == result.local_energy_upper_j == pytest.approx(1.6)


@pytest.mark.parametrize("gamma", [2.0, 3.5, 6.0])
def test_gamma_ge_two_matches_closed_form(gamma: float) -> None:
    cycles, frequency, kappa = Interval(2e9, 6e9), Interval(1e9, 3e9), 1e-28
    result = local_response_bounds_si(cycles, frequency, kappa, gamma)
    assert result.method == "closed_form_gamma_ge_2"
    assert result.local_latency_lower_s == pytest.approx(cycles.lower / frequency.upper)
    assert result.local_latency_upper_s == pytest.approx(cycles.upper / frequency.lower)
    assert result.local_energy_lower_j == pytest.approx(kappa * cycles.lower * frequency.lower**2)
    assert result.local_energy_upper_j == pytest.approx(kappa * cycles.upper * frequency.upper**2)


@pytest.mark.parametrize("gamma", [0.1, 0.5, 1.0, 1.5, 1.9])
def test_intermediate_gamma_is_at_least_as_extreme_as_dense_grid(gamma: float) -> None:
    cycles, frequency, kappa = Interval(2e9, 6e9), Interval(1e9, 3e9), 1e-28
    analytic = local_response_bounds_si(cycles, frequency, kappa, gamma)
    grid = _grid_bounds(cycles, frequency, kappa, gamma)
    scale_t = max(1.0, analytic.local_latency_upper_s)
    scale_e = max(1.0, analytic.local_energy_upper_j)
    assert analytic.local_latency_lower_s <= grid[0] + 1e-12 * scale_t
    assert analytic.local_latency_upper_s >= grid[1] - 1e-12 * scale_t
    assert analytic.local_energy_lower_j <= grid[2] + 1e-12 * scale_e
    assert analytic.local_energy_upper_j >= grid[3] - 1e-12 * scale_e
    assert _max_differences(analytic, grid)[1] <= GRID_RELATIVE_TOLERANCE


def test_fixed_seed_random_intervals_against_dense_grid_and_write_audit_csv() -> None:
    generator = random.Random(RANDOM_SEED)
    rows: list[dict[str, object]] = []
    gamma_values = (0.1, 0.5, 1.0, 1.5, 1.9)
    for case_id in range(RANDOM_CASES):
        c_lower = generator.uniform(1e7, 5e9)
        c_upper = c_lower + generator.uniform(0.0, 5e9)
        f_lower = generator.uniform(1e8, 3e9)
        f_upper = f_lower + generator.uniform(0.0, 3e9)
        kappa = generator.uniform(1e-30, 5e-28)
        gamma = gamma_values[case_id % len(gamma_values)]
        cycles, frequency = Interval(c_lower, c_upper), Interval(f_lower, f_upper)
        analytic = local_response_bounds_si(cycles, frequency, kappa, gamma)
        grid = _grid_bounds(cycles, frequency, kappa, gamma)
        absolute_difference, relative_difference = _max_differences(analytic, grid)
        assert analytic.local_latency_lower_s <= grid[0] + 1e-11 * max(1.0, grid[0])
        assert analytic.local_latency_upper_s >= grid[1] - 1e-11 * max(1.0, grid[1])
        assert analytic.local_energy_lower_j <= grid[2] + 1e-11 * max(1.0, grid[2])
        assert analytic.local_energy_upper_j >= grid[3] - 1e-11 * max(1.0, grid[3])
        assert relative_difference <= GRID_RELATIVE_TOLERANCE
        rows.append(
            {
                "case_id": case_id,
                "gamma": gamma,
                "C_lower": c_lower,
                "C_upper": c_upper,
                "F_lower": f_lower,
                "F_upper": f_upper,
                "analytic_T_lower": analytic.local_latency_lower_s,
                "grid_T_lower": grid[0],
                "analytic_T_upper": analytic.local_latency_upper_s,
                "grid_T_upper": grid[1],
                "analytic_E_lower": analytic.local_energy_lower_j,
                "grid_E_lower": grid[2],
                "analytic_E_upper": analytic.local_energy_upper_j,
                "grid_E_upper": grid[3],
                "absolute_difference": absolute_difference,
                "relative_difference": relative_difference,
                "method": analytic.method,
            }
        )
    destination = os.environ.get("MEC_DUALBP_VALIDATION_CSV")
    if destination:
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def test_bounds_expand_monotonically_with_gamma() -> None:
    cycles, frequency = Interval(1e9, 5e9), Interval(1e9, 4e9)
    results = [local_response_bounds_si(cycles, frequency, 2e-28, gamma) for gamma in (0.0, 0.1, 0.5, 1.0, 1.5, 1.9, 2.0, 6.0)]
    for previous, current in zip(results, results[1:]):
        assert current.local_latency_lower_s <= previous.local_latency_lower_s + 1e-14
        assert current.local_latency_upper_s >= previous.local_latency_upper_s - 1e-14
        assert current.local_energy_lower_j <= previous.local_energy_lower_j + 1e-14
        assert current.local_energy_upper_j >= previous.local_energy_upper_j - 1e-14


def test_continuity_near_zero_and_two() -> None:
    cycles, frequency = Interval(2e9, 6e9), Interval(1e9, 3e9)
    center = local_response_bounds_si(cycles, frequency, 1e-28, 0.0)
    near_zero = local_response_bounds_si(cycles, frequency, 1e-28, 1e-10)
    closed = local_response_bounds_si(cycles, frequency, 1e-28, 2.0)
    near_two = local_response_bounds_si(cycles, frequency, 1e-28, 2.0 - 1e-10)
    for field in ("local_latency_lower_s", "local_latency_upper_s", "local_energy_lower_j", "local_energy_upper_j"):
        assert getattr(near_zero, field) == pytest.approx(getattr(center, field), rel=1e-9, abs=1e-12)
        assert getattr(near_two, field) == pytest.approx(getattr(closed, field), rel=1e-9, abs=1e-12)


def test_degenerate_interval_radii_are_supported() -> None:
    for cycles, frequency in (
        (Interval(2e9, 2e9), Interval(1e9, 3e9)),
        (Interval(2e9, 6e9), Interval(2e9, 2e9)),
        (Interval(2e9, 2e9), Interval(2e9, 2e9)),
    ):
        result = local_response_bounds_si(cycles, frequency, 1e-28, 1.0)
        assert 0.0 <= result.local_latency_lower_s <= result.local_latency_upper_s < math.inf
        assert 0.0 <= result.local_energy_lower_j <= result.local_energy_upper_j < math.inf


@pytest.mark.parametrize(
    "call",
    [
        lambda: local_latency_point_s(-1.0, 1.0),
        lambda: local_latency_point_s(1.0, 0.0),
        lambda: local_energy_point_j(1.0, 1.0, -1.0),
        lambda: local_response_bounds_si(Interval(-1.0, 1.0), Interval(1.0, 2.0), 1.0, 1.0),
        lambda: local_response_bounds_si(Interval(1.0, 2.0), Interval(0.0, 2.0), 1.0, 1.0),
        lambda: local_response_bounds_si(Interval(1.0, 2.0), Interval(1.0, 2.0), 1.0, math.nan),
    ],
)
def test_invalid_local_inputs_raise(call: object) -> None:
    with pytest.raises(ValueError):
        call()  # type: ignore[operator]
