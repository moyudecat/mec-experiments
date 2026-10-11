"""普通 LHS 与预算修复样本测试。"""

import numpy as np
import pytest

from mec_dualbp.uncertainty.budget_set import is_user_budget_feasible
from mec_dualbp.uncertainty.sampling import lhs_user_disturbances, ordinary_lhs_unit
from mec_dualbp.uncertainty.schema import EDGE_COMM_ENERGY_ACTIVE_MASK


def test_raw_ordinary_lhs_has_exactly_one_point_per_stratum() -> None:
    count = 40
    raw = ordinary_lhs_unit(count, 4, np.random.default_rng(7))
    for dimension in range(raw.shape[1]):
        bins = np.floor(raw[:, dimension] * count).astype(int)
        assert sorted(bins.tolist()) == list(range(count))


def test_repaired_samples_are_budget_feasible_and_inactive_zero() -> None:
    samples = lhs_user_disturbances(30, 1.0, EDGE_COMM_ENERGY_ACTIVE_MASK, 11)
    assert all(is_user_budget_feasible(row.tolist(), 1.0) for row in samples.repaired_lhs)
    assert np.all(samples.raw_lhs[:, [2, 5]] == 0.0)
    assert np.all(samples.repaired_lhs[:, [2, 5]] == 0.0)


def test_sampling_seed_reproducibility_and_independence() -> None:
    first = lhs_user_disturbances(20, 2.0, EDGE_COMM_ENERGY_ACTIVE_MASK, 8)
    second = lhs_user_disturbances(20, 2.0, EDGE_COMM_ENERGY_ACTIVE_MASK, 8)
    third = lhs_user_disturbances(20, 2.0, EDGE_COMM_ENERGY_ACTIVE_MASK, 9)
    assert np.array_equal(first.raw_lhs, second.raw_lhs)
    assert np.array_equal(first.repaired_lhs, second.repaired_lhs)
    assert not np.array_equal(first.raw_lhs, third.raw_lhs)


def test_center_and_anchors_are_extra_and_source_labeled() -> None:
    samples = lhs_user_disturbances(
        10, 3.0, EDGE_COMM_ENERGY_ACTIVE_MASK, 5, True, True
    )
    assert samples.raw_lhs.shape == (10, 6)
    assert samples.sources.count("ordinary_lhs_repaired") == 10
    assert samples.sources.count("center") == 1
    assert samples.sources.count("boundary_anchor") == 2
    assert samples.initialization_samples.shape == (13, 6)


def test_repair_is_not_claimed_to_preserve_strata() -> None:
    samples = lhs_user_disturbances(50, 0.2, EDGE_COMM_ENERGY_ACTIVE_MASK, 21)
    assert np.any(samples.raw_lhs != samples.repaired_lhs)
    assert "repaired" in samples.sources[-1]


@pytest.mark.parametrize("args", [(0, 2), (2, 0), (2.5, 2)])
def test_invalid_lhs_shape_raises(args: tuple[object, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        ordinary_lhs_unit(*args, np.random.default_rng(1))  # type: ignore[arg-type]

