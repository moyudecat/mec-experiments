import numpy as np

from mec_dualbp.experiments.exp45_oe_rvea_v2 import LEVELS, SEEDS, _rank_biserial, _protocol_bodies


def test_frozen_exp45_counts_and_seeds():
    assert SEEDS == (72001, 72002, 72003, 72004, 72005)
    assert all(len(v) == 5 for v in LEVELS.values())
    exp4, stress, exp5 = _protocol_bodies(None)
    assert exp4["planned_runs"] == 30
    assert stress["points_per_representative_gamma"] == 10003
    assert exp5["independent_runs"] == 85 and exp5["exact_default_reuse_records"] == 15


def test_rank_biserial_ties_and_direction():
    assert _rank_biserial(np.ones(4), np.ones(4)) == 0
    assert _rank_biserial(np.array([2., 3.]), np.array([1., 1.])) == 1
