from __future__ import annotations

import numpy as np
import pytest

from mec_dualbp.optimization.adapters import FORMAL_ALGORITHMS
from mec_dualbp.optimization.evaluation_archive import EvaluationArchive


def _row(t, e, *, feasible=True, cv=0.0):
    return {"prediction_mode": "corr_pred", "objectives": [t, e],
            "constraint_violation": cv, "feasible": feasible,
            "qos_min_possibility": 1.0 if feasible else 0.0}


def test_historical_tradeoff_points_survive_official_archive() -> None:
    cache = {(0., 0., 0.): _row(1., 3.), (1., 0., 1.): _row(2., 2.),
             (1., 1., 1.): _row(3., 1.)}
    front = EvaluationArchive.from_cache(cache).official_front()
    assert {(r["Phi_T"], r["Phi_E"]) for r in front} == {(1., 3.), (2., 2.), (3., 1.)}


def test_dominated_and_infeasible_history_excluded() -> None:
    cache = {(0.,): _row(1., 1.), (1.,): _row(2., 2.), (2.,): _row(.5, .5, feasible=False, cv=1.)}
    front = EvaluationArchive.from_cache(cache).official_front()
    assert len(front) == 1 and (front[0]["Phi_T"], front[0]["Phi_E"]) == (1., 1.)


def test_canonical_dedup_and_metadata_account_for_paid_cache_hits() -> None:
    archive = EvaluationArchive.from_cache({(0.,): _row(1., 2.)})
    metadata = archive.metadata(paid_requests=3, cache_hits=2)
    assert metadata["paid_requests"] == 3
    assert metadata["canonical_unique_evaluated"] == 1
    assert metadata["official_front_size"] == 1


def test_fatal_inconsistent_duplicate_hash(monkeypatch) -> None:
    monkeypatch.setattr("mec_dualbp.optimization.evaluation_archive.canonical_decision_hash", lambda _: "same")
    with pytest.raises(RuntimeError, match="fatal deterministic consistency"):
        EvaluationArchive.from_cache({(0.,): _row(1., 2.), (1.,): _row(2., 1.)})


def test_all_algorithms_bind_one_reporting_archive_contract() -> None:
    assert len(FORMAL_ALGORITHMS) == 5
    for _algorithm in FORMAL_ALGORITHMS:
        assert EvaluationArchive.from_cache({(0.,): _row(1., 1.)}).official_front()


def test_archive_is_not_algorithm_search_state() -> None:
    archive = EvaluationArchive.from_cache({(0.,): _row(1., 1.)})
    assert not hasattr(archive, "leader")
    assert not hasattr(archive, "tell")
    assert not hasattr(archive, "ask")


def test_reference_oracle_two_point_minimization() -> None:
    archive = EvaluationArchive.from_cache({(0.,): _row(1., 2.), (1.,): _row(2., 1.), (2.,): _row(3., 3.)})
    observed = np.asarray([[r["Phi_T"], r["Phi_E"]] for r in archive.official_front()])
    assert {tuple(row) for row in observed} == {(1., 2.), (2., 1.)}
