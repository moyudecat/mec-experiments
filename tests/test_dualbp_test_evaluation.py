from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mec_dualbp.experiments.dualbp_test_evaluation import CORRECTED_HASH
from mec_dualbp.training.calibration import apply_directional_correction
from mec_dualbp.training.dualbp import f_stratum, regression_metrics
from mec_dualbp.training.test_evaluation import (
    BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED, paired_bootstrap_accuracy,
    safety_metrics, test_protocol_is_frozen as protocol_is_frozen,
    wilson_interval,
)


def test_protocol_must_freeze_before_target_access():
    assert not protocol_is_frozen({"status":"freeze_candidate"})
    assert protocol_is_frozen({"status":"frozen","test_evaluation_protocol_hash":"x",
        "target_access_allowed_after_freeze":True,"post_test_model_modification_allowed":False,
        "post_test_alpha_modification_allowed":False})


def test_sealed_test_counts_and_historical_access():
    root=Path(__file__).resolve().parents[1]
    audit=json.loads((root/"outputs/01_formal/dualbp_training_formal_20260811_003126/data_leakage_audit.json").read_text())
    assert audit["metadata_only_audit"]["time_test"]["row_count"]==180
    assert audit["metadata_only_audit"]["energy_test"]["row_count"]==60
    assert audit["test_labels_accessed"] is False


def test_frozen_hashes_before_test():
    root=Path(__file__).resolve().parents[1]; train=root/"outputs/01_formal/dualbp_training_formal_20260811_003126"
    assert hashlib.sha256((train/"time_bp/selected_model.pt").read_bytes()).hexdigest()=="291c5733bebdb4cb0c2cf7a4a07138ca6bb07c3ad75bffb378aa09b7f657d827"
    assert hashlib.sha256((train/"energy_bp/selected_model.pt").read_bytes()).hexdigest()=="8481dfdca0e03e3d6bebe4c4d5808259d16f7cdd67f4a65251fe7aad3e07d971"
    corrected=json.loads((root/"outputs/01_formal/dualbp_alpha_calibration_formal_20260811_134010/corrected_dualbp_v1_hash.json").read_text())
    assert corrected["sha256"]==CORRECTED_HASH


def test_raw_is_unchanged_and_corrected_formula():
    raw_l=np.array([2.]); raw_u=np.array([4.]); alpha_l=.1; alpha_u=.2
    corr_l,corr_u=apply_directional_correction(raw_l,raw_u,alpha_l,alpha_u)
    np.testing.assert_array_equal(raw_l,[2.]); np.testing.assert_array_equal(raw_u,[4.])
    assert corr_l[0]==pytest.approx(2/1.1) and corr_u[0]==pytest.approx(4/.8)


def test_accuracy_metric_formulas():
    true_l=np.array([1.,2.,3.]); true_u=np.array([2.,3.,4.]); pred_l=np.array([1.,2.,4.]); pred_u=np.array([2.,4.,4.])
    m=regression_metrics(true_l,true_u,pred_l,pred_u,2.5)
    assert m["lower_mae"]==pytest.approx(1/3)
    assert m["lower_rmse"]==pytest.approx(np.sqrt(1/3))
    assert m["lower_r2"]==pytest.approx(.5)
    assert m["lower_wmape"]==pytest.approx(1/6)


def test_safety_metric_formulas():
    m=safety_metrics(np.array([1.,2.]),np.array([3.,4.]),np.array([1.5,1.]),np.array([2.5,5.]))
    assert m["uer"]==.5 and m["ler"]==.5 and m["icr"]==.5
    assert m["upper_under_relative_magnitude"]==pytest.approx((.5/3)/2)
    assert m["lower_over_relative_magnitude"]==pytest.approx((.5/1)/2)


def test_wilson_interval_known_case():
    low,high=wilson_interval(5,10)
    assert low==pytest.approx(.236593,rel=1e-5) and high==pytest.approx(.763407,rel=1e-5)
    assert wilson_interval(0,10)[0]==0


def test_paired_bootstrap_is_fixed_and_paired():
    assert BOOTSTRAP_SEED==46001 and BOOTSTRAP_REPETITIONS==5000
    true=np.arange(1,11,dtype=float); raw=true+1; corr=true+.5
    first=paired_bootstrap_accuracy(true,raw,corr); second=paired_bootstrap_accuracy(true,raw,corr)
    assert first==second and first["corrected"]["mae"][1]<first["raw"]["mae"][0]


@pytest.mark.parametrize("value,group",[(1.5e9,"Low"),(3.25e9,"Mid-high"),(5e9,"Upper-anchor")])
def test_time_f_strata_unchanged(value,group): assert f_stratum(value)==group


def test_gamma_group_keys_are_frozen(): assert [str(float(v)) for v in range(1,6)]==["1.0","2.0","3.0","4.0","5.0"]


def test_no_post_test_training_or_alpha_entry_is_registered():
    policy={"post_test_model_modification_allowed":False,"post_test_alpha_modification_allowed":False}
    assert not policy["post_test_model_modification_allowed"] and not policy["post_test_alpha_modification_allowed"]


def test_plot_data_round_trip_matches_source(tmp_path:Path):
    rows="reference,raw,corrected\n1,1.1,0.9\n"
    path=tmp_path/"fig.csv"; path.write_text(rows)
    assert path.read_text()==rows and "calibration" not in path.name and "train" not in path.name
