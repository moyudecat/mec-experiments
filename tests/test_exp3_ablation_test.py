from pathlib import Path
import numpy as np
from mec_dualbp.experiments.exp1_formal_protocol import file_sha256
from mec_dualbp.experiments.exp3_ablation_test import CONTRASTS, MODEL_HASHES, _model_binding
from mec_dualbp.training.exp3_evaluation import common_bootstrap_indices, contribution_status, full_test_metrics, paired_metric_difference

ROOT=Path(__file__).resolve().parents[1]

def test_models_frozen_and_hashes_match():assert _model_binding(ROOT)["passed"]
def test_primary_contrasts_exact():assert [(a,b) for a,b,_ in CONTRASTS]==[("A","B"),("B","C"),("B","D"),("B","E")]
def test_metric_definitions_and_invalid_preservation():
    m=full_test_metrics(np.array([1.,1.]),np.array([2.,2.]),np.array([.5,2.5]),np.array([2.5,1.5]),1.)
    assert m["uer"]==.5 and m["ler"]==.5 and m["icr"]==.5 and m["invalid_interval_count"]==1
def test_common_paired_bootstrap_indices():
    a=common_bootstrap_indices(5,10);b=common_bootstrap_indices(5,10);assert np.array_equal(a,b)
    ci=paired_metric_difference(np.arange(5.),np.arange(5.),np.arange(5.)+1,a);assert ci["MAE"]==( -1.0,-1.0)
def test_no_overall_rank_contract():
    protocol=ROOT/"outputs/05_audit/exp3_ablation_protocol_20260811_144345/exp3_ablation_protocol.json";assert file_sha256(protocol)=="a4d7a4a86c4618754534495609364f6debf20ed5e3ca6f814085d543d79d8544"
def test_contribution_rules_synthetic():
    metrics={}
    base={"uer":.2,"ler":.2,"icr":.6,"upper_under_relative_magnitude":.02,"lower_over_relative_magnitude":.02,"lower_wmape":.02,"upper_wmape":.02,"invalid_interval_count":0}
    for method in "ABCDE":
        for network in ("Time","Energy"):metrics[(method,network)]=dict(base)
    for network in ("Time","Energy"):metrics[("A",network)].update(uer=.1,ler=.1,icr=.8);metrics[("C",network)].update(uer=.3,ler=.3,icr=.4,upper_under_relative_magnitude=.03,lower_over_relative_magnitude=.03)
    metrics[("E","Time")]["invalid_interval_count"]=1
    status=contribution_status(metrics,.01);assert status["alpha_contribution"]=="supported" and status["center_radius_structural_validity_contribution"]=="supported"
