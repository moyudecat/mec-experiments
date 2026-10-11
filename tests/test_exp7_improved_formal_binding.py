from mec_dualbp.experiments.exp7_formal_runner import FINAL_ALGORITHMS, SCENARIO_HASHES
from mec_dualbp.optimization.adapters import V2_ALGORITHM
from mec_dualbp.optimization.dualbp_engine_registry import IMPROVED_MODE
from mec_dualbp.optimization.outer_contract import IMPROVED_CORRECTED_DUALBP_HASH


def test_final_algorithm_set_uses_frozen_v2_only() -> None:
    assert FINAL_ALGORITHMS == (V2_ALGORITHM, "MOEA/D", "RVEA", "MOPSO", "Random")
    assert "CorrectedDualBP-CNSGA-II" not in FINAL_ALGORITHMS


def test_improved_evaluator_and_scenario_bindings_are_frozen() -> None:
    assert IMPROVED_MODE == "improved_corrected_dualbp_v1"
    assert IMPROVED_CORRECTED_DUALBP_HASH == "b505197af2f99f526d24f31ca0c4be1c6d5082612b19fe94a589f823ff1e35d7"
    assert tuple(SCENARIO_HASHES) == ("S1", "S2", "S3", "S4", "S5")
