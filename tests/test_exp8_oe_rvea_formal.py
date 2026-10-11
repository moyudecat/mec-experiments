from pathlib import Path

from mec_dualbp.experiments.exp8_oe_rvea_formal import (
    EXPECTED_A,
    EXPECTED_B,
    EXPECTED_OVERALL,
    preflight,
)


def test_exp8_frozen_protocol_preflight() -> None:
    project = Path(__file__).resolve().parents[1]
    protocol = project / "outputs/05_audit/exp8_oe_rvea_protocol_freeze_20260821_124744"
    result = preflight(project, protocol)
    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["checks"]["G2_N20_exact"] is True


def test_exp8_protocol_identity_constants() -> None:
    assert EXPECTED_OVERALL == "effaad44c47b4ffdf481817f741d48d76e105a8b5e44bf31896149681c624117"
    assert EXPECTED_A == "7f6b99cc2d337dc45152f22635941fe42b685cb991616e56b992efe1cbe5f6ff"
    assert EXPECTED_B == "10f63c8167a94fba69943f28ca25f81ca58723a4daaec178d9816954982448ed"
