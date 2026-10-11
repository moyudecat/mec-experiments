import json
from pathlib import Path

from mec_dualbp.experiments.exp7_persistence_guard import Exp7PersistenceGuard


PROTOCOL = "p" * 64
EVALUATOR = "e" * 64


def _completed(root: Path) -> Path:
    run = root / "scenarios" / "S1" / "runs" / "Random__seed_52001"
    run.mkdir(parents=True)
    status = {"status": "completed", "requests": 20000, "scenario": "S1",
              "algorithm": "Random", "seed": 52001,
              "evaluator_hash": EVALUATOR, "effective_protocol_hash": PROTOCOL}
    (run / "status.json").write_text(json.dumps(status), encoding="utf-8")
    for name in ("checkpoint.pkl", "evaluation_archive_metadata.json", "final_front.csv", "history.csv"):
        (run / name).write_bytes(name.encode())
    return run


def test_active_formal_is_resumed_without_clearing(tmp_path: Path) -> None:
    formal, mirror = tmp_path / "formal", tmp_path / "mirror"
    guard = Exp7PersistenceGuard(formal, mirror, PROTOCOL, EVALUATOR)
    active = {"status": "active", "protocol_hash": PROTOCOL}
    guard.initialize(active)
    sentinel = formal / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    guard.initialize(active)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_completed_count_only_after_validation(tmp_path: Path) -> None:
    guard = Exp7PersistenceGuard(tmp_path / "formal", tmp_path / "mirror", PROTOCOL, EVALUATOR)
    guard.initialize({"status": "active", "protocol_hash": PROTOCOL})
    run = _completed(tmp_path / "formal")
    (run / "final_front.csv").unlink()
    try:
        guard.mirror_run(run)
    except RuntimeError:
        pass
    else:
        raise AssertionError("incomplete run was mirrored")
    assert guard.manifests() == []


def test_watchdog_detects_and_restores_deleted_completed_run(tmp_path: Path) -> None:
    formal, mirror = tmp_path / "formal", tmp_path / "mirror"
    guard = Exp7PersistenceGuard(formal, mirror, PROTOCOL, EVALUATOR)
    guard.initialize({"status": "active", "protocol_hash": PROTOCOL})
    run = _completed(formal)
    manifest = guard.mirror_run(run)
    expected = manifest["source_artifact_hashes"]
    for file in run.iterdir():
        file.unlink()
    run.rmdir()
    count, recovered = guard.watchdog(1)
    assert (count, recovered) == (1, 1)
    assert guard.validate(run)["source_artifact_hashes"] == expected


def test_mirror_hash_identity_and_recovery_do_not_evaluate(tmp_path: Path) -> None:
    formal, mirror = tmp_path / "formal", tmp_path / "mirror"
    guard = Exp7PersistenceGuard(formal, mirror, PROTOCOL, EVALUATOR)
    guard.initialize({"status": "active", "protocol_hash": PROTOCOL})
    run = _completed(formal)
    manifest = guard.mirror_run(run)
    assert manifest["mirror_valid"] is True
    assert manifest["source_artifact_hashes"] == manifest["mirror_artifact_hashes"]
    # Recovery operates solely on persisted bytes; the guard has no evaluator API.
    assert not hasattr(guard, "evaluate")


def test_resume_keeps_completed_mirror_singleton(tmp_path: Path) -> None:
    formal, mirror = tmp_path / "formal", tmp_path / "mirror"
    guard = Exp7PersistenceGuard(formal, mirror, PROTOCOL, EVALUATOR)
    guard.initialize({"status": "active", "protocol_hash": PROTOCOL})
    run = _completed(formal)
    guard.mirror_run(run)
    guard.mirror_run(run)
    assert len(guard.manifests()) == 1
