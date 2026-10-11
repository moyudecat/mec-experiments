from datetime import datetime, timedelta, timezone

import pytest

from mec_dualbp.utils.sharded_checkpoint import ShardedCheckpoint, ShardedUnitKey, _atomic_json


def _key(seed=101):
    return ShardedUnitKey("FM1", "edge_core_latency_lower", "B4_formal_candidate_v1", seed)


def _hashes(code="code"):
    return {"formal_config_hash": "formal", "code_tree_hash": code, "sample_content_hash": "sample", "task_hash": "task", "method_hash": "method"}


def test_sharded_completed_skip_and_result(tmp_path) -> None:
    store = ShardedCheckpoint(tmp_path / "checkpoint", 10)
    key = _key(); store.register(key, _hashes()); store.mark_running(key); store.mark_completed(key, {"x": 1})
    assert store.should_skip(key, _hashes()) and store.load_result(key) == {"x": 1}
    assert store.summary()["completed"] == 1


def test_sharded_hash_changes_reject_reuse(tmp_path) -> None:
    store = ShardedCheckpoint(tmp_path / "checkpoint", 10)
    store.register(_key(), _hashes())
    with pytest.raises(ValueError):
        store.register(_key(), _hashes("changed-code"))


def test_sharded_failed_history_and_rerun(tmp_path) -> None:
    store = ShardedCheckpoint(tmp_path / "checkpoint", 10); key = _key()
    store.register(key, _hashes()); store.mark_running(key)
    try: raise RuntimeError("probe")
    except RuntimeError as error: store.mark_failed(key, error)
    store.mark_running(key); store.mark_completed(key, {"ok": True})
    history = __import__("json").loads((store.unit_dir(key) / "history.json").read_text())
    assert any(item["status"] == "failed" for item in history) and store.should_skip(key, _hashes())


def test_sharded_stale_requires_dead_process_and_timeout(tmp_path) -> None:
    store = ShardedCheckpoint(tmp_path / "checkpoint", 1); key = _key()
    store.register(key, _hashes()); store.mark_running(key)
    status = store.status(key); status["process_id"] = 2_147_483_647; status["started_at"] = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
    _atomic_json(store.unit_dir(key) / "status.json", status)
    assert store.mark_stale_if_eligible(key)


def test_atomic_writer_leaves_no_temporary_files(tmp_path) -> None:
    target = tmp_path / "status.json"; _atomic_json(target, {"ok": True})
    assert target.exists() and not list(tmp_path.glob("*.tmp.*"))
