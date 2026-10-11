"""配置哈希绑定的结果级断点测试。"""

from datetime import datetime, timedelta, timezone

import pytest

from mec_dualbp.utils.checkpoint import CheckpointStore, TaskUnitKey


def _key(seed: int = 101) -> TaskUnitKey:
    return TaskUnitKey("S1", "edge_core_latency_upper", "B1_preview_candidate", seed)


def test_completed_result_is_atomic_loadable_and_skipped(tmp_path) -> None:
    store = CheckpointStore(tmp_path / "checkpoint", "abc", 60)
    key = _key()
    store.register([key])
    store.mark_running(key)
    store.mark_completed(key, {"response_value": 1.25})
    assert store.should_skip(key)
    assert store.load_result(key)["response_value"] == 1.25
    record = store.snapshot()[key.unit_id]
    assert store.load_result_from_record(record)["response_value"] == 1.25
    assert not list((tmp_path / "checkpoint").rglob("*.tmp"))
    assert store.partial_summary()["completed"] == 1


def test_config_hash_mismatch_forbids_reuse(tmp_path) -> None:
    root = tmp_path / "checkpoint"
    CheckpointStore(root, "first", 60)
    with pytest.raises(ValueError):
        CheckpointStore(root, "second", 60)


def test_running_timeout_becomes_stale_and_can_be_selected(tmp_path) -> None:
    store = CheckpointStore(tmp_path / "checkpoint", "abc", 10)
    key = _key()
    store.register([key])
    started = datetime.now(timezone.utc) - timedelta(seconds=20)
    store.mark_running(key, started, process_id=2_147_483_647)
    assert store.mark_stale_running() == [key.unit_id]
    assert store.get(key)["status"] == "stale"
    assert store.selected_for_rerun() == [key]
    assert not store.should_skip(key)


def test_failed_traceback_is_preserved_and_failed_only_rerun_supported(tmp_path) -> None:
    store = CheckpointStore(tmp_path / "checkpoint", "abc", 60)
    failed, pending = _key(101), _key(202)
    store.register([failed, pending])
    store.mark_running(failed)
    try:
        raise RuntimeError("intentional")
    except RuntimeError as error:
        store.mark_failed(failed, error)
    record = store.get(failed)
    assert record["status"] == "failed"
    assert "intentional" in record["error"] and "Traceback" in record["traceback"]
    assert "Traceback" in record["history"][-1]["traceback"]
    assert store.selected_for_rerun(("failed",)) == [failed]
    summary = store.partial_summary()
    assert summary["failed"] == 1 and summary["pending"] == 1 and summary["total"] == 2
