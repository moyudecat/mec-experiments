"""Independent per-unit checkpoint shards for safe spawned-process execution."""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _atomic_json(path: Path, payload: object, retries: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(retries):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt + 1 == retries:
                temporary.unlink(missing_ok=True)
                raise
            time.sleep(0.025 * (attempt + 1))


def _process_exists(process_id: int | None) -> bool:
    if not process_id or process_id <= 0:
        return False
    try:
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, process_id)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    except (AttributeError, OSError):
        try:
            os.kill(process_id, 0)
            return True
        except OSError:
            return False


@dataclass(frozen=True, slots=True)
class ShardedUnitKey:
    sample_id: str
    task_name: str
    method_name: str
    seed: int

    @property
    def unit_id(self) -> str:
        return f"{self.sample_id}::{self.task_name}::{self.method_name}::{self.seed}"

    @property
    def unit_hash(self) -> str:
        return hashlib.sha256(self.unit_id.encode("utf-8")).hexdigest()


class ShardedCheckpoint:
    """Checkpoint store in which workers never share a mutable status file."""

    def __init__(self, root: str | Path, stale_after_seconds: float = 3600.0) -> None:
        self.root = Path(root)
        self.units_root = self.root / "units"
        self.units_root.mkdir(parents=True, exist_ok=True)
        self.stale_after_seconds = float(stale_after_seconds)

    def unit_dir(self, key: ShardedUnitKey) -> Path:
        return self.units_root / key.unit_hash

    def register(self, key: ShardedUnitKey, hashes: dict[str, str]) -> dict[str, Any]:
        directory = self.unit_dir(key)
        status_path = directory / "status.json"
        if status_path.exists():
            record = json.loads(status_path.read_text(encoding="utf-8"))
            if record["hashes"] != hashes:
                raise ValueError("completed or existing unit hash bundle mismatch; reuse forbidden")
            return record
        now = _now().isoformat()
        record = {
            "key": asdict(key), "unit_id": key.unit_id, "unit_hash": key.unit_hash,
            "status": "pending", "hashes": hashes, "process_id": None,
            "started_at": None, "ended_at": None, "error": None,
        }
        _atomic_json(status_path, record)
        _atomic_json(directory / "history.json", [{"status": "pending", "at": now}])
        return record

    def status(self, key: ShardedUnitKey) -> dict[str, Any]:
        return json.loads((self.unit_dir(key) / "status.json").read_text(encoding="utf-8"))

    def _transition(self, key: ShardedUnitKey, status: str, **updates: Any) -> None:
        directory = self.unit_dir(key)
        record = self.status(key)
        record.update(status=status, **updates)
        _atomic_json(directory / "status.json", record)
        history_path = directory / "history.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        history.append({"status": status, "at": _now().isoformat(), **{k: v for k, v in updates.items() if k in ("error", "traceback")}})
        _atomic_json(history_path, history)

    def should_skip(self, key: ShardedUnitKey, hashes: dict[str, str]) -> bool:
        record = self.register(key, hashes)
        return record["status"] == "completed" and record["hashes"] == hashes and (self.unit_dir(key) / "result.json").exists()

    def mark_running(self, key: ShardedUnitKey) -> None:
        self._transition(key, "running", process_id=os.getpid(), started_at=_now().isoformat(), ended_at=None, error=None)

    def mark_completed(self, key: ShardedUnitKey, result: dict[str, Any]) -> None:
        directory = self.unit_dir(key)
        _atomic_json(directory / "result.json", result)
        self._transition(key, "completed", process_id=None, ended_at=_now().isoformat(), error=None)

    def mark_failed(self, key: ShardedUnitKey, error: BaseException) -> None:
        trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        payload = {"error_type": type(error).__name__, "message": str(error), "traceback": trace, "at": _now().isoformat()}
        _atomic_json(self.unit_dir(key) / "error.json", payload)
        self._transition(key, "failed", process_id=None, ended_at=_now().isoformat(), error=f"{type(error).__name__}: {error}", traceback=trace)

    def mark_stale_if_eligible(self, key: ShardedUnitKey, now: datetime | None = None) -> bool:
        record = self.status(key)
        if record["status"] != "running":
            return False
        current = now or _now()
        elapsed = (current - datetime.fromisoformat(record["started_at"])).total_seconds()
        if elapsed <= self.stale_after_seconds or _process_exists(record.get("process_id")):
            return False
        self._transition(key, "stale", process_id=None, ended_at=current.isoformat(), error="stale process no longer exists")
        return True

    def load_result(self, key: ShardedUnitKey) -> dict[str, Any]:
        if self.status(key)["status"] != "completed":
            raise ValueError("unit is not completed")
        return json.loads((self.unit_dir(key) / "result.json").read_text(encoding="utf-8"))

    def summary(self) -> dict[str, int]:
        counts = {name: 0 for name in ("pending", "running", "completed", "failed", "stale")}
        for path in self.units_root.glob("*/status.json"):
            counts[json.loads(path.read_text(encoding="utf-8"))["status"]] += 1
        counts["total"] = sum(counts.values())
        return counts
