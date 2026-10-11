"""配置哈希绑定的任务级原子断点与结果持久化。"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

CheckpointStatus = Literal["pending", "running", "completed", "failed", "stale"]


def _process_exists(process_id: int | None) -> bool:
    """保守检查本机进程是否存在；无法确认时视为仍存在。"""

    if process_id is None or process_id <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            query_limited_information, False, process_id
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return ctypes.get_last_error() == 5
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def _utc_now() -> datetime:
    """返回带时区的UTC当前时间。"""

    return datetime.now(timezone.utc)


def _atomic_json_write(path: Path, payload: object) -> None:
    """先写同目录临时文件并fsync，再用replace原子替换目标。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.05 * (attempt + 1))


@dataclass(frozen=True, slots=True)
class TaskUnitKey:
    """Preview断点的最小执行单元。"""

    sample_id: str
    task_name: str
    method_name: str
    seed: int

    @property
    def unit_id(self) -> str:
        """返回稳定且可读的单元标识。"""

        return f"{self.sample_id}::{self.task_name}::{self.method_name}::{self.seed}"


class CheckpointStore:
    """维护status.json、结果文件、失败历史和stale转换。"""

    def __init__(self, root: str | Path, config_hash: str, stale_after_seconds: float) -> None:
        """打开或创建与配置哈希严格绑定的断点目录。"""

        if not config_hash or stale_after_seconds <= 0:
            raise ValueError("config_hash and positive stale timeout are required")
        self.root = Path(root)
        self.status_path = self.root / "status.json"
        self.results_dir = self.root / "results"
        self.config_hash = config_hash
        self.stale_after_seconds = float(stale_after_seconds)
        self.root.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if self.status_path.exists():
            state = self._read_state()
            if state["config_hash"] != config_hash:
                raise ValueError("checkpoint config hash mismatch; reuse is forbidden")
        else:
            _atomic_json_write(
                self.status_path,
                {"config_hash": config_hash, "stale_after_seconds": self.stale_after_seconds, "units": {}},
            )

    def _read_state(self) -> dict[str, Any]:
        """读取并验证status.json基本结构。"""

        state = json.loads(self.status_path.read_text(encoding="utf-8"))
        if not isinstance(state.get("units"), dict):
            raise ValueError("invalid checkpoint status schema")
        return state

    def _write_state(self, state: dict[str, Any]) -> None:
        """原子更新status.json。"""

        _atomic_json_write(self.status_path, state)

    def register(self, keys: list[TaskUnitKey]) -> None:
        """注册pending单元；已有记录及不利历史原样保留。"""

        state = self._read_state()
        for key in keys:
            state["units"].setdefault(
                key.unit_id,
                {
                    "key": asdict(key),
                    "status": "pending",
                    "config_hash": self.config_hash,
                    "started_at": None,
                    "ended_at": None,
                    "wall_time_seconds": None,
                    "process_id": None,
                    "error": None,
                    "traceback": None,
                    "result_file": None,
                    "history": [],
                },
            )
        self._write_state(state)

    def get(self, key: TaskUnitKey) -> dict[str, Any]:
        """返回单元记录副本。"""

        return dict(self._read_state()["units"][key.unit_id])

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return one consistent in-memory snapshot for bulk read-only operations."""

        return {
            unit_id: dict(record)
            for unit_id, record in self._read_state()["units"].items()
        }

    def load_result_from_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Load a completed result using a record from :meth:`snapshot`."""

        if record["status"] != "completed" or not record["result_file"]:
            raise ValueError("unit has no completed result")
        return json.loads((self.root / record["result_file"]).read_text(encoding="utf-8"))

    def should_skip(self, key: TaskUnitKey) -> bool:
        """仅当同一配置哈希的单元已completed时自动跳过。"""

        record = self.get(key)
        return record["status"] == "completed" and record["config_hash"] == self.config_hash

    def mark_running(
        self,
        key: TaskUnitKey,
        started_at: datetime | None = None,
        process_id: int | None = None,
    ) -> None:
        """将单元原子标记为running并记录开始时间。"""

        state = self._read_state()
        record = state["units"][key.unit_id]
        now = started_at or _utc_now()
        record.update(
            status="running",
            started_at=now.isoformat(),
            ended_at=None,
            wall_time_seconds=None,
            process_id=os.getpid() if process_id is None else process_id,
        )
        record["history"].append({"status": "running", "at": now.isoformat()})
        self._write_state(state)

    def mark_completed(self, key: TaskUnitKey, result: dict[str, Any]) -> None:
        """先原子保存结果，再将状态标记为completed。"""

        state = self._read_state()
        record = state["units"][key.unit_id]
        if record["status"] != "running":
            raise ValueError("only running units can complete")
        ended = _utc_now()
        started = datetime.fromisoformat(record["started_at"])
        result_name = hashlib.sha256(key.unit_id.encode("utf-8")).hexdigest() + ".json"
        result_path = self.results_dir / result_name
        _atomic_json_write(result_path, result)
        record.update(
            status="completed",
            ended_at=ended.isoformat(),
            wall_time_seconds=(ended - started).total_seconds(),
            error=None,
            traceback=None,
            result_file=str(result_path.relative_to(self.root)),
            process_id=None,
        )
        record["history"].append({"status": "completed", "at": ended.isoformat()})
        self._write_state(state)

    def mark_failed(self, key: TaskUnitKey, error: BaseException) -> None:
        """保留错误类型、消息和完整traceback，不删除失败记录。"""

        state = self._read_state()
        record = state["units"][key.unit_id]
        ended = _utc_now()
        started = datetime.fromisoformat(record["started_at"])
        error_text = f"{type(error).__name__}: {error}"
        trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        record.update(
            status="failed",
            ended_at=ended.isoformat(),
            wall_time_seconds=(ended - started).total_seconds(),
            error=error_text,
            traceback=trace,
            process_id=None,
        )
        record["history"].append(
            {
                "status": "failed",
                "at": ended.isoformat(),
                "error": error_text,
                "traceback": trace,
            }
        )
        self._write_state(state)

    def mark_stale_running(self, now: datetime | None = None) -> list[str]:
        """将超时running单元标记为stale，绝不视为completed。"""

        state = self._read_state()
        current = now or _utc_now()
        changed = []
        for unit_id, record in state["units"].items():
            if record["status"] != "running":
                continue
            elapsed = (current - datetime.fromisoformat(record["started_at"])).total_seconds()
            if elapsed > self.stale_after_seconds and not _process_exists(record.get("process_id")):
                record.update(status="stale", ended_at=current.isoformat(), wall_time_seconds=elapsed)
                record["history"].append({"status": "stale", "at": current.isoformat()})
                changed.append(unit_id)
        self._write_state(state)
        return changed

    def selected_for_rerun(self, statuses: tuple[str, ...] = ("failed", "stale")) -> list[TaskUnitKey]:
        """返回指定失败/stale状态的单元，支持定向重跑。"""

        state = self._read_state()
        return [TaskUnitKey(**record["key"]) for record in state["units"].values() if record["status"] in statuses]

    def load_result(self, key: TaskUnitKey) -> dict[str, Any]:
        """读取completed单元的结果文件。"""

        record = self.get(key)
        if record["status"] != "completed" or not record["result_file"]:
            raise ValueError("unit has no completed result")
        return json.loads((self.root / record["result_file"]).read_text(encoding="utf-8"))

    def partial_summary(self) -> dict[str, int]:
        """按状态生成可用于中途展示的临时汇总。"""

        units = self._read_state()["units"].values()
        counts = {status: 0 for status in ("pending", "running", "completed", "failed", "stale")}
        for record in units:
            counts[record["status"]] += 1
        counts["total"] = sum(counts.values())
        return counts
