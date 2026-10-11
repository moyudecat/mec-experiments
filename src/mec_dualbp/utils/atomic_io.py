"""Windows-safe atomic writers for critical experiment state."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Callable, Sequence


DEFAULT_BACKOFF_SECONDS = (0.05, 0.10, 0.20, 0.40, 0.80, 1.00, 1.00, 1.00)


def _record_retry(target: Path, attempt: int, error: PermissionError, wait: float) -> None:
    diagnostic = target.parent / "atomic_write_diagnostics.log"
    line = json.dumps({
        "attempt": attempt,
        "exception": repr(error),
        "wait_seconds": wait,
        "target": str(target),
    }, ensure_ascii=False)
    try:
        with diagnostic.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line + "\n")
    except OSError:
        # Diagnostics are best-effort and must not hide the original replace error.
        pass


def _replace_with_retry(temp: Path, target: Path, *,
                        backoff_seconds: Sequence[float],
                        replace: Callable[[str | bytes | os.PathLike[str] | os.PathLike[bytes],
                                           str | bytes | os.PathLike[str] | os.PathLike[bytes]], None] = os.replace) -> None:
    attempts = len(backoff_seconds)
    if attempts == 0:
        raise ValueError("at least one replace attempt is required")
    for index, wait in enumerate(backoff_seconds, 1):
        try:
            replace(temp, target)
            return
        except PermissionError as error:
            _record_retry(target, index, error, float(wait))
            if index == attempts:
                raise
            time.sleep(float(wait))


def safe_atomic_write_bytes(path: Path, data: bytes, *,
                            backoff_seconds: Sequence[float] = DEFAULT_BACKOFF_SECONDS,
                            replace: Callable[..., None] = os.replace) -> None:
    """Atomically replace *path* after closing and fsyncing a unique sibling temp."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, dir=target.parent,
            prefix=f".{target.name}.{os.getpid()}.", suffix=".tmp",
        ) as stream:
            temp_path = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temp_path, target, backoff_seconds=backoff_seconds, replace=replace)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def safe_atomic_write_json(path: Path, value: object, *,
                           backoff_seconds: Sequence[float] = DEFAULT_BACKOFF_SECONDS,
                           replace: Callable[..., None] = os.replace) -> None:
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    safe_atomic_write_bytes(Path(path), encoded, backoff_seconds=backoff_seconds, replace=replace)

