from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mec_dualbp.utils.atomic_io import safe_atomic_write_json


def test_repeated_atomic_status_updates_are_valid(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    for index in range(25):
        safe_atomic_write_json(path, {"index": index})
    assert json.loads(path.read_text(encoding="utf-8")) == {"index": 24}


def test_temp_is_closed_before_replace(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    observed = []

    def replace(source, target):
        # On Windows this succeeds only after NamedTemporaryFile has closed.
        observed.append(Path(source))
        os.replace(source, target)

    safe_atomic_write_json(path, {"closed": True}, replace=replace)
    assert observed and not observed[0].exists()


def test_permission_error_retries_then_succeeds(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    attempts = 0

    def replace(source, target):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(5, "temporary lock")
        os.replace(source, target)

    safe_atomic_write_json(path, {"ok": True}, backoff_seconds=(0, 0, 0), replace=replace)
    assert attempts == 3 and json.loads(path.read_text()) == {"ok": True}
    assert len((tmp_path / "atomic_write_diagnostics.log").read_text(encoding="utf-8").splitlines()) == 2


def test_retry_exhaustion_preserves_existing_final(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    path.write_text('{"old": true}\n', encoding="utf-8")

    def locked(source, target):
        raise PermissionError(5, "persistent lock")

    with pytest.raises(PermissionError):
        safe_atomic_write_json(path, {"new": True}, backoff_seconds=(0, 0), replace=locked)
    assert json.loads(path.read_text(encoding="utf-8")) == {"old": True}
    assert not list(tmp_path.glob(".status.json.*.tmp"))


def test_unique_temp_names_avoid_fixed_status_tmp(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    names = []

    def replace(source, target):
        names.append(Path(source).name)
        os.replace(source, target)

    safe_atomic_write_json(path, {"n": 1}, replace=replace)
    safe_atomic_write_json(path, {"n": 2}, replace=replace)
    assert len(set(names)) == 2
    assert all(name != "status.json.tmp" for name in names)


def test_rapid_250_updates_leave_no_temp_debris(tmp_path: Path) -> None:
    nested = tmp_path / "windows style path" / "status.json"
    for index in range(250):
        safe_atomic_write_json(nested, {"index": index})
    assert json.loads(nested.read_text(encoding="utf-8"))["index"] == 249
    assert not list(nested.parent.glob(".status.json.*.tmp"))

