"""Independent durability guard for the Exp-7 clean recovery rerun."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from mec_dualbp.utils.atomic_io import safe_atomic_write_json

REQUIRED_RUN_FILES = (
    "status.json", "checkpoint.pkl", "evaluation_archive_metadata.json",
    "final_front.csv", "history.csv",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): _sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


class Exp7PersistenceGuard:
    """Validate, mirror, watch, and byte-for-byte restore completed run assets."""

    def __init__(self, formal: Path, mirror: Path, protocol_hash: str,
                 evaluator_hash: str) -> None:
        self.formal = Path(formal)
        self.mirror = Path(mirror)
        self.protocol_hash = protocol_hash
        self.evaluator_hash = evaluator_hash
        self.manifest_dir = self.mirror / "manifests"

    def initialize(self, active: Mapping[str, Any]) -> None:
        marker = self.formal / "FORMAL_ACTIVE.json"
        if marker.exists():
            existing = json.loads(marker.read_text(encoding="utf-8"))
            if existing.get("protocol_hash") != self.protocol_hash:
                raise RuntimeError("active Formal protocol mismatch")
        else:
            self.formal.mkdir(parents=True, exist_ok=True)
            safe_atomic_write_json(marker, dict(active))
        self.mirror.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        safe_atomic_write_json(self.mirror / "FORMAL_ACTIVE.json", dict(active))

    def validate(self, run_dir: Path) -> dict[str, Any]:
        missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"completed run missing durable artifacts: {missing}")
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if status.get("status") != "completed" or int(status.get("requests", -1)) != 20_000:
            raise RuntimeError("completed run status/request mismatch")
        if status.get("evaluator_hash") != self.evaluator_hash:
            raise RuntimeError("completed run evaluator hash mismatch")
        if status.get("effective_protocol_hash") != self.protocol_hash:
            raise RuntimeError("completed run protocol hash mismatch")
        hashes = _files(run_dir)
        if not hashes:
            raise RuntimeError("completed run has no durable files")
        return {"identity": {k: status[k] for k in ("scenario", "algorithm", "seed")},
                "source_artifact_hashes": hashes}

    def _manifest_path(self, identity: Mapping[str, Any]) -> Path:
        algorithm = str(identity["algorithm"]).replace("/", "_")
        return self.manifest_dir / f"{identity['scenario']}__{algorithm}__{identity['seed']}.json"

    def mirror_run(self, run_dir: Path) -> dict[str, Any]:
        checked = self.validate(run_dir)
        identity = checked["identity"]
        target = self.mirror / "scenarios" / str(identity["scenario"]) / "runs" / run_dir.name
        temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(run_dir, temporary)
        mirror_hashes = _files(temporary)
        if mirror_hashes != checked["source_artifact_hashes"]:
            raise RuntimeError("recovery mirror hash mismatch")
        if target.exists():
            existing = _files(target)
            if existing != mirror_hashes:
                raise RuntimeError("existing recovery mirror drift")
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, target)
        manifest = {
            **checked, "mirror_artifact_hashes": mirror_hashes,
            "mirror_relative_path": target.relative_to(self.mirror).as_posix(),
            "mirrored_at": datetime.now().astimezone().isoformat(), "mirror_valid": True,
        }
        safe_atomic_write_json(self._manifest_path(identity), manifest)
        return manifest

    def manifests(self) -> list[dict[str, Any]]:
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self.manifest_dir.glob("*.json"))]

    def _restore(self, manifest: Mapping[str, Any]) -> None:
        source = self.mirror / str(manifest["mirror_relative_path"])
        identity = manifest["identity"]
        target = self.formal / "scenarios" / str(identity["scenario"]) / "runs" / source.name
        for relative, digest in manifest["mirror_artifact_hashes"].items():
            source_file, target_file = source / relative, target / relative
            if _sha(source_file) != digest:
                raise RuntimeError("mirror corrupted during recovery")
            if target_file.exists() and _sha(target_file) == digest:
                continue
            target_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = target_file.with_name(target_file.name + f".restore-{os.getpid()}")
            shutil.copy2(source_file, temporary)
            os.replace(temporary, target_file)

    def watchdog(self, previous_count: int) -> tuple[int, int]:
        manifests = self.manifests()
        if len(manifests) < previous_count:
            raise RuntimeError("mirrored completed count decreased")
        recovered = 0
        marker = self.formal / "FORMAL_ACTIVE.json"
        if not marker.exists():
            active = json.loads((self.mirror / "FORMAL_ACTIVE.json").read_text(encoding="utf-8"))
            safe_atomic_write_json(marker, active)
            recovered += 1
        for manifest in manifests:
            identity = manifest["identity"]
            name = f"{str(identity['algorithm']).replace('/', '_')}__seed_{identity['seed']}"
            target = self.formal / "scenarios" / str(identity["scenario"]) / "runs" / name
            expected = manifest["source_artifact_hashes"]
            actual = _files(target) if target.exists() else {}
            if any(actual.get(path) != digest for path, digest in expected.items()):
                self._restore(manifest)
                recovered += 1
            self.validate(target)
        return len(manifests), recovered

    def write_progress(self, *, failed: int, running: str | None,
                       last_completed: Mapping[str, Any] | None, recovered: int) -> None:
        count = len(self.manifests())
        safe_atomic_write_json(self.mirror / "durable_progress.json", {
            "persisted_completed": count, "mirrored_completed": count,
            "failed": failed, "running": running, "last_completed": last_completed,
            "recovered_from_mirror_count": recovered,
            "timestamp": datetime.now().astimezone().isoformat(),
        })
