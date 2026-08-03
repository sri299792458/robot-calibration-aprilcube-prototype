"""Atomic YAML storage for versioned manually taught pose sets."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import yaml

from g1_aprilcube_calibration.models import utc_now_iso
from g1_aprilcube_calibration.pose_schema import (
    PoseAuditEvent,
    PoseRecord,
    PoseSet,
)


class PoseStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.backup_path = self.path.with_suffix(self.path.suffix + ".bak")

    def initialize(self, pose_set: PoseSet, *, overwrite: bool = False) -> None:
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"pose store already exists: {self.path}")
        self._write(pose_set, create_backup=overwrite)

    def load(self) -> PoseSet:
        with self.path.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, dict):
            raise TypeError(f"pose store must contain a mapping: {self.path}")
        return PoseSet.from_dict(data)

    def append(self, pose: PoseRecord, *, details: dict | None = None) -> PoseSet:
        pose_set = self.load()
        event = PoseAuditEvent(
            action="add",
            pose_id=pose.id,
            occurred_at_utc=utc_now_iso(),
            details={} if details is None else details,
        )
        updated = pose_set.with_pose(pose, event)
        self._write(updated)
        return updated

    def undo_last(self, *, reason: str) -> PoseSet:
        if not reason.strip():
            raise ValueError("undo reason must be non-empty")
        pose_set = self.load()
        if not pose_set.poses:
            raise ValueError("cannot undo an empty pose store")
        event = PoseAuditEvent(
            action="undo",
            pose_id=pose_set.poses[-1].id,
            occurred_at_utc=utc_now_iso(),
            details={"reason": reason},
        )
        updated = pose_set.without_last_pose(event)
        self._write(updated)
        return updated

    def _write(self, pose_set: PoseSet, *, create_backup: bool = True) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = yaml.safe_dump(
            pose_set.to_dict(),
            sort_keys=False,
            allow_unicode=True,
        )
        if create_backup and self.path.exists():
            shutil.copy2(self.path, self.backup_path)

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
            text=True,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.path)
            directory_descriptor = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
