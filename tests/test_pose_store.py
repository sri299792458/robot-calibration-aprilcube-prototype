from copy import deepcopy

import numpy as np
import pytest
import yaml
from jsonschema import ValidationError

from g1_aprilcube_calibration.pose_schema import (
    PoseAuditEvent,
    PoseRecord,
    PoseSet,
)
from g1_aprilcube_calibration.pose_store import PoseStore

UTC = "2026-08-02T12:00:00Z"


def make_pose(pose_id: str = "pose_001") -> PoseRecord:
    full = np.arange(29, dtype=float) / 100.0
    return PoseRecord(
        id=pose_id,
        group="center",
        measured_calibration_q=tuple(full[15:22]),
        measured_full_q=tuple(full),
        calibration_q_spread=(0.001,) * 7,
        recorded_at_utc=UTC,
        recorded_monotonic_s=10.0,
        preview_path="previews/pose_001.png",
        visual_quality={"grade": "green", "visible_ids": [0, 5]},
    )


def empty_pose_set() -> PoseSet:
    return PoseSet(
        robot_model="g1_29dof_rev_1_0",
        mode_machine=5,
        urdf_sha256="a" * 64,
        calibration_arm="left",
        handoff_q=(0.0,) * 7,
        hold_q=(0.0,) * 7,
    )


def test_pose_record_excludes_camera_witness_state() -> None:
    assert "head_witness_ack" not in make_pose().to_dict()


def test_pose_set_requires_calibration_arm_to_match_full_state() -> None:
    pose = make_pose()
    data = empty_pose_set().to_dict()
    data["poses"] = [pose.to_dict()]
    data["poses"][0]["measured_calibration_q"][0] += 0.1
    data["audit_log"] = [PoseAuditEvent("add", pose.id, UTC).to_dict()]
    data["content_sha256"] = empty_pose_set().content_sha256
    with pytest.raises(ValueError, match="does not match"):
        PoseSet.from_dict(data, verify_hash=False)


def test_atomic_store_append_backup_and_undo(tmp_path) -> None:
    path = tmp_path / "poses.yaml"
    store = PoseStore(path)
    store.initialize(empty_pose_set())
    original_text = path.read_text()

    appended = store.append(make_pose(), details={"operator": "test"})

    assert [pose.id for pose in appended.poses] == ["pose_001"]
    assert appended.audit_log[-1].action == "add"
    assert store.backup_path.read_text() == original_text
    assert store.load() == appended
    assert not list(tmp_path.glob("*.tmp"))

    undone = store.undo_last(reason="bad visual coverage")
    assert undone.poses == ()
    assert [event.action for event in undone.audit_log] == ["add", "undo"]
    assert undone.audit_log[-1].details["reason"] == "bad visual coverage"


def test_store_refuses_overwrite_by_default(tmp_path) -> None:
    store = PoseStore(tmp_path / "poses.yaml")
    store.initialize(empty_pose_set())
    with pytest.raises(FileExistsError):
        store.initialize(empty_pose_set())


def test_duplicate_pose_id_is_rejected(tmp_path) -> None:
    store = PoseStore(tmp_path / "poses.yaml")
    store.initialize(empty_pose_set())
    store.append(make_pose())
    with pytest.raises(ValueError, match="duplicate"):
        store.append(make_pose())


def test_content_tampering_is_detected(tmp_path) -> None:
    path = tmp_path / "poses.yaml"
    store = PoseStore(path)
    store.initialize(empty_pose_set())
    store.append(make_pose())
    data = yaml.safe_load(path.read_text())
    data["poses"][0]["measured_full_q"][0] = 123.0
    path.write_text(yaml.safe_dump(data, sort_keys=False))

    with pytest.raises(ValueError, match="SHA-256"):
        store.load()


def test_schema_and_audit_sequence_are_validated() -> None:
    pose = make_pose()
    event = PoseAuditEvent("add", pose.id, UTC)
    pose_set = empty_pose_set().with_pose(pose, event)
    data = pose_set.to_dict()

    wrong_version = deepcopy(data)
    wrong_version["schema_version"] = 1
    with pytest.raises(ValidationError):
        PoseSet.from_dict(wrong_version)

    with pytest.raises(ValueError, match="audit log"):
        PoseSet(
            robot_model=pose_set.robot_model,
            mode_machine=5,
            urdf_sha256=pose_set.urdf_sha256,
            calibration_arm=pose_set.calibration_arm,
            handoff_q=pose_set.handoff_q,
            hold_q=pose_set.hold_q,
            poses=(pose,),
            audit_log=(),
        )
