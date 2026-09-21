from pathlib import Path

import numpy as np
import pytest

from g1_aprilcube_calibration.pose_schema import PoseAuditEvent, PoseRecord, PoseSet
from g1_aprilcube_calibration.replay_targets import back_off_replay_targets
from g1_aprilcube_calibration.urdf_model import URDFModel

ROOT = Path(__file__).parents[1]
URDF = ROOT / "unitree_ros" / "robots" / "g1_description" / "g1_29dof_rev_1_0.urdf"
UTC = "2026-08-02T12:00:00Z"


def _pose_set(model: URDFModel, calibration_q: np.ndarray) -> PoseSet:
    full = np.zeros(29)
    full[15:22] = calibration_q
    pose = PoseRecord(
        id="target",
        group="test",
        measured_calibration_q=tuple(calibration_q),
        measured_full_q=tuple(full),
        calibration_q_spread=(0.0,) * 7,
        recorded_at_utc=UTC,
        recorded_monotonic_s=1.0,
    )
    source = PoseSet(
        robot_model="g1_29dof_rev_1_0",
        mode_machine=5,
        urdf_sha256=model.sha256,
        calibration_arm="left",
    )
    return source.with_pose(pose, PoseAuditEvent("add", pose.id, UTC))


def test_backoff_applies_to_every_joint_and_preserves_measurements() -> None:
    model = URDFModel(URDF)
    names = (
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    )
    limits = model.joint_limits(names)
    measured = np.zeros(7)
    measured[0] = limits[0].lower + 0.001
    measured[4] = limits[4].upper - 0.002
    measured[6] = limits[6].lower + 0.003
    source = _pose_set(model, measured)
    source_hash = source.content_sha256

    derived, adjustments = back_off_replay_targets(
        source,
        model,
        joint_limit_margin_rad=0.03,
    )

    result = derived.poses[0]
    assert source.content_sha256 == source_hash
    assert result.measured_calibration_q == tuple(measured)
    assert result.command_calibration_q[0] == pytest.approx(limits[0].lower + 0.03)
    assert result.command_calibration_q[4] == pytest.approx(limits[4].upper - 0.03)
    assert result.command_calibration_q[6] == pytest.approx(limits[6].lower + 0.03)
    assert result.command_calibration_q[1:4] == (0.0, 0.0, 0.0)
    assert {item.joint_name for item in adjustments} == {
        names[0],
        names[4],
        names[6],
    }
    assert {item.limit_side for item in adjustments} == {"lower", "upper"}
    assert derived.content_sha256 != source_hash


def test_backoff_leaves_safe_pose_without_replay_override() -> None:
    model = URDFModel(URDF)
    source = _pose_set(model, np.zeros(7))

    derived, adjustments = back_off_replay_targets(
        source,
        model,
        joint_limit_margin_rad=0.03,
    )

    assert adjustments == ()
    assert derived.poses[0].replay_calibration_q is None
    assert derived.content_sha256 == source.content_sha256
