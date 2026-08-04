import numpy as np
import pytest

from g1_aprilcube_calibration.models import RobotStateSample
from g1_aprilcube_calibration.pose_recorder import (
    PoseRecorder,
    PoseRecordingRequest,
)
from g1_aprilcube_calibration.pose_schema import PoseSet
from g1_aprilcube_calibration.pose_store import PoseStore
from g1_aprilcube_calibration.quality import QualityGrade, QualityReport
from g1_aprilcube_calibration.readiness import RecordingGateConfig, StateSampleBuffer
from g1_aprilcube_calibration.timestamp_pairing import ImageTiming, PairingConfig

UTC = "2026-08-02T12:00:00Z"


def report(grade: QualityGrade) -> QualityReport:
    return QualityReport(
        grade=grade,
        hard_failures=("no tag",) if grade is QualityGrade.RED else (),
        warnings=("only one face",) if grade is QualityGrade.YELLOW else (),
        metrics={"tag_ids": [0, 5]},
        signature=None,
        pose_diagnostic=None,
    )


def recorder(tmp_path, *, hold_offset_rad: float = 0.0) -> PoseRecorder:
    store = PoseStore(tmp_path / "poses.yaml")
    store.initialize(
        PoseSet(
            robot_model="g1_29dof_rev_1_0",
            mode_machine=5,
            urdf_sha256="a" * 64,
            calibration_arm="left",
        )
    )
    buffer = StateSampleBuffer()
    for index, time_s in enumerate(np.arange(9.7, 10.31, 0.05)):
        q = np.arange(29, dtype=float) / 100.0
        q[15:22] += (index % 3 - 1) * 0.0005
        q[22:29] += hold_offset_rad
        buffer.add(RobotStateSample(time_s, UTC, 5, q, np.zeros(29), np.zeros(29)))
    return PoseRecorder(
        store=store,
        state_buffer=buffer,
        gate_config=RecordingGateConfig(
            calibration_arm="left",
            state_freshness_timeout_s=0.1,
            stationary_duration_s=0.4,
            maximum_state_gap_s=0.06,
            maximum_calibration_position_spread_rad=0.01,
            minimum_samples=5,
        ),
        pairing_config=PairingConfig(0.05, 0.1),
    )


def request(grade: QualityGrade = QualityGrade.GREEN, **kwargs):
    defaults = {
        "pose_id": "pose_001",
        "group": "center",
        "image_timing": ImageTiming(10.1, UTC, 123),
        "visual_report": report(grade),
        "preview_path": "previews/pose_001.png",
    }
    defaults.update(kwargs)
    return PoseRecordingRequest(**defaults)


def test_green_pose_records_measured_median_and_evidence(tmp_path) -> None:
    subject = recorder(tmp_path)
    pose_set = subject.record(request(), now_monotonic_s=10.31)
    pose = pose_set.poses[0]

    assert pose_set.calibration_arm == "left"
    assert np.allclose(pose.measured_calibration_q, np.arange(15, 22) / 100.0)
    assert max(pose.calibration_q_spread) == pytest.approx(0.001)
    assert pose.visual_quality["state_readiness"]["ready"]
    assert pose.visual_quality["timestamp_pairing"]["image_header_stamp_ns"] == 123
    assert subject.store.load().content_sha256 == pose_set.content_sha256


def test_stationary_hold_arm_may_settle_away_from_ready_handoff(tmp_path) -> None:
    subject = recorder(tmp_path, hold_offset_rad=0.25)

    assessment = subject.assess(request(), now_monotonic_s=10.31)

    assert assessment.allowed
    assert assessment.readiness is not None
    assert assessment.readiness.maximum_hold_position_spread_rad == pytest.approx(0)


@pytest.mark.parametrize(
    ("pose_request", "message"),
    [
        (request(QualityGrade.RED), "visual"),
        (request(QualityGrade.YELLOW), "override reason"),
    ],
)
def test_hard_gates_do_not_write_pose(tmp_path, pose_request, message: str) -> None:
    subject = recorder(tmp_path)
    with pytest.raises(ValueError, match=message):
        subject.record(pose_request, now_monotonic_s=10.31)
    assert subject.store.load().poses == ()


def test_yellow_pose_requires_and_records_reason(tmp_path) -> None:
    subject = recorder(tmp_path)
    pose_set = subject.record(
        request(QualityGrade.YELLOW, yellow_override_reason="needed edge coverage"),
        now_monotonic_s=10.31,
    )
    assert pose_set.poses[0].visual_quality["yellow_override_reason"] == (
        "needed edge coverage"
    )


def test_nonstationary_or_unpaired_state_cannot_be_recorded(tmp_path) -> None:
    subject = recorder(tmp_path)
    subject.state_buffer.add(
        RobotStateSample(
            10.35,
            UTC,
            5,
            np.zeros(29),
            np.ones(29) * 0.1,
            np.zeros(29),
        )
    )
    assessment = subject.assess(
        request(image_timing=ImageTiming(10.34, UTC)), now_monotonic_s=10.36
    )
    assert not assessment.allowed
    assert any(
        "window" in reason or "spread" in reason for reason in assessment.failures
    )
