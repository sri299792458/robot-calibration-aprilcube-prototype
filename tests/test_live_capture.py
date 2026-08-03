from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from aprilcube.generate import DICT_MAP

from aprilcube import CorrespondenceDetector
from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.clock import ManualClock
from g1_aprilcube_calibration.config import QualityThresholds
from g1_aprilcube_calibration.live_capture import LiveBurstConfig, LiveBurstFrameSource
from g1_aprilcube_calibration.models import RobotStateSample
from g1_aprilcube_calibration.quality import PoseQualityEvaluator
from g1_aprilcube_calibration.readiness import RecordingGateConfig, StateSampleBuffer
from g1_aprilcube_calibration.ros.camera_adapter import ROSFrameBuffer, ROSImageFrame
from g1_aprilcube_calibration.timestamp_pairing import ImageTiming, PairingConfig

ROOT = Path(__file__).parents[1]
TARGET = ROOT / "aprilcube" / "models" / "dex3_safe_cube" / "config.json"
UTC = "2026-08-02T12:00:00Z"


def marker_image():
    image = np.full((480, 640, 3), 220, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(DICT_MAP["4x4_100"])
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 180)
    image[140:320, 230:410] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return image


def info():
    return RectifiedCameraInfo(
        640,
        480,
        "color_optical",
        "head_color",
        "TEST",
        "plumb_bob",
        (0.0,) * 5,
        (600.0, 0.0, 319.5, 0.0, 600.0, 239.5, 0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        (600.0, 0.0, 319.5, 0.0, 0.0, 600.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0),
    )


def thresholds():
    return QualityThresholds.from_yaml(ROOT / "config" / "capture_quality.yaml")


def test_live_source_waits_for_new_stationary_confirmed_frames():
    frames = ROSFrameBuffer()
    states = StateSampleBuffer()
    for time_s in np.arange(0.5, 2.01, 0.02):
        states.add(RobotStateSample(time_s, UTC, 5, np.zeros(29), np.zeros(29)))
    scheduled = [
        ROSImageFrame(marker_image(), ImageTiming(time_s, UTC, index), info())
        for index, time_s in enumerate((1.0, 1.5), start=1)
    ]
    clock = ManualClock()

    def wait_once(duration):
        clock.advance(duration)
        if scheduled:
            frames.add(scheduled.pop(0))

    source = LiveBurstFrameSource(
        camera_frames=frames,
        robot_states=states,
        detector=CorrespondenceDetector(TARGET),
        quality_evaluator=PoseQualityEvaluator(thresholds()),
        recording_config=RecordingGateConfig(
            calibration_arm="left",
            state_freshness_timeout_s=0.1,
            stationary_duration_s=0.5,
            maximum_state_gap_s=0.1,
            maximum_calibration_position_spread_rad=0.01,
            minimum_samples=5,
        ),
        pairing_config=PairingConfig(0.05, 0.1),
        config=LiveBurstConfig(frame_count=2, timeout_s=1, poll_interval_s=0.01),
        clock=clock,
        wait_once=wait_once,
        accept_yellow=lambda _frame: True,
    )
    burst = source.capture_burst(pose_id="pose", capture_id="capture_001")
    assert [frame.frame_id for frame in burst] == [
        "capture_001_000",
        "capture_001_001",
    ]
    assert all(frame.correspondences.valid for frame in burst)
    assert all(len(frame.state_window) >= 25 for frame in burst)


def test_live_source_keeps_frame_until_future_state_bracket_arrives():
    frames = ROSFrameBuffer()
    states = StateSampleBuffer()
    for time_s in np.arange(0.5, 1.01, 0.02):
        states.add(RobotStateSample(time_s, UTC, 5, np.zeros(29), np.zeros(29)))
    clock = ManualClock()
    frame_added = [False]

    def wait_once(duration):
        clock.advance(duration)
        if not frame_added[0]:
            frames.add(
                ROSImageFrame(marker_image(), ImageTiming(1.0, UTC, 1), info())
            )
            frame_added[0] = True
            return
        next_time = states.latest.receipt_monotonic_s + 0.02
        states.add(
            RobotStateSample(next_time, UTC, 5, np.zeros(29), np.zeros(29))
        )

    source = LiveBurstFrameSource(
        camera_frames=frames,
        robot_states=states,
        detector=CorrespondenceDetector(TARGET),
        quality_evaluator=PoseQualityEvaluator(thresholds()),
        recording_config=RecordingGateConfig(
            calibration_arm="left",
            state_freshness_timeout_s=0.1,
            stationary_duration_s=0.5,
            maximum_state_gap_s=0.1,
            maximum_calibration_position_spread_rad=0.01,
            minimum_samples=5,
        ),
        pairing_config=PairingConfig(0.05, 0.1),
        config=LiveBurstConfig(frame_count=1, timeout_s=1, poll_interval_s=0.01),
        clock=clock,
        wait_once=wait_once,
        accept_yellow=lambda _frame: True,
    )
    burst = source.capture_burst(pose_id="pose", capture_id="capture_001")
    assert burst[0].frame_id == "capture_001_000"
    assert burst[0].state_window[-1].receipt_monotonic_s >= 1.25
