from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from g1_aprilcube_calibration.camera_initialization import (
    realsense_link_T_color_optical,
)
from g1_aprilcube_calibration.dataset_builder import CalibrationSample
from g1_aprilcube_calibration.joint_map import G1_29_JOINT_NAMES
from g1_aprilcube_calibration.robot_calibration_bridge import (
    CAMERA_OPTICAL_FRAME,
    CAMERA_OPTICAL_JOINT,
    add_color_optical_frame,
    build_optimizer_config,
    sample_to_observation_record,
)
from g1_aprilcube_calibration.urdf_model import URDFModel

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"


def _sample() -> CalibrationSample:
    return CalibrationSample(
        capture_id="capture_001",
        pose_id="pose_001",
        frame_id="capture_001_000",
        raw_image_path="raw/images/capture_001_000.png",
        raw_image_sha256="a" * 64,
        camera_info={
            "frame_id": CAMERA_OPTICAL_FRAME,
            "height": 720,
            "width": 1280,
            "distortion_model": "plumb_bob",
            "d": [0.0] * 5,
            "k": [1.0] * 9,
            "r": [1.0] * 9,
            "p": [1.0] * 12,
        },
        measured_state={
            "mode_machine": 5,
            "position": [index / 100.0 for index in range(29)],
        },
        pairing={},
        visible_tag_ids=(1,),
        corner_tag_ids=(1, 1, 1, 1),
        image_points_px=((1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)),
        object_points_m=(
            (0.0, 0.0, 0.0),
            (0.01, 0.0, 0.0),
            (0.01, 0.01, 0.0),
            (0.0, 0.01, 0.0),
        ),
        correspondence_sha256="b" * 64,
    )


def test_optical_overlay_matches_existing_nominal_convention(tmp_path: Path) -> None:
    augmented = add_color_optical_frame(URDF.read_text(encoding="utf-8"))
    path = tmp_path / "robot_calibration_overlay.urdf"
    path.write_text(augmented, encoding="utf-8")
    try:
        model = URDFModel(path)
        actual = model.transform("d435_link", CAMERA_OPTICAL_FRAME, {})
    finally:
        path.unlink(missing_ok=True)
    np.testing.assert_allclose(actual, realsense_link_T_color_optical(), atol=1e-12)
    root = ET.fromstring(augmented)
    assert root.find(f"./link[@name='{CAMERA_OPTICAL_FRAME}']") is not None
    assert root.find(f"./joint[@name='{CAMERA_OPTICAL_JOINT}']") is not None


def test_sample_maps_to_paired_native_observations() -> None:
    record = sample_to_observation_record(_sample())
    assert record["joint_names"] == list(G1_29_JOINT_NAMES)
    assert len(record["joint_positions"]) == 29
    assert record["arm_feature_frame"] == "aprilcube_target"
    assert record["camera_feature_frame"] == CAMERA_OPTICAL_FRAME
    assert len(record["object_points_m"]) == len(record["image_points_px"]) == 4


def test_optimizer_config_uses_native_models_and_aggregate_joint_prior() -> None:
    target = np.eye(4)
    target[:3, 3] = [0.02, -0.03, 0.04]
    document = build_optimizer_config(
        hand_T_target=target,
        calibration_arm="left",
        sample_count=25,
        calibrate_shoulder_roll=True,
        shoulder_roll_prior_sigma_deg=5.0,
    )
    parameters = document["robot_calibration"]["ros__parameters"]
    step = parameters["aprilcube_calibration"]
    assert step["arm"] == {"type": "chain3d", "frame": "left_rubber_hand"}
    assert step["camera"]["type"] == "camera2d"
    assert step["free_frames"] == ["d435_joint", "aprilcube_target"]
    assert step["free_params"] == ["left_shoulder_roll_joint"]
    expected_scale = 1.0 / (math.radians(5.0) * math.sqrt(25))
    assert step["shoulder_roll_prior"]["joint_scale"] == pytest.approx(
        expected_scale
    )


def test_extrinsics_config_has_no_joint_offset() -> None:
    document = build_optimizer_config(
        hand_T_target=np.eye(4),
        calibration_arm="right",
        sample_count=39,
        calibrate_shoulder_roll=False,
    )
    step = document["robot_calibration"]["ros__parameters"][
        "aprilcube_calibration"
    ]
    assert "free_params" not in step
    assert step["arm"]["frame"] == "right_rubber_hand"
    assert step["error_blocks"] == ["aprilcube_reprojection"]
