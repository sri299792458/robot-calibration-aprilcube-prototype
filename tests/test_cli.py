import json
from pathlib import Path

import cv2
import numpy as np
from aprilcube.generate import DICT_MAP

from g1_aprilcube_calibration.cli import main
from g1_aprilcube_calibration.hardware_cli import (
    _load_session_plan,
    _manual_pose_set_for_session,
    _next_pose_id,
    _unique_route_edges,
)
from g1_aprilcube_calibration.pose_schema import PoseRecord, PoseSet
from g1_aprilcube_calibration.pose_store import PoseStore

ROOT = Path(__file__).parents[1]


def _write_marker(path: Path, tag_id: int = 0) -> None:
    image = np.full((480, 640, 3), 220, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(DICT_MAP["4x4_100"])
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, 180)
    image[120:300, 230:410] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    assert cv2.imwrite(str(path), image)


def test_headless_preview_writes_image_and_report(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "preview.png"
    report = tmp_path / "report.json"
    _write_marker(source)

    exit_code = main(
        [
            "preview",
            "--image",
            str(source),
            "--output",
            str(output),
            "--report-json",
            str(report),
            "--no-window",
        ]
    )

    assert exit_code == 0
    assert output.is_file()
    assert report.is_file()
    assert '"grade": "yellow"' in report.read_text()


def test_headless_preview_returns_nonzero_for_no_detection(tmp_path: Path) -> None:
    source = tmp_path / "blank.png"
    assert cv2.imwrite(str(source), np.full((480, 640, 3), 220, dtype=np.uint8))

    exit_code = main(
        [
            "preview",
            "--image",
            str(source),
            "--no-window",
        ]
    )

    assert exit_code == 1


def test_pose_set_cli_summarizes_session_owned_pose_set(tmp_path, capsys):
    pose_path = tmp_path / "poses.yaml"
    pose_set = PoseSet(
        robot_model="g1_29dof_rev_1_0",
        mode_machine=5,
        urdf_sha256="a" * 64,
        calibration_arm="left",
    )
    PoseStore(pose_path).initialize(pose_set)

    assert main(["pose-summary", "--pose-set", str(pose_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["pose_count"] == 0
    assert len(output["content_sha256"]) == 64


def test_manual_teacher_builds_empty_pose_set_from_hardware(tmp_path):
    args = type(
        "Args",
        (),
        {
            "session_directory": tmp_path / "session",
            "hardware_config": ROOT / "config" / "hardware.yaml",
        },
    )()

    pose_set = _manual_pose_set_for_session(args)

    assert pose_set.schema_version == 3
    assert pose_set.robot_model == "g1_29dof_rev_1_0"
    assert pose_set.calibration_arm == "left"
    assert pose_set.poses == ()
    assert "handoff_q" not in pose_set.to_dict()
    assert "hold_q" not in pose_set.to_dict()


def test_session_plan_contains_only_camera_capture_poses(tmp_path):
    path = tmp_path / "session_plan.yaml"
    path.write_text("schema_version: 2\ncapture_pose_ids: [pose_001, pose_002]\n")
    plan = _load_session_plan(path)
    assert plan.capture_pose_ids == ("pose_001", "pose_002")


def test_dynamic_route_edges_preserve_order_and_remove_repeated_traversals():
    assert _unique_route_edges(
        [
            "__handoff__",
            "pose_001",
            "pose_004",
            "pose_005",
            "pose_001",
            "pose_004",
            "pose_005",
            "pose_001",
            "__handoff__",
        ]
    ) == (
        ("__handoff__", "pose_001"),
        ("pose_001", "pose_004"),
        ("pose_004", "pose_005"),
        ("pose_005", "pose_001"),
        ("pose_001", "__handoff__"),
    )


def test_artifact_inspection_reports_hardware_ready_left_arm_mount(capsys):
    assert main(["inspect-artifacts"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["hardware_ready"] is True
    assert output["blocking_reasons"] == []


def test_commissioning_acknowledgement_fails_before_sdk_import(capsys):
    exit_code = main(
        [
            "commission-weight-zero",
            "--network-interface",
            "not-a-real-interface",
            "--confirm",
            "wrong phrase",
        ]
    )
    assert exit_code == 2
    assert "must exactly equal" in capsys.readouterr().err


def test_pose_teacher_allocates_first_calibration_pose_then_next_number(tmp_path):
    pose_path = tmp_path / "poses.yaml"
    PoseStore(pose_path).initialize(
        PoseSet(
            robot_model="g1_29dof_rev_1_0",
            mode_machine=5,
            urdf_sha256="a" * 64,
            calibration_arm="left",
        )
    )
    store = PoseStore(pose_path)
    assert _next_pose_id(store, "pose_001") == "pose_001"
    full = np.zeros(29)
    pose = PoseRecord(
        "pose_001",
        "calibration",
        (0.0,) * 7,
        tuple(full),
        (0.0,) * 7,
        "2026-08-02T12:00:00Z",
        1.0,
    )
    store.append(pose, details={})
    second = PoseRecord(
        "pose_003",
        "calibration",
        (0.0,) * 7,
        tuple(full),
        (0.0,) * 7,
        "2026-08-02T12:00:00Z",
        2.0,
    )
    store.append(second, details={})
    assert _next_pose_id(store, "pose_001") == "pose_002"
