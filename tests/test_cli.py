import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from aprilcube.generate import DICT_MAP

from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.cli import build_parser, main
from g1_aprilcube_calibration.dataset_builder import CalibrationDataset
from g1_aprilcube_calibration.hardware_cli import (
    MOTION_ACK,
    _collect_new_camera_frames,
    _DeferredSIGINT,
    _executor_config,
    _hardware_document_sha256,
    _load_session_plan,
    _manual_pose_set_for_session,
    _next_pose_id,
    _raise_safety_cleanup_failures,
    _retry_table_observation,
    _run_isolated_control_work,
    _shutdown_rclpy_once,
    _unique_route_edges,
    _wait_for_space,
    _write_table_failure_bundle,
    _write_table_image_burst,
)
from g1_aprilcube_calibration.pose_schema import PoseAuditEvent, PoseRecord, PoseSet
from g1_aprilcube_calibration.pose_store import PoseStore
from g1_aprilcube_calibration.ros.camera_adapter import ROSFrameBuffer, ROSImageFrame
from g1_aprilcube_calibration.timestamp_pairing import ImageTiming
from g1_aprilcube_calibration.urdf_model import URDFModel
from g1_aprilcube_calibration.workflow_cli import (
    _validated_modeled_target_initialization,
)

ROOT = Path(__file__).parents[1]


def _delayed_isolated_sum(*, values):
    time.sleep(0.2)
    return sum(values)


def test_automatic_collection_commands_have_unambiguous_defaults() -> None:
    parser = build_parser()
    design = parser.parse_args(
        [
            "design-auto-collection",
            "--reference-state-json",
            "state.json",
            "--output-directory",
            "work/plan",
            "--calibration-result",
            "result.json",
        ]
    )
    assert design.calibration_result == Path("result.json")
    collect = parser.parse_args(
        [
            "collect-auto",
            "--network-interface",
            "eth0",
            "--plan",
            "work/plan/authored_plan.json",
            "--session-directory",
            "sessions/auto",
            "--session-id",
            "auto",
            "--image-topic",
            "/camera/image",
            "--camera-info-topic",
            "/camera/info",
            "--camera-name",
            "head",
            "--camera-serial",
            "123",
            "--confirm",
            MOTION_ACK,
        ]
    )

    assert design.target_count == 80
    assert design.candidate_count == 1600
    assert design.workers is None
    assert design.hardware_config == ROOT / "config/hardware_dex3_aruco.yaml"
    assert design.target_config == ROOT / "config/dex3_dorsal_aruco_target.json"
    assert design.collision_config == ROOT / "config/collision_pairs_dex3_aruco.yaml"
    assert design.urdf == ROOT / "config/urdf/g1_29dof_rev_1_0_g1pilot_collision.urdf"
    assert collect.burst_timeout_s == 3.0
    assert collect.hardware_config == ROOT / "config/hardware_dex3_aruco.yaml"
    assert collect.target_config == ROOT / "config/dex3_dorsal_aruco_target.json"
    assert collect.quality_config == ROOT / "config/capture_quality_dex3_aruco.yaml"
    commission = parser.parse_args(
        [
            "commission-dex3-middle-close",
            "--network-interface",
            "eth0",
            "--confirm",
            MOTION_ACK,
        ]
    )
    assert commission.hardware_config == ROOT / "config/hardware_dex3_aruco.yaml"
    assert (
        commission.collision_config == ROOT / "config/collision_pairs_dex3_aruco.yaml"
    )
    solve = parser.parse_args(
        [
            "solve",
            "--dataset",
            "dataset.json",
            "--output-directory",
            "work/solve",
        ]
    )
    assert solve.hardware_config == ROOT / "config/hardware_dex3_aruco.yaml"
    assert solve.target_config == ROOT / "config/dex3_dorsal_aruco_target.json"
    assert solve.holdout_fraction == pytest.approx(0.2)
    assert solve.target_transform_mode == "fixed"
    assert solve.native_runner == ROOT / "tools/g1_robot_calibration.sh"
    assert solve.robot_calibration_directory == ROOT / "robot_calibration"
    assert solve.free_joint_offset == []
    assert solve.all_arm_joint_offsets is False
    assert not hasattr(solve, "pnp_sample_index")
    with pytest.raises(SystemExit):
        parser.parse_args(["export-robot-calibration"])


def test_dex3_hardware_runtime_control_config_loads() -> None:
    config, rate_hz = _executor_config(
        ROOT / "config" / "hardware_dex3_aruco.yaml"
    )

    assert rate_hz == pytest.approx(250.0)
    assert config.nominal_tick_period_s == pytest.approx(0.004)
    assert config.control_gap_fault_s == pytest.approx(0.25)


def test_dex3_solver_initialization_is_bound_to_frozen_target_artifact() -> None:
    target = ROOT / "config/dex3_dorsal_aruco_target.json"
    dataset = CalibrationDataset(
        session_id="dex3",
        session_manifest_sha256="a" * 64,
        target_artifact_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        pose_set_sha256="b" * 64,
        urdf_sha256="c" * 64,
        calibration_arm="right",
        observation_phase="held",
        samples=(),
    )

    hardware, hand_T_target = _validated_modeled_target_initialization(
        dataset,
        hardware_path=ROOT / "config/hardware_dex3_aruco.yaml",
        target_path=target,
    )

    np.testing.assert_allclose(
        hand_T_target,
        hardware["robot"]["calibration_target_modeled_hand_T_target"],
    )

    with pytest.raises(ValueError, match="dataset target differs"):
        _validated_modeled_target_initialization(
            dataset,
            hardware_path=ROOT / "config/hardware_dex3_aruco.yaml",
            target_path=ROOT / "aprilcube/models/dex3_safe_cube/config.json",
        )


def _write_marker(path: Path, tag_id: int = 4) -> None:
    image = np.full((480, 640, 3), 220, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(DICT_MAP["6x6_50"])
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


def test_prepare_replay_emits_adjusted_pose_set_and_full_order(tmp_path, capsys):
    urdf = ROOT / "unitree_ros" / "robots" / "g1_description" / "g1_29dof_rev_1_0.urdf"
    model = URDFModel(urdf)
    source = PoseSet(
        robot_model="g1_29dof_rev_1_0",
        mode_machine=5,
        urdf_sha256=model.sha256,
        calibration_arm="left",
    )
    wrist_limit = model.joint_limits(source.joint_order)[4].upper
    for index, wrist_roll in enumerate((wrist_limit - 0.001, 0.5), start=1):
        measured = np.zeros(7)
        measured[4] = wrist_roll
        full = np.zeros(29)
        full[15:22] = measured
        pose = PoseRecord(
            id=f"pose_{index:03d}",
            group="test",
            measured_calibration_q=tuple(measured),
            measured_full_q=tuple(full),
            calibration_q_spread=(0.0,) * 7,
            recorded_at_utc="2026-08-02T12:00:00Z",
            recorded_monotonic_s=float(index),
        )
        source = source.with_pose(
            pose, PoseAuditEvent("add", pose.id, "2026-08-02T12:00:00Z")
        )
    source_path = tmp_path / "source.yaml"
    output = tmp_path / "replay"
    PoseStore(source_path).initialize(source)

    assert (
        main(
            [
                "prepare-replay",
                "--pose-set",
                str(source_path),
                "--output-directory",
                str(output),
                "--urdf",
                str(urdf),
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    derived = PoseStore(output / "pose_set.yaml").load()
    edges = yaml.safe_load((output / "edges.yaml").read_text())
    plan = yaml.safe_load((output / "session_plan.yaml").read_text())
    assert summary["pose_count"] == 2
    assert summary["adjusted_pose_count"] == 1
    assert derived.poses[0].measured_calibration_q[4] == wrist_limit - 0.001
    assert derived.poses[0].command_calibration_q[4] == pytest.approx(
        wrist_limit - 0.03
    )
    assert derived.poses[1].replay_calibration_q is None
    assert edges["edges"] == [["pose_001", "pose_002"]]
    assert plan["capture_pose_ids"] == ["pose_001", "pose_002"]
    assert (output / "adjustments.json").is_file()


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


def test_replay_move_confirmation_uses_only_space(monkeypatch, capsys):
    keys = iter((b"q", b"\x1b", b" "))
    monkeypatch.setattr(
        "g1_aprilcube_calibration.hardware_cli._read_terminal_key",
        lambda: next(keys),
    )

    assert _wait_for_space("Press SPACE: ") is True
    assert capsys.readouterr().out == "Press SPACE: SPACE\n"


def test_fcl_sigint_is_deferred_to_a_polled_flag() -> None:
    with _DeferredSIGINT() as cancellation:
        cancellation._handle(None, None)
        assert cancellation.requested is True


def test_ros_shutdown_is_idempotent() -> None:
    class FakeRclpy:
        active = True
        shutdown_count = 0

        @classmethod
        def try_shutdown(cls):
            if cls.active:
                cls.active = False
                cls.shutdown_count += 1

    _shutdown_rclpy_once(FakeRclpy)
    _shutdown_rclpy_once(FakeRclpy)

    assert FakeRclpy.shutdown_count == 1


def test_replay_accepts_yellow_without_an_override_flag(tmp_path):
    parsed = build_parser().parse_args(
        [
            "collect-session",
            "--network-interface",
            "enp3s0",
            "--pose-set",
            str(tmp_path / "poses.yaml"),
            "--validation-report",
            str(tmp_path / "validation.json"),
            "--plan-yaml",
            str(tmp_path / "plan.yaml"),
            "--session-directory",
            str(tmp_path / "session"),
            "--session-id",
            "test_session",
            "--image-topic",
            "/camera/color/image_raw",
            "--camera-info-topic",
            "/camera/color/camera_info",
            "--camera-name",
            "head_color",
            "--camera-serial",
            "D435-TEST",
            "--head-witness-ack",
            "--confirm",
            MOTION_ACK,
        ]
    )

    assert parsed.burst_timeout_s == 3.0
    assert not hasattr(parsed, "yellow_override_reason")


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


def test_automatic_collection_rejects_existing_session_before_plan_or_sdk(
    tmp_path, capsys
):
    session = tmp_path / "existing"
    session.mkdir()

    exit_code = main(
        [
            "collect-auto",
            "--network-interface",
            "not-a-real-interface",
            "--plan",
            str(tmp_path / "missing-plan.json"),
            "--session-directory",
            str(session),
            "--session-id",
            "existing",
            "--image-topic",
            "/camera/color/image_raw",
            "--camera-info-topic",
            "/camera/color/camera_info",
            "--camera-name",
            "head_color",
            "--camera-serial",
            "D435-TEST",
            "--confirm",
            MOTION_ACK,
        ]
    )

    assert exit_code == 2
    assert "session directory already exists" in capsys.readouterr().err


def test_table_execution_acknowledgement_fails_before_plan_or_sdk(capsys, tmp_path):
    exit_code = main(
        [
            "execute-table-accuracy",
            "--network-interface",
            "not-a-real-interface",
            "--image-topic",
            "/camera/color/image_raw",
            "--camera-info-topic",
            "/camera/color/camera_info",
            "--camera-name",
            "head_color",
            "--camera-serial",
            "D435-TEST",
            "--plan",
            str(tmp_path / "missing-plan.json"),
            "--output-directory",
            str(tmp_path / "run"),
            "--confirm",
            "wrong phrase",
        ]
    )

    assert exit_code == 2
    assert "must exactly equal" in capsys.readouterr().err


def test_table_image_burst_is_lossless_hash_bound_and_robot_labeled(tmp_path):
    info = RectifiedCameraInfo(
        width=8,
        height=6,
        frame_id="camera_color_optical_frame",
        camera_name="test",
        serial_number="TEST",
        distortion_model="plumb_bob",
        d=(0.0,) * 5,
        k=(10.0, 0.0, 4.0, 0.0, 10.0, 3.0, 0.0, 0.0, 1.0),
        r=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        p=(10.0, 0.0, 4.0, 0.0, 0.0, 10.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    )
    frames = tuple(
        ROSImageFrame(
            image_bgr=np.full((6, 8, 3), index, dtype=np.uint8),
            timing=ImageTiming(
                receipt_monotonic_s=1.0 + index * 0.1,
                receipt_utc=f"2026-08-08T12:00:0{index}Z",
                header_stamp_ns=index,
            ),
            camera_info=info,
        )
        for index in range(3)
    )

    manifest = _write_table_image_burst(
        tmp_path / "burst",
        frames,
        commands_robot=True,
    )

    canonical = dict(manifest)
    content_sha256 = canonical.pop("content_sha256")
    assert content_sha256 == _hardware_document_sha256(canonical)
    assert manifest["commands_robot"] is True
    assert manifest["frame_count"] == 3
    for index, frame in enumerate(frames, start=1):
        decoded = cv2.imread(str(tmp_path / "burst" / f"frame_{index:03d}.png"))
        assert np.array_equal(decoded, frame.image_bgr)


def test_table_failure_bundle_preserves_primary_error_without_motion(tmp_path):
    output = tmp_path / "failed_run"
    _write_table_failure_bundle(
        output=output,
        error=RuntimeError("measured route failed"),
        plan={"content_sha256": "a" * 64},
        input_plan={"content_sha256": "a" * 64},
        hardware_bytes=b"control: {}\n",
        collision_bytes=b"pairs: []\n",
        preflight_frames=(),
        loaded_planning_frames=(),
        verification_frames=(),
        achieved_frames=(),
        motion_preflight=None,
        final_report=None,
        synchronized=None,
        watchdog=None,
        transport=None,
        attained_states={"table_target": None},
        cleanup_errors=[],
    )

    document = json.loads((output / "failure.json").read_text())
    assert document["error"] == "RuntimeError: measured route failed"
    assert not document["commands_published"]
    assert document["attained_states"] == {"table_target": None}
    assert (output / "plan.json").is_file()
    assert (output / "input_plan.json").is_file()
    assert (output / "hardware.yaml").read_bytes() == b"control: {}\n"


def test_table_camera_collector_excludes_buffered_frames_and_checks_control():
    info = RectifiedCameraInfo(
        width=2,
        height=2,
        frame_id="camera_color_optical_frame",
        camera_name="test",
        serial_number="TEST",
        distortion_model="plumb_bob",
        d=(0.0,) * 5,
        k=(10.0, 0.0, 1.0, 0.0, 10.0, 1.0, 0.0, 0.0, 1.0),
        r=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        p=(10.0, 0.0, 1.0, 0.0, 0.0, 10.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    )

    def frame(index: int) -> ROSImageFrame:
        return ROSImageFrame(
            image_bgr=np.full((2, 2, 3), index, dtype=np.uint8),
            timing=ImageTiming(
                receipt_monotonic_s=1.0 + index * 0.1,
                receipt_utc=f"2026-08-08T12:00:0{index}Z",
                header_stamp_ns=index,
            ),
            camera_info=info,
        )

    buffer = ROSFrameBuffer(maximum_frames=10)
    buffer.add(frame(0))
    queued = [frame(1), frame(2), frame(3)]

    class FakeRclpy:
        @staticmethod
        def spin_once(_node, *, timeout_sec):
            assert timeout_sec == 0.05
            if queued:
                buffer.add(queued.pop(0))

    camera = type("Camera", (), {"frames": buffer})()
    checks = []
    captured = _collect_new_camera_frames(
        FakeRclpy,
        object(),
        camera,
        frame_count=3,
        timeout_s=1.0,
        maximum_duration_s=1.0,
        control_check=lambda: checks.append(True),
    )

    assert [item.timing.header_stamp_ns for item in captured] == [1, 2, 3]
    assert len(checks) >= 6


def test_table_observation_retries_fresh_frames_without_loosening_gate(capsys):
    collected = []

    def collect():
        frames = (object(),)
        collected.append(frames)
        return frames

    attempts = iter((ValueError("spread is 3.052mm"), {"stable": True}))

    def observe(_frames):
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    frames, observation, rejections = _retry_table_observation(
        collect=collect,
        observe=observe,
        label="verification",
    )

    assert frames is collected[1]
    assert observation == {"stable": True}
    assert rejections == ("spread is 3.052mm",)
    assert len(collected) == 2
    assert "collecting a fresh burst before any changing target" in (
        capsys.readouterr().out
    )


def test_isolated_planning_work_keeps_polling_control_health():
    class Driver:
        def __init__(self):
            self.check_count = 0

        def check(self):
            self.check_count += 1

    driver = Driver()

    result = _run_isolated_control_work(
        label="test planning",
        worker=_delayed_isolated_sum,
        worker_kwargs={"values": (1, 2, 3)},
        driver=driver,
    )

    assert result == 6
    assert driver.check_count >= 2


def test_table_observation_retry_preserves_failure_after_three_fresh_bursts():
    attempts = []

    def collect():
        attempts.append(True)
        return (object(),)

    with pytest.raises(ValueError, match="failed after 3 fresh bursts"):
        _retry_table_observation(
            collect=collect,
            observe=lambda _frames: (_ for _ in ()).throw(ValueError("unstable")),
            label="verification",
        )

    assert len(attempts) == 3


def test_pose_teacher_requires_motion_ack_and_configures_hold_safety(tmp_path):
    arguments = [
        "teach-poses",
        "--network-interface",
        "enp3s0",
        "--session-directory",
        str(tmp_path / "session"),
        "--session-id",
        "test_session",
        "--image-topic",
        "/camera/color/image_rect_raw",
        "--camera-info-topic",
        "/camera/color/camera_info",
        "--camera-name",
        "head_color",
        "--camera-serial",
        "D435-TEST",
    ]
    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)

    parsed = build_parser().parse_args([*arguments, "--confirm", MOTION_ACK])

    assert parsed.confirm == MOTION_ACK
    assert parsed.pc2_host
    assert parsed.lock_file.name == "g1-aprilcube-calibration-command.lock"


def test_seated_debug_hold_requires_normal_motion_acknowledgement():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "commission-seated-debug-hold",
                "--network-interface",
                "enp3s0",
            ]
        )
    commissioned = build_parser().parse_args(
        [
            "commission-seated-debug-hold",
            "--network-interface",
            "enp3s0",
            "--confirm",
            MOTION_ACK,
        ]
    )
    assert commissioned.confirm == MOTION_ACK
    assert commissioned.duration_s == 2.0
    assert commissioned.trigger == "restore-seated"
    assert commissioned.lock_file.name == "g1-aprilcube-calibration-command.lock"


def test_seated_debug_hold_exposes_explicit_heartbeat_recovery_commission():
    commissioned = build_parser().parse_args(
        [
            "commission-seated-debug-hold",
            "--network-interface",
            "enp3s0",
            "--trigger",
            "heartbeat-zero-torque",
            "--confirm",
            MOTION_ACK,
        ]
    )

    assert commissioned.trigger == "heartbeat-zero-torque"


def test_table_execution_is_blocked_by_uncommissioned_debug_gate(capsys, tmp_path):
    hardware = yaml.safe_load((ROOT / "config/hardware.yaml").read_text())
    hardware["control"]["seated_debug_lowcmd_commissioned"] = False
    hardware_path = tmp_path / "uncommissioned_hardware.yaml"
    hardware_path.write_text(yaml.safe_dump(hardware, sort_keys=False))

    exit_code = main(
        [
            "execute-table-accuracy",
            "--hardware-config",
            str(hardware_path),
            "--network-interface",
            "enp3s0",
            "--image-topic",
            "/camera/image",
            "--camera-info-topic",
            "/camera/info",
            "--camera-name",
            "head",
            "--camera-serial",
            "123",
            "--plan",
            "missing-plan.json",
            "--output-directory",
            "missing-run",
            "--confirm",
            MOTION_ACK,
        ]
    )

    assert exit_code == 2
    assert "blocked before command creation" in capsys.readouterr().err


def test_repository_hardware_enables_physically_verified_debug_gate():
    hardware = yaml.safe_load((ROOT / "config/hardware.yaml").read_text())

    assert hardware["control"]["seated_debug_lowcmd_commissioned"] is True
    assert (
        hardware["control"]["seated_debug_failure_policy"]
        == "restore_ai_verify_zero_torque_fsm_0"
    )
    assert hardware["control"]["gravity_feedforward"] == "pinocchio_rnea_at_commanded_q"
    assert hardware["control"]["seated_gravity_feedforward_commissioned"] is True


def test_safety_cleanup_failure_preserves_primary_error() -> None:
    primary = RuntimeError("motion switch failed")
    cleanup = ValueError("PC2 Damp was not verified")

    with pytest.raises(RuntimeError) as failure:
        _raise_safety_cleanup_failures(
            primary,
            [("debug ownership cleanup", cleanup)],
        )

    assert failure.value.__cause__ is primary
    assert "primary failure: RuntimeError: motion switch failed" in str(failure.value)
    assert "PC2 Damp was not verified" in str(failure.value)


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
