"""Explicitly armed G1 commissioning and rectified-session collection commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import tempfile
import termios
import time
import tty
from collections.abc import Callable
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from concurrent.futures import (
    TimeoutError as FuturesTimeoutError,
)
from dataclasses import replace
from itertools import pairwise
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from aprilcube import CorrespondenceDetector
from g1_aprilcube_calibration.activation_handoff import (
    ActivationHandoff,
    build_activation_handoff,
)
from g1_aprilcube_calibration.authored_collection import (
    AuthoredCollectionPlan,
    exposed_target_normal_from_hardware,
    modeled_hand_T_target_from_hardware,
    validate_exposed_camera_views,
    validate_hardware_target_profile,
    validate_modeled_hand_target_binding,
)
from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.capture_diagnostics import supported_vs_held_metrics
from g1_aprilcube_calibration.clock import SystemClock
from g1_aprilcube_calibration.collision import (
    CollisionConfig,
    FCLCollisionChecker,
)
from g1_aprilcube_calibration.config import QualityThresholds
from g1_aprilcube_calibration.dual_arm_clearance import (
    DUAL_CLEARANCE_POSE_ID,
    RIGHT_CLEARANCE_POSE_ID,
    DualArmClearanceExecutor,
    live_dex3_collision_config,
    plan_dual_arm_shoulder_clearance,
    validate_dex3_finger_sweep_at_state,
)
from g1_aprilcube_calibration.executor_driver import (
    ExecutorControlDriver,
    SynchronizedPoseExecutor,
)
from g1_aprilcube_calibration.executor_state_machine import (
    ExecutorConfig,
    ExecutorState,
    PoseExecutor,
)
from g1_aprilcube_calibration.gravity_compensation import (
    UNITREE_XR_GRAVITY_REFERENCE,
    G1PinocchioGravityFeedforward,
)
from g1_aprilcube_calibration.inverse_kinematics import IKConfig
from g1_aprilcube_calibration.joint_map import (
    G1_29_JOINT_NAMES,
    LEFT_ARM_INDICES,
    RIGHT_ARM_INDICES,
    arm_hand_link,
    arm_joint_names,
    validate_full_joint_vector,
)
from g1_aprilcube_calibration.live_capture import LiveBurstConfig, LiveBurstFrameSource
from g1_aprilcube_calibration.models import utc_now_iso
from g1_aprilcube_calibration.pc2_safety import (
    PC2DampingWatchdog,
    PC2SafetyConfig,
)
from g1_aprilcube_calibration.pose_recorder import (
    PoseRecorder,
    PoseRecordingRequest,
)
from g1_aprilcube_calibration.pose_schema import (
    HANDOFF_POSE_ID,
    PoseSet,
)
from g1_aprilcube_calibration.pose_store import PoseStore, TransientPoseStore
from g1_aprilcube_calibration.pose_validator import (
    DirectedEdgeResult,
    PathValidationConfig,
    PosePathValidator,
    ValidationReport,
)
from g1_aprilcube_calibration.preview import render_operator_preview
from g1_aprilcube_calibration.process_lock import CommandOwnerLock
from g1_aprilcube_calibration.quality import (
    CameraIntrinsics,
    PoseQualityEvaluator,
    QualityGrade,
    ViewSignature,
)
from g1_aprilcube_calibration.readiness import RecordingGateConfig, StateSampleBuffer
from g1_aprilcube_calibration.ros.camera_adapter import (
    ROSCameraSubscriber,
    ROSImageFrame,
)
from g1_aprilcube_calibration.session_runner import (
    ApprovedSessionOrchestrator,
    AuthoredCollectionOrchestrator,
    CaptureSessionRunner,
    SessionExecutionPlan,
)
from g1_aprilcube_calibration.session_runtime_log import SessionRuntimeLog
from g1_aprilcube_calibration.session_store import IsolatedSessionStore, SessionStore
from g1_aprilcube_calibration.table_accuracy import (
    CharucoBoardPoseDetector,
    CharucoBoardSpec,
    create_evaluation_document,
    create_plan_document,
    hand_target_from_board,
    load_plan_document,
    observe_burst,
    predicted_board_T_cube_from_model,
)
from g1_aprilcube_calibration.table_motion import (
    TABLE_ESCAPE_POSE_ID,
    TABLE_TARGET_POSE_ID,
    TablePlaneConfig,
    build_lifted_start_cube_target,
    build_table_motion_preflight,
    validate_table_plane_escape_path,
    validate_table_plane_path,
)
from g1_aprilcube_calibration.teaching_controller import (
    TeachingArmController,
    TeachingConfig,
    TeachingState,
)
from g1_aprilcube_calibration.teaching_driver import (
    SynchronizedTeachingController,
    TeachingControlDriver,
)
from g1_aprilcube_calibration.timestamp_pairing import PairingConfig
from g1_aprilcube_calibration.transforms import validate_transform
from g1_aprilcube_calibration.transports.base import ArmCommand
from g1_aprilcube_calibration.transports.unitree_arm_sdk import (
    UnitreeArmSDKTransport,
    UnitreeLowStateObserver,
    UnitreeTransportConfig,
)
from g1_aprilcube_calibration.transports.unitree_debug_lowcmd import (
    UnitreeDebugLowCmdConfig,
    UnitreeDebugLowCmdTransport,
)
from g1_aprilcube_calibration.transports.unitree_dex3 import (
    DEX3_CALIBRATION_POSTURE_SOURCE,
    DEX3_MOTOR_JOINT_SUFFIXES,
    Dex3ControlConfig,
    UnitreeDex3PostureController,
    UnitreeDex3StateObserver,
    dex3_motor_joint_name,
)
from g1_aprilcube_calibration.urdf_model import URDFModel

WRITE_ACK = "I UNDERSTAND THIS WRITES RT/ARM_SDK"
DAMP_ACK = "DAMP THE LOAD-BEARING-HARNESSED G1 NOW"
MOTION_ACK = (
    "I CONFIRM THE G1 IS SECURED BY THE LOAD-BEARING HARNESS AND THE WORKSPACE IS CLEAR"
)
LIVE_TEACHING_YELLOW_REASON = "operator accepted yellow warning in live teaching UI"
_MAX_RUNTIME_VALIDATION_WORKERS = 8
_MINIMUM_PARALLEL_VALIDATION_EDGES = 8
_runtime_validation_worker: tuple[PosePathValidator, object, np.ndarray] | None = None
REPLAY_YELLOW_REASON = "yellow visual warning accepted by replay default"


def add_hardware_subparsers(
    subparsers: argparse._SubParsersAction,
    *,
    workspace_root: Path,
    default_target: Path,
    default_quality: Path,
) -> None:
    default_hardware = workspace_root / "config" / "hardware_dex3_aruco.yaml"
    default_collision = workspace_root / "config" / "collision_pairs_dex3_aruco.yaml"
    default_lock = Path("/tmp/g1-aprilcube-calibration-command.lock")

    table_images = subparsers.add_parser(
        "capture-table-images",
        help="save a camera-only lossless burst for the tabletop accuracy test",
    )
    _add_network_arguments(table_images)
    _add_camera_arguments(table_images)
    table_images.add_argument("--output-directory", type=Path, required=True)
    table_images.add_argument("--frame-count", type=int, default=7)
    table_images.add_argument("--camera-timeout-s", type=float, default=10.0)
    table_images.add_argument("--maximum-duration-s", type=float, default=2.0)
    table_images.set_defaults(handler=run_capture_table_images)

    table_execute = subparsers.add_parser(
        "execute-table-accuracy",
        help=(
            "solve a supported-start escape and table target, capture, reverse "
            "to the supported start, restore seated control, and score accuracy"
        ),
    )
    _add_network_arguments(table_execute)
    _add_motion_safety_arguments(table_execute)
    _add_camera_arguments(table_execute)
    table_execute.add_argument("--plan", type=Path, required=True)
    table_execute.add_argument("--output-directory", type=Path, required=True)
    table_execute.add_argument("--hardware-config", type=Path, default=default_hardware)
    table_execute.add_argument(
        "--collision-config", type=Path, default=default_collision
    )
    table_execute.add_argument("--frame-count", type=int, default=7)
    table_execute.add_argument("--camera-timeout-s", type=float, default=10.0)
    table_execute.add_argument("--maximum-burst-duration-s", type=float, default=2.0)
    table_execute.add_argument("--ik-restarts", type=int, default=24)
    table_execute.add_argument("--minimum-board-clearance-mm", type=float, default=10.0)
    table_execute.add_argument("--maximum-board-shift-mm", type=float, default=5.0)
    table_execute.add_argument("--maximum-board-shift-deg", type=float, default=1.0)
    table_execute.add_argument(
        "--confirm", required=True, help=f"must equal: {MOTION_ACK}"
    )
    table_execute.add_argument("--lock-file", type=Path, default=default_lock)
    table_execute.set_defaults(handler=run_execute_table_accuracy)

    zero = subparsers.add_parser(
        "commission-weight-zero",
        help="publish one measured arm target with blend weight zero",
    )
    _add_network_arguments(zero)
    zero.add_argument("--confirm", required=True, help=f"must equal: {WRITE_ACK}")
    zero.add_argument("--timeout-s", type=float, default=5.0)
    zero.add_argument("--lock-file", type=Path, default=default_lock)
    zero.set_defaults(handler=run_commission_weight_zero)

    damping = subparsers.add_parser(
        "commission-damping",
        help="physically verify the PC2 heartbeat watchdog and G1 Damp RPC",
    )
    _add_network_arguments(damping)
    _add_motion_safety_arguments(damping)
    damping.add_argument("--hardware-config", type=Path, default=default_hardware)
    damping.add_argument(
        "--trigger",
        choices=("heartbeat-timeout", "explicit"),
        default="heartbeat-timeout",
    )
    damping.add_argument("--confirm", required=True, help=f"must equal: {DAMP_ACK}")
    damping.set_defaults(handler=run_commission_damping)

    dex3_posture = subparsers.add_parser(
        "commission-dex3-middle-close",
        help=(
            "derive and execute a live collision-validated shoulder-clearance "
            "route, verify NVIDIA middle-close, restore the original fingers "
            "and shoulders, and send terminal Dex3 timeout"
        ),
    )
    _add_network_arguments(dex3_posture)
    _add_motion_safety_arguments(dex3_posture)
    dex3_posture.add_argument("--hardware-config", type=Path, default=default_hardware)
    dex3_posture.add_argument(
        "--collision-config", type=Path, default=default_collision
    )
    dex3_posture.add_argument("--duration-s", type=float, default=2.0)
    dex3_posture.add_argument(
        "--confirm", required=True, help=f"must equal: {MOTION_ACK}"
    )
    dex3_posture.add_argument("--lock-file", type=Path, default=default_lock)
    dex3_posture.set_defaults(handler=run_commission_dex3_middle_close)

    seated_debug_hold = subparsers.add_parser(
        "commission-seated-debug-hold",
        help=(
            "commission zero-displacement seated debug lowcmd takeover and "
            "verified normal FSM 0 -> 1 -> 3 restoration"
        ),
    )
    _add_network_arguments(seated_debug_hold)
    _add_motion_safety_arguments(seated_debug_hold)
    seated_debug_hold.add_argument(
        "--hardware-config", type=Path, default=default_hardware
    )
    seated_debug_hold.add_argument(
        "--collision-config", type=Path, default=default_collision
    )
    seated_debug_hold.add_argument("--duration-s", type=float, default=2.0)
    seated_debug_hold.add_argument(
        "--trigger",
        choices=("restore-seated", "heartbeat-zero-torque"),
        default="restore-seated",
        help=(
            "restore AI through verified FSM 0 -> 1 -> 3 or "
            "deliberately stop command/heartbeat publication and verify PC2 "
            "independently restores AI in zero-torque FSM 0"
        ),
    )
    seated_debug_hold.add_argument(
        "--confirm", required=True, help=f"must equal: {MOTION_ACK}"
    )
    seated_debug_hold.add_argument("--lock-file", type=Path, default=default_lock)
    seated_debug_hold.set_defaults(handler=run_commission_seated_debug_hold)

    teach = subparsers.add_parser(
        "teach-poses",
        help=(
            "continuously guide the arm through arm_sdk, freeze measured poses, "
            "and record paired supported/held calibration bursts"
        ),
    )
    _add_network_arguments(teach)
    _add_motion_safety_arguments(teach)
    teach.add_argument("--session-directory", type=Path, required=True)
    teach.add_argument("--session-id", required=True)
    _add_camera_arguments(teach)
    teach.add_argument("--hardware-config", type=Path, default=default_hardware)
    teach.add_argument("--collision-config", type=Path, default=default_collision)
    teach.add_argument("--target-config", type=Path, default=default_target)
    teach.add_argument("--quality-config", type=Path, default=default_quality)
    teach.add_argument("--camera-timeout-s", type=float, default=10.0)
    teach.add_argument("--guide-camera-timeout-s", type=float, default=1.0)
    teach.add_argument("--burst-timeout-s", type=float, default=15.0)
    teach.add_argument("--group", default="calibration")
    teach.add_argument("--first-pose-id", default="pose_001")
    teach.add_argument("--preview-directory", type=Path)
    teach.add_argument(
        "--yellow-override-reason",
        help=(
            "optional custom note for yellow views; without it, SPACE accepts yellow "
            "and stores the standard live-operator warning"
        ),
    )
    teach.add_argument("--confirm", required=True, help=f"must equal: {MOTION_ACK}")
    teach.add_argument("--lock-file", type=Path, default=default_lock)
    teach.set_defaults(handler=run_teach_poses)

    hold = subparsers.add_parser(
        "commission-hold",
        help="derive a stationary handoff, then acquire/release arm_sdk there",
    )
    _add_network_arguments(hold)
    _add_motion_safety_arguments(hold)
    hold.add_argument("--pose-set", type=Path, required=True)
    hold.add_argument("--hardware-config", type=Path, default=default_hardware)
    hold.add_argument("--collision-config", type=Path, default=default_collision)
    hold.add_argument("--duration-s", type=float, default=2.0)
    hold.add_argument("--confirm", required=True, help=f"must equal: {MOTION_ACK}")
    hold.add_argument("--lock-file", type=Path, default=default_lock)
    hold.set_defaults(handler=run_commission_hold)

    pose = subparsers.add_parser(
        "commission-pose",
        help="validate a live handoff, visit one pose, return, and release",
    )
    _add_network_arguments(pose)
    _add_motion_safety_arguments(pose)
    pose.add_argument("--pose-set", type=Path, required=True)
    pose.add_argument("--validation-report", type=Path, required=True)
    pose.add_argument("--target-pose", required=True)
    pose.add_argument("--hardware-config", type=Path, default=default_hardware)
    pose.add_argument("--collision-config", type=Path, default=default_collision)
    pose.add_argument("--confirm", required=True, help=f"must equal: {MOTION_ACK}")
    pose.add_argument("--lock-file", type=Path, default=default_lock)
    pose.set_defaults(handler=run_commission_pose)

    collect = subparsers.add_parser(
        "collect-session",
        help="execute a passed pose plan and write an immutable rectified session",
    )
    _add_network_arguments(collect)
    _add_motion_safety_arguments(collect)
    collect.add_argument("--pose-set", type=Path, required=True)
    collect.add_argument("--validation-report", type=Path, required=True)
    collect.add_argument("--plan-yaml", type=Path, required=True)
    collect.add_argument("--session-directory", type=Path, required=True)
    collect.add_argument("--session-id", required=True)
    _add_camera_arguments(collect)
    collect.add_argument("--hardware-config", type=Path, default=default_hardware)
    collect.add_argument("--collision-config", type=Path, default=default_collision)
    collect.add_argument("--target-config", type=Path, default=default_target)
    collect.add_argument("--quality-config", type=Path, default=default_quality)
    collect.add_argument("--camera-timeout-s", type=float, default=10.0)
    collect.add_argument("--burst-timeout-s", type=float, default=3.0)
    collect.add_argument(
        "--head-witness-ack",
        action="store_true",
        required=True,
        help="confirm the current fixed camera/head witness mark before collection",
    )
    collect.add_argument("--auto-confirm-transitions", action="store_true")
    collect.add_argument("--no-window", action="store_true")
    collect.add_argument("--confirm", required=True, help=f"must equal: {MOTION_ACK}")
    collect.add_argument("--lock-file", type=Path, default=default_lock)
    collect.set_defaults(handler=run_collect_session)

    collect_auto = subparsers.add_parser(
        "collect-auto",
        help=(
            "execute an authored standing calibration route, automatically "
            "capture valid views, and skip red/invisible targets"
        ),
    )
    _add_network_arguments(collect_auto)
    _add_motion_safety_arguments(collect_auto)
    collect_auto.add_argument("--plan", type=Path, required=True)
    collect_auto.add_argument("--session-directory", type=Path, required=True)
    collect_auto.add_argument("--session-id", required=True)
    _add_camera_arguments(collect_auto)
    collect_auto.add_argument("--hardware-config", type=Path, default=default_hardware)
    collect_auto.add_argument(
        "--collision-config", type=Path, default=default_collision
    )
    collect_auto.add_argument("--target-config", type=Path, default=default_target)
    collect_auto.add_argument("--quality-config", type=Path, default=default_quality)
    collect_auto.add_argument("--camera-timeout-s", type=float, default=10.0)
    collect_auto.add_argument("--burst-timeout-s", type=float, default=3.0)
    collect_auto.add_argument("--no-window", action="store_true")
    collect_auto.add_argument(
        "--confirm", required=True, help=f"must equal: {MOTION_ACK}"
    )
    collect_auto.add_argument("--lock-file", type=Path, default=default_lock)
    collect_auto.set_defaults(handler=run_collect_auto)


def _add_network_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--network-interface", required=True)
    parser.add_argument("--domain-id", type=int, default=0)


def _add_motion_safety_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pc2-host",
        default=os.environ.get("G1_PC2_HOST", "unitree@192.168.123.164"),
    )
    parser.add_argument(
        "--pc2-ssh-identity",
        type=Path,
        default=Path(
            os.environ.get(
                "G1_PC2_SSH_IDENTITY",
                str(Path.home() / ".ssh" / "g1_pc2_ed25519"),
            )
        ),
    )


def _add_camera_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image-topic", required=True)
    parser.add_argument("--camera-info-topic", required=True)
    parser.add_argument("--camera-name", required=True)
    parser.add_argument("--camera-serial", required=True)
    parser.add_argument(
        "--ros-camera-reliability",
        choices=("reliable", "best-effort"),
        default="reliable",
        help=(
            "ROS Image/CameraInfo subscription reliability; calibration defaults "
            "to reliable"
        ),
    )


def run_capture_table_images(args: argparse.Namespace) -> int:
    """Capture rectified images without constructing any robot transport."""
    if args.frame_count < 3:
        raise ValueError("--frame-count must be at least 3")
    if args.maximum_duration_s <= 0:
        raise ValueError("--maximum-duration-s must be positive")
    output = args.output_directory.resolve()
    if output.exists():
        raise FileExistsError(f"table image burst already exists: {output}")
    try:
        import rclpy
    except ImportError as error:
        raise RuntimeError(
            "rclpy is unavailable; source the ROS environment"
        ) from error

    rclpy.init(args=None)
    node = rclpy.create_node("g1_table_accuracy_image_capture")
    camera = None
    try:
        camera = ROSCameraSubscriber(
            node,
            image_topic=args.image_topic,
            camera_info_topic=args.camera_info_topic,
            camera_name=args.camera_name,
            serial_number=args.camera_serial,
            reliability=args.ros_camera_reliability,
            maximum_frames=max(30, args.frame_count * 2),
        )
        _wait_for_camera(rclpy, node, camera, args.camera_timeout_s)
        frames = _collect_new_camera_frames(
            rclpy,
            node,
            camera,
            frame_count=args.frame_count,
            timeout_s=args.camera_timeout_s,
            maximum_duration_s=args.maximum_duration_s,
        )
    finally:
        if camera is not None:
            camera.close()
        node.destroy_node()
        _shutdown_rclpy_once(rclpy)

    manifest = _write_table_image_burst(output, frames)
    print(json.dumps({**manifest, "output": str(output)}, indent=2, sort_keys=True))
    return 0


def run_execute_table_accuracy(args: argparse.Namespace) -> int:
    """Acquire while seated, run one table trial, and release at the start pose."""

    _require_ack(args.confirm, MOTION_ACK)
    _require_dex3_posture_commissioned(args.hardware_config)
    _validate_table_execution_arguments(args)
    with args.hardware_config.open(encoding="utf-8") as stream:
        table_hardware = yaml.safe_load(stream)
    table_control = table_hardware["control"]
    if table_control.get("seated_table_control") != "debug_lowcmd":
        raise RuntimeError(
            "seated tabletop execution requires control.seated_table_control="
            "debug_lowcmd"
        )
    if table_control.get("seated_debug_lowcmd_commissioned") is not True:
        raise RuntimeError(
            "seated tabletop execution is blocked before command creation: the "
            "debug rt/lowcmd takeover, normal FSM 0 -> 1 -> 3 restoration, and "
            "heartbeat-loss AI restoration to zero-torque FSM 0 have not been "
            "commissioned on this robot"
        )
    if table_control.get("seated_gravity_feedforward_commissioned") is not True:
        raise RuntimeError(
            "seated tabletop execution is blocked before command creation: "
            "Unitree-style Pinocchio gravity feedforward has not passed its "
            "zero-displacement seated debug commissioning run"
        )
    output = args.output_directory.resolve()
    if output.exists():
        raise FileExistsError(f"table-accuracy run already exists: {output}")

    input_plan = load_plan_document(args.plan)
    plan = input_plan
    model = URDFModel(_configured_urdf(args.hardware_config))
    _validate_table_plan_artifacts(plan, model=model)
    empty_pose_set = PoseSet(
        robot_model=model.name,
        mode_machine=5,
        urdf_sha256=model.sha256,
        calibration_arm=plan["calibration_arm"],
    )
    hardware_bytes = args.hardware_config.read_bytes()
    _validate_hardware_preflight(
        hardware_bytes,
        empty_pose_set,
        require_poses=False,
    )
    collision_bytes = args.collision_config.read_bytes()
    collision = _validate_collision_preflight(args.collision_config)
    if args.collision_config.read_bytes() != collision_bytes:
        raise RuntimeError("collision configuration changed during preflight")
    recording, _, executor_config, rate_hz = _runtime_configs(args.hardware_config)
    camera_info = RectifiedCameraInfo.from_dict(plan["camera_info"])
    board_spec = CharucoBoardSpec.from_dict(plan["board_spec"])
    board_detector = CharucoBoardPoseDetector(board_spec)
    table_plane_config = TablePlaneConfig(
        board_width_m=board_spec.width_mm / 1000.0,
        board_height_m=board_spec.height_mm / 1000.0,
        table_margin_m=0.2,
        minimum_clearance_m=args.minimum_board_clearance_mm / 1000.0,
    )
    hand_cube_path = Path(plan["sources"]["hand_cube_config_path"])
    hand_cube_detector = CorrespondenceDetector(hand_cube_path)
    torso_T_camera = validate_transform(
        np.asarray(plan["calibration"]["torso_T_camera"], dtype=np.float64)
    )
    hand_T_cube = validate_transform(
        np.asarray(plan["calibration"]["hand_T_cube"], dtype=np.float64)
    )
    try:
        import rclpy
    except ImportError as error:
        raise RuntimeError(
            "rclpy is unavailable; source the ROS environment"
        ) from error

    rclpy.init(args=None)
    node = rclpy.create_node("g1_table_accuracy_executor")
    camera = None
    observer = None
    dex3_observer = None
    dex3_controller = None
    dex3_posture_evidence = None
    transport = None
    watchdog = None
    synchronized = None
    driver = None
    preflight_frames: tuple[ROSImageFrame, ...] = ()
    loaded_planning_frames: tuple[ROSImageFrame, ...] = ()
    verification_frames: tuple[ROSImageFrame, ...] = ()
    loaded_planning_rejections: tuple[str, ...] = ()
    verification_rejections: tuple[str, ...] = ()
    achieved_frames: tuple[ROSImageFrame, ...] = ()
    initial_observation = None
    loaded_planning_observation = None
    verification_observation = None
    execution_plan = None
    motion_preflight = None
    final_report = None
    final_escape_clearance = None
    final_elevated_route_clearance = None
    final_activation = None
    board_shift = None
    initial_cube_shift = None
    takeover_board_shift = None
    takeover_cube_shift = None
    approval_cube_shift = None
    escape_outbound_duration_s = None
    supported_return_timeout_s = None
    capture_failure = None
    attained_escape_outbound = None
    attained_target_before_capture = None
    attained_target_after_capture = None
    attained_escape_return = None
    attained_supported_return = None
    target_capture_position_spread_rad = None
    primary_error: BaseException | None = None
    command_lock = CommandOwnerLock(args.lock_file)
    command_lock.acquire()
    try:
        camera = ROSCameraSubscriber(
            node,
            image_topic=args.image_topic,
            camera_info_topic=args.camera_info_topic,
            camera_name=args.camera_name,
            serial_number=args.camera_serial,
            reliability=args.ros_camera_reliability,
            maximum_frames=max(30, args.frame_count * 3),
        )
        states = StateSampleBuffer()
        observer = UnitreeLowStateObserver(
            _transport_config(args),
            on_sample=states.add,
        )
        _wait_for_state(observer, 5.0)
        dex3_observer = UnitreeDex3StateObserver(
            _dex3_control_config(args),
            initialize_factory=False,
        )
        _wait_for_dex3_state(dex3_observer, 5.0)
        _wait_for_camera(rclpy, node, camera, args.camera_timeout_s)
        live_camera_info = camera.frames.latest.camera_info
        if live_camera_info.profile_sha256 != camera_info.profile_sha256:
            raise ValueError("live camera profile differs from the table-accuracy plan")

        print(
            "PRECHECK — keep the robot and fixed table board still; "
            "collecting a live board/cube burst"
        )
        preflight_frames = _collect_new_camera_frames(
            rclpy,
            node,
            camera,
            frame_count=args.frame_count,
            timeout_s=args.camera_timeout_s,
            maximum_duration_s=args.maximum_burst_duration_s,
        )
        initial_observation = _observe_table_frames(
            preflight_frames,
            camera_info=camera_info,
            hand_cube_detector=hand_cube_detector,
            board_detector=board_detector,
        )
        initial_camera_T_board = _observation_transform(
            initial_observation,
            "camera_T_board",
        )
        initial_board_T_cube = _observation_transform(
            initial_observation,
            "board_T_hand_cube",
        )
        planned_initial_board_T_cube = validate_transform(
            np.asarray(
                plan["target"]["initial_board_T_hand_cube"],
                dtype=np.float64,
            )
        )
        initial_cube_shift = _transform_delta(
            planned_initial_board_T_cube,
            initial_board_T_cube,
        )
        if initial_cube_shift["translation_mm"] > args.maximum_board_shift_mm:
            raise ValueError(
                "live supported cube differs from the planned initial cube by "
                f"{initial_cube_shift['translation_mm']:.3f}mm; limit is "
                f"{args.maximum_board_shift_mm:.3f}mm. No robot command was sent"
            )
        if initial_cube_shift["rotation_deg"] > args.maximum_board_shift_deg:
            raise ValueError(
                "live supported cube orientation differs from the planned initial "
                f"orientation by {initial_cube_shift['rotation_deg']:.3f}deg; "
                f"limit is {args.maximum_board_shift_deg:.3f}deg. No robot command "
                "was sent"
            )
        activation = _wait_for_activation_handoff(
            observer,
            states,
            empty_pose_set,
            recording,
        )
        print(
            "READ-ONLY SETUP PASSED — camera, board, cube, artifacts, and "
            "stationary arms are valid; no IK/FCL route has been frozen and no "
            "robot command has been published"
        )
        _wait_for_space(
            "START — confirm the G1 is seated; leave both forearms supported "
            "and stationary with every finger clear to curl; clear the complete "
            "lift/target/return sweep. "
            "Press SPACE once to acquire, settle under load, observe and plan "
            "from that loaded state, execute one open-loop trial, return, and "
            "restore seated control through FSM 0 -> 1 -> 3: "
        )

        if args.hardware_config.read_bytes() != hardware_bytes:
            raise RuntimeError(
                "hardware configuration changed during preflight; no robot command was sent"
            )
        if args.collision_config.read_bytes() != collision_bytes:
            raise RuntimeError(
                "collision configuration changed during preflight; no robot command was sent"
            )
        live_urdf_sha256 = hashlib.sha256(
            _configured_urdf(args.hardware_config).read_bytes()
        ).hexdigest()
        if live_urdf_sha256 != model.sha256:
            raise RuntimeError(
                "URDF changed during preflight; no robot command was sent"
            )
        activation = _wait_for_activation_handoff(
            observer,
            states,
            empty_pose_set,
            recording,
        )
        gravity_feedforward = _gravity_feedforward(args.hardware_config)
        gravity_feedforward.seed_reference(activation.reference_state.position)
        takeover_q14 = np.concatenate(
            (
                activation.reference_state.left_q,
                activation.reference_state.right_q,
            )
        )
        takeover_gravity_tau = gravity_feedforward.torque_for(takeover_q14)
        _print_gravity_preflight(gravity_feedforward, takeover_gravity_tau)
        print(
            "TAKEOVER CHECK PASSED — PC2 will verify seated FSM 3 and the laptop "
            "will acquire complete 29-joint debug lowcmd control at the exact "
            "measured state; IK/FCL planning will happen only after loaded settling"
        )

        watchdog = _pc2_damping_watchdog(
            args,
            args.hardware_config,
            require_regular=False,
            required_initial_fsm_id=_seated_fsm_id(args.hardware_config),
            restore_motion_service_before_loco=True,
        )
        transport = UnitreeDebugLowCmdTransport(
            _transport_config(args),
            _debug_lowcmd_config(args.hardware_config),
            observer=observer,
            ownership_keepalive=watchdog.pulse,
        )
        observer = None
        _wait_for_state(transport, 5.0)
        dex3_controller = UnitreeDex3PostureController(
            _dex3_control_config(args),
            observer=dex3_observer,
        )
        dex3_observer = None
        table_executor_config = replace(
            executor_config,
            require_motion_endpoint_tolerance=False,
        )
        raw_executor = PoseExecutor(
            transport=transport,
            clock=SystemClock(),
            pose_set=empty_pose_set,
            handoff_q=activation.handoff_q,
            hold_q=activation.hold_q,
            approved_validation_report_sha256=input_plan["content_sha256"],
            config=table_executor_config,
            gravity_feedforward=gravity_feedforward,
        )
        synchronized = SynchronizedPoseExecutor(raw_executor)
        driver = ExecutorControlDriver(
            synchronized,
            rate_hz=rate_hz,
            safety_heartbeat=_dex3_safety_heartbeat(
                watchdog,
                dex3_controller,
            ),
        )
        # Complete every non-commanding DDS and executor setup before arming
        # PC2's 0.5 s lease. The next actions start the control thread and
        # execute the guarded ReleaseMode/first-lowcmd transition.
        watchdog.start()
        dex3_posture_evidence = dex3_controller.acquire_posture(
            safety_heartbeat=watchdog.pulse,
        )
        print(
            "DEX3 CALIBRATION POSTURE ACQUIRED — both hands reached NVIDIA's "
            "middle-close target; posture commands remain active throughout "
            "seated body control"
        )
        driver.start()
        synchronized.acquire(operator_confirmed=True)
        _wait_for_driven_state(
            synchronized,
            driver,
            ExecutorState.READY,
            rclpy=rclpy,
            node=node,
            timeout_s=executor_config.acquisition_ramp_s + 5.0,
        )
        print(
            "CONTROL ACQUIRED — holding the measured state under full gravity "
            "feedforward; collecting the loaded board/cube observation now. "
            "No changing arm target has been issued"
        )
        (
            loaded_planning_frames,
            loaded_planning_observation,
            loaded_planning_rejections,
        ) = _capture_controlled_table_observation(
            synchronized=synchronized,
            driver=driver,
            collect=lambda: _collect_new_camera_frames(
                rclpy,
                node,
                camera,
                frame_count=args.frame_count,
                timeout_s=args.camera_timeout_s,
                maximum_duration_s=args.maximum_burst_duration_s,
                control_check=driver.check,
            ),
            observe=lambda frames: _observe_table_frames(
                frames,
                camera_info=camera_info,
                hand_cube_detector=hand_cube_detector,
                board_detector=board_detector,
            ),
            label="post-takeover loaded-state planning",
        )
        loaded_camera_T_board = _observation_transform(
            loaded_planning_observation,
            "camera_T_board",
        )
        loaded_board_T_cube = _observation_transform(
            loaded_planning_observation,
            "board_T_hand_cube",
        )
        takeover_board_shift = _transform_delta(
            initial_camera_T_board,
            loaded_camera_T_board,
        )
        takeover_cube_shift = _transform_delta(
            initial_board_T_cube,
            loaded_board_T_cube,
        )
        loaded_activation = _wait_for_activation_handoff(
            synchronized,
            states,
            empty_pose_set,
            recording,
        )
        print(
            "POST-TAKEOVER REPLAN — the controller is holding the loaded handoff "
            "while IK and FCL run; no changing arm target has been issued"
        )
        target_xy = tuple(float(value) for value in input_plan["target"]["board_xy_mm"])
        execution_plan = create_plan_document(
            dataset_path=input_plan["sources"]["dataset_path"],
            result_path=input_plan["sources"]["result_path"],
            hand_cube_config_path=input_plan["sources"]["hand_cube_config_path"],
            planning_burst=loaded_planning_observation,
            board_spec=board_spec,
            lift_mm=float(input_plan["target"]["lift_above_initial_cube_mm"]),
            board_xy_mm=(target_xy[0], target_xy[1]),
        )
        _validate_table_plan_artifacts(execution_plan, model=model)
        plan = execution_plan
        desired_board_T_cube = validate_transform(
            np.asarray(
                plan["target"]["desired_board_T_hand_cube"],
                dtype=np.float64,
            )
        )
        desired_board_T_escape = build_lifted_start_cube_target(
            current_board_T_cube=loaded_board_T_cube,
            desired_board_T_cube=desired_board_T_cube,
        )
        desired_torso_T_escape = hand_target_from_board(
            torso_T_camera=torso_T_camera,
            hand_T_cube=hand_T_cube,
            camera_T_board=loaded_camera_T_board,
            desired_board_T_cube=desired_board_T_escape,
        )
        desired_torso_T_hand = hand_target_from_board(
            torso_T_camera=torso_T_camera,
            hand_T_cube=hand_T_cube,
            camera_T_board=loaded_camera_T_board,
            desired_board_T_cube=desired_board_T_cube,
        )
        motion_preflight = _run_isolated_control_work(
            label="post-takeover IK/FCL planning",
            worker=_build_table_motion_preflight_worker,
            worker_kwargs={
                "urdf_path": str(model.path),
                "collision_config": collision,
                "desired_torso_T_escape": desired_torso_T_escape,
                "desired_torso_T_hand": desired_torso_T_hand,
                "torso_T_board": torso_T_camera @ loaded_camera_T_board,
                "reference_full_q": loaded_activation.reference_state.position,
                "calibration_arm": plan["calibration_arm"],
                "plan_sha256": plan["content_sha256"],
                "recorded_at_utc": loaded_activation.reference_state.receipt_utc,
                "recorded_monotonic_s": (
                    loaded_activation.reference_state.receipt_monotonic_s
                ),
                "ik_config": IKConfig(restart_count=args.ik_restarts),
                "table_plane_config": table_plane_config,
            },
            driver=driver,
        )
        driver.check()
        _print_table_motion_preflight(loaded_activation, motion_preflight)

        # Planning can take several seconds.  Observe once more before motion
        # and reject a changed camera/board or supported cube instead of silently
        # turning this one-shot calibration test into feedback control.
        (
            verification_frames,
            verification_observation,
            verification_rejections,
        ) = _capture_controlled_table_observation(
            synchronized=synchronized,
            driver=driver,
            collect=lambda: _collect_new_camera_frames(
                rclpy,
                node,
                camera,
                frame_count=max(3, min(args.frame_count, 5)),
                timeout_s=args.camera_timeout_s,
                maximum_duration_s=args.maximum_burst_duration_s,
                control_check=driver.check,
            ),
            observe=lambda frames: _observe_table_frames(
                frames,
                camera_info=camera_info,
                hand_cube_detector=hand_cube_detector,
                board_detector=board_detector,
            ),
            label="post-replan loaded-state verification",
        )
        verified_camera_T_board = _observation_transform(
            verification_observation,
            "camera_T_board",
        )
        verified_board_T_cube = _observation_transform(
            verification_observation,
            "board_T_hand_cube",
        )
        board_shift = _transform_delta(
            loaded_camera_T_board,
            verified_camera_T_board,
        )
        if board_shift["translation_mm"] > args.maximum_board_shift_mm:
            raise ValueError(
                "loaded camera-to-board translation changed during planning by "
                f"{board_shift['translation_mm']:.3f}mm; limit is "
                f"{args.maximum_board_shift_mm:.3f}mm. No changing arm target "
                "was sent"
            )
        if board_shift["rotation_deg"] > args.maximum_board_shift_deg:
            raise ValueError(
                "loaded camera-to-board rotation changed during planning by "
                f"{board_shift['rotation_deg']:.3f}deg; limit is "
                f"{args.maximum_board_shift_deg:.3f}deg. No changing arm target "
                "was sent"
            )
        approval_cube_shift = _transform_delta(
            loaded_board_T_cube,
            verified_board_T_cube,
        )
        if approval_cube_shift["translation_mm"] > args.maximum_board_shift_mm:
            raise ValueError(
                "loaded supported cube moved during planning by "
                f"{approval_cube_shift['translation_mm']:.3f}mm; limit is "
                f"{args.maximum_board_shift_mm:.3f}mm. No changing arm target "
                "was sent"
            )
        if approval_cube_shift["rotation_deg"] > args.maximum_board_shift_deg:
            raise ValueError(
                "loaded supported cube rotated during planning by "
                f"{approval_cube_shift['rotation_deg']:.3f}deg; limit is "
                f"{args.maximum_board_shift_deg:.3f}deg. No changing arm target "
                "was sent"
            )

        final_activation = _wait_for_activation_handoff(
            synchronized,
            states,
            motion_preflight.pose_set,
            recording,
        )
        escape_target_q = np.asarray(
            motion_preflight.escape_ik_solution.calibration_q,
            dtype=np.float64,
        )
        table_target_q = np.asarray(
            motion_preflight.target_ik_solution.calibration_q,
            dtype=np.float64,
        )
        (
            final_report,
            final_escape_clearance,
            final_elevated_route_clearance,
        ) = _run_isolated_control_work(
            label="final loaded-state route validation",
            worker=_validate_final_table_motion_worker,
            worker_kwargs={
                "urdf_path": str(model.path),
                "collision_config": collision,
                "pose_set": motion_preflight.pose_set,
                "reference_full_q": final_activation.reference_state.position,
                "torso_T_board": torso_T_camera @ verified_camera_T_board,
                "calibration_arm": plan["calibration_arm"],
                "escape_target_q": escape_target_q,
                "table_target_q": table_target_q,
                "table_plane_config": table_plane_config,
            },
            driver=driver,
        )
        if not final_escape_clearance.passed:
            raise ValueError(
                "post-takeover supported-start escape either deepens the modeled "
                "table contact or does not finish clear of the table plane. No "
                "changing arm target was sent"
            )
        if not final_elevated_route_clearance.passed:
            raise ValueError(
                "post-takeover elevated route puts "
                f"{final_elevated_route_clearance.minimum_link} only "
                f"{final_elevated_route_clearance.minimum_clearance_m:.4f}m above "
                "the table plane; require "
                f"{final_elevated_route_clearance.required_clearance_m:.4f}m. No "
                "changing arm target was sent"
            )
        driver.check()
        synchronized.install_validated_plan(
            pose_set=motion_preflight.pose_set,
            approved_validation_report_sha256=final_report.content_sha256,
            validated_reference_state=final_activation.reference_state,
        )
        driver.check()
        _print_dynamic_preflight(final_activation, final_report)
        print(
            "POST-TAKEOVER PLAN INSTALLED — beginning the validated one-shot "
            "open-loop motion now; the achieved image is used only for scoring, "
            "never for endpoint correction"
        )
        escape_outbound_started_s = time.monotonic()
        synchronized.start_pose(
            TABLE_ESCAPE_POSE_ID,
            approval=final_report.approval(
                HANDOFF_POSE_ID,
                TABLE_ESCAPE_POSE_ID,
            ),
            operator_confirmed=True,
        )
        _wait_for_driven_state(
            synchronized,
            driver,
            ExecutorState.READY,
            rclpy=rclpy,
            node=node,
            timeout_s=executor_config.motion_timeout_s + 5.0,
        )
        escape_outbound_duration_s = time.monotonic() - escape_outbound_started_s
        attained_escape_outbound = synchronized.observe_state()
        synchronized.start_pose(
            TABLE_TARGET_POSE_ID,
            approval=final_report.approval(
                TABLE_ESCAPE_POSE_ID,
                TABLE_TARGET_POSE_ID,
            ),
            operator_confirmed=True,
        )
        _wait_for_driven_state(
            synchronized,
            driver,
            ExecutorState.READY,
            rclpy=rclpy,
            node=node,
            timeout_s=executor_config.motion_timeout_s + 5.0,
        )
        attained_target_before_capture = synchronized.observe_state()
        # The complete command-space route, including both return edges, was
        # validated from the loaded state after ownership and before this first
        # changing target. Keep the nominal command at the target, capture, then
        # traverse those same segments in reverse. Do not create a different
        # return segment by rebasing on the measured tracking offset.
        synchronized.begin_capture()
        try:
            achieved_frames = _collect_new_camera_frames(
                rclpy,
                node,
                camera,
                frame_count=args.frame_count,
                timeout_s=args.camera_timeout_s,
                maximum_duration_s=args.maximum_burst_duration_s,
                control_check=driver.check,
            )
        except KeyboardInterrupt:
            if synchronized.state is ExecutorState.CAPTURING:
                synchronized.finish_capture(outcome="operator interrupted capture")
            raise
        except Exception as error:  # noqa: BLE001 - any visual failure is unscored.
            if synchronized.state is ExecutorState.CAPTURING:
                synchronized.finish_capture(outcome="visual capture failed")
            # A control-thread error is not a recoverable data-quality failure.
            # Re-raise it so the ownership cleanup runs immediately.
            driver.check()
            capture_failure = f"{type(error).__name__}: {error}"
            print(
                "TABLE IMAGE CAPTURE INVALID — completing the validated reverse "
                "motion before clean release; this trial will not produce an "
                "accuracy score"
            )
        else:
            synchronized.finish_capture(outcome="captured lossless table burst")
        attained_target_after_capture = synchronized.observe_state()
        target_capture_position_spread_rad = float(
            np.max(
                np.abs(
                    attained_target_after_capture.position
                    - attained_target_before_capture.position
                )
            )
        )
        synchronized.start_pose(
            TABLE_ESCAPE_POSE_ID,
            approval=final_report.approval(
                TABLE_TARGET_POSE_ID,
                TABLE_ESCAPE_POSE_ID,
            ),
            operator_confirmed=True,
        )
        _wait_for_driven_state(
            synchronized,
            driver,
            ExecutorState.READY,
            rclpy=rclpy,
            node=node,
            timeout_s=executor_config.motion_timeout_s + 5.0,
        )
        attained_escape_return = synchronized.observe_state()
        supported_return_timeout_s = (
            escape_outbound_duration_s
            + executor_config.acquisition_ramp_s
            + executor_config.settle_dwell_s
        )
        synchronized.start_pose(
            HANDOFF_POSE_ID,
            approval=final_report.approval(
                TABLE_ESCAPE_POSE_ID,
                HANDOFF_POSE_ID,
            ),
            operator_confirmed=True,
        )
        _wait_for_driven_state(
            synchronized,
            driver,
            ExecutorState.READY,
            rclpy=rclpy,
            node=node,
            timeout_s=supported_return_timeout_s,
        )
        attained_supported_return = synchronized.observe_state()
        # Stop rt/lowcmd before asking PC2 to select the AI motion service.  The
        # watchdog remains armed during this no-command handoff interval.
        driver.close()
        driver.check()
        dex3_controller.timeout()
        watchdog.restore_seated()
        synchronized.confirm_external_takeover(
            "PC2 verified AI takeover through FSM 0 -> 1 -> seated FSM 3"
        )
        print(
            "returned to the measured table-supported pose; PC2 verified "
            "AI FSM 0 -> 1 -> seated FSM 3 before debug lowcmd closed"
        )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_errors: list[tuple[str, BaseException]] = []
        if driver is not None and driver.is_alive:
            _attempt_safety_cleanup(
                cleanup_errors,
                "control driver stop",
                driver.close,
            )
        local_dex3_timeout_error = None
        if dex3_controller is not None and not dex3_controller.timed_out:
            try:
                dex3_controller.timeout()
            except BaseException as error:  # noqa: BLE001 - finish PC2 recovery.
                local_dex3_timeout_error = error

        def finish_guard_ownership() -> None:
            if watchdog is None:
                return
            if not (
                watchdog.armed
                or (transport is not None and transport.requires_external_takeover)
            ):
                return
            if transport is None:
                watchdog.disarm()
                return
            _terminate_debug_lowcmd_safely(
                watchdog,
                synchronized,
                transport,
                reason="table-accuracy debug lowcmd failed or was interrupted",
            )

        _attempt_safety_cleanup(
            cleanup_errors,
            "PC2/controller ownership handoff",
            finish_guard_ownership,
        )

        def close_dex3_control() -> None:
            if dex3_controller is None:
                return
            if (
                not dex3_controller.timed_out
                and watchdog is not None
                and (watchdog.terminal_action is not None)
            ):
                dex3_controller.close_after_external_timeout()
            else:
                dex3_controller.close()

        _attempt_safety_cleanup(
            cleanup_errors,
            "Dex3 controller close",
            close_dex3_control,
        )
        if local_dex3_timeout_error is not None and (
            watchdog is None or watchdog.terminal_action is None
        ):
            cleanup_errors.append(("local Dex3 timeout", local_dex3_timeout_error))

        def close_unowned_transport() -> None:
            if transport is not None and not transport.requires_external_takeover:
                transport.close()

        _attempt_safety_cleanup(
            cleanup_errors,
            "unowned debug transport close",
            close_unowned_transport,
        )
        if watchdog is not None and (watchdog.armed and transport is None):
            _attempt_safety_cleanup(
                cleanup_errors,
                "orphaned watchdog disarm",
                watchdog.disarm,
            )
        if observer is not None:
            _attempt_safety_cleanup(
                cleanup_errors,
                "lowstate observer close",
                observer.close,
            )
        if dex3_observer is not None:
            _attempt_safety_cleanup(
                cleanup_errors,
                "Dex3 observer close",
                dex3_observer.close,
            )
        if camera is not None:
            _attempt_safety_cleanup(
                cleanup_errors,
                "camera close",
                camera.close,
            )
        _attempt_safety_cleanup(cleanup_errors, "ROS node destroy", node.destroy_node)
        _attempt_safety_cleanup(
            cleanup_errors,
            "ROS shutdown",
            lambda: _shutdown_rclpy_once(rclpy),
        )
        _attempt_safety_cleanup(
            cleanup_errors,
            "command lock release",
            command_lock.release,
        )
        if primary_error is not None:
            _attempt_safety_cleanup(
                cleanup_errors,
                "table failure evidence write",
                lambda: _write_table_failure_bundle(
                    output=output,
                    error=primary_error,
                    plan=plan,
                    input_plan=input_plan,
                    hardware_bytes=hardware_bytes,
                    collision_bytes=collision_bytes,
                    preflight_frames=preflight_frames,
                    loaded_planning_frames=loaded_planning_frames,
                    verification_frames=verification_frames,
                    achieved_frames=achieved_frames,
                    motion_preflight=motion_preflight,
                    final_report=final_report,
                    synchronized=synchronized,
                    watchdog=watchdog,
                    transport=transport,
                    attained_states={
                        "lifted_start_outbound": attained_escape_outbound,
                        "table_target_before_capture": (attained_target_before_capture),
                        "table_target_after_capture": attained_target_after_capture,
                        "lifted_start_return": attained_escape_return,
                        "supported_return": attained_supported_return,
                    },
                    cleanup_errors=cleanup_errors,
                ),
            )
        _raise_safety_cleanup_failures(primary_error, cleanup_errors)

    assert initial_observation is not None
    assert loaded_planning_observation is not None
    assert verification_observation is not None
    assert execution_plan is not None
    assert motion_preflight is not None
    assert final_report is not None
    assert final_escape_clearance is not None
    assert final_elevated_route_clearance is not None
    assert final_activation is not None
    assert board_shift is not None
    assert initial_cube_shift is not None
    assert takeover_board_shift is not None
    assert takeover_cube_shift is not None
    assert approval_cube_shift is not None
    assert escape_outbound_duration_s is not None
    assert supported_return_timeout_s is not None
    assert attained_escape_outbound is not None
    assert attained_target_before_capture is not None
    assert attained_target_after_capture is not None
    assert attained_escape_return is not None
    assert attained_supported_return is not None
    assert target_capture_position_spread_rad is not None
    assert dex3_posture_evidence is not None
    gravity_reference_q = gravity_feedforward.reference_full_q
    takeover_q14 = np.concatenate(
        (gravity_reference_q[15:22], gravity_reference_q[22:29])
    )
    takeover_gravity_tau = gravity_feedforward.torque_for(takeover_q14)
    escape_command_q14 = takeover_q14.copy()
    table_command_q14 = takeover_q14.copy()
    calibration_slice = (
        slice(0, 7) if plan["calibration_arm"] == "left" else slice(7, 14)
    )
    escape_command_q14[calibration_slice] = escape_target_q
    table_command_q14[calibration_slice] = table_target_q
    escape_gravity_tau = gravity_feedforward.torque_for(escape_command_q14)
    table_gravity_tau = gravity_feedforward.torque_for(table_command_q14)
    preflight_manifest = _write_table_image_burst(
        output / "preflight",
        preflight_frames,
        commands_robot=False,
    )
    loaded_planning_manifest = _write_table_image_burst(
        output / "loaded_planning",
        loaded_planning_frames,
        commands_robot=True,
    )
    verification_manifest = _write_table_image_burst(
        output / "loaded_verification",
        verification_frames,
        commands_robot=True,
    )
    achieved_manifest = None
    evaluation = None
    evaluation_failure = capture_failure
    if achieved_frames:
        achieved_manifest = _write_table_image_burst(
            output / "achieved",
            achieved_frames,
            commands_robot=True,
        )
        if evaluation_failure is None:
            try:
                achieved_observation = observe_burst(
                    sorted((output / "achieved").glob("frame_*.png")),
                    camera_info=camera_info,
                    hand_cube_detector=hand_cube_detector,
                    board_detector=board_detector,
                    minimum_accepted_frames=3,
                    reject_unstable=False,
                )
                measured_capture_q = 0.5 * (
                    attained_target_before_capture.position
                    + attained_target_after_capture.position
                )
                torso_T_hand_at_measured_q = model.transform(
                    "torso_link",
                    arm_hand_link(plan["calibration_arm"]),
                    dict(
                        zip(
                            G1_29_JOINT_NAMES,
                            measured_capture_q,
                            strict=True,
                        )
                    ),
                )
                predicted_board_T_cube = predicted_board_T_cube_from_model(
                    torso_T_camera=torso_T_camera,
                    hand_T_cube=hand_T_cube,
                    camera_T_board=_observation_transform(
                        achieved_observation,
                        "camera_T_board",
                    ),
                    torso_T_hand_at_measured_q=torso_T_hand_at_measured_q,
                )
                evaluation = create_evaluation_document(
                    plan=plan,
                    achieved_burst=achieved_observation,
                    predicted_board_T_cube=predicted_board_T_cube,
                )
            except Exception as error:  # noqa: BLE001 - preserve scoring failure.
                evaluation_failure = f"{type(error).__name__}: {error}"
    if evaluation_failure is not None:
        failure_document = {
            "schema_version": 1,
            "kind": "g1_table_accuracy_evaluation_failure",
            "plan_sha256": plan["content_sha256"],
            "reason": evaluation_failure,
            "motion_completed_safely": True,
        }
        failure_document["content_sha256"] = _hardware_document_sha256(failure_document)
        _write_hardware_json(output / "evaluation_error.json", failure_document)
    (output / "pose_set.yaml").write_bytes(_pose_set_bytes(motion_preflight.pose_set))
    (output / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "input_plan.json").write_text(
        json.dumps(input_plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "hardware.yaml").write_bytes(hardware_bytes)
    (output / "collision_pairs.yaml").write_bytes(collision_bytes)
    (output / "validation_report.json").write_bytes(
        _validation_report_bytes(final_report)
    )
    if evaluation is not None:
        _write_hardware_json(output / "evaluation.json", evaluation)
    runtime = {
        "schema_version": 3,
        "kind": "g1_table_accuracy_execution",
        "plan_sha256": plan["content_sha256"],
        "input_plan_sha256": input_plan["content_sha256"],
        "planning_policy": (
            "capture_loaded_state_and_build_ik_fcl_route_after_lowcmd_ownership"
        ),
        "final_validation_report_sha256": final_report.content_sha256,
        "hardware_config_sha256": hashlib.sha256(hardware_bytes).hexdigest(),
        "collision_config_sha256": collision.content_sha256,
        "table_plane_config": {
            name: getattr(table_plane_config, name)
            for name in table_plane_config.__dataclass_fields__
        },
        "escape_target": {
            "desired_board_T_hand_cube": desired_board_T_escape.tolist(),
            "desired_torso_T_hand": desired_torso_T_escape.tolist(),
            "policy": "preserve_live_xy_orientation_and_lift_to_planned_relative_z",
        },
        "motion_preflight": motion_preflight.to_dict(),
        "final_escape_clearance": final_escape_clearance.to_dict(),
        "final_elevated_route_clearance": final_elevated_route_clearance.to_dict(),
        "final_activation_state": final_activation.reference_state.to_dict(),
        "final_activation_readiness": final_activation.readiness.to_dict(),
        "gravity_feedforward": {
            "implementation": "pinocchio_rnea_at_commanded_q",
            "upstream_reference": UNITREE_XR_GRAVITY_REFERENCE,
            "pinocchio_version": gravity_feedforward.backend_version,
            "urdf_sha256": gravity_feedforward.urdf_sha256,
            "velocity_rad_s": [0.0] * 29,
            "acceleration_rad_s2": [0.0] * 29,
            "non_arm_reference": "measured_seated_takeover_configuration",
            "reference_full_q": gravity_reference_q.tolist(),
            "external_payload_model": "none",
            "lowcmd_torque_ramp": "command_weight_zero_to_one_during_acquisition",
            "takeover_tau14_nm": takeover_gravity_tau.tolist(),
            "escape_tau14_nm": escape_gravity_tau.tolist(),
            "table_target_tau14_nm": table_gravity_tau.tolist(),
        },
        "dex3_posture_control": {
            "posture_name": "nvidia_groot_middle_close",
            "posture_source": DEX3_CALIBRATION_POSTURE_SOURCE,
            "left_commanded_q_rad": list(dex3_controller.config.left_target_q_rad),
            "right_commanded_q_rad": list(dex3_controller.config.right_target_q_rad),
            "measured_settled_state": dex3_posture_evidence.to_dict(),
            "terminal_policy": "unitree_timeout_bit_on_both_hands",
        },
        "loaded_planning_to_verification_camera_shift": board_shift,
        "planned_to_live_initial_cube_shift": initial_cube_shift,
        "takeover_camera_to_board_shift": takeover_board_shift,
        "takeover_board_to_cube_shift": takeover_cube_shift,
        "loaded_planning_to_verification_cube_shift": approval_cube_shift,
        "escape_outbound_duration_s": escape_outbound_duration_s,
        "supported_return_timeout_s": supported_return_timeout_s,
        "preflight_observation": initial_observation,
        "loaded_planning_observation": loaded_planning_observation,
        "loaded_planning_rejections": list(loaded_planning_rejections),
        "loaded_verification_observation": verification_observation,
        "loaded_verification_rejections": list(verification_rejections),
        "capture_manifests": {
            "preflight_sha256": preflight_manifest["content_sha256"],
            "loaded_planning_sha256": loaded_planning_manifest["content_sha256"],
            "loaded_verification_sha256": verification_manifest["content_sha256"],
            "achieved_sha256": (
                None
                if achieved_manifest is None
                else achieved_manifest["content_sha256"]
            ),
        },
        "evaluation_status": (
            "failed"
            if evaluation is None
            else (
                "passed"
                if evaluation["measurement_quality"]["passed"]
                else "measured_with_repeatability_warning"
            )
        ),
        "evaluation_failure": evaluation_failure,
        "endpoint_policy": "measured_stationarity_with_tracking_error_recorded",
        "capture_policy": "capture_immediately_after_target_settle_before_return",
        "command_path_policy": (
            "post_takeover_loaded_state_plan_then_prevalidated_nominal_segments_"
            "reversed_without_endpoint_visual_correction"
        ),
        "endpoint_tracking_evidence": {
            "lifted_start_outbound": _joint_endpoint_evidence(
                sample=attained_escape_outbound,
                target_q=escape_target_q,
                calibration_arm=plan["calibration_arm"],
            ),
            "table_target_before_capture": _joint_endpoint_evidence(
                sample=attained_target_before_capture,
                target_q=table_target_q,
                calibration_arm=plan["calibration_arm"],
            ),
            "table_target_after_capture": _joint_endpoint_evidence(
                sample=attained_target_after_capture,
                target_q=table_target_q,
                calibration_arm=plan["calibration_arm"],
            ),
            "lifted_start_return": _joint_endpoint_evidence(
                sample=attained_escape_return,
                target_q=escape_target_q,
                calibration_arm=plan["calibration_arm"],
            ),
            "supported_return": _joint_endpoint_evidence(
                sample=attained_supported_return,
                target_q=final_activation.handoff_q,
                calibration_arm=plan["calibration_arm"],
            ),
            "maximum_full_joint_change_during_capture_rad": (
                target_capture_position_spread_rad
            ),
        },
        "executor_events": [_executor_event_dict(item) for item in synchronized.events],
        "terminal_state": synchronized.state.value,
        "terminal_external_takeover_verified": watchdog.terminal_action
        in {"seated", "zero_torque"},
        "terminal_watchdog_action": watchdog.terminal_action,
        "verified_zero_torque": watchdog.terminal_action == "zero_torque",
        "verified_seated": watchdog.terminal_action == "seated",
        "returned_to_dynamic_handoff": (
            synchronized.current_pose_id == HANDOFF_POSE_ID
        ),
        "commands_robot": True,
    }
    runtime["content_sha256"] = _hardware_document_sha256(runtime)
    _write_hardware_json(output / "runtime.json", runtime)
    result = {
        "output": str(output),
        "plan_sha256": plan["content_sha256"],
        "terminal_external_takeover_verified": watchdog.terminal_action
        in {"seated", "zero_torque"},
        "verified_zero_torque": watchdog.terminal_action == "zero_torque",
        "verified_seated": watchdog.terminal_action == "seated",
        "returned_to_dynamic_handoff": True,
        "evaluation_status": (
            "failed"
            if evaluation is None
            else (
                "passed"
                if evaluation["measurement_quality"]["passed"]
                else "measured_with_repeatability_warning"
            )
        ),
    }
    if evaluation is not None:
        result["evaluation_sha256"] = evaluation["content_sha256"]
        result.update(evaluation["task_error"])
        result["measurement_quality"] = evaluation["measurement_quality"]
        result["model_implied_tracking_error"] = evaluation[
            "model_implied_tracking_error"
        ]
        result["held_out_composite_model_error"] = evaluation[
            "held_out_composite_model_error"
        ]
    else:
        result["evaluation_error"] = evaluation_failure
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if evaluation is not None else 1


def _validate_table_execution_arguments(args: argparse.Namespace) -> None:
    if args.frame_count < 3:
        raise ValueError("--frame-count must be at least 3")
    if args.camera_timeout_s <= 0:
        raise ValueError("--camera-timeout-s must be positive")
    if args.maximum_burst_duration_s <= 0:
        raise ValueError("--maximum-burst-duration-s must be positive")
    if args.ik_restarts < 1:
        raise ValueError("--ik-restarts must be positive")
    if args.minimum_board_clearance_mm <= 0:
        raise ValueError("--minimum-board-clearance-mm must be positive")
    if args.maximum_board_shift_mm <= 0:
        raise ValueError("--maximum-board-shift-mm must be positive")
    if args.maximum_board_shift_deg <= 0:
        raise ValueError("--maximum-board-shift-deg must be positive")


def _validate_table_plan_artifacts(plan: dict, *, model: URDFModel) -> None:
    if plan["sources"]["urdf_sha256"] != model.sha256:
        raise ValueError("table-accuracy plan belongs to a different URDF")
    hand_cube_path = Path(plan["sources"]["hand_cube_config_path"])
    if not hand_cube_path.is_file():
        raise FileNotFoundError(
            f"hand-cube config from plan is missing: {hand_cube_path}"
        )
    hand_cube_sha256 = hashlib.sha256(hand_cube_path.read_bytes()).hexdigest()
    if hand_cube_sha256 != plan["sources"]["hand_cube_config_sha256"]:
        raise ValueError("hand-cube config has changed since the plan was created")


def _collect_new_camera_frames(
    rclpy,
    node,
    camera,
    *,
    frame_count: int,
    timeout_s: float,
    maximum_duration_s: float,
    control_check=None,
) -> tuple[ROSImageFrame, ...]:
    """Collect only frames received after this function is entered."""

    if frame_count < 1:
        raise ValueError("camera frame count must be positive")
    if timeout_s <= 0 or maximum_duration_s <= 0:
        raise ValueError("camera timing limits must be positive")
    existing = camera.frames.snapshot()
    seen = {
        (item.timing.receipt_monotonic_s, item.timing.header_stamp_ns)
        for item in existing
    }
    expected_profile = None if not existing else existing[-1].camera_info.profile_sha256
    collected: list[ROSImageFrame] = []
    deadline = time.monotonic() + timeout_s
    while len(collected) < frame_count:
        if control_check is not None:
            control_check()
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"timed out after collecting {len(collected)}/{frame_count} "
                "new rectified camera frames"
            )
        rclpy.spin_once(node, timeout_sec=0.05)
        if control_check is not None:
            control_check()
        for frame in camera.frames.snapshot():
            key = (
                frame.timing.receipt_monotonic_s,
                frame.timing.header_stamp_ns,
            )
            if key in seen:
                continue
            seen.add(key)
            if expected_profile is None:
                expected_profile = frame.camera_info.profile_sha256
            if frame.camera_info.profile_sha256 != expected_profile:
                raise ValueError("camera profile changed during the image burst")
            collected.append(frame)
            if len(collected) == frame_count:
                break
        if len(collected) >= 2:
            duration = (
                collected[-1].timing.receipt_monotonic_s
                - collected[0].timing.receipt_monotonic_s
            )
            if duration > maximum_duration_s:
                raise RuntimeError(
                    f"camera burst duration is {duration:.3f}s; limit is "
                    f"{maximum_duration_s:.3f}s"
                )
    return tuple(collected)


def _retry_table_observation(
    *,
    collect: Callable[[], tuple[ROSImageFrame, ...]],
    observe: Callable[[tuple[ROSImageFrame, ...]], dict],
    label: str,
    maximum_attempts: int = 3,
) -> tuple[tuple[ROSImageFrame, ...], dict, tuple[str, ...]]:
    """Retry a visual rejection without changing the robot target."""

    if not label:
        raise ValueError("table-observation retry label must be non-empty")
    if maximum_attempts < 1:
        raise ValueError("table-observation maximum attempts must be positive")
    rejections: list[str] = []
    for attempt in range(1, maximum_attempts + 1):
        frames = collect()
        try:
            observation = observe(frames)
        except ValueError as error:
            reason = str(error)
            rejections.append(reason)
            if attempt < maximum_attempts:
                print(
                    f"{label} burst {attempt}/{maximum_attempts} rejected: "
                    f"{reason}; collecting a fresh burst before any changing target"
                )
                continue
            raise ValueError(
                f"{label} failed after {maximum_attempts} fresh bursts; "
                f"last rejection: {reason}"
            ) from error
        return frames, observation, tuple(rejections)
    raise AssertionError("unreachable table-observation retry state")


def _capture_controlled_table_observation(
    *,
    synchronized: SynchronizedPoseExecutor,
    driver: ExecutorControlDriver,
    collect: Callable[[], tuple[ROSImageFrame, ...]],
    observe: Callable[[tuple[ROSImageFrame, ...]], dict],
    label: str,
) -> tuple[tuple[ROSImageFrame, ...], dict, tuple[str, ...]]:
    """Capture while the fixed-rate driver holds one unchanged command target."""

    synchronized.begin_capture()
    try:
        result = _retry_table_observation(
            collect=collect,
            observe=observe,
            label=label,
        )
    except BaseException:
        if synchronized.state is ExecutorState.CAPTURING:
            synchronized.finish_capture(outcome=f"{label} failed")
        driver.check()
        raise
    synchronized.finish_capture(outcome=f"{label} accepted")
    driver.check()
    return result


def _write_table_image_burst(
    output: Path,
    frames: tuple[ROSImageFrame, ...],
    *,
    commands_robot: bool = False,
) -> dict:
    if not frames:
        raise ValueError("cannot write an empty table image burst")
    if output.exists():
        raise FileExistsError(f"table image burst already exists: {output}")
    profiles = {item.camera_info.profile_sha256 for item in frames}
    if len(profiles) != 1:
        raise ValueError("table image burst mixes camera profiles")
    output.mkdir(parents=True)
    records = []
    for index, frame in enumerate(frames, start=1):
        name = f"frame_{index:03d}.png"
        ok, encoded = cv2.imencode(
            ".png",
            frame.image_bgr,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )
        if not ok:
            raise RuntimeError(f"failed to losslessly encode table frame {index}")
        content = encoded.tobytes()
        (output / name).write_bytes(content)
        records.append(
            {
                "file": name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "receipt_monotonic_s": frame.timing.receipt_monotonic_s,
                "receipt_utc": frame.timing.receipt_utc,
                "header_stamp_ns": frame.timing.header_stamp_ns,
            }
        )
    duration_s = (
        frames[-1].timing.receipt_monotonic_s - frames[0].timing.receipt_monotonic_s
    )
    manifest = {
        "schema_version": 1,
        "kind": "g1_table_accuracy_image_burst",
        "created_at_utc": utc_now_iso(),
        "camera_info": frames[0].camera_info.to_dict(),
        "camera_profile_sha256": frames[0].camera_info.profile_sha256,
        "frame_count": len(frames),
        "duration_s": duration_s,
        "commands_robot": bool(commands_robot),
        "frames": records,
    }
    manifest["content_sha256"] = _hardware_document_sha256(manifest)
    _write_hardware_json(output / "manifest.json", manifest)
    return manifest


def _write_table_failure_bundle(
    *,
    output: Path,
    error: BaseException,
    plan: dict,
    input_plan: dict,
    hardware_bytes: bytes,
    collision_bytes: bytes,
    preflight_frames: tuple[ROSImageFrame, ...],
    loaded_planning_frames: tuple[ROSImageFrame, ...],
    verification_frames: tuple[ROSImageFrame, ...],
    achieved_frames: tuple[ROSImageFrame, ...],
    motion_preflight,
    final_report,
    synchronized,
    watchdog,
    transport,
    attained_states: dict,
    cleanup_errors: list[tuple[str, BaseException]],
) -> None:
    """Persist a failed physical trial after command ownership cleanup completes."""

    if output.exists():
        raise FileExistsError(f"table failure output already exists: {output}")
    output.mkdir(parents=True)
    manifests = {}
    for name, frames, commands_robot in (
        ("preflight", preflight_frames, False),
        ("loaded_planning", loaded_planning_frames, True),
        ("loaded_verification", verification_frames, True),
        ("achieved", achieved_frames, True),
    ):
        if frames:
            manifests[name] = _write_table_image_burst(
                output / name,
                frames,
                commands_robot=commands_robot,
            )["content_sha256"]
    (output / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "input_plan.json").write_text(
        json.dumps(input_plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "hardware.yaml").write_bytes(hardware_bytes)
    (output / "collision_pairs.yaml").write_bytes(collision_bytes)
    if motion_preflight is not None:
        (output / "pose_set.yaml").write_bytes(
            _pose_set_bytes(motion_preflight.pose_set)
        )
    if final_report is not None:
        (output / "validation_report.json").write_bytes(
            _validation_report_bytes(final_report)
        )
    document = {
        "schema_version": 3,
        "kind": "g1_table_accuracy_execution_failure",
        "occurred_at_utc": utc_now_iso(),
        "plan_sha256": plan["content_sha256"],
        "input_plan_sha256": input_plan["content_sha256"],
        "error": f"{type(error).__name__}: {error}",
        "commands_published": bool(
            transport is not None and transport.command_count > 0
        ),
        "command_count": None if transport is None else transport.command_count,
        "executor_state": (None if synchronized is None else synchronized.state.value),
        "executor_fault_reason": (
            None if synchronized is None else synchronized.fault_reason
        ),
        "executor_events": (
            []
            if synchronized is None
            else [_executor_event_dict(item) for item in synchronized.events]
        ),
        "terminal_watchdog_action": (
            None if watchdog is None else watchdog.terminal_action
        ),
        "attained_states": {
            name: None if sample is None else sample.to_dict()
            for name, sample in attained_states.items()
        },
        "capture_manifest_sha256": manifests,
        "cleanup_errors_before_failure_log": [
            {"action": action, "error": f"{type(item).__name__}: {item}"}
            for action, item in cleanup_errors
        ],
        "interpretation": (
            "failure evidence only; absence of an evaluation.json means no "
            "tabletop landing score was produced"
        ),
    }
    document["content_sha256"] = _hardware_document_sha256(document)
    _write_hardware_json(output / "failure.json", document)


def _observe_table_frames(
    frames: tuple[ROSImageFrame, ...],
    *,
    camera_info: RectifiedCameraInfo,
    hand_cube_detector: CorrespondenceDetector,
    board_detector: CharucoBoardPoseDetector,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="g1-table-observation-") as temporary:
        directory = Path(temporary) / "burst"
        _write_table_image_burst(directory, frames)
        return observe_burst(
            sorted(directory.glob("frame_*.png")),
            camera_info=camera_info,
            hand_cube_detector=hand_cube_detector,
            board_detector=board_detector,
            minimum_accepted_frames=3,
        )


def _observation_transform(observation: dict, name: str) -> np.ndarray:
    try:
        raw = observation["aggregate"][name]
    except (KeyError, TypeError) as error:
        raise ValueError(f"table observation is missing aggregate {name}") from error
    return validate_transform(np.asarray(raw, dtype=np.float64))


def _transform_delta(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    first = validate_transform(first)
    second = validate_transform(second)
    translation_mm = 1000.0 * float(np.linalg.norm(second[:3, 3] - first[:3, 3]))
    relative = first[:3, :3].T @ second[:3, :3]
    rotation_deg = float(np.degrees(Rotation.from_matrix(relative.copy()).magnitude()))
    return {
        "translation_mm": translation_mm,
        "rotation_deg": rotation_deg,
    }


def _print_table_motion_preflight(
    activation: ActivationHandoff,
    motion_preflight,
) -> None:
    report = motion_preflight.validation_report
    clearances = [
        edge.minimum_clearance_m
        for edge in report.edges
        if edge.minimum_clearance_m is not None
    ]
    minimum_clearance = min(clearances) if clearances else None
    escape = motion_preflight.escape_ik_solution
    target = motion_preflight.target_ik_solution
    escape_clearance = motion_preflight.escape_clearance
    elevated_route_clearance = motion_preflight.elevated_route_clearance
    escape_joint_text = ", ".join(
        f"{name}={value:.4f}"
        for name, value in zip(
            arm_joint_names(motion_preflight.pose_set.calibration_arm),
            escape.calibration_q,
            strict=True,
        )
    )
    target_joint_text = ", ".join(
        f"{name}={value:.4f}"
        for name, value in zip(
            arm_joint_names(motion_preflight.pose_set.calibration_arm),
            target.calibration_q,
            strict=True,
        )
    )
    print(
        "LIVE SUPPORTED-START ROUTE PASSED IK + FCL: "
        f"escape IK error={escape.translation_error_m * 1000.0:.3f}mm/"
        f"{escape.rotation_error_deg:.3f}deg, "
        f"target IK error={target.translation_error_m * 1000.0:.3f}mm/"
        f"{target.rotation_error_deg:.3f}deg, "
        f"minimum clearance="
        f"{'n/a' if minimum_clearance is None else f'{minimum_clearance:.4f}m'}, "
        "escape destination board-plane clearance="
        f"{_optional_clearance(escape_clearance.destination_clearance_m)}, "
        "elevated-route board-plane clearance="
        f"{elevated_route_clearance.minimum_clearance_m:.4f}m "
        f"at {elevated_route_clearance.minimum_link}, "
        f"stationary samples={activation.readiness.sample_count}"
    )
    print("LIFTED START JOINTS — " + escape_joint_text)
    print("BOARD TARGET JOINTS — " + target_joint_text)


def _build_table_motion_preflight_worker(
    *,
    urdf_path: str,
    collision_config: CollisionConfig,
    desired_torso_T_escape: np.ndarray,
    desired_torso_T_hand: np.ndarray,
    torso_T_board: np.ndarray,
    reference_full_q: np.ndarray,
    calibration_arm: str,
    plan_sha256: str,
    recorded_at_utc: str,
    recorded_monotonic_s: float,
    ik_config: IKConfig,
    table_plane_config: TablePlaneConfig,
):
    """Build the heavy IK/FCL plan in a process isolated from control timing."""

    return build_table_motion_preflight(
        model=URDFModel(urdf_path),
        collision_config=collision_config,
        desired_torso_T_escape=desired_torso_T_escape,
        desired_torso_T_hand=desired_torso_T_hand,
        torso_T_board=torso_T_board,
        reference_full_q=reference_full_q,
        calibration_arm=calibration_arm,
        plan_sha256=plan_sha256,
        recorded_at_utc=recorded_at_utc,
        recorded_monotonic_s=recorded_monotonic_s,
        ik_config=ik_config,
        table_plane_config=table_plane_config,
    )


def _validate_final_table_motion_worker(
    *,
    urdf_path: str,
    collision_config: CollisionConfig,
    pose_set: PoseSet,
    reference_full_q: np.ndarray,
    torso_T_board: np.ndarray,
    calibration_arm: str,
    escape_target_q: np.ndarray,
    table_target_q: np.ndarray,
    table_plane_config: TablePlaneConfig,
) -> tuple[ValidationReport, object, object]:
    """Revalidate the exact loaded handoff and final observed table plane."""

    model = URDFModel(urdf_path)
    path_config = PathValidationConfig()
    report = PosePathValidator(
        model=model,
        collision_checker=FCLCollisionChecker(model, collision_config),
        config=path_config,
    ).validate(
        pose_set,
        directed_edges=(
            (HANDOFF_POSE_ID, TABLE_ESCAPE_POSE_ID),
            (TABLE_ESCAPE_POSE_ID, TABLE_TARGET_POSE_ID),
            (TABLE_TARGET_POSE_ID, TABLE_ESCAPE_POSE_ID),
            (TABLE_ESCAPE_POSE_ID, HANDOFF_POSE_ID),
        ),
        reference_full_q=reference_full_q,
    )
    if not report.passed:
        failures = [
            f"{edge.from_pose_id}->{edge.to_pose_id}: " + "; ".join(edge.failures)
            for edge in report.edges
            if not edge.passed
        ]
        raise ValueError("dynamic route validation failed: " + " | ".join(failures))
    escape_clearance = validate_table_plane_escape_path(
        model=model,
        collision_config=collision_config,
        torso_T_board=torso_T_board,
        reference_full_q=reference_full_q,
        calibration_arm=calibration_arm,
        target_calibration_q=escape_target_q,
        config=table_plane_config,
    )
    elevated_route_clearance = validate_table_plane_path(
        model=model,
        collision_config=collision_config,
        torso_T_board=torso_T_board,
        reference_full_q=reference_full_q,
        calibration_arm=calibration_arm,
        source_calibration_q=escape_target_q,
        target_calibration_q=table_target_q,
        config=table_plane_config,
    )
    return report, escape_clearance, elevated_route_clearance


def _validate_loaded_dex3_finger_sweep_worker(
    *,
    urdf_path: str,
    collision_config: CollisionConfig,
    body_q: np.ndarray,
    initial_left_hand_q_rad: np.ndarray,
    initial_right_hand_q_rad: np.ndarray,
    target_left_hand_q_rad: tuple[float, ...],
    target_right_hand_q_rad: tuple[float, ...],
    path_config: PathValidationConfig,
):
    """Run articulated Dex3 FCL away from the fixed-rate control process."""

    model = URDFModel(urdf_path)
    return validate_dex3_finger_sweep_at_state(
        model=model,
        collision_checker=FCLCollisionChecker(model, collision_config),
        body_q=body_q,
        initial_left_hand_q_rad=initial_left_hand_q_rad,
        initial_right_hand_q_rad=initial_right_hand_q_rad,
        target_left_hand_q_rad=target_left_hand_q_rad,
        target_right_hand_q_rad=target_right_hand_q_rad,
        config=path_config,
    )


def _run_isolated_control_work(
    *,
    label: str,
    worker: Callable,
    worker_kwargs: dict,
    driver: ExecutorControlDriver,
    control_maintenance: Callable[[], None] | None = None,
):
    """Poll control health while CPU/native planning runs in a spawned process."""

    if not label.strip():
        raise ValueError("isolated control-work label must be non-empty")
    pool = ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn"))
    future = pool.submit(worker, **worker_kwargs)
    started_s = time.monotonic()
    next_status_s = started_s + 5.0
    try:
        while True:
            try:
                result = future.result(
                    timeout=0.01 if control_maintenance is not None else 0.1
                )
                break
            except FuturesTimeoutError:
                driver.check()
                if control_maintenance is not None:
                    control_maintenance()
                now = time.monotonic()
                if now >= next_status_s:
                    print(
                        f"{label} still running after {now - started_s:.1f}s; "
                        "the fixed-rate controller is holding the unchanged target"
                    )
                    next_status_s = now + 5.0
        driver.check()
    except BaseException:
        future.cancel()
        pool.shutdown(wait=future.done(), cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return result


def _optional_clearance(value: float | None) -> str:
    return "no moving geometry" if value is None else f"{value:.4f}m"


def _executor_event_dict(event) -> dict:
    return {
        "sequence": event.sequence,
        "occurred_monotonic_s": event.occurred_monotonic_s,
        "previous_state": event.previous_state.value,
        "state": event.state.value,
        "reason": event.reason,
    }


def _hardware_document_sha256(document: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _write_hardware_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_commission_weight_zero(args: argparse.Namespace) -> int:
    _require_ack(args.confirm, WRITE_ACK)
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be positive")
    clock = SystemClock()
    with CommandOwnerLock(args.lock_file):
        transport = UnitreeArmSDKTransport(_transport_config(args), clock=clock)
        try:
            state = _wait_for_state(transport, args.timeout_s)
            command = ArmCommand.create(
                [*state.left_q, *state.right_q],
                weight=0.0,
                issued_monotonic_s=clock.monotonic(),
            )
            transport.send_command(command)
        finally:
            transport.close()
    print("weight-zero arm_sdk packet published and transport closed")
    return 0


def run_commission_damping(args: argparse.Namespace) -> int:
    """Verify the actual supported damping transition without arm ownership."""

    _require_ack(args.confirm, DAMP_ACK)
    observer = UnitreeLowStateObserver(_transport_config(args))
    watchdog = _pc2_damping_watchdog(
        args,
        args.hardware_config,
        require_regular=False,
    )
    try:
        initial = _wait_for_state(observer, 5.0)
        watchdog.start()
        print(
            f"PC2 damping watchdog armed from mode_machine={initial.mode_machine}; "
            f"trigger={args.trigger}"
        )
        if args.trigger == "heartbeat-timeout":
            watchdog.wait_for_automatic_damping()
        else:
            watchdog.damp("explicit physical commissioning test")
        print(
            "physical damping commissioning passed; locomotion fsm_id=1 "
            f"(mode_machine remains {initial.mode_machine})"
        )
        return 0
    except BaseException:
        if watchdog.armed:
            watchdog.damp("damping commissioning interrupted")
        raise
    finally:
        observer.close()


def run_commission_dex3_middle_close(args: argparse.Namespace) -> int:
    """Commission middle-close only after a live-derived clearance route."""

    _require_ack(args.confirm, MOTION_ACK)
    if args.duration_s <= 0.0:
        raise ValueError("--duration-s must be positive")
    states = StateSampleBuffer()
    lowstate = None
    dex3_observer = None
    dex3_controller = None
    transport = None
    synchronized = None
    driver = None
    watchdog = None
    primary_error = None
    cleanup_errors: list[tuple[str, BaseException]] = []
    with CommandOwnerLock(args.lock_file):
        try:
            collision = _validate_collision_preflight(args.collision_config)
            recording, _, executor_config, rate_hz = _runtime_configs(
                args.hardware_config
            )
            pose_set = _hardware_pose_set(args.hardware_config)
            lowstate = UnitreeLowStateObserver(
                _transport_config(args), on_sample=states.add
            )
            dex3_config = _dex3_control_config(args)
            base_path_config = PathValidationConfig()
            dex3_observer = UnitreeDex3StateObserver(
                dex3_config,
                initialize_factory=False,
            )
            activation = _wait_for_activation_handoff(
                lowstate,
                states,
                pose_set,
                recording,
            )
            initial_hands = _wait_for_dex3_state(dex3_observer, 5.0)
            model = URDFModel(_configured_urdf(args.hardware_config))
            dex3_model = URDFModel(_configured_gravity_urdf(args.hardware_config))
            live_collision = live_dex3_collision_config(collision)
            print(
                "READ-ONLY DEX3 CLEARANCE PLANNING — searching outward from the "
                "fresh stationary shoulder positions and checking the complete "
                "measured-to-target finger sweep; no command publisher exists",
                flush=True,
            )
            clearance_plan = plan_dual_arm_shoulder_clearance(
                model=model,
                collision_checker=FCLCollisionChecker(model, collision),
                dex3_model=dex3_model,
                dex3_collision_checker=FCLCollisionChecker(dex3_model, live_collision),
                reference_full_q=activation.reference_state.position,
                initial_left_hand_q_rad=initial_hands.left.position,
                initial_right_hand_q_rad=initial_hands.right.position,
                target_left_hand_q_rad=dex3_config.left_target_q_rad,
                target_right_hand_q_rad=dex3_config.right_target_q_rad,
                path_config=base_path_config,
                progress=lambda index, total, offset: print(
                    "READ-ONLY DEX3 CLEARANCE PLANNING: candidate "
                    f"{index}/{total}, outward offset={offset:.4f}rad; NO MOTION",
                    flush=True,
                ),
                rejection=lambda offset, reason: print(
                    "READ-ONLY DEX3 CLEARANCE REJECTED: "
                    f"outward offset={offset:.4f}rad; {reason}; NO MOTION",
                    flush=True,
                ),
            )
            left_target = clearance_plan.dual_clearance_q14[1]
            right_target = clearance_plan.dual_clearance_q14[8]
            pair = clearance_plan.finger_sweep_minimum_pair
            pair_text = "unknown" if pair is None else "/".join(pair)
            recovery = clearance_plan.finger_sweep_recovered_start_limit_joints
            recovery_text = "none" if not recovery else ",".join(recovery)
            print(
                "READ-ONLY DEX3 CLEARANCE PLAN PASSED — derived from the live "
                "stationary Ready state and measured finger joints; selected "
                "the first valid candidate from the 0.0800rad outward search start; "
                f"outward shoulder-roll offset="
                f"{clearance_plan.shoulder_roll_offset_rad:.4f}rad; "
                f"left target={left_target:.4f}rad, right target="
                f"{right_target:.4f}rad; arm-route minimum clearance="
                f"{clearance_plan.minimum_clearance_m:.4f}m; complete finger-sweep "
                f"minimum clearance={clearance_plan.finger_sweep_minimum_clearance_m:.4f}m "
                f"at {pair_text} across "
                f"{clearance_plan.finger_sweep_sample_count} samples; measured "
                f"start-limit recovery={recovery_text}"
            )
            _wait_for_space(
                "No command publisher exists yet. Verify the harness and complete "
                "validated shoulder/finger sweep are clear. Press SPACE to acquire the "
                "hands and arms at their exact measured states, hold the fingers "
                "through the validated right-then-left shoulder route, close and "
                "test both hands, restore and hold the measured finger posture "
                "through the reverse arm route, and release arm_sdk: "
            )
            gravity_feedforward = _prepare_gravity_feedforward(
                args.hardware_config,
                activation.reference_state.position,
            )
            transport = UnitreeArmSDKTransport(
                _transport_config(args), observer=lowstate
            )
            lowstate = None
            dex3_controller = UnitreeDex3PostureController(
                dex3_config,
                observer=dex3_observer,
            )
            dex3_observer = None
            watchdog = _pc2_damping_watchdog(args, args.hardware_config)
            raw_executor = DualArmClearanceExecutor(
                transport=transport,
                clock=SystemClock(),
                plan=clearance_plan,
                config=executor_config,
                gravity_feedforward=gravity_feedforward,
            )
            synchronized = SynchronizedPoseExecutor(raw_executor)
            driver = ExecutorControlDriver(
                synchronized,
                rate_hz=rate_hz,
                safety_heartbeat=watchdog.pulse,
            )
            watchdog.start()
            held_hands = dex3_controller.acquire_measured_hold(
                safety_heartbeat=watchdog.pulse,
            )
            held_error = max(
                held_hands.maximum_target_error(
                    clearance_plan.initial_left_hand_q_rad,
                    clearance_plan.initial_right_hand_q_rad,
                )[2],
                0.0,
            )
            if held_error > dex3_config.posture_position_tolerance_rad:
                raise RuntimeError(
                    "live Dex3 state changed between read-only planning and "
                    f"measured-state hold acquisition by {held_error:.4f}rad; "
                    f"limit is {dex3_config.posture_position_tolerance_rad:.4f}rad"
                )
            driver.start()
            synchronized.acquire(operator_confirmed=True)
            _wait_for_control_state(
                synchronized,
                driver,
                ExecutorState.READY,
                timeout_s=executor_config.motion_timeout_s + 5.0,
                control_maintenance=dex3_controller.maintain_initial_posture,
            )
            synchronized.start_pose(
                RIGHT_CLEARANCE_POSE_ID,
                operator_confirmed=True,
            )
            _wait_for_control_state(
                synchronized,
                driver,
                ExecutorState.READY,
                timeout_s=executor_config.motion_timeout_s + 5.0,
                control_maintenance=dex3_controller.maintain_initial_posture,
            )
            synchronized.start_pose(
                DUAL_CLEARANCE_POSE_ID,
                operator_confirmed=True,
            )
            _wait_for_control_state(
                synchronized,
                driver,
                ExecutorState.READY,
                timeout_s=executor_config.motion_timeout_s + 5.0,
                control_maintenance=dex3_controller.maintain_initial_posture,
            )
            print(
                "ARM CLEARANCE ACTIVE — both shoulders reached the validated "
                "outward position; rechecking the complete finger "
                "sweep at the measured loaded arm state"
            )
            loaded_state = synchronized.observe()
            loaded_hands = _wait_for_dex3_state(dex3_controller.observer, 5.0)
            loaded_sweep = _run_isolated_control_work(
                label="measured loaded-state Dex3 finger-sweep validation",
                worker=_validate_loaded_dex3_finger_sweep_worker,
                worker_kwargs={
                    "urdf_path": str(_configured_gravity_urdf(args.hardware_config)),
                    "collision_config": live_collision,
                    "body_q": loaded_state.position,
                    "initial_left_hand_q_rad": loaded_hands.left.position,
                    "initial_right_hand_q_rad": loaded_hands.right.position,
                    "target_left_hand_q_rad": dex3_config.left_target_q_rad,
                    "target_right_hand_q_rad": dex3_config.right_target_q_rad,
                    "path_config": base_path_config,
                },
                driver=driver,
                control_maintenance=dex3_controller.maintain_initial_posture,
            )
            if not loaded_sweep.passed:
                raise RuntimeError(
                    "measured loaded-state Dex3 finger sweep failed: "
                    f"{loaded_sweep.failure}"
                )
            loaded_pair = loaded_sweep.minimum_pair
            loaded_pair_text = (
                "unknown" if loaded_pair is None else "/".join(loaded_pair)
            )
            loaded_recovery = loaded_sweep.recovered_start_limit_joints
            loaded_recovery_text = (
                "none" if not loaded_recovery else ",".join(loaded_recovery)
            )
            print(
                "MEASURED LOADED-STATE FINGER SWEEP PASSED — minimum clearance="
                f"{loaded_sweep.minimum_clearance_m:.4f}m at "
                f"{loaded_pair_text} across {loaded_sweep.sample_count} samples; "
                f"measured start-limit recovery={loaded_recovery_text}; "
                "beginning the measured-state-limited Dex3 command"
            )
            settled = dex3_controller.acquire_posture(
                safety_heartbeat=driver.check,
            )
            deadline = time.monotonic() + args.duration_s
            while time.monotonic() < deadline:
                dex3_controller.maintain_posture()
                driver.check()
                time.sleep(1.0 / dex3_config.command_rate_hz)
            restored = dex3_controller.restore_initial_posture(
                safety_heartbeat=driver.check,
            )
            synchronized.start_pose(
                RIGHT_CLEARANCE_POSE_ID,
                operator_confirmed=True,
            )
            _wait_for_control_state(
                synchronized,
                driver,
                ExecutorState.READY,
                timeout_s=executor_config.motion_timeout_s + 5.0,
                control_maintenance=dex3_controller.maintain_initial_posture,
            )
            synchronized.start_pose(HANDOFF_POSE_ID, operator_confirmed=True)
            _wait_for_control_state(
                synchronized,
                driver,
                ExecutorState.READY,
                timeout_s=executor_config.motion_timeout_s + 5.0,
                control_maintenance=dex3_controller.maintain_initial_posture,
            )
            synchronized.begin_clean_release(operator_confirmed=True)
            _wait_for_control_state(
                synchronized,
                driver,
                ExecutorState.STOPPED,
                timeout_s=executor_config.release_ramp_s + 5.0,
                control_maintenance=dex3_controller.maintain_initial_posture,
            )
            dex3_controller.timeout()
            dex3_controller.close()
            dex3_controller = None
            driver.close()
            driver.check()
            driver = None
            watchdog.disarm()
            side, motor_index, maximum_error = settled.maximum_target_error(
                dex3_config.left_target_q_rad,
                dex3_config.right_target_q_rad,
            )
            restore_error = max(
                restored.maximum_target_error(
                    clearance_plan.initial_left_hand_q_rad,
                    clearance_plan.initial_right_hand_q_rad,
                )[2],
                0.0,
            )
            print(
                "Dex3 middle-close commissioning passed: both hands reached and "
                "settled at the configured NVIDIA target; worst final target "
                f"error={maximum_error:.6f}rad at {side} motor {motor_index} "
                f"({dex3_motor_joint_name(side, motor_index)}), "
                "maximum final velocity="
                f"{settled.maximum_abs_velocity_rad_s:.6f}rad/s; restored initial "
                f"finger posture with worst error={restore_error:.6f}rad; reversed "
                "the validated shoulder route; terminal hand timeout and arm_sdk "
                "weight zero completed"
            )
            return 0
        except BaseException as error:
            primary_error = error
            local_timeout_error = None
            if dex3_controller is not None:
                try:
                    dex3_controller.timeout()
                except BaseException as error:  # noqa: BLE001 - PC2 is backup.
                    local_timeout_error = error
            if driver is not None:
                _attempt_safety_cleanup(
                    cleanup_errors,
                    "clearance control driver stop",
                    driver.close,
                )
            if watchdog is not None and watchdog.armed:
                try:
                    watchdog.damp(
                        "Dex3 middle-close commissioning failed or was interrupted"
                    )
                    if synchronized is not None:
                        synchronized.confirm_external_damping(
                            "PC2 verified whole-body damping after commissioning fault"
                        )
                except BaseException as cleanup_error:  # noqa: BLE001
                    cleanup_errors.append(("PC2 damping", cleanup_error))
            if local_timeout_error is not None and (
                watchdog is None or watchdog.terminal_action is None
            ):
                cleanup_errors.append(("local Dex3 timeout", local_timeout_error))
            _raise_safety_cleanup_failures(primary_error, cleanup_errors)
            raise
        finally:
            if driver is not None:
                _attempt_safety_cleanup(
                    cleanup_errors,
                    "clearance control driver close",
                    driver.close,
                )
            if dex3_controller is not None:
                if (
                    not dex3_controller.timed_out
                    and watchdog is not None
                    and (watchdog.terminal_action is not None)
                ):
                    dex3_controller.close_after_external_timeout()
                elif dex3_controller.timed_out or dex3_controller.command_count == 0:
                    dex3_controller.close()
            if dex3_observer is not None:
                dex3_observer.close()
            if transport is not None and transport.command_count == 0:
                transport.close()
            if lowstate is not None:
                lowstate.close()


def run_teach_poses(args: argparse.Namespace) -> int:
    _require_ack(args.confirm, MOTION_ACK)
    if args.guide_camera_timeout_s <= 0:
        raise ValueError("--guide-camera-timeout-s must be positive")
    hardware_bytes = args.hardware_config.read_bytes()
    target_bytes = args.target_config.read_bytes()
    collision_bytes = args.collision_config.read_bytes()
    quality_bytes = args.quality_config.read_bytes()
    _validate_hardware_target_preflight(hardware_bytes, target_bytes)
    pose_set = _manual_pose_set_for_session(args)
    _validate_calibration_arm(args.hardware_config, pose_set)
    collision = _validate_collision_preflight(args.collision_config)
    _validate_hardware_preflight(hardware_bytes, pose_set, require_poses=False)
    recording, pairing, _, rate_hz = _runtime_configs(args.hardware_config)
    teaching_config = _teaching_config(args.hardware_config)
    thresholds = QualityThresholds.from_yaml(args.quality_config)
    states = StateSampleBuffer()

    try:
        import rclpy
    except ImportError as error:
        raise RuntimeError(
            "rclpy is unavailable; source the ROS Jazzy environment"
        ) from error
    rclpy.init(args=None)
    node = rclpy.create_node("g1_aprilcube_pose_teacher")
    camera = None
    observer = None
    transport = None
    teaching = None
    driver = None
    watchdog = None
    activation_reference_state = None
    pending_supported_frames = ()
    pending_pose_id = None
    pending_capture_id = None
    held_pose_saved = False
    command_lock = CommandOwnerLock(args.lock_file)
    command_lock.acquire()
    try:
        camera = ROSCameraSubscriber(
            node,
            image_topic=args.image_topic,
            camera_info_topic=args.camera_info_topic,
            camera_name=args.camera_name,
            serial_number=args.camera_serial,
            reliability=args.ros_camera_reliability,
        )
        observer = UnitreeLowStateObserver(
            _transport_config(args), on_sample=states.add
        )
        _wait_for_state(observer, 5.0)
        _wait_for_camera(rclpy, node, camera, args.camera_timeout_s)
        camera_info = camera.frames.latest.camera_info
        pose_store = TransientPoseStore(pose_set)
        detector = CorrespondenceDetector(args.target_config)
        evaluator = PoseQualityEvaluator(thresholds)
        recorder = PoseRecorder(
            store=pose_store,
            state_buffer=states,
            gate_config=recording,
            pairing_config=pairing,
        )
        anchor_next = not pose_set.poses
        latest_evaluation = None
        last_frame_key = None
        last_fresh_camera_s = time.monotonic()
        capture_camera_key = None
        capture_fresh_camera_s = time.monotonic()

        def wait_during_capture(duration: float) -> None:
            nonlocal capture_camera_key, capture_fresh_camera_s
            _spin_driver_and_wait(rclpy, node, duration, driver)
            current = camera.frames.latest
            if current is not None:
                current_key = (
                    current.timing.receipt_monotonic_s,
                    current.timing.header_stamp_ns,
                )
                if current_key != capture_camera_key:
                    capture_camera_key = current_key
                    capture_fresh_camera_s = time.monotonic()
            if (
                teaching is not None
                and teaching.state is TeachingState.GUIDE
                and time.monotonic() - capture_fresh_camera_s
                > args.guide_camera_timeout_s
            ):
                try:
                    teaching.protective_hold(
                        "camera frames became stale during supported capture"
                    )
                except RuntimeError:
                    driver.check()
                    if teaching.state is not TeachingState.HOLDING:
                        raise
                raise RuntimeError("camera frames became stale; the arm was secured")

        preview_directory = args.preview_directory or (
            args.session_directory / "preview" / "poses"
        )
        history = _pose_history(pose_set)
        capture_source = LiveBurstFrameSource(
            camera_frames=camera.frames,
            robot_states=states,
            detector=detector,
            quality_evaluator=PoseQualityEvaluator(thresholds),
            recording_config=recording,
            pairing_config=pairing,
            config=LiveBurstConfig(
                frame_count=thresholds.stationary_burst_frames,
                timeout_s=args.burst_timeout_s,
                poll_interval_s=0.01,
                maximum_duration_s=(thresholds.stationary_burst_maximum_duration_s),
            ),
            wait_once=wait_during_capture,
            accept_yellow=lambda _frame: True,
            history=tuple(history),
        )

        activation = _wait_for_activation_handoff(
            observer,
            states,
            pose_store.load(),
            recording,
        )
        activation_report = _runtime_validation_report(
            pose_set=activation.pose_set,
            reference_full_q=activation.reference_state.position,
            directed_edges=((HANDOFF_POSE_ID, HANDOFF_POSE_ID),),
            hardware_config=args.hardware_config,
            collision_config=collision,
        )
        _print_dynamic_preflight(activation, activation_report)
        activation_reference_state = activation.reference_state
        watchdog = _pc2_damping_watchdog(args, args.hardware_config)
        watchdog.start()
        session_store = _open_or_resume_manual_session(
            args,
            camera_info=camera_info,
            pose_set=pose_set,
            recording=recording,
            pairing=pairing,
            artifacts={
                "pose_set.yaml": _pose_set_bytes(pose_set),
                "hardware.yaml": hardware_bytes,
                "target.json": target_bytes,
                "collision_pairs.yaml": collision_bytes,
                "capture_quality.yaml": quality_bytes,
            },
        )
        transport = UnitreeArmSDKTransport(_transport_config(args), observer=observer)
        observer = None
        lower_q, upper_q = _calibration_arm_limits(
            args.hardware_config,
            pose_set.calibration_arm,
        )
        raw_teaching = TeachingArmController(
            transport=transport,
            clock=SystemClock(),
            calibration_arm=pose_set.calibration_arm,
            activation_q14=(
                *activation.reference_state.left_q,
                *activation.reference_state.right_q,
            ),
            calibration_lower_q=lower_q,
            calibration_upper_q=upper_q,
            config=teaching_config,
        )
        teaching = SynchronizedTeachingController(raw_teaching)
        driver = TeachingControlDriver(
            teaching,
            rate_hz=rate_hz,
            safety_heartbeat=watchdog.pulse,
        )
        driver.start()
        teaching.acquire(operator_confirmed=True)
        _wait_for_driven_state(
            teaching,
            driver,
            TeachingState.GUIDE,
            rclpy=rclpy,
            node=node,
            timeout_s=teaching_config.acquisition_ramp_s + 5.0,
        )

        def finish_control_in_damp(reason: str) -> None:
            nonlocal driver, watchdog, transport
            assert driver is not None
            assert watchdog is not None
            assert transport is not None
            driver.close()
            _terminate_motion_safely(
                watchdog,
                teaching,
                transport,
                reason=reason,
            )
            driver.check()
            driver = None
            watchdog = None
            transport = None
            print("verified whole-body Damp; arm_sdk control is closed")

        print("SUPPORT ARM — move to a pose — SPACE records it — Q stops safely")
        while True:
            if driver is not None:
                driver.check()
            rclpy.spin_once(node, timeout_sec=0.01)
            frame = camera.frames.latest
            if frame is not None:
                frame_key = (
                    frame.timing.receipt_monotonic_s,
                    frame.timing.header_stamp_ns,
                )
                if frame_key != last_frame_key:
                    last_fresh_camera_s = time.monotonic()
                    correspondences = detector.detect(frame.image_bgr)
                    intrinsics = CameraIntrinsics(
                        _camera_matrix(frame.camera_info), frame.camera_info.d
                    )
                    quality = evaluator.evaluate(
                        correspondences,
                        intrinsics=intrinsics,
                        history=history,
                    )
                    state = teaching.state
                    if state is TeachingState.GUIDE:
                        control_status = "SUPPORT ARM — MOVE TO POSE — SPACE TO RECORD"
                        footer_lines = (
                            "SPACE = RECORD THIS POSE",
                            "Q = STOP SAFELY (support arm first)",
                        )
                        if teaching.near_joint_limit:
                            control_status += " | NEAR URDF LIMIT"
                    elif state is TeachingState.ENTERING_HOLD:
                        control_status = "KEEP SUPPORTING — SECURING POSE"
                        footer_lines = (
                            "WAIT — keep supporting the arm",
                            "Q = STOP SAFELY (support arm first)",
                        )
                    elif state is TeachingState.HOLDING:
                        if held_pose_saved:
                            control_status = "POSE SAVED — SUPPORT ARM — SPACE FOR NEXT"
                            footer_lines = (
                                "SPACE = READY FOR THE NEXT POSE",
                                "Q = STOP SAFELY (support arm first)",
                            )
                        elif (
                            pending_supported_frames
                            and teaching.hold_command_count >= 10
                        ):
                            control_status = (
                                "HOLD ACTIVE — REMOVE HAND — SPACE SAVE | "
                                "SUPPORT + R REJECT"
                            )
                            footer_lines = (
                                "SPACE = SAVE THIS POSE",
                                "R = REJECT (support arm first)",
                                "Q = STOP SAFELY (support arm first)",
                            )
                        elif pending_supported_frames:
                            control_status = "KEEP SUPPORTING — VERIFYING HOLD"
                            footer_lines = (
                                "WAIT — keep supporting the arm",
                                "Q = STOP SAFELY (support arm first)",
                            )
                        else:
                            control_status = (
                                "CAPTURE CANCELLED — SUPPORT ARM — SPACE TO CONTINUE"
                            )
                            footer_lines = (
                                "SPACE = CONTINUE",
                                "Q = STOP SAFELY (support arm first)",
                            )
                    elif state is TeachingState.CAPTURING:
                        control_status = "HANDS OFF — SAVING POSE"
                        footer_lines = ("WAIT — pose capture is in progress",)
                    elif state is TeachingState.ENTERING_GUIDE:
                        control_status = "KEEP SUPPORTING — PREPARING NEXT POSE"
                        footer_lines = ("WAIT — keep supporting the arm",)
                    elif state is TeachingState.STOPPED:
                        control_status = "DAMP CONFIRMED"
                        footer_lines = ("Control session has ended",)
                    else:
                        control_status = "PLEASE WAIT"
                        footer_lines = ("Wait for the current transition",)
                    rendered = render_operator_preview(
                        frame.image_bgr,
                        correspondences,
                        quality,
                        intrinsics=intrinsics,
                        saved_view_count=len(history),
                        footer_lines=footer_lines,
                    )
                    cv2.putText(
                        rendered,
                        f"next={_next_pose_id(pose_store, args.first_pose_id)}",
                        (16, rendered.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    if (
                        state is TeachingState.HOLDING
                        and pending_supported_frames
                        and not held_pose_saved
                        and teaching.hold_command_count >= 10
                    ):
                        banner_color = (0, 145, 0)
                    elif state in {
                        TeachingState.ENTERING_HOLD,
                        TeachingState.CAPTURING,
                        TeachingState.ENTERING_GUIDE,
                    }:
                        banner_color = (0, 130, 210)
                    else:
                        banner_color = (35, 35, 35)
                    cv2.rectangle(
                        rendered,
                        (0, 0),
                        (frame.image_bgr.shape[1], 44),
                        banner_color,
                        -1,
                    )
                    cv2.putText(
                        rendered,
                        control_status,
                        (16, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    latest_evaluation = (
                        frame,
                        quality,
                        rendered,
                    )
                    last_frame_key = frame_key
                cv2.imshow("G1 pose collection", latest_evaluation[2])
            if (
                teaching.state is TeachingState.GUIDE
                and time.monotonic() - last_fresh_camera_s > args.guide_camera_timeout_s
            ):
                teaching.protective_hold("camera frames became stale during GUIDE")
                pending_supported_frames = ()
                pending_pose_id = None
                pending_capture_id = None
                held_pose_saved = False
                print(
                    "camera became stale: protective hold active; restore the "
                    "camera, support the arm, then press SPACE"
                )
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                print("support confirmed; requesting verified Damp...")
                finish_control_in_damp("operator stopped manual teaching")
                break
            elif key in (ord("r"), ord("R")):
                if (
                    teaching.state is not TeachingState.HOLDING
                    or held_pose_saved
                    or not pending_supported_frames
                ):
                    print("nothing unsaved is available to reject")
                    continue
                if teaching.hold_command_count < 10:
                    print("keep supporting; hold verification is still running")
                    continue
                rejected_pose_id = pending_pose_id
                print("support confirmed; rejecting this unsaved pose...")
                teaching.resume_guide(operator_confirmed=True)
                _wait_for_driven_state(
                    teaching,
                    driver,
                    TeachingState.GUIDE,
                    rclpy=rclpy,
                    node=node,
                    timeout_s=teaching_config.gain_transition_ramp_s + 5.0,
                )
                pending_supported_frames = ()
                pending_pose_id = None
                pending_capture_id = None
                held_pose_saved = False
                last_frame_key = None
                print(
                    f"POSE REJECTED: {rejected_pose_id}; move the supported arm "
                    "and press SPACE to try again"
                )
                continue
            elif key == ord(" "):
                if teaching.state is TeachingState.STOPPED:
                    print("arm_sdk is released; rerun the command to teach more poses")
                    continue
                if teaching.state is TeachingState.HOLDING and (
                    held_pose_saved or not pending_supported_frames
                ):
                    print("support confirmed; preparing the arm for the next pose...")
                    teaching.resume_guide(operator_confirmed=True)
                    _wait_for_driven_state(
                        teaching,
                        driver,
                        TeachingState.GUIDE,
                        rclpy=rclpy,
                        node=node,
                        timeout_s=teaching_config.gain_transition_ramp_s + 5.0,
                    )
                    pending_supported_frames = ()
                    pending_pose_id = None
                    pending_capture_id = None
                    held_pose_saved = False
                    last_frame_key = None
                    print("STEP 1 — move the arm; SPACE records the next pose")
                    continue
                if teaching.state not in {
                    TeachingState.GUIDE,
                    TeachingState.HOLDING,
                }:
                    print("please wait for the current transition to finish")
                    continue
                if latest_evaluation is None:
                    print("waiting for a camera frame")
                    continue
                quality = latest_evaluation[1]
                if quality.grade is QualityGrade.RED:
                    print("pose not saved: visual quality is red")
                    continue
                if quality.grade is QualityGrade.YELLOW:
                    detail = "; ".join(quality.warnings) or "quality warning"
                    print(f"visual warning recorded: {detail}")
                if teaching.state is TeachingState.GUIDE:
                    pending_pose_id = _next_pose_id(pose_store, args.first_pose_id)
                    pending_capture_id = _next_manual_capture_id(session_store)
                    print(
                        f"keep supporting and completely still: collecting "
                        f"{thresholds.stationary_burst_frames} supported frames..."
                    )
                    try:
                        pending_supported_frames = capture_source.capture_burst(
                            pose_id=pending_pose_id,
                            capture_id=f"{pending_capture_id}_supported",
                            remember_signature=False,
                        )
                    except RuntimeError as error:
                        pending_supported_frames = ()
                        pending_pose_id = None
                        pending_capture_id = None
                        print("pose not acquired: " + str(error))
                        continue
                    try:
                        teaching.begin_hold(operator_confirmed=True)
                    except ValueError as error:
                        pending_supported_frames = ()
                        pending_pose_id = None
                        pending_capture_id = None
                        print("supported burst discarded: " + str(error))
                        continue
                    except RuntimeError:
                        driver.check()
                        pending_supported_frames = ()
                        pending_pose_id = None
                        pending_capture_id = None
                        raise
                    last_frame_key = None
                    print(
                        "KEEP SUPPORTING — wait for the green REMOVE YOUR HAND banner"
                    )
                    continue
                if teaching.state is not TeachingState.HOLDING:
                    print("please wait for the current transition to finish")
                    continue
                if teaching.hold_command_count < 10:
                    print("keep supporting; hold verification is still running")
                    continue
                if held_pose_saved:
                    print("pose already saved; support the arm and press SPACE")
                    continue
                if (
                    not pending_supported_frames
                    or pending_pose_id is None
                    or pending_capture_id is None
                ):
                    print(
                        "capture was cancelled; support the arm and press SPACE "
                        "to continue"
                    )
                    continue
                pose_id = pending_pose_id
                capture_id = pending_capture_id
                print(
                    f"collecting {thresholds.stationary_burst_frames} "
                    f"lossless stationary frames for {pose_id}..."
                )
                teaching.begin_capture()
                try:
                    burst = capture_source.capture_burst(
                        pose_id=pose_id,
                        capture_id=capture_id,
                    )
                except RuntimeError as error:
                    driver.check()
                    teaching.finish_capture(outcome="raw burst rejected")
                    print(f"pose not saved: {error}")
                    continue
                teaching.finish_capture(outcome="raw burst collected")
                selected = SessionStore.select_medoid_frame(burst)
                selected_supported = SessionStore.select_medoid_frame(
                    pending_supported_frames
                )
                paired_metrics = supported_vs_held_metrics(
                    selected_supported,
                    selected,
                    calibration_arm=pose_set.calibration_arm,
                )
                preview_directory.mkdir(parents=True, exist_ok=True)
                preview_path = preview_directory / f"{pose_id}_{capture_id}.png"
                if preview_path.exists():
                    raise FileExistsError(
                        f"pose preview already exists: {preview_path}"
                    )
                request = PoseRecordingRequest(
                    pose_id=pose_id,
                    group=args.group,
                    image_timing=selected.image_timing,
                    visual_report=selected.quality,
                    anchor=anchor_next,
                    preview_path=preview_path,
                    yellow_override_reason=(
                        args.yellow_override_reason or LIVE_TEACHING_YELLOW_REASON
                        if selected.quality.grade is QualityGrade.YELLOW
                        else None
                    ),
                )
                now = selected.state_window[-1].receipt_monotonic_s
                assessment = recorder.assess(request, now_monotonic_s=now)
                if not assessment.allowed:
                    capture_source.undo_last_signature()
                    print("pose not saved: " + "; ".join(assessment.failures))
                    continue
                selected_preview = render_operator_preview(
                    selected.image_bgr,
                    selected.correspondences,
                    selected.quality,
                    intrinsics=CameraIntrinsics(
                        _camera_matrix(selected.camera_info),
                        selected.camera_info.d,
                    ),
                    saved_view_count=len(history),
                )
                if not cv2.imwrite(str(preview_path), selected_preview):
                    raise RuntimeError(f"failed to write pose preview: {preview_path}")
                previous = pose_store.load()
                updated = recorder.record(request, now_monotonic_s=now)
                try:
                    session_store.update_manual_pose_set(updated)
                    manifest = session_store.append_capture(
                        capture_id=capture_id,
                        pose_id=pose_id,
                        outcome="accepted",
                        reason="manual stationary burst passed",
                        frames=burst,
                        supported_frames=pending_supported_frames,
                        metadata={
                            "capture_phase": "continuous_guide_to_weight_1_hold",
                            "operator_hands_off_confirmed": True,
                            "activation_reference_state": (
                                activation_reference_state.to_dict()
                            ),
                            "hold_target_calibration_q": list(
                                teaching.held_calibration_q or ()
                            ),
                            "supported_vs_held": paired_metrics,
                        },
                    )
                except Exception:
                    rolled_back = pose_store.undo_last(
                        reason="manual session capture transaction failed"
                    )
                    try:
                        session_store.update_manual_pose_set(rolled_back)
                    except Exception as rollback_error:
                        pose_store.initialize(previous, overwrite=True)
                        raise RuntimeError(
                            "manual session rollback failed; session requires repair"
                        ) from rollback_error
                    raise
                saved_capture = manifest.captures[-1]
                if saved_capture.selected_frame_id != selected.frame_id:
                    raise RuntimeError("manual capture medoid selection changed")
                if (
                    saved_capture.selected_supported_frame_id
                    != selected_supported.frame_id
                ):
                    raise RuntimeError(
                        "manual supported-capture medoid selection changed"
                    )
                if selected.quality.signature is not None:
                    history.append(selected.quality.signature)
                anchor_next = False
                held_pose_saved = True
                print(
                    f"POSE SAVED: {pose_id} (total {len(updated.poses)}). "
                    "Support the arm and press SPACE for the next pose."
                )
        session_store.validate_manual_alignment(pose_store.load())
        manifest = session_store.load()
        print(
            f"stopped resumable manual session with {len(manifest.captures)} "
            f"saved poses: {args.session_directory}"
        )
        return 0
    finally:
        if driver is not None and driver.is_alive:
            try:
                driver.close()
            except RuntimeError:
                # Stopping the heartbeat is deliberate; PC2 remains able to damp.
                pass
        if watchdog is not None and watchdog.armed:
            if transport is None or transport.command_count == 0:
                watchdog.disarm()
            elif teaching is None:
                watchdog.damp("manual teaching cleanup before executor creation")
            else:
                _terminate_motion_safely(
                    watchdog,
                    teaching,
                    transport,
                    reason="manual teaching failed or cleanup was required",
                )
        if transport is not None and transport.command_count == 0:
            transport.close()
        if observer is not None:
            observer.close()
        if camera is not None:
            camera.close()
        node.destroy_node()
        _shutdown_rclpy_once(rclpy)
        cv2.destroyAllWindows()
        command_lock.release()


def run_commission_hold(args: argparse.Namespace) -> int:
    _require_ack(args.confirm, MOTION_ACK)
    if args.duration_s <= 0:
        raise ValueError("--duration-s must be positive")
    source_pose_set = PoseStore(args.pose_set).load()
    _validate_calibration_arm(args.hardware_config, source_pose_set)
    collision = _validate_collision_preflight(args.collision_config)
    recording, _, config, rate_hz = _runtime_configs(args.hardware_config)
    clock = SystemClock()
    with CommandOwnerLock(args.lock_file):
        states = StateSampleBuffer()
        observer = UnitreeLowStateObserver(
            _transport_config(args), clock=clock, on_sample=states.add
        )
        transport = None
        watchdog = None
        executor = None
        try:
            activation = _wait_for_activation_handoff(
                observer, states, source_pose_set, recording
            )
            report = _runtime_validation_report(
                pose_set=activation.pose_set,
                reference_full_q=activation.reference_state.position,
                directed_edges=((HANDOFF_POSE_ID, HANDOFF_POSE_ID),),
                hardware_config=args.hardware_config,
                collision_config=collision,
            )
            _print_dynamic_preflight(activation, report)
            gravity_feedforward = _prepare_gravity_feedforward(
                args.hardware_config,
                activation.reference_state.position,
            )
            transport = UnitreeArmSDKTransport(
                _transport_config(args), observer=observer
            )
            observer = None
            watchdog = _pc2_damping_watchdog(args, args.hardware_config)
            executor = PoseExecutor(
                transport=transport,
                clock=clock,
                pose_set=activation.pose_set,
                handoff_q=activation.handoff_q,
                hold_q=activation.hold_q,
                approved_validation_report_sha256=report.content_sha256,
                config=config,
                gravity_feedforward=gravity_feedforward,
            )
            _wait_for_state(transport, 5.0)
            watchdog.start()
            executor.acquire(operator_confirmed=True)
            _drive_direct(
                executor,
                ExecutorState.READY,
                rate_hz=rate_hz,
                safety_heartbeat=watchdog.pulse,
            )
            deadline = time.monotonic() + args.duration_s
            while time.monotonic() < deadline:
                time.sleep(1.0 / rate_hz)
                state = executor.tick()
                if state is ExecutorState.FAULT:
                    raise RuntimeError(
                        "executor faulted: "
                        f"{executor.fault_reason or 'unknown reason'}; "
                        "PC2 heartbeat intentionally stopped"
                    )
                watchdog.pulse()
            executor.begin_clean_release(operator_confirmed=True)
            _drive_direct(
                executor,
                ExecutorState.STOPPED,
                rate_hz=rate_hz,
                safety_heartbeat=watchdog.pulse,
            )
            watchdog.disarm()
        except BaseException:
            if watchdog is not None and transport is not None and executor is not None:
                _terminate_motion_safely(
                    watchdog,
                    executor,
                    transport,
                    reason="commission-hold failed or was interrupted",
                )
            elif transport is not None and transport.command_count == 0:
                transport.close()
            raise
        finally:
            if observer is not None:
                observer.close()
    print("handoff acquisition/hold completed with terminal weight zero")
    return 0


def run_commission_seated_debug_hold(args: argparse.Namespace) -> int:
    """Commission zero-displacement debug lowcmd and verified recovery."""

    _require_ack(args.confirm, MOTION_ACK)
    if args.duration_s <= 0:
        raise ValueError("--duration-s must be positive")
    collision = _validate_collision_preflight(args.collision_config)
    recording, _, executor_config, rate_hz = _runtime_configs(args.hardware_config)
    empty_pose_set = _hardware_pose_set(args.hardware_config)
    clock = SystemClock()
    with CommandOwnerLock(args.lock_file):
        states = StateSampleBuffer()
        observer = UnitreeLowStateObserver(
            _transport_config(args), clock=clock, on_sample=states.add
        )
        transport = None
        watchdog = None
        executor = None
        try:
            activation = _wait_for_activation_handoff(
                observer,
                states,
                empty_pose_set,
                recording,
            )
            report = _runtime_validation_report(
                pose_set=activation.pose_set,
                reference_full_q=activation.reference_state.position,
                directed_edges=((HANDOFF_POSE_ID, HANDOFF_POSE_ID),),
                hardware_config=args.hardware_config,
                collision_config=collision,
            )
            _print_dynamic_preflight(activation, report)
            gravity_feedforward = _gravity_feedforward(args.hardware_config)
            gravity_feedforward.seed_reference(activation.reference_state.position)
            takeover_q14 = np.concatenate(
                (activation.reference_state.left_q, activation.reference_state.right_q)
            )
            takeover_gravity_tau = gravity_feedforward.torque_for(takeover_q14)
            _print_gravity_preflight(gravity_feedforward, takeover_gravity_tau)
            print(
                "ZERO-MOTION DEBUG COMMISSION — PC2 will verify seated FSM 3; "
                "all 29 measured joints will then be held through rt/lowcmd; "
                "the arm position target will not change while Unitree-style "
                "gravity torque ramps from zero to full"
            )

            watchdog = _pc2_damping_watchdog(
                args,
                args.hardware_config,
                require_regular=False,
                required_initial_fsm_id=_seated_fsm_id(args.hardware_config),
                restore_motion_service_before_loco=True,
            )
            transport = UnitreeDebugLowCmdTransport(
                _transport_config(args),
                _debug_lowcmd_config(args.hardware_config),
                observer=observer,
                ownership_keepalive=watchdog.pulse,
            )
            observer = None
            executor = PoseExecutor(
                transport=transport,
                clock=clock,
                pose_set=activation.pose_set,
                handoff_q=activation.handoff_q,
                hold_q=activation.hold_q,
                approved_validation_report_sha256=report.content_sha256,
                config=executor_config,
                gravity_feedforward=gravity_feedforward,
            )
            print(
                "TAKEOVER — maintaining an independent PC2 heartbeat while "
                "releasing the seated motion service; the first lowcmd target "
                "is the exact measured 29-joint state"
            )
            # Unitree's client setup performs blocking DDS endpoint matching.
            # Arm PC2 only after that setup, immediately before guarded takeover.
            watchdog.start()
            executor.acquire(operator_confirmed=True)
            print(
                "debug ownership and first measured-state lowcmd packet passed; "
                "holding the zero-displacement target"
            )
            _drive_direct(
                executor,
                ExecutorState.READY,
                rate_hz=rate_hz,
                safety_heartbeat=watchdog.pulse,
            )
            gravity_reference_q = gravity_feedforward.reference_full_q
            gravity_reference_q14 = np.concatenate(
                (gravity_reference_q[15:22], gravity_reference_q[22:29])
            )
            maximum_gravity_commission_drift_rad = (
                executor.maximum_acquisition_position_change_rad
            )
            deadline = time.monotonic() + args.duration_s
            while time.monotonic() < deadline:
                time.sleep(1.0 / rate_hz)
                state = executor.tick()
                if state is ExecutorState.FAULT:
                    raise RuntimeError(
                        f"executor faulted: {executor.fault_reason or 'unknown reason'}"
                    )
                sample = executor.observe_state()
                measured_q14 = np.concatenate((sample.left_q, sample.right_q))
                drift = float(np.max(np.abs(measured_q14 - gravity_reference_q14)))
                maximum_gravity_commission_drift_rad = max(
                    maximum_gravity_commission_drift_rad,
                    drift,
                )
                if drift > executor_config.ownership_transition_position_tolerance_rad:
                    raise RuntimeError(
                        "zero-displacement gravity commissioning arm drift is "
                        f"{drift:.4f}rad; limit is "
                        f"{executor_config.ownership_transition_position_tolerance_rad:.4f}rad"
                    )
                watchdog.pulse()
            if args.trigger == "restore-seated":
                watchdog.restore_seated()
                executor.confirm_external_takeover(
                    "PC2 verified AI takeover through FSM 0 -> 1 -> seated FSM 3"
                )
            else:
                # Deliberately publish neither lowcmd nor heartbeat.  This is
                # the laptop-loss case; PC2 must independently restore the AI
                # service and verify its zero-torque FSM 0 initialization.
                watchdog.wait_for_automatic_zero_torque()
                executor.confirm_external_takeover(
                    "PC2 independently restored the motion service and verified "
                    "zero-torque FSM 0 after heartbeat loss"
                )
        except BaseException as primary_error:
            try:
                if watchdog is not None and transport is not None:
                    _terminate_debug_lowcmd_safely(
                        watchdog,
                        executor,
                        transport,
                        reason=(
                            "seated debug lowcmd commissioning failed or was "
                            "interrupted"
                        ),
                    )
                elif watchdog is not None and watchdog.armed:
                    watchdog.disarm()
                elif transport is not None and not transport.requires_external_takeover:
                    transport.close()
            except BaseException as cleanup_error:  # noqa: BLE001
                _raise_safety_cleanup_failures(
                    primary_error,
                    [("seated debug ownership cleanup", cleanup_error)],
                )
            raise
        finally:
            if observer is not None:
                observer.close()

    print(
        "seated debug lowcmd zero-motion hold passed; PC2 verified "
        + (
            "AI FSM 0 -> 1 -> seated FSM 3"
            if args.trigger == "restore-seated"
            else "AI service in zero-torque FSM 0"
        )
        + " before the lowcmd publisher closed; maximum gravity-ramp arm drift="
        + f"{maximum_gravity_commission_drift_rad:.6f}rad"
    )
    return 0


def run_commission_pose(args: argparse.Namespace) -> int:
    _require_ack(args.confirm, MOTION_ACK)
    if args.target_pose == HANDOFF_POSE_ID:
        raise ValueError("target pose must be a taught calibration pose")
    source_pose_set = PoseStore(args.pose_set).load()
    _validate_calibration_arm(args.hardware_config, source_pose_set)
    policy_report = _load_validation_report(args.validation_report)
    _validate_report_binding(source_pose_set, policy_report)
    collision = _validate_collision_preflight(
        args.collision_config, report=policy_report
    )
    recording, _, config, rate_hz = _runtime_configs(args.hardware_config)
    clock = SystemClock()
    with CommandOwnerLock(args.lock_file):
        states = StateSampleBuffer()
        observer = UnitreeLowStateObserver(
            _transport_config(args), clock=clock, on_sample=states.add
        )
        transport = None
        watchdog = None
        executor = None
        try:
            activation = _wait_for_activation_handoff(
                observer, states, source_pose_set, recording
            )
            report = _runtime_validation_report(
                pose_set=activation.pose_set,
                reference_full_q=activation.reference_state.position,
                directed_edges=(
                    (HANDOFF_POSE_ID, args.target_pose),
                    (args.target_pose, HANDOFF_POSE_ID),
                ),
                hardware_config=args.hardware_config,
                collision_config=collision,
                policy_config=policy_report.config,
            )
            _print_dynamic_preflight(activation, report)
            outward = report.approval(HANDOFF_POSE_ID, args.target_pose)
            returning = report.approval(args.target_pose, HANDOFF_POSE_ID)
            gravity_feedforward = _prepare_gravity_feedforward(
                args.hardware_config,
                activation.reference_state.position,
            )
            transport = UnitreeArmSDKTransport(
                _transport_config(args), observer=observer
            )
            observer = None
            watchdog = _pc2_damping_watchdog(args, args.hardware_config)
            executor = PoseExecutor(
                transport=transport,
                clock=clock,
                pose_set=activation.pose_set,
                handoff_q=activation.handoff_q,
                hold_q=activation.hold_q,
                approved_validation_report_sha256=report.content_sha256,
                config=config,
                gravity_feedforward=gravity_feedforward,
            )
            _wait_for_state(transport, 5.0)
            watchdog.start()
            executor.acquire(operator_confirmed=True)
            _drive_direct(
                executor,
                ExecutorState.READY,
                rate_hz=rate_hz,
                safety_heartbeat=watchdog.pulse,
            )
            executor.start_pose(
                args.target_pose, approval=outward, operator_confirmed=True
            )
            _drive_direct(
                executor,
                ExecutorState.READY,
                rate_hz=rate_hz,
                safety_heartbeat=watchdog.pulse,
            )
            executor.start_pose(
                HANDOFF_POSE_ID, approval=returning, operator_confirmed=True
            )
            _drive_direct(
                executor,
                ExecutorState.READY,
                rate_hz=rate_hz,
                safety_heartbeat=watchdog.pulse,
            )
            executor.begin_clean_release(operator_confirmed=True)
            _drive_direct(
                executor,
                ExecutorState.STOPPED,
                rate_hz=rate_hz,
                safety_heartbeat=watchdog.pulse,
            )
            watchdog.disarm()
        except BaseException:
            if watchdog is not None and transport is not None and executor is not None:
                _terminate_motion_safely(
                    watchdog,
                    executor,
                    transport,
                    reason="commission-pose failed or was interrupted",
                )
            elif transport is not None and transport.command_count == 0:
                transport.close()
            raise
        finally:
            if observer is not None:
                observer.close()
    print("single-pose round trip passed with terminal weight zero")
    return 0


def run_collect_session(args: argparse.Namespace) -> int:
    _require_ack(args.confirm, MOTION_ACK)
    source_pose_set = PoseStore(args.pose_set).load()
    policy_report = _load_validation_report(args.validation_report)
    _validate_report_binding(source_pose_set, policy_report)
    collision = _validate_collision_preflight(
        args.collision_config, report=policy_report
    )
    plan = _load_session_plan(args.plan_yaml)
    _preflight_plan(plan, policy_report)
    hardware_bytes = args.hardware_config.read_bytes()
    target_bytes = args.target_config.read_bytes()
    collision_bytes = args.collision_config.read_bytes()
    quality_bytes = args.quality_config.read_bytes()
    _validate_hardware_target_preflight(hardware_bytes, target_bytes)
    recording, pairing, executor_config, rate_hz = _runtime_configs(
        args.hardware_config
    )
    _validate_hardware_preflight(hardware_bytes, source_pose_set)
    thresholds = QualityThresholds.from_yaml(args.quality_config)

    try:
        import rclpy
    except ImportError as error:
        raise RuntimeError(
            "rclpy is unavailable; source the ROS Jazzy environment"
        ) from error

    rclpy.init(args=None)
    node = rclpy.create_node("g1_aprilcube_calibration_collector")
    camera = None
    transport = None
    driver = None
    synchronized = None
    watchdog = None
    observer = None
    try:
        camera = ROSCameraSubscriber(
            node,
            image_topic=args.image_topic,
            camera_info_topic=args.camera_info_topic,
            camera_name=args.camera_name,
            serial_number=args.camera_serial,
            reliability=args.ros_camera_reliability,
        )
        _wait_for_camera(rclpy, node, camera, args.camera_timeout_s)
        camera_info = camera.frames.latest.camera_info
        states = StateSampleBuffer()
        with CommandOwnerLock(args.lock_file):
            observer = UnitreeLowStateObserver(
                _transport_config(args), on_sample=states.add
            )
            activation = _wait_for_activation_handoff(
                observer, states, source_pose_set, recording
            )
            route = [
                HANDOFF_POSE_ID,
                *plan.capture_pose_ids,
                HANDOFF_POSE_ID,
            ]
            report = _runtime_validation_report(
                pose_set=activation.pose_set,
                reference_full_q=activation.reference_state.position,
                directed_edges=_unique_route_edges(route),
                hardware_config=args.hardware_config,
                collision_config=collision,
                policy_config=policy_report.config,
            )
            _print_dynamic_preflight(activation, report)
            gravity_feedforward = _prepare_gravity_feedforward(
                args.hardware_config,
                activation.reference_state.position,
            )
            pose_bytes = _pose_set_bytes(activation.pose_set)
            validation_bytes = _validation_report_bytes(report)
            transport = UnitreeArmSDKTransport(
                _transport_config(args), observer=observer
            )
            observer = None
            _wait_for_state(transport, 5.0)
            watchdog = _pc2_damping_watchdog(args, args.hardware_config)
            watchdog.start()
            store = SessionStore(args.session_directory)
            store.create(
                session_id=args.session_id,
                created_at_utc=utc_now_iso(),
                camera_info=camera_info,
                pose_set_content_sha256=activation.pose_set.content_sha256,
                artifacts={
                    "pose_set.yaml": pose_bytes,
                    "hardware.yaml": hardware_bytes,
                    "target.json": target_bytes,
                    "collision_pairs.yaml": collision_bytes,
                    "capture_quality.yaml": quality_bytes,
                    "validation_report.json": validation_bytes,
                },
                pairing_config=pairing,
                recording_gate_config=recording,
                provenance={
                    "command": "g1-calib collect-session",
                    "mode_machine": 5,
                    "head_witness_ack": args.head_witness_ack,
                    "yellow_override_reason": REPLAY_YELLOW_REASON,
                    "camera_image_topic": args.image_topic,
                    "camera_info_topic": args.camera_info_topic,
                    "ros_camera_reliability": args.ros_camera_reliability,
                    "network_interface": args.network_interface,
                    "source_pose_set_sha256": (activation.source_pose_set_sha256),
                    "policy_validation_report_sha256": (policy_report.content_sha256),
                    "motion_completion_policy": (
                        "complete_command_then_measured_stationarity"
                    ),
                    "dynamic_handoff_state": (activation.reference_state.to_dict()),
                    "dynamic_handoff_readiness": (activation.readiness.to_dict()),
                    "gravity_feedforward": _gravity_provenance(gravity_feedforward),
                },
                collection_method="replay",
            )
            raw_executor = PoseExecutor(
                transport=transport,
                clock=SystemClock(),
                pose_set=activation.pose_set,
                handoff_q=activation.handoff_q,
                hold_q=activation.hold_q,
                approved_validation_report_sha256=report.content_sha256,
                config=executor_config,
                gravity_feedforward=gravity_feedforward,
            )
            synchronized = SynchronizedPoseExecutor(raw_executor)
            driver = ExecutorControlDriver(
                synchronized,
                rate_hz=rate_hz,
                safety_heartbeat=watchdog.pulse,
            )
            driver.start()

            def preview(frame, correspondences, quality):
                if args.no_window:
                    return
                intrinsics = CameraIntrinsics(
                    _camera_matrix(camera_info),
                    camera_info.d,
                )
                rendered = render_operator_preview(
                    frame.image_bgr,
                    correspondences,
                    quality,
                    intrinsics=intrinsics,
                    saved_view_count=0,
                    footer_lines=(
                        "CAPTURE IS AUTOMATIC — NO KEY REQUIRED",
                        "RED/TIMEOUT = REJECT POSE AND CONTINUE",
                        "CTRL+C IN TERMINAL = ABORT AND DAMP",
                    ),
                )
                cv2.imshow("G1 calibration collection", rendered)
                cv2.waitKey(1)

            def wait_once(duration_s: float) -> None:
                rclpy.spin_once(node, timeout_sec=0.0)
                driver.check()
                time.sleep(duration_s)

            source = LiveBurstFrameSource(
                camera_frames=camera.frames,
                robot_states=states,
                detector=CorrespondenceDetector(args.target_config),
                quality_evaluator=PoseQualityEvaluator(thresholds),
                recording_config=recording,
                pairing_config=pairing,
                config=LiveBurstConfig(
                    frame_count=thresholds.stationary_burst_frames,
                    timeout_s=args.burst_timeout_s,
                    poll_interval_s=1.0 / rate_hz,
                    maximum_duration_s=(thresholds.stationary_burst_maximum_duration_s),
                ),
                wait_once=wait_once,
                accept_yellow=lambda _frame: True,
                preview=preview,
            )
            runner = CaptureSessionRunner(executor=synchronized, store=store)

            def confirm_move(source_pose: str, target_pose: str) -> bool:
                if args.auto_confirm_transitions:
                    return True
                return _wait_for_space(
                    f"Validated move {source_pose} -> {target_pose}. Press SPACE: "
                )

            def report_capture_rejection(pose_id: str, reason: str) -> None:
                print(f"REJECTED {pose_id}: {reason}; continuing replay")

            orchestrator = ApprovedSessionOrchestrator(
                executor=synchronized,
                validation_report=report,
                capture_runner=runner,
                frame_source=source,
                control_step=lambda: wait_once(1.0 / rate_hz),
                confirm_move=confirm_move,
                plan=plan,
                report_capture_rejection=report_capture_rejection,
            )
            try:
                orchestrator.run(
                    confirm_acquisition=True,
                    confirm_release=True,
                )
            except KeyboardInterrupt:
                if driver.is_alive:
                    driver.close()
                _terminate_motion_safely(
                    watchdog,
                    synchronized,
                    transport,
                    reason="operator interrupted collection",
                )
                return 130
            driver.close()
            driver.check()
            watchdog.disarm()
        print(f"finalized session: {args.session_directory}")
        return 0
    finally:
        if driver is not None and driver.is_alive:
            try:
                driver.close()
            except RuntimeError:
                # Stopping the heartbeat is deliberate; the PC2 guard is still
                # able to damp without a functioning laptop control thread.
                pass
        if watchdog is not None and watchdog.armed:
            if transport is not None and transport.command_count == 0:
                watchdog.disarm()
            elif synchronized is None or transport is None:
                watchdog.damp("collection cleanup before executor creation")
            else:
                _terminate_motion_safely(
                    watchdog,
                    synchronized,
                    transport,
                    reason="collection failed or cleanup was required",
                )
        if transport is not None and transport.command_count == 0:
            transport.close()
        if observer is not None:
            observer.close()
        if camera is not None:
            camera.close()
        node.destroy_node()
        _shutdown_rclpy_once(rclpy)
        cv2.destroyAllWindows()


def run_collect_auto(args: argparse.Namespace) -> int:
    """Run a predesigned standing route without per-transition keypresses."""

    _require_ack(args.confirm, MOTION_ACK)
    if args.session_directory.exists():
        raise FileExistsError(
            f"session directory already exists: {args.session_directory}"
        )
    _require_dex3_posture_commissioned(args.hardware_config)
    with args.plan.open(encoding="utf-8") as stream:
        plan = AuthoredCollectionPlan.from_dict(json.load(stream))
    authored_target_count = len(plan.capture_pose_ids)
    policy_path = args.plan.with_name("validation_report.json")
    policy_report = _load_validation_report(policy_path)
    _validate_report_binding(plan, policy_report)
    collision = _validate_collision_preflight(
        args.collision_config, report=policy_report
    )
    for source, target in pairwise(plan.route_pose_ids):
        if not policy_report.edge(source, target).passed:
            raise ValueError(f"authored route edge failed: {source}->{target}")
    hardware_bytes = args.hardware_config.read_bytes()
    target_bytes = args.target_config.read_bytes()
    collision_bytes = args.collision_config.read_bytes()
    quality_bytes = args.quality_config.read_bytes()
    _validate_hardware_target_preflight(hardware_bytes, target_bytes)
    plan_bytes = _authored_plan_bytes(plan)
    recording, pairing, executor_config, rate_hz = _runtime_configs(
        args.hardware_config
    )
    _validate_hardware_preflight(hardware_bytes, plan)
    thresholds = QualityThresholds.from_yaml(args.quality_config)

    try:
        import rclpy
    except ImportError as error:
        raise RuntimeError(
            "rclpy is unavailable; source the ROS Jazzy environment"
        ) from error

    rclpy.init(args=None)
    node = rclpy.create_node("g1_aprilcube_authored_collector")
    camera = None
    transport = None
    driver = None
    synchronized = None
    clearance_driver = None
    clearance_synchronized = None
    active_synchronized = None
    watchdog = None
    observer = None
    dex3_observer = None
    dex3_controller = None
    dex3_posture_evidence = None
    result = None
    runtime_log = None
    isolated_store = None
    record_executor_events = None
    logged_executor_event_count = 0
    progress = {"accepted": 0, "rejected": 0, "message": "waiting"}
    try:
        camera = ROSCameraSubscriber(
            node,
            image_topic=args.image_topic,
            camera_info_topic=args.camera_info_topic,
            camera_name=args.camera_name,
            serial_number=args.camera_serial,
            reliability=args.ros_camera_reliability,
        )
        _wait_for_camera(rclpy, node, camera, args.camera_timeout_s)
        camera_info = camera.frames.latest.camera_info
        if camera_info.profile_sha256 != plan.camera_profile_sha256:
            raise ValueError("live rectified CameraInfo differs from the authored plan")
        states = StateSampleBuffer()
        with CommandOwnerLock(args.lock_file):
            observer = UnitreeLowStateObserver(
                _transport_config(args), on_sample=states.add
            )
            dex3_observer = UnitreeDex3StateObserver(
                _dex3_control_config(args),
                initialize_factory=False,
            )
            dex3_config = dex3_observer.config
            initial_hands = _wait_for_dex3_state(dex3_observer, 5.0)
            activation = _wait_for_activation_handoff(observer, states, plan, recording)
            # Use the exact same 5 mm clearance policy that passed the physical
            # Dex3 close/restore commissioning.  The authored calibration route
            # retains its separate, stricter frozen 10 mm policy below.
            clearance_path_config = PathValidationConfig()
            body_model = URDFModel(_configured_urdf(args.hardware_config))
            dex3_model = URDFModel(_configured_gravity_urdf(args.hardware_config))
            live_collision = live_dex3_collision_config(collision)
            print(
                "READ-ONLY DEX3 CLEARANCE PLANNING — starting at 0.0800rad "
                "outward from the fresh stationary shoulders and checking the "
                "complete measured-to-target finger sweep; no command publisher "
                "exists",
                flush=True,
            )
            clearance_plan = plan_dual_arm_shoulder_clearance(
                model=body_model,
                collision_checker=FCLCollisionChecker(body_model, collision),
                dex3_model=dex3_model,
                dex3_collision_checker=FCLCollisionChecker(dex3_model, live_collision),
                reference_full_q=activation.reference_state.position,
                initial_left_hand_q_rad=initial_hands.left.position,
                initial_right_hand_q_rad=initial_hands.right.position,
                target_left_hand_q_rad=dex3_config.left_target_q_rad,
                target_right_hand_q_rad=dex3_config.right_target_q_rad,
                path_config=clearance_path_config,
                progress=lambda index, total, offset: print(
                    "READ-ONLY DEX3 CLEARANCE PLANNING: candidate "
                    f"{index}/{total}, outward offset={offset:.4f}rad; NO MOTION",
                    flush=True,
                ),
                rejection=lambda offset, reason: print(
                    "READ-ONLY DEX3 CLEARANCE REJECTED: "
                    f"outward offset={offset:.4f}rad; {reason}; NO MOTION",
                    flush=True,
                ),
            )
            clearance_reference_q = activation.reference_state.position.copy()
            clearance_reference_q[np.asarray(LEFT_ARM_INDICES)] = np.asarray(
                clearance_plan.dual_clearance_q14[:7]
            )
            clearance_reference_q[np.asarray(RIGHT_ARM_INDICES)] = np.asarray(
                clearance_plan.dual_clearance_q14[7:]
            )
            directed_edges = _unique_route_edges(plan.route_pose_ids)
            validation_started_at = time.monotonic()
            last_progress_at = validation_started_at

            def report_validation_progress(
                completed: int,
                total: int,
                source_pose: str,
                target_pose: str,
            ) -> None:
                nonlocal last_progress_at
                now = time.monotonic()
                if completed not in {1, total} and now - last_progress_at < 2.0:
                    return
                elapsed = now - validation_started_at
                remaining = (
                    0.0
                    if completed == total
                    else elapsed * (total - completed) / completed
                )
                print(
                    "READ-ONLY COLLISION PREFLIGHT: "
                    f"{completed}/{total} edges; elapsed={elapsed:.1f}s; "
                    f"estimated remaining={remaining:.1f}s; "
                    f"last={source_pose}->{target_pose}; NO MOTION",
                    flush=True,
                )
                last_progress_at = now

            print(
                "READ-ONLY COLLISION PREFLIGHT: validating "
                f"{len(directed_edges)} live-state trajectory edges. "
                "No command publisher exists and no motion can occur. "
                "Ctrl+C cancels.",
                flush=True,
            )
            try:
                with _DeferredSIGINT() as cancellation:
                    report = _runtime_validation_report(
                        pose_set=plan,
                        reference_full_q=clearance_reference_q,
                        directed_edges=directed_edges,
                        hardware_config=args.hardware_config,
                        collision_config=collision,
                        policy_config=policy_report.config,
                        progress=report_validation_progress,
                        cancellation_requested=lambda: cancellation.requested,
                    )
                    if cancellation.requested:
                        raise KeyboardInterrupt
                _print_dynamic_preflight(activation, report)
                gravity_feedforward = _prepare_gravity_feedforward(
                    args.hardware_config,
                    activation.reference_state.position,
                )
                isolated_store = IsolatedSessionStore(
                    args.session_directory,
                    poll_interval_s=1.0 / rate_hz,
                )
                isolated_store.start()
                print(
                    "ISOLATED SESSION WRITER READY — capture validation, PNG/JSON "
                    "encoding, fsync, and manifest commits will run outside the "
                    "control process",
                    flush=True,
                )
                _wait_for_space(
                    "READ-ONLY PREFLIGHT COMPLETE. No arm command has been "
                    "published. Verify the complete shoulder/finger/calibration "
                    "sweep is clear. Press SPACE to create the arm_sdk and Dex3 "
                    "publishers, move both shoulders through the commissioned "
                    "outward route, acquire NVIDIA's middle-close posture, and begin "
                    f"automatic {plan.calibration_arm.upper()}-ARM motion: "
                )
                if not rclpy.ok():
                    raise KeyboardInterrupt
            except KeyboardInterrupt:
                print(
                    "automatic collection cancelled before command publisher "
                    "creation; no arm command was published",
                    file=sys.stderr,
                )
                return 130
            validation_bytes = _validation_report_bytes(report)
            transport = UnitreeArmSDKTransport(
                _transport_config(args), observer=observer
            )
            observer = None
            _wait_for_state(transport, 5.0)
            dex3_controller = UnitreeDex3PostureController(
                _dex3_control_config(args),
                observer=dex3_observer,
            )
            dex3_observer = None
            watchdog = _pc2_damping_watchdog(args, args.hardware_config)
            watchdog.start()
            clearance_raw_executor = DualArmClearanceExecutor(
                transport=transport,
                clock=SystemClock(),
                plan=clearance_plan,
                config=executor_config,
                gravity_feedforward=gravity_feedforward,
            )
            clearance_synchronized = SynchronizedPoseExecutor(clearance_raw_executor)
            active_synchronized = clearance_synchronized
            clearance_driver = ExecutorControlDriver(
                clearance_synchronized,
                rate_hz=rate_hz,
                safety_heartbeat=watchdog.pulse,
            )
            held_hands = dex3_controller.acquire_measured_hold(
                safety_heartbeat=watchdog.pulse,
            )
            held_error = held_hands.maximum_target_error(
                clearance_plan.initial_left_hand_q_rad,
                clearance_plan.initial_right_hand_q_rad,
            )[2]
            if held_error > dex3_config.posture_position_tolerance_rad:
                raise RuntimeError(
                    "live Dex3 state changed between read-only planning and "
                    f"measured-state hold acquisition by {held_error:.4f}rad; "
                    f"limit is {dex3_config.posture_position_tolerance_rad:.4f}rad"
                )
            clearance_driver.start()
            clearance_synchronized.acquire(operator_confirmed=True)
            _wait_for_control_state(
                clearance_synchronized,
                clearance_driver,
                ExecutorState.READY,
                timeout_s=executor_config.motion_timeout_s + 5.0,
                control_maintenance=dex3_controller.maintain_initial_posture,
            )
            for clearance_pose_id in (
                RIGHT_CLEARANCE_POSE_ID,
                DUAL_CLEARANCE_POSE_ID,
            ):
                clearance_synchronized.start_pose(
                    clearance_pose_id,
                    operator_confirmed=True,
                )
                _wait_for_control_state(
                    clearance_synchronized,
                    clearance_driver,
                    ExecutorState.READY,
                    timeout_s=executor_config.motion_timeout_s + 5.0,
                    control_maintenance=dex3_controller.maintain_initial_posture,
                )
            loaded_state = clearance_synchronized.observe()
            loaded_hands = _wait_for_dex3_state(dex3_controller.observer, 5.0)
            loaded_sweep = _run_isolated_control_work(
                label="automatic-collection loaded Dex3 finger-sweep validation",
                worker=_validate_loaded_dex3_finger_sweep_worker,
                worker_kwargs={
                    "urdf_path": str(_configured_gravity_urdf(args.hardware_config)),
                    "collision_config": live_collision,
                    "body_q": loaded_state.position,
                    "initial_left_hand_q_rad": loaded_hands.left.position,
                    "initial_right_hand_q_rad": loaded_hands.right.position,
                    "target_left_hand_q_rad": dex3_config.left_target_q_rad,
                    "target_right_hand_q_rad": dex3_config.right_target_q_rad,
                    "path_config": clearance_path_config,
                },
                driver=clearance_driver,
                control_maintenance=dex3_controller.maintain_initial_posture,
            )
            if not loaded_sweep.passed:
                raise RuntimeError(
                    "automatic-collection loaded Dex3 finger sweep failed: "
                    f"{loaded_sweep.failure}"
                )
            dex3_posture_evidence = dex3_controller.acquire_posture(
                safety_heartbeat=clearance_driver.check,
            )
            clearance_driver.safety_heartbeat = _dex3_safety_heartbeat(
                watchdog,
                dex3_controller,
            )
            print(
                "DEX3 CALIBRATION POSTURE ACQUIRED AT OUTWARD HANDOFF — both "
                "hands reached NVIDIA's middle-close target; posture commands "
                "remain active for the complete calibration interval"
            )
            store = SessionStore(args.session_directory)
            store.create(
                session_id=args.session_id,
                created_at_utc=utc_now_iso(),
                camera_info=camera_info,
                pose_set_content_sha256=plan.content_sha256,
                artifacts={
                    "authored_plan.json": plan_bytes,
                    "hardware.yaml": hardware_bytes,
                    "target.json": target_bytes,
                    "collision_pairs.yaml": collision_bytes,
                    "capture_quality.yaml": quality_bytes,
                    "validation_report.json": validation_bytes,
                },
                pairing_config=pairing,
                recording_gate_config=recording,
                provenance={
                    "command": "g1-calib collect-auto",
                    "mode_machine": 5,
                    "authored_target_count": authored_target_count,
                    "yellow_override_reason": REPLAY_YELLOW_REASON,
                    "yellow_policy": "automatic_accept_with_recorded_warning",
                    "red_policy": "reject_target_and_continue",
                    "camera_image_topic": args.image_topic,
                    "camera_info_topic": args.camera_info_topic,
                    "ros_camera_reliability": args.ros_camera_reliability,
                    "network_interface": args.network_interface,
                    "source_authored_plan_sha256": plan.content_sha256,
                    "policy_validation_report_sha256": policy_report.content_sha256,
                    "motion_completion_policy": (
                        "complete_command_then_measured_stationarity"
                    ),
                    "dynamic_handoff_state": activation.reference_state.to_dict(),
                    "dynamic_handoff_readiness": activation.readiness.to_dict(),
                    "dex3_clearance_handoff": {
                        "plan_sha256": clearance_plan.content_sha256,
                        "shoulder_roll_offset_rad": (
                            clearance_plan.shoulder_roll_offset_rad
                        ),
                        "command_q14": list(clearance_plan.dual_clearance_q14),
                        "loaded_finger_sweep_minimum_clearance_m": (
                            loaded_sweep.minimum_clearance_m
                        ),
                        "loaded_finger_sweep_minimum_pair": (loaded_sweep.minimum_pair),
                    },
                    "lowstate_fields": ["q", "dq", "tau_est"],
                    "gravity_feedforward": _gravity_provenance(gravity_feedforward),
                    "dex3_posture_control": {
                        "posture_name": "nvidia_groot_middle_close",
                        "posture_source": DEX3_CALIBRATION_POSTURE_SOURCE,
                        "left_commanded_q_rad": list(
                            dex3_controller.config.left_target_q_rad
                        ),
                        "right_commanded_q_rad": list(
                            dex3_controller.config.right_target_q_rad
                        ),
                        "measured_settled_state": dex3_posture_evidence.to_dict(),
                        "terminal_policy": "unitree_timeout_bit_on_both_hands",
                    },
                },
                collection_method="authored_automatic",
            )
            runtime_log = SessionRuntimeLog(store.directory / "runtime.jsonl")
            runtime_log.append(
                "run_started",
                session_id=args.session_id,
                authored_target_count=authored_target_count,
                plan_sha256=plan.content_sha256,
                validation_report_sha256=report.content_sha256,
                motion_completion_policy=(
                    "complete_command_then_measured_stationarity"
                ),
            )
            raw_executor = PoseExecutor(
                transport=transport,
                clock=SystemClock(),
                pose_set=plan,
                handoff_q=(
                    clearance_plan.dual_clearance_q14[:7]
                    if plan.calibration_arm == "left"
                    else clearance_plan.dual_clearance_q14[7:]
                ),
                hold_q=(
                    clearance_plan.dual_clearance_q14[7:]
                    if plan.calibration_arm == "left"
                    else clearance_plan.dual_clearance_q14[:7]
                ),
                approved_validation_report_sha256=report.content_sha256,
                config=executor_config,
                gravity_feedforward=gravity_feedforward,
            )
            synchronized = SynchronizedPoseExecutor(raw_executor)
            driver = ExecutorControlDriver(
                synchronized,
                rate_hz=rate_hz,
                safety_heartbeat=_dex3_safety_heartbeat(
                    watchdog,
                    dex3_controller,
                ),
            )
            assert isolated_store is not None
            isolated_store.set_health_check(driver.check)
            clearance_driver.close()
            clearance_driver.check()
            clearance_driver = None
            watchdog.pulse()
            synchronized.adopt_owned_control(
                previous_command_q14=clearance_plan.dual_clearance_q14,
            )
            active_synchronized = synchronized
            driver.start()
            detector = CorrespondenceDetector(args.target_config)
            quality_evaluator = PoseQualityEvaluator(thresholds)
            last_preview_key = None

            def record_executor_events() -> None:
                nonlocal logged_executor_event_count
                events = synchronized.events
                for event in events[logged_executor_event_count:]:
                    runtime_log.append(
                        "executor_transition",
                        sequence=event.sequence,
                        occurred_monotonic_s=event.occurred_monotonic_s,
                        previous_state=event.previous_state.value,
                        state=event.state.value,
                        reason=event.reason,
                    )
                logged_executor_event_count = len(events)

            def preview(frame, correspondences, quality):
                if args.no_window:
                    return
                intrinsics = CameraIntrinsics(
                    _camera_matrix(camera_info), camera_info.d
                )
                rendered = render_operator_preview(
                    frame.image_bgr,
                    correspondences,
                    quality,
                    intrinsics=intrinsics,
                    saved_view_count=progress["accepted"],
                    footer_lines=(
                        (
                            f"{progress['message']}  |  CONTROL "
                            f"{synchronized.state.value.upper()}"
                        ),
                        (
                            f"ACCEPTED {progress['accepted']}/"
                            f"{authored_target_count}  REJECTED "
                            f"{progress['rejected']}"
                        ),
                        "RED/NO CUBE = SKIP  |  CTRL+C = ABORT AND DAMP",
                    ),
                )
                cv2.imshow("G1 automatic calibration collection", rendered)
                cv2.waitKey(1)

            def wait_once(duration_s: float) -> None:
                nonlocal last_preview_key
                rclpy.spin_once(node, timeout_sec=0.0)
                record_executor_events()
                driver.check()
                latest = camera.frames.latest
                if not args.no_window and latest is not None:
                    key = (
                        latest.timing.receipt_monotonic_s,
                        latest.timing.header_stamp_ns,
                    )
                    if key != last_preview_key:
                        correspondences = detector.detect(latest.image_bgr)
                        intrinsics = CameraIntrinsics(
                            latest.camera_info.rectified_camera_matrix,
                            latest.camera_info.d,
                        )
                        quality = quality_evaluator.evaluate(
                            correspondences,
                            intrinsics=intrinsics,
                            history=(),
                        )
                        preview(latest, correspondences, quality)
                        last_preview_key = key
                time.sleep(duration_s)

            source = LiveBurstFrameSource(
                camera_frames=camera.frames,
                robot_states=states,
                detector=detector,
                quality_evaluator=quality_evaluator,
                recording_config=recording,
                pairing_config=pairing,
                config=LiveBurstConfig(
                    frame_count=thresholds.stationary_burst_frames,
                    timeout_s=args.burst_timeout_s,
                    poll_interval_s=1.0 / rate_hz,
                    maximum_duration_s=(thresholds.stationary_burst_maximum_duration_s),
                ),
                wait_once=wait_once,
                accept_yellow=lambda _frame: True,
                preview=preview,
                report_burst_restart=lambda pose_id, capture_id, reason: (
                    runtime_log.append(
                        "burst_restarted",
                        pose_id=pose_id,
                        capture_id=capture_id,
                        reason=reason,
                    )
                ),
            )
            runner = CaptureSessionRunner(
                executor=synchronized,
                store=isolated_store,
            )

            def report_progress(message: str, accepted: int, rejected: int) -> None:
                progress.update(message=message, accepted=accepted, rejected=rejected)
                print(
                    f"{message}; accepted={accepted}/{authored_target_count}, "
                    f"rejected={rejected}"
                )
                runtime_log.append(
                    "capture_progress",
                    message=message,
                    accepted=accepted,
                    authored_target_count=authored_target_count,
                    rejected=rejected,
                )

            orchestrator = AuthoredCollectionOrchestrator(
                executor=synchronized,
                validation_report=report,
                capture_runner=runner,
                frame_source=source,
                control_step=lambda: wait_once(1.0 / rate_hz),
                plan=plan,
                accepted_pose_count=authored_target_count,
                report_progress=report_progress,
            )
            try:
                result = orchestrator.run(
                    confirm_acquisition=False,
                    confirm_release=False,
                    control_already_acquired=True,
                    retain_control_at_handoff=True,
                )
            except KeyboardInterrupt:
                record_executor_events()
                runtime_log.append(
                    "operator_interrupted",
                    accepted=progress["accepted"],
                    rejected=progress["rejected"],
                    executor_state=synchronized.state.value,
                )
                if driver.is_alive:
                    driver.close()
                runtime_log.append(
                    "safety_action_started",
                    action="verified_damp",
                    reason="operator interrupted automatic collection",
                )
                _terminate_motion_safely(
                    watchdog,
                    synchronized,
                    transport,
                    reason="operator interrupted automatic collection",
                )
                runtime_log.append(
                    "safety_action_completed",
                    action="verified_damp",
                    executor_state=synchronized.state.value,
                )
                return 130
            driver.safety_heartbeat = watchdog.pulse
            restored_hands = dex3_controller.restore_initial_posture(
                safety_heartbeat=driver.check,
            )
            restore_error = restored_hands.maximum_target_error(
                clearance_plan.initial_left_hand_q_rad,
                clearance_plan.initial_right_hand_q_rad,
            )[2]
            driver.close()
            driver.check()
            record_executor_events()
            driver = None
            watchdog.pulse()
            active_synchronized = clearance_synchronized
            clearance_synchronized.resume_owned_control()
            clearance_driver = ExecutorControlDriver(
                clearance_synchronized,
                rate_hz=rate_hz,
                safety_heartbeat=watchdog.pulse,
            )
            clearance_driver.start()
            for clearance_pose_id in (
                RIGHT_CLEARANCE_POSE_ID,
                HANDOFF_POSE_ID,
            ):
                clearance_synchronized.start_pose(
                    clearance_pose_id,
                    operator_confirmed=True,
                )
                _wait_for_control_state(
                    clearance_synchronized,
                    clearance_driver,
                    ExecutorState.READY,
                    timeout_s=executor_config.motion_timeout_s + 5.0,
                    control_maintenance=dex3_controller.maintain_initial_posture,
                )
            clearance_synchronized.begin_clean_release(operator_confirmed=True)
            _wait_for_control_state(
                clearance_synchronized,
                clearance_driver,
                ExecutorState.STOPPED,
                timeout_s=executor_config.release_ramp_s + 5.0,
                control_maintenance=dex3_controller.maintain_initial_posture,
            )
            clearance_driver.close()
            clearance_driver.check()
            clearance_driver = None
            dex3_controller.timeout()
            watchdog.disarm()
            dex3_controller.close()
            dex3_controller = None
        print(
            f"finalized session: {args.session_directory}; "
            f"accepted={result.accepted_count}, rejected={result.rejected_count}"
        )
        runtime_log.append(
            "run_completed",
            accepted=result.accepted_count,
            rejected=result.rejected_count,
            attempted=result.attempted_count,
            route_exhausted=result.route_exhausted,
            executor_state=synchronized.state.value,
            terminal_action=("restore_fingers_reverse_clearance_arm_sdk_weight_zero"),
            restored_finger_maximum_error_rad=restore_error,
        )
        return 0
    except Exception as error:
        if record_executor_events is not None:
            record_executor_events()
        if runtime_log is not None:
            runtime_log.append(
                "run_failed",
                error_type=type(error).__name__,
                message=str(error),
                accepted=progress["accepted"],
                rejected=progress["rejected"],
                executor_state=(
                    None if synchronized is None else synchronized.state.value
                ),
                executor_fault=(
                    None if synchronized is None else synchronized.fault_reason
                ),
            )
        raise
    finally:
        if clearance_driver is not None and clearance_driver.is_alive:
            try:
                clearance_driver.close()
            except RuntimeError:
                pass
        if driver is not None and driver.is_alive:
            try:
                driver.close()
            except RuntimeError:
                pass
        local_dex3_timeout_error = None
        if dex3_controller is not None and not dex3_controller.timed_out:
            try:
                dex3_controller.timeout()
            except BaseException as error:  # noqa: BLE001 - finish PC2 recovery.
                local_dex3_timeout_error = error
        if watchdog is not None and watchdog.armed:
            cleanup_action = (
                "disarm_without_commands"
                if transport is not None and transport.command_count == 0
                else "verified_damp"
            )
            if runtime_log is not None:
                runtime_log.append(
                    "safety_action_started",
                    action=cleanup_action,
                    reason="automatic collection failed or cleanup was required",
                )
            try:
                if transport is not None and transport.command_count == 0:
                    watchdog.disarm()
                elif active_synchronized is None or transport is None:
                    watchdog.damp(
                        "automatic collection cleanup before executor creation"
                    )
                else:
                    _terminate_motion_safely(
                        watchdog,
                        active_synchronized,
                        transport,
                        reason=("automatic collection failed or cleanup was required"),
                    )
            except BaseException as cleanup_error:
                if runtime_log is not None:
                    runtime_log.append(
                        "safety_action_failed",
                        action=cleanup_action,
                        error_type=type(cleanup_error).__name__,
                        message=str(cleanup_error),
                    )
                raise
            else:
                if runtime_log is not None:
                    runtime_log.append(
                        "safety_action_completed",
                        action=cleanup_action,
                        executor_state=(
                            None
                            if active_synchronized is None
                            else active_synchronized.state.value
                        ),
                        executor_fault=(
                            None
                            if active_synchronized is None
                            else active_synchronized.fault_reason
                        ),
                    )
        if dex3_controller is not None:
            if (
                not dex3_controller.timed_out
                and watchdog is not None
                and (watchdog.terminal_action is not None)
            ):
                dex3_controller.close_after_external_timeout()
            else:
                dex3_controller.close()
        if local_dex3_timeout_error is not None and (
            watchdog is None or watchdog.terminal_action is None
        ):
            raise local_dex3_timeout_error
        if dex3_observer is not None:
            dex3_observer.close()
        if transport is not None and transport.command_count == 0:
            transport.close()
        if observer is not None:
            observer.close()
        if camera is not None:
            camera.close()
        if isolated_store is not None:
            isolated_store.close()
        node.destroy_node()
        _shutdown_rclpy_once(rclpy)
        cv2.destroyAllWindows()


def _transport_config(args: argparse.Namespace) -> UnitreeTransportConfig:
    with args.hardware_config.open(encoding="utf-8") as stream:
        control = yaml.safe_load(stream)["control"]
    return UnitreeTransportConfig(
        network_interface=args.network_interface,
        domain_id=args.domain_id,
        shoulder_elbow_kp=float(control["hold_shoulder_elbow_kp"]),
        shoulder_elbow_kd=float(control["hold_shoulder_elbow_kd"]),
        wrist_kp=float(control["hold_wrist_kp"]),
        wrist_kd=float(control["hold_wrist_kd"]),
    )


def _dex3_control_config(args: argparse.Namespace) -> Dex3ControlConfig:
    with args.hardware_config.open(encoding="utf-8") as stream:
        hardware = yaml.safe_load(stream)
    dex3 = hardware.get("dex3_control")
    if not isinstance(dex3, dict) or dex3.get("enabled") is not True:
        raise ValueError("Dex3 calibration hardware requires dex3_control.enabled=true")
    if dex3.get("posture_name") != "nvidia_groot_middle_close":
        raise ValueError(
            "Dex3 calibration requires posture_name=nvidia_groot_middle_close"
        )
    if dex3.get("posture_source") != DEX3_CALIBRATION_POSTURE_SOURCE:
        raise ValueError("Dex3 calibration posture source differs from NVIDIA GR00T")
    if (
        hardware.get("robot", {}).get("calibration_finger_posture_model")
        != dex3["posture_name"]
    ):
        raise ValueError(
            "robot finger-posture model differs from the Dex3 command target"
        )
    config = Dex3ControlConfig(
        network_interface=args.network_interface,
        domain_id=args.domain_id,
        left_command_topic=str(dex3["left_command_topic"]),
        right_command_topic=str(dex3["right_command_topic"]),
        left_state_topic=str(dex3["left_state_topic"]),
        right_state_topic=str(dex3["right_state_topic"]),
        command_rate_hz=float(dex3["command_rate_hz"]),
        kp=float(dex3["kp"]),
        kd=float(dex3["kd"]),
        left_target_q_rad=tuple(float(value) for value in dex3["left_target_q_rad"]),
        right_target_q_rad=tuple(float(value) for value in dex3["right_target_q_rad"]),
        state_freshness_timeout_s=float(dex3["state_freshness_timeout_s"]),
        maximum_measured_command_delta_rad=float(
            dex3["maximum_measured_command_delta_rad"]
        ),
        posture_ramp_s=float(dex3["posture_ramp_s"]),
        posture_position_tolerance_rad=float(dex3["posture_position_tolerance_rad"]),
        posture_position_spread_rad=float(
            hardware["control"]["settled_position_spread_rad"]
        ),
        posture_settle_dwell_s=float(dex3["posture_settle_dwell_s"]),
        posture_timeout_s=float(dex3["posture_timeout_s"]),
        timeout_repetitions=int(dex3["timeout_repetitions"]),
    )
    locked = hardware.get("control", {}).get("gravity_locked_joint_positions_rad")
    if not isinstance(locked, dict):
        raise TypeError("Dex3 calibration requires an explicit gravity posture")
    for side, target in (
        ("left", config.left_target_q_rad),
        ("right", config.right_target_q_rad),
    ):
        for suffix, expected in zip(
            DEX3_MOTOR_JOINT_SUFFIXES[side], target, strict=True
        ):
            name = f"{side}_hand_{suffix}_joint"
            if name not in locked or not np.isclose(
                float(locked[name]), expected, atol=1e-12, rtol=0.0
            ):
                raise ValueError(
                    "Dex3 command and gravity postures disagree at "
                    f"{name}: command={expected}rad, gravity={locked.get(name)!r}"
                )
    return config


def _require_dex3_posture_commissioned(hardware_config: Path) -> None:
    with hardware_config.open(encoding="utf-8") as stream:
        hardware = yaml.safe_load(stream)
    dex3 = hardware.get("dex3_control")
    if (
        not isinstance(dex3, dict)
        or dex3.get("posture_control_commissioned") is not True
    ):
        raise RuntimeError(
            "Dex3 motion is blocked before command creation: the measured-state "
            "NVIDIA middle-close ramp, continuous posture hold, and terminal "
            "timeout have not passed commission-dex3-middle-close on this robot"
        )


def _dex3_safety_heartbeat(
    watchdog: PC2DampingWatchdog,
    dex3: UnitreeDex3PostureController,
) -> Callable[[], None]:
    def maintain_and_pulse() -> None:
        dex3.maintain_posture()
        watchdog.pulse()

    return maintain_and_pulse


def _debug_lowcmd_config(hardware_config: Path) -> UnitreeDebugLowCmdConfig:
    with hardware_config.open(encoding="utf-8") as stream:
        control = yaml.safe_load(stream)["control"]
    if (
        control.get("seated_debug_failure_policy")
        != "restore_ai_verify_zero_torque_fsm_0"
    ):
        raise ValueError(
            "seated debug failure policy must restore AI and verify zero-torque FSM 0"
        )
    return UnitreeDebugLowCmdConfig(
        command_topic=str(control["seated_debug_command_topic"]),
        body_strong_kp=float(control["debug_body_strong_kp"]),
        body_strong_kd=float(control["debug_body_strong_kd"]),
        body_weak_kp=float(control["debug_body_weak_kp"]),
        body_weak_kd=float(control["debug_body_weak_kd"]),
        activation_position_tolerance_rad=float(
            control["activation_position_tolerance_rad"]
        ),
        body_hold_position_tolerance_rad=float(
            control["debug_body_hold_position_tolerance_rad"]
        ),
        motion_switch_timeout_s=float(control["debug_motion_switch_timeout_s"]),
        motion_switch_poll_interval_s=float(
            control["debug_motion_switch_poll_interval_s"]
        ),
    )


def _require_ack(actual: str, expected: str) -> None:
    if actual != expected:
        raise ValueError(f"--confirm must exactly equal: {expected}")


def _read_terminal_key() -> bytes:
    if not sys.stdin.isatty():
        raise RuntimeError("SPACE confirmation requires an interactive terminal")
    descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        key = os.read(descriptor, 1)
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)
    if not key:
        raise RuntimeError("terminal input closed while waiting for SPACE")
    return key


def _wait_for_space(prompt: str) -> bool:
    print(prompt, end="", flush=True)
    while True:
        key = _read_terminal_key()
        if key == b" ":
            print("SPACE")
            return True
        if key == b"\x03":
            raise KeyboardInterrupt


class _DeferredSIGINT:
    """Turn SIGINT into a polled flag while a C extension is executing."""

    def __init__(self) -> None:
        self.requested = False
        self._previous_handler = None

    def __enter__(self):
        self._previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle)
        return self

    def __exit__(self, *_exc_info) -> None:
        signal.signal(signal.SIGINT, self._previous_handler)

    def _handle(self, _signum, _frame) -> None:
        self.requested = True


def _shutdown_rclpy_once(rclpy) -> None:
    """Shut ROS down without failing if its SIGINT handler already did so."""

    rclpy.try_shutdown()


def _wait_for_state(transport, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            state = transport.observe()
            if not state.is_mode5:
                raise ValueError(
                    f"robot is mode_machine={state.mode_machine}; expected 5"
                )
            return state
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise RuntimeError("timed out waiting for rt/lowstate") from None
            time.sleep(0.02)


def _wait_for_dex3_state(observer: UnitreeDex3StateObserver, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    last_error = "no valid dual-hand state"
    while time.monotonic() < deadline:
        try:
            return observer.observe()
        except RuntimeError as error:
            last_error = str(error)
        time.sleep(0.02)
    raise RuntimeError("timed out waiting for Dex3 state: " + last_error)


def _wait_for_activation_handoff(
    observer,
    states: StateSampleBuffer,
    pose_set,
    recording: RecordingGateConfig,
    *,
    timeout_s: float = 5.0,
) -> ActivationHandoff:
    """Select a stationary measured reference from the trailing state window."""

    if timeout_s <= 0:
        raise ValueError("activation timeout must be positive")
    deadline = time.monotonic() + timeout_s
    last_error = "no valid rt/lowstate sample"
    while True:
        try:
            observer.observe()
            return build_activation_handoff(
                pose_set,
                states.snapshot(),
                now_monotonic_s=time.monotonic(),
                config=recording,
            )
        except (RuntimeError, ValueError) as error:
            last_error = str(error)
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "timed out waiting for a stationary dynamic handoff: " + last_error
            ) from None
        time.sleep(0.01)


def _unique_route_edges(
    route: list[str] | tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source, target in pairwise(route):
        edge = (source, target)
        if source == target or edge in seen:
            continue
        seen.add(edge)
        edges.append(edge)
    return tuple(edges)


def _configured_urdf(hardware_config: Path) -> Path:
    with hardware_config.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return _resolve_hardware_path(hardware_config, data["robot"]["urdf"])


def _configured_gravity_urdf(hardware_config: Path) -> Path:
    """Resolve the articulated model used by gravity and Dex3 collision checks."""

    with hardware_config.open(encoding="utf-8") as stream:
        hardware = yaml.safe_load(stream)
    return _resolve_hardware_path(
        hardware_config,
        hardware["control"].get(
            "gravity_model_urdf",
            hardware["robot"]["urdf"],
        ),
    )


def _resolve_hardware_path(hardware_config: Path, value: str | Path) -> Path:
    configured = Path(value)
    if configured.is_absolute():
        return configured
    return hardware_config.resolve().parent.parent / configured


def _gravity_feedforward(hardware_config: Path) -> G1PinocchioGravityFeedforward:
    with hardware_config.open(encoding="utf-8") as stream:
        hardware = yaml.safe_load(stream)
    control = hardware["control"]
    implementation = str(control.get("gravity_feedforward", ""))
    if implementation != "pinocchio_rnea_at_commanded_q":
        raise ValueError(
            "control.gravity_feedforward must be pinocchio_rnea_at_commanded_q"
        )
    gravity_urdf = _configured_gravity_urdf(hardware_config)
    locked = control.get("gravity_locked_joint_positions_rad")
    if locked is not None and not isinstance(locked, dict):
        raise TypeError("gravity_locked_joint_positions_rad must be a mapping")
    return G1PinocchioGravityFeedforward(
        gravity_urdf,
        locked_joint_positions_rad=locked,
    )


def _prepare_gravity_feedforward(
    hardware_config: Path,
    reference_full_q: np.ndarray,
) -> G1PinocchioGravityFeedforward:
    reference = validate_full_joint_vector(
        reference_full_q,
        name="gravity-feedforward reference full q",
    )
    gravity = _gravity_feedforward(hardware_config)
    gravity.seed_reference(reference)
    takeover_q14 = np.concatenate(
        (
            reference[np.asarray(LEFT_ARM_INDICES)],
            reference[np.asarray(RIGHT_ARM_INDICES)],
        )
    )
    _print_gravity_preflight(gravity, gravity.torque_for(takeover_q14))
    return gravity


def _gravity_provenance(
    gravity: G1PinocchioGravityFeedforward,
) -> dict[str, object]:
    return {
        "implementation": UNITREE_XR_GRAVITY_REFERENCE,
        "pinocchio_version": gravity.backend_version,
        "urdf_path": str(gravity.urdf_path),
        "urdf_sha256": gravity.urdf_sha256,
        "source_model_nq": gravity.full_model_nq,
        "reduced_model_nq": gravity.reduced_model_nq,
        "locked_joint_positions_rad": dict(gravity.locked_joint_positions_rad),
    }


def _print_gravity_preflight(
    gravity_feedforward: G1PinocchioGravityFeedforward,
    torque: np.ndarray,
) -> None:
    left_values = ", ".join(
        f"{name}={value:+.3f}Nm"
        for name, value in zip(G1_29_JOINT_NAMES[15:22], torque[:7], strict=True)
    )
    right_peak_index = 7 + int(np.argmax(np.abs(torque[7:])))
    print(
        "GRAVITY FEEDFORWARD READY — "
        f"{UNITREE_XR_GRAVITY_REFERENCE}; Pinocchio "
        f"{gravity_feedforward.backend_version}; source model "
        f"nq={gravity_feedforward.full_model_nq}, locked joints="
        f"{len(gravity_feedforward.locked_joint_positions_rad)}; URDF "
        f"sha256={gravity_feedforward.urdf_sha256}; left takeover torque: "
        f"{left_values}; held-right peak: "
        f"{G1_29_JOINT_NAMES[15 + right_peak_index]}="
        f"{torque[right_peak_index]:+.3f}Nm"
    )


def _hardware_pose_set(hardware_config: Path) -> PoseSet:
    with hardware_config.open(encoding="utf-8") as stream:
        hardware = yaml.safe_load(stream)
    if not isinstance(hardware, dict):
        raise TypeError("hardware configuration must contain a mapping")
    model = URDFModel(_configured_urdf(hardware_config))
    return PoseSet(
        robot_model=model.name,
        mode_machine=int(hardware["robot"]["mode_machine"]),
        urdf_sha256=model.sha256,
        calibration_arm=str(hardware["control"]["calibration_arm"]),
    )


def _initialize_runtime_validation_worker(
    urdf_path: str,
    collision_config: CollisionConfig,
    path_config: PathValidationConfig,
    pose_set,
    reference_full_q: np.ndarray,
) -> None:
    global _runtime_validation_worker
    model = URDFModel(urdf_path)
    _runtime_validation_worker = (
        PosePathValidator(
            model=model,
            collision_checker=FCLCollisionChecker(model, collision_config),
            config=path_config,
        ),
        pose_set,
        reference_full_q,
    )


def _validate_runtime_edge_worker(
    edge: tuple[str, str],
) -> DirectedEdgeResult:
    if _runtime_validation_worker is None:
        raise RuntimeError("runtime validation worker was not initialized")
    validator, pose_set, reference_full_q = _runtime_validation_worker
    return validator.validate(
        pose_set,
        directed_edges=(edge,),
        reference_full_q=reference_full_q,
    ).edges[0]


def _parallel_runtime_edge_results(
    *,
    model: URDFModel,
    collision_config: CollisionConfig,
    path_config: PathValidationConfig,
    pose_set,
    reference_full_q: np.ndarray,
    directed_edges: tuple[tuple[str, str], ...],
    worker_count: int,
    progress: Callable[[int, int, str, str], None] | None,
    cancellation_requested: Callable[[], bool] | None,
) -> tuple[DirectedEdgeResult, ...]:
    if cancellation_requested is not None and cancellation_requested():
        raise KeyboardInterrupt
    results: list[DirectedEdgeResult | None] = [None] * len(directed_edges)
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=get_context("spawn"),
        initializer=_initialize_runtime_validation_worker,
        initargs=(
            str(model.path),
            collision_config,
            path_config,
            pose_set,
            reference_full_q,
        ),
    ) as executor:
        futures = {
            executor.submit(_validate_runtime_edge_worker, edge): index
            for index, edge in enumerate(directed_edges)
        }
        try:
            for completed, future in enumerate(as_completed(futures), start=1):
                if cancellation_requested is not None and cancellation_requested():
                    raise KeyboardInterrupt
                index = futures[future]
                results[index] = future.result()
                if progress is not None:
                    source, target = directed_edges[index]
                    progress(completed, len(directed_edges), source, target)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    if any(result is None for result in results):
        raise RuntimeError("parallel validation did not return every edge")
    return tuple(result for result in results if result is not None)


def _runtime_validation_report(
    *,
    pose_set,
    reference_full_q,
    directed_edges: tuple[tuple[str, str], ...],
    hardware_config: Path,
    collision_config: CollisionConfig,
    policy_config: dict | None = None,
    progress: Callable[[int, int, str, str], None] | None = None,
    cancellation_requested: Callable[[], bool] | None = None,
    parallel_workers: int | None = None,
) -> ValidationReport:
    if not directed_edges:
        raise ValueError("runtime validation requires at least one directed edge")
    model = URDFModel(_configured_urdf(hardware_config))
    path_config = (
        PathValidationConfig()
        if policy_config is None
        else PathValidationConfig(**policy_config)
    )
    reference = validate_full_joint_vector(
        reference_full_q,
        name="reference_full_q",
    )
    worker_count = (
        min(
            _MAX_RUNTIME_VALIDATION_WORKERS,
            len(directed_edges),
            os.cpu_count() or 1,
        )
        if parallel_workers is None
        else parallel_workers
    )
    if worker_count < 1:
        raise ValueError("parallel validation worker count must be positive")
    use_parallel = worker_count > 1 and (
        parallel_workers is not None
        or len(directed_edges) >= _MINIMUM_PARALLEL_VALIDATION_EDGES
    )
    if use_parallel:
        edges = _parallel_runtime_edge_results(
            model=model,
            collision_config=collision_config,
            path_config=path_config,
            pose_set=pose_set,
            reference_full_q=reference,
            directed_edges=directed_edges,
            worker_count=worker_count,
            progress=progress,
            cancellation_requested=cancellation_requested,
        )
        report = ValidationReport(
            pose_set_sha256=pose_set.content_sha256,
            urdf_sha256=model.sha256,
            collision_config_sha256=collision_config.content_sha256,
            reference_full_q_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
            config={
                name: getattr(path_config, name)
                for name in path_config.__dataclass_fields__
            },
            edges=edges,
        )
    else:
        validator = PosePathValidator(
            model=model,
            collision_checker=FCLCollisionChecker(model, collision_config),
            config=path_config,
        )
        report = validator.validate(
            pose_set,
            directed_edges=directed_edges,
            reference_full_q=reference,
            progress=progress,
            cancellation_requested=cancellation_requested,
        )
    if not report.passed:
        failures = [
            f"{edge.from_pose_id}->{edge.to_pose_id}: " + "; ".join(edge.failures)
            for edge in report.edges
            if not edge.passed
        ]
        raise ValueError("dynamic route validation failed: " + " | ".join(failures))
    return report


def _joint_endpoint_evidence(*, sample, target_q: np.ndarray, calibration_arm: str):
    target = np.asarray(target_q, dtype=np.float64).reshape(-1)
    if target.shape != (7,) or not np.all(np.isfinite(target)):
        raise ValueError("endpoint target must contain seven finite joints")
    measured = sample.arm_q(calibration_arm)
    errors = np.abs(measured - target)
    worst_index = int(np.argmax(errors))
    return {
        "target_q": target.tolist(),
        "measured_q": measured.tolist(),
        "absolute_error_rad": errors.tolist(),
        "maximum_error_rad": float(errors[worst_index]),
        "maximum_error_joint": arm_joint_names(calibration_arm)[worst_index],
        "measured_state": sample.to_dict(),
        "interpretation": (
            "joint tracking evidence only; it is not the board-relative task "
            "accuracy score"
        ),
    }


def _print_dynamic_preflight(
    activation: ActivationHandoff,
    report: ValidationReport,
) -> None:
    clearances = [
        edge.minimum_clearance_m
        for edge in report.edges
        if edge.minimum_clearance_m is not None
    ]
    minimum_clearance = None if not clearances else min(clearances)
    print(
        "dynamic handoff passed before command publisher creation: "
        f"samples={activation.readiness.sample_count}, "
        f"position_spread="
        f"{activation.readiness.maximum_calibration_position_spread_rad:.6f}rad, "
        f"edges={len(report.edges)}, "
        f"minimum_clearance="
        f"{'n/a' if minimum_clearance is None else f'{minimum_clearance:.6f}m'}"
    )


def _pose_set_bytes(pose_set) -> bytes:
    return yaml.safe_dump(
        pose_set.to_dict(), sort_keys=False, allow_unicode=True
    ).encode()


def _authored_plan_bytes(plan: AuthoredCollectionPlan) -> bytes:
    return json.dumps(plan.to_dict(), indent=2, sort_keys=True).encode() + b"\n"


def _validation_report_bytes(report: ValidationReport) -> bytes:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True).encode() + b"\n"


def _drive_direct(
    executor: PoseExecutor,
    desired: ExecutorState,
    *,
    rate_hz: float,
    safety_heartbeat=None,
) -> None:
    deadline = time.monotonic() + executor.config.motion_timeout_s + 5.0
    last_motion_status_s = time.monotonic()
    while executor.state is not desired:
        if time.monotonic() >= deadline:
            raise RuntimeError(f"timed out waiting for {desired.value}")
        if executor.state is ExecutorState.STOPPED:
            raise RuntimeError(
                "executor stopped: " + (executor.fault_reason or "unknown reason")
            )
        time.sleep(1.0 / rate_hz)
        state = executor.tick()
        if state is ExecutorState.FAULT:
            raise RuntimeError(
                f"executor faulted: {executor.fault_reason or 'unknown reason'}"
            )
        if state in {ExecutorState.MOVING, ExecutorState.SETTLING}:
            now = time.monotonic()
            if now - last_motion_status_s >= 1.0:
                print(executor.motion_diagnostic())
                last_motion_status_s = now
        if safety_heartbeat is not None:
            safety_heartbeat()


def _attempt_safety_cleanup(
    errors: list[tuple[str, BaseException]],
    label: str,
    action: Callable[[], None],
) -> None:
    """Run every cleanup action so one failure cannot suppress the rest."""

    try:
        action()
    except BaseException as error:  # noqa: BLE001
        errors.append((label, error))


def _raise_safety_cleanup_failures(
    primary_error: BaseException | None,
    cleanup_errors: list[tuple[str, BaseException]],
) -> None:
    """Report the initiating fault and every cleanup fault in one exception."""

    if not cleanup_errors:
        return
    cleanup_detail = " || ".join(
        f"{label}: {type(error).__name__}: {error}" for label, error in cleanup_errors
    )
    if primary_error is None:
        raise RuntimeError(f"safety cleanup failed: {cleanup_detail}")
    raise RuntimeError(
        "primary failure: "
        f"{type(primary_error).__name__}: {primary_error}; "
        f"safety cleanup also failed: {cleanup_detail}"
    ) from primary_error


def _terminate_motion_safely(
    watchdog: PC2DampingWatchdog,
    executor,
    transport: UnitreeArmSDKTransport,
    *,
    reason: str,
) -> None:
    """Disarm before ownership, otherwise require accepted whole-body damping."""

    if not watchdog.armed:
        if transport.command_count == 0:
            transport.close()
            return
        raise RuntimeError("watchdog is not armed after arm commands were published")
    if transport.command_count == 0:
        watchdog.disarm()
        transport.close()
        return
    watchdog.damp(reason)
    executor.confirm_external_damping(reason)


def _terminate_debug_lowcmd_safely(
    watchdog: PC2DampingWatchdog,
    executor,
    transport: UnitreeDebugLowCmdTransport,
    *,
    reason: str,
) -> None:
    """Require verified external ownership after any debug transition attempt."""

    if not transport.requires_external_takeover:
        if watchdog.armed:
            watchdog.disarm()
        transport.close()
        return
    if not watchdog.armed:
        if watchdog.terminal_action in {"zero_torque", "seated"}:
            if executor is not None:
                executor.confirm_external_takeover(reason)
            else:
                transport.close_after_external_takeover()
            return
        raise RuntimeError(
            "debug lowcmd ownership was attempted without an armed PC2 guard"
        )
    watchdog.restore_zero_torque(reason)
    if executor is not None:
        executor.confirm_external_takeover(reason)
    else:
        transport.close_after_external_takeover()


def _pc2_damping_watchdog(
    args: argparse.Namespace,
    hardware_config: Path,
    *,
    require_regular: bool = True,
    query_initial_fsm_id: bool = True,
    required_initial_fsm_id: int | None = None,
    restore_motion_service_before_loco: bool = False,
) -> PC2DampingWatchdog:
    if require_regular and required_initial_fsm_id is not None:
        raise ValueError("cannot combine Regular and explicit initial FSM requirements")
    required_fsm_id = (
        _regular_fsm_id(hardware_config) if require_regular else required_initial_fsm_id
    )
    return PC2DampingWatchdog(
        _pc2_safety_config(
            args,
            hardware_config,
            required_initial_fsm_id=required_fsm_id,
            query_initial_fsm_id=query_initial_fsm_id,
            restore_motion_service_before_loco=restore_motion_service_before_loco,
        )
    )


def _regular_fsm_id(hardware_config: Path) -> int:
    with hardware_config.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or not isinstance(data.get("control"), dict):
        raise TypeError("hardware config is missing the control section")
    value = data["control"]["required_regular_fsm_id"]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("hardware Regular locomotion FSM ID must be an integer")
    if value < 0:
        raise ValueError("hardware Regular locomotion FSM ID must be non-negative")
    return value


def _seated_fsm_id(hardware_config: Path) -> int:
    with hardware_config.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or not isinstance(data.get("control"), dict):
        raise TypeError("hardware config is missing the control section")
    value = data["control"]["required_seated_fsm_id"]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("hardware seated locomotion FSM ID must be an integer")
    if value < 0:
        raise ValueError("hardware seated locomotion FSM ID must be non-negative")
    return value


def _pc2_safety_config(
    args: argparse.Namespace,
    hardware_config: Path,
    *,
    required_initial_fsm_id: int | None,
    query_initial_fsm_id: bool = True,
    restore_motion_service_before_loco: bool = False,
) -> PC2SafetyConfig:
    with hardware_config.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise TypeError("hardware configuration must contain a mapping")
    safety = data.get("safety")
    if not isinstance(safety, dict):
        raise TypeError("hardware config is missing the PC2 damping safety section")
    if safety.get("emergency_policy") != "pc2_g1_loco_damp":
        raise ValueError("hardware emergency policy must be pc2_g1_loco_damp")
    if safety.get("physical_support") != "load_bearing_harness":
        raise ValueError("whole-body damping requires the load-bearing harness")
    return PC2SafetyConfig(
        host=args.pc2_host,
        ssh_identity=args.pc2_ssh_identity,
        heartbeat_interval_s=float(safety["heartbeat_interval_s"]),
        heartbeat_timeout_s=float(safety["heartbeat_timeout_s"]),
        connect_timeout_s=float(safety["connect_timeout_s"]),
        client_timeout_s=float(safety["loco_client_timeout_s"]),
        required_initial_fsm_id=required_initial_fsm_id,
        query_initial_fsm_id=query_initial_fsm_id,
        restore_motion_service_before_loco=restore_motion_service_before_loco,
        manage_dex3=bool(safety.get("manage_dex3", False)),
        remote_ros_setup=Path(safety["pc2_ros_setup"]),
        remote_cyclonedds_setup=Path(safety["pc2_cyclonedds_setup"]),
        remote_unitree_setup=Path(safety["pc2_unitree_ros2_setup"]),
        remote_cyclonedds_uri=Path(safety["pc2_cyclonedds_uri"]),
        remote_watchdog_python=Path(safety["pc2_watchdog_python"]),
        remote_cyclonedds_home=Path(safety["pc2_cyclonedds_home"]),
        remote_motion_switcher_interface=str(safety["pc2_motion_switcher_interface"]),
        remote_dex3_interface=str(
            safety.get(
                "pc2_dex3_interface",
                safety["pc2_motion_switcher_interface"],
            )
        ),
        remote_damp_executable=Path(safety["pc2_g1_loco_client"]),
    )


def _teaching_config(path: Path) -> TeachingConfig:
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    control = data["control"]
    return TeachingConfig(
        state_freshness_timeout_s=float(control["lowstate_timeout_s"]),
        acquisition_ramp_s=float(control["acquisition_weight_ramp_s"]),
        gain_transition_ramp_s=float(control["gain_transition_ramp_s"]),
        guide_kp_scale=float(control["guide_kp_scale"]),
        guide_kd_scale=float(control["guide_kd_scale"]),
        activation_position_tolerance_rad=float(
            control["activation_position_tolerance_rad"]
        ),
        opposite_arm_hold_tolerance_rad=float(
            control["held_arm_position_tolerance_rad"]
        ),
        calibration_arm_hold_tolerance_rad=float(
            control["ownership_transition_position_tolerance_rad"]
        ),
        guide_joint_limit_margin_rad=float(control["guide_joint_limit_margin_rad"]),
    )


def _calibration_arm_limits(
    hardware_config: Path,
    calibration_arm: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    model = URDFModel(_configured_urdf(hardware_config))
    limits = model.joint_limits(arm_joint_names(calibration_arm))
    return (
        tuple(limit.lower for limit in limits),
        tuple(limit.upper for limit in limits),
    )


def _executor_config(path: Path) -> tuple[ExecutorConfig, float]:
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    control = data["control"]
    rate_hz = float(control["command_rate_hz"])
    heartbeat_timeout_s = float(data["safety"]["heartbeat_timeout_s"])
    control_gap_fault_s = float(control["control_gap_fault_s"])
    if control_gap_fault_s > heartbeat_timeout_s / 2.0:
        raise ValueError(
            "control gap fault limit must be no more than half the PC2 "
            "heartbeat timeout"
        )
    config = ExecutorConfig(
        maximum_joint_velocity_rad_s=float(control["max_joint_velocity_rad_s"]),
        motion_position_tolerance_rad=float(control["motion_position_tolerance_rad"]),
        ownership_transition_position_tolerance_rad=float(
            control["ownership_transition_position_tolerance_rad"]
        ),
        activation_position_tolerance_rad=float(
            control["activation_position_tolerance_rad"]
        ),
        held_arm_position_tolerance_rad=float(
            control["held_arm_position_tolerance_rad"]
        ),
        settled_position_spread_rad=float(control["settled_position_spread_rad"]),
        settle_dwell_s=float(control["settle_dwell_s"]),
        state_freshness_timeout_s=float(control["lowstate_timeout_s"]),
        nominal_tick_period_s=1.0 / rate_hz,
        control_gap_fault_s=control_gap_fault_s,
        acquisition_ramp_s=float(control["acquisition_weight_ramp_s"]),
        release_ramp_s=float(control["release_weight_ramp_s"]),
        motion_timeout_s=float(control["motion_timeout_s"]),
    )
    return config, rate_hz


def _runtime_configs(path: Path):
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    recording_data = data["pose_recording"]
    executor, rate = _executor_config(path)
    recording = RecordingGateConfig(
        calibration_arm=str(data["control"]["calibration_arm"]),
        state_freshness_timeout_s=float(data["control"]["lowstate_timeout_s"]),
        stationary_duration_s=float(recording_data["stationary_duration_s"]),
        maximum_state_gap_s=float(recording_data["maximum_state_gap_s"]),
        maximum_calibration_position_spread_rad=float(
            recording_data["maximum_calibration_position_spread_rad"]
        ),
        maximum_hold_position_spread_rad=float(
            recording_data["maximum_hold_position_spread_rad"]
        ),
        minimum_samples=int(recording_data["minimum_state_samples"]),
    )
    pairing = PairingConfig(
        maximum_nearest_delta_s=float(recording_data["maximum_image_state_delta_s"]),
        maximum_bracket_span_s=float(recording_data["maximum_image_state_bracket_s"]),
    )
    return recording, pairing, executor, rate


def _load_validation_report(path: Path) -> ValidationReport:
    with path.open(encoding="utf-8") as stream:
        return ValidationReport.from_dict(json.load(stream))


def _validate_report_binding(pose_set, report: ValidationReport) -> None:
    if report.pose_set_sha256 != pose_set.content_sha256:
        raise ValueError("validation report belongs to a different pose set")
    if report.urdf_sha256 != pose_set.urdf_sha256:
        raise ValueError("validation report belongs to a different URDF")
    if not report.passed:
        raise ValueError("validation report did not pass")


def _load_session_plan(path: Path) -> SessionExecutionPlan:
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or set(data) != {
        "schema_version",
        "capture_pose_ids",
    }:
        raise ValueError("session plan fields do not match schema version 2")
    if data["schema_version"] != 2:
        raise ValueError("unsupported session plan schema version")
    return SessionExecutionPlan(
        capture_pose_ids=tuple(str(item) for item in data["capture_pose_ids"]),
    )


def _preflight_plan(plan: SessionExecutionPlan, report: ValidationReport) -> None:
    # The persisted report approves taught-pose geometry. Per-run handoff edges
    # are derived from the live stationary activation state and revalidated.
    for source, target in pairwise(plan.capture_pose_ids):
        if source == target:
            continue
        edge = report.edge(source, target)
        if not edge.passed:
            raise ValueError(f"session plan edge failed: {source}->{target}")


def _wait_for_camera(rclpy, node, camera, timeout_s: float) -> None:
    if timeout_s <= 0:
        raise ValueError("camera timeout must be positive")
    deadline = time.monotonic() + timeout_s
    while camera.frames.latest is None:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "timed out waiting for matching rectified Image and CameraInfo: "
                + (camera.last_error or "no messages")
            )
        rclpy.spin_once(node, timeout_sec=0.1)


def _next_pose_id(store, first_pose_id: str) -> str:
    pose_set = store.load()
    if not pose_set.poses:
        return first_pose_id
    used = {pose.id for pose in pose_set.poses}
    index = 1
    while f"pose_{index:03d}" in used:
        index += 1
    return f"pose_{index:03d}"


def _next_manual_capture_id(store: SessionStore) -> str:
    used = {capture.capture_id for capture in store.load().captures}
    index = 1
    while f"capture_{index:03d}" in used:
        index += 1
    return f"capture_{index:03d}"


def _pose_history(pose_set) -> list[ViewSignature]:
    history = []
    for pose in pose_set.poses:
        quality = pose.visual_quality
        if not isinstance(quality, dict):
            continue
        signature = quality.get("signature")
        if isinstance(signature, dict):
            history.append(ViewSignature.from_dict(signature))
    return history


def _spin_driver_and_wait(rclpy, node, duration_s: float, driver) -> None:
    rclpy.spin_once(node, timeout_sec=0.0)
    if driver is not None:
        driver.check()
    time.sleep(duration_s)


def _wait_for_driven_state(
    executor,
    driver,
    desired,
    *,
    rclpy,
    node,
    timeout_s: float,
) -> None:
    if timeout_s <= 0:
        raise ValueError("executor wait timeout must be positive")
    deadline = time.monotonic() + timeout_s
    last_motion_status_s = time.monotonic()
    while executor.state is not desired:
        if time.monotonic() >= deadline:
            raise RuntimeError(f"timed out waiting for {desired.value}")
        rclpy.spin_once(node, timeout_sec=0.0)
        driver.check()
        if executor.state in {ExecutorState.MOVING, ExecutorState.SETTLING}:
            now = time.monotonic()
            if now - last_motion_status_s >= 1.0:
                print(executor.motion_diagnostic())
                last_motion_status_s = now
        time.sleep(0.01)
    driver.check()


def _wait_for_control_state(
    executor,
    driver,
    desired,
    *,
    timeout_s: float,
    control_maintenance: Callable[[], None] | None = None,
) -> None:
    """Wait for a driver-owned executor without requiring a ROS event loop."""

    if timeout_s <= 0:
        raise ValueError("executor wait timeout must be positive")
    deadline = time.monotonic() + timeout_s
    last_motion_status_s = time.monotonic()
    while True:
        driver.check()
        if control_maintenance is not None:
            control_maintenance()
        if executor.state is desired:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f"timed out waiting for {desired.value}")
        if executor.state in {ExecutorState.MOVING, ExecutorState.SETTLING}:
            now = time.monotonic()
            if now - last_motion_status_s >= 1.0:
                print(executor.motion_diagnostic())
                last_motion_status_s = now
        time.sleep(0.01)
    driver.check()


def _open_or_resume_manual_session(
    args,
    *,
    camera_info,
    pose_set,
    recording: RecordingGateConfig,
    pairing: PairingConfig,
    artifacts: dict[str, bytes],
) -> SessionStore:
    store = SessionStore(args.session_directory)
    if not args.session_directory.exists():
        if pose_set.poses:
            raise ValueError("new manual session pose set must be empty")
        store.create(
            session_id=args.session_id,
            created_at_utc=utc_now_iso(),
            camera_info=camera_info,
            pose_set_content_sha256=pose_set.content_sha256,
            artifacts=artifacts,
            pairing_config=pairing,
            recording_gate_config=recording,
            provenance={
                "command": "g1-calib teach-poses",
                "mode_machine": 5,
                "camera_image_topic": args.image_topic,
                "camera_info_topic": args.camera_info_topic,
                "ros_camera_reliability": args.ros_camera_reliability,
                "network_interface": args.network_interface,
                "yellow_override_reason": (
                    args.yellow_override_reason or LIVE_TEACHING_YELLOW_REASON
                ),
                "yellow_policy": "SPACE_accepts_and_records_warning",
                "capture_control_mode": "arm_sdk_continuous_guide_hold",
                "capture_control_weight": 1.0,
                "supported_and_held_bursts_are_equal_length": True,
                "lowstate_fields": ["q", "dq", "tau_est"],
            },
            collection_method="manual_teaching",
        )
        return store

    if not store.manifest_path.is_file():
        raise ValueError(
            f"session directory exists without a manifest: {args.session_directory}"
        )
    manifest = store.load()
    if manifest.finalized:
        raise RuntimeError("manual calibration session is already finalized")
    if manifest.session_id != args.session_id:
        raise ValueError("--session-id does not match the resumable session")
    if manifest.provenance.get("collection_method") != "manual_teaching":
        raise ValueError("existing session was not created by manual teaching")
    if (
        manifest.provenance.get("capture_control_mode")
        != "arm_sdk_continuous_guide_hold"
    ):
        raise ValueError(
            "existing session does not use continuous arm_sdk GUIDE/HOLD teaching"
        )
    if manifest.camera_profile_sha256 != camera_info.profile_sha256:
        raise ValueError("live camera profile changed from the resumable session")
    if manifest.pose_set_content_sha256 != pose_set.content_sha256:
        raise ValueError("session pose set differs from its manifest")
    for name, content in artifacts.items():
        expected = manifest.artifact_sha256.get(name)
        if expected != hashlib.sha256(content).hexdigest():
            raise ValueError(f"current {name} differs from the resumable session")
    store.verify_artifacts()
    store.validate_manual_alignment(pose_set)
    return store


def _manual_pose_set_for_session(args) -> PoseSet:
    session_directory = Path(args.session_directory)
    if session_directory.exists():
        manifest_path = session_directory / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(
                f"session directory exists without a manifest: {session_directory}"
            )
        return PoseStore(session_directory / "pose_set.yaml").load()

    with args.hardware_config.open(encoding="utf-8") as stream:
        hardware = yaml.safe_load(stream)
    if not isinstance(hardware, dict):
        raise TypeError("hardware configuration must contain a mapping")
    model = URDFModel(_configured_urdf(args.hardware_config))
    return PoseSet(
        robot_model=model.name,
        mode_machine=int(hardware["robot"]["mode_machine"]),
        urdf_sha256=model.sha256,
        calibration_arm=str(hardware["control"]["calibration_arm"]),
    )


def _validate_hardware_preflight(
    hardware_bytes: bytes, pose_set, *, require_poses: bool = True
) -> None:
    data = yaml.safe_load(hardware_bytes)
    if not isinstance(data, dict):
        raise TypeError("hardware configuration must contain a mapping")
    robot = data["robot"]
    control = data["control"]
    mount = data["camera"]["mount"]
    if robot["mode_machine"] != 5 or robot["dof"] != 29:
        raise ValueError("hardware configuration must select 29-DoF mode_machine=5")
    if control["command_topic"] != "rt/arm_sdk":
        raise ValueError("hardware command topic must be exactly rt/arm_sdk")
    if control.get("seated_table_control") != "debug_lowcmd":
        raise ValueError("seated table control must be exactly debug_lowcmd")
    if control.get("seated_debug_command_topic") != "rt/lowcmd":
        raise ValueError("seated debug command topic must be exactly rt/lowcmd")
    if (
        control.get("seated_debug_failure_policy")
        != "restore_ai_verify_zero_torque_fsm_0"
    ):
        raise ValueError(
            "seated debug failure policy must restore AI and verify zero-torque FSM 0"
        )
    if not isinstance(control.get("seated_debug_lowcmd_commissioned"), bool):
        raise TypeError("seated debug commissioning gate must be boolean")
    required_fsm_id = control["required_regular_fsm_id"]
    if not isinstance(required_fsm_id, int) or isinstance(required_fsm_id, bool):
        raise TypeError("hardware Regular locomotion FSM ID must be an integer")
    if required_fsm_id < 0:
        raise ValueError("hardware Regular locomotion FSM ID must be non-negative")
    seated_fsm_id = control["required_seated_fsm_id"]
    if not isinstance(seated_fsm_id, int) or isinstance(seated_fsm_id, bool):
        raise TypeError("hardware seated locomotion FSM ID must be an integer")
    if seated_fsm_id < 0:
        raise ValueError("hardware seated locomotion FSM ID must be non-negative")
    if float(control["guide_joint_limit_margin_rad"]) <= 0:
        raise ValueError("guide joint-limit warning margin must be positive")
    if control["state_topic"] != "rt/lowstate":
        raise ValueError("hardware state topic must be exactly rt/lowstate")
    if control["left_arm_motor_indices"] != list(range(15, 22)):
        raise ValueError("hardware left arm indices do not match mode 5")
    if control["right_arm_motor_indices"] != list(range(22, 29)):
        raise ValueError("hardware right arm indices do not match mode 5")
    if robot["calibration_arm"] != control["calibration_arm"]:
        raise ValueError("robot/control calibration-arm configuration disagrees")
    if pose_set.calibration_arm != control["calibration_arm"]:
        raise ValueError("pose set and hardware calibration arms do not match")
    if isinstance(pose_set, AuthoredCollectionPlan):
        validate_modeled_hand_target_binding(
            pose_set,
            modeled_hand_T_target=modeled_hand_T_target_from_hardware(data),
        )
        validate_exposed_camera_views(
            pose_set,
            exposed_target_normal=exposed_target_normal_from_hardware(data),
        )
    if not mount["mechanically_fixed"] or not mount["witness_marked"]:
        raise ValueError("camera pitch must be mechanically fixed and witness marked")
    if require_poses and not pose_set.poses:
        raise ValueError("collection requires a non-empty pose set")


def _validate_hardware_target_preflight(
    hardware_bytes: bytes, target_bytes: bytes
) -> None:
    hardware = yaml.safe_load(hardware_bytes)
    target = json.loads(target_bytes)
    validate_hardware_target_profile(hardware, target)


def _validate_calibration_arm(path: Path, pose_set) -> None:
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    configured = data["control"]["calibration_arm"]
    if data["robot"]["calibration_arm"] != configured:
        raise ValueError("robot/control calibration-arm configuration disagrees")
    if pose_set.calibration_arm != configured:
        raise ValueError(
            f"pose set uses {pose_set.calibration_arm} arm but hardware config "
            f"selects {configured}"
        )


def _validate_collision_preflight(
    path: Path,
    *,
    report: ValidationReport | None = None,
) -> CollisionConfig:
    config = CollisionConfig.from_yaml(path)
    if not config.hardware_ready:
        raise ValueError(
            "collision configuration is not hardware-ready: "
            + ", ".join(config.blocking_reasons)
        )
    if report is not None and report.collision_config_sha256 != config.content_sha256:
        raise ValueError(
            "validation report belongs to a different collision configuration"
        )
    return config


def _camera_matrix(camera_info):
    import numpy as np

    return np.asarray(camera_info.p).reshape(3, 4)[:, :3]
