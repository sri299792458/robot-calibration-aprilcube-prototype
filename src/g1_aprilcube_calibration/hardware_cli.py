"""Explicitly armed G1 commissioning and rectified-session collection commands."""

from __future__ import annotations

import argparse
import json
import time
from itertools import pairwise
from pathlib import Path

import cv2
import yaml

from aprilcube import CorrespondenceDetector
from g1_aprilcube_calibration.clock import SystemClock
from g1_aprilcube_calibration.collision import CollisionConfig
from g1_aprilcube_calibration.config import QualityThresholds
from g1_aprilcube_calibration.executor_driver import (
    ExecutorControlDriver,
    SynchronizedPoseExecutor,
)
from g1_aprilcube_calibration.executor_state_machine import (
    ExecutorConfig,
    ExecutorState,
    PoseExecutor,
)
from g1_aprilcube_calibration.live_capture import LiveBurstConfig, LiveBurstFrameSource
from g1_aprilcube_calibration.models import utc_now_iso
from g1_aprilcube_calibration.pose_recorder import (
    PoseRecorder,
    PoseRecordingRequest,
)
from g1_aprilcube_calibration.pose_store import PoseStore
from g1_aprilcube_calibration.pose_validator import ValidationReport
from g1_aprilcube_calibration.preview import render_operator_preview
from g1_aprilcube_calibration.process_lock import CommandOwnerLock
from g1_aprilcube_calibration.quality import (
    CameraIntrinsics,
    PoseQualityEvaluator,
    QualityGrade,
)
from g1_aprilcube_calibration.readiness import RecordingGateConfig, StateSampleBuffer
from g1_aprilcube_calibration.ros.camera_adapter import ROSCameraSubscriber
from g1_aprilcube_calibration.session_runner import (
    ApprovedSessionOrchestrator,
    CaptureSessionRunner,
    SessionExecutionPlan,
)
from g1_aprilcube_calibration.session_store import SessionStore
from g1_aprilcube_calibration.timestamp_pairing import PairingConfig
from g1_aprilcube_calibration.transports.base import ArmCommand
from g1_aprilcube_calibration.transports.unitree_arm_sdk import (
    UnitreeArmSDKTransport,
    UnitreeLowStateObserver,
    UnitreeTransportConfig,
)

WRITE_ACK = "I UNDERSTAND THIS WRITES RT/ARM_SDK"
MOTION_ACK = "I CONFIRM THE G1 WORKSPACE IS CLEAR"


def add_hardware_subparsers(
    subparsers: argparse._SubParsersAction,
    *,
    workspace_root: Path,
    default_target: Path,
    default_quality: Path,
) -> None:
    default_hardware = workspace_root / "config" / "hardware.yaml"
    default_collision = workspace_root / "config" / "collision_pairs.yaml"
    default_lock = Path("/tmp/g1-aprilcube-calibration-command.lock")

    zero = subparsers.add_parser(
        "commission-weight-zero",
        help="publish one measured arm target with blend weight zero",
    )
    _add_network_arguments(zero)
    zero.add_argument("--confirm", required=True, help=f"must equal: {WRITE_ACK}")
    zero.add_argument("--timeout-s", type=float, default=5.0)
    zero.add_argument("--lock-file", type=Path, default=default_lock)
    zero.set_defaults(handler=run_commission_weight_zero)

    teach = subparsers.add_parser(
        "teach-poses",
        help="record stationary measured poses with live rectified quality preview",
    )
    _add_network_arguments(teach)
    teach.add_argument("--pose-set", type=Path, required=True)
    _add_camera_arguments(teach)
    teach.add_argument("--hardware-config", type=Path, default=default_hardware)
    teach.add_argument("--target-config", type=Path, default=default_target)
    teach.add_argument("--quality-config", type=Path, default=default_quality)
    teach.add_argument("--group", default="calibration")
    teach.add_argument("--first-pose-id", default="home")
    teach.add_argument("--preview-directory", type=Path)
    teach.add_argument("--yellow-override-reason")
    teach.add_argument("--head-witness-ack", action="store_true", required=True)
    teach.set_defaults(handler=run_teach_poses)

    hold = subparsers.add_parser(
        "commission-hold",
        help="ramp arm_sdk ownership at measured pose, hold briefly, ramp to zero",
    )
    _add_network_arguments(hold)
    hold.add_argument("--pose-set", type=Path, required=True)
    hold.add_argument("--hardware-config", type=Path, default=default_hardware)
    hold.add_argument("--collision-config", type=Path, default=default_collision)
    hold.add_argument("--duration-s", type=float, default=2.0)
    hold.add_argument("--confirm", required=True, help=f"must equal: {MOTION_ACK}")
    hold.add_argument("--lock-file", type=Path, default=default_lock)
    hold.set_defaults(handler=run_commission_hold)

    pose = subparsers.add_parser(
        "commission-pose",
        help="acquire at home, execute one passed pose edge, return, and release",
    )
    _add_network_arguments(pose)
    pose.add_argument("--pose-set", type=Path, required=True)
    pose.add_argument("--validation-report", type=Path, required=True)
    pose.add_argument("--home-pose", required=True)
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
    collect.add_argument("--burst-timeout-s", type=float, default=15.0)
    collect.add_argument("--yellow-override-reason")
    collect.add_argument("--auto-confirm-transitions", action="store_true")
    collect.add_argument("--no-window", action="store_true")
    collect.add_argument("--confirm", required=True, help=f"must equal: {MOTION_ACK}")
    collect.add_argument("--lock-file", type=Path, default=default_lock)
    collect.set_defaults(handler=run_collect_session)


def _add_network_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--network-interface", required=True)
    parser.add_argument("--domain-id", type=int, default=0)


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


def run_teach_poses(args: argparse.Namespace) -> int:
    if not args.head_witness_ack:
        raise ValueError("--head-witness-ack is required after checking the mark")
    store = PoseStore(args.pose_set)
    pose_set = store.load()
    _validate_calibration_arm(args.hardware_config, pose_set)
    recording, pairing, _, _ = _runtime_configs(args.hardware_config)
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
        _wait_for_camera(rclpy, node, camera, 10.0)
        detector = CorrespondenceDetector(args.target_config)
        evaluator = PoseQualityEvaluator(thresholds)
        recorder = PoseRecorder(
            store=store,
            state_buffer=states,
            gate_config=recording,
            pairing_config=pairing,
        )
        preview_directory = args.preview_directory or (
            args.pose_set.parent / "pose_previews"
        )
        history = []
        anchor_next = not pose_set.poses
        pending_yellow = None
        latest_evaluation = None
        last_frame_key = None
        print("pose teaching is read-only: S=save, A=toggle anchor, U=undo, Q=quit")
        while True:
            rclpy.spin_once(node, timeout_sec=0.01)
            frame = camera.frames.latest
            if frame is not None:
                frame_key = (
                    frame.timing.receipt_monotonic_s,
                    frame.timing.header_stamp_ns,
                )
                if frame_key != last_frame_key:
                    correspondences = detector.detect(frame.image_bgr)
                    intrinsics = CameraIntrinsics(
                        _camera_matrix(frame.camera_info), frame.camera_info.d
                    )
                    quality = evaluator.evaluate(
                        correspondences,
                        intrinsics=intrinsics,
                        history=history,
                    )
                    rendered = render_operator_preview(
                        frame.image_bgr,
                        correspondences,
                        quality,
                        intrinsics=intrinsics,
                        saved_view_count=len(history),
                    )
                    cv2.putText(
                        rendered,
                        f"next={_next_pose_id(store, args.first_pose_id)} "
                        f"anchor={'yes' if anchor_next else 'no'}",
                        (16, rendered.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    latest_evaluation = (
                        frame,
                        quality,
                        rendered,
                    )
                    last_frame_key = frame_key
                cv2.imshow("G1 read-only pose teaching", latest_evaluation[2])
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("a"):
                anchor_next = not anchor_next
                pending_yellow = None
                print(f"next pose anchor={anchor_next}")
            elif key == ord("u"):
                updated = store.undo_last(reason="operator live undo")
                if history:
                    history.pop()
                pending_yellow = None
                print(f"undid final pose; {len(updated.poses)} remain")
            elif key == ord("s") and latest_evaluation is not None:
                selected = latest_evaluation
                quality = selected[1]
                if quality.grade is QualityGrade.RED:
                    print("pose not saved: visual quality is red")
                    pending_yellow = None
                    continue
                if quality.grade is QualityGrade.YELLOW:
                    if not args.yellow_override_reason:
                        print(
                            "pose not saved: yellow requires --yellow-override-reason"
                        )
                        pending_yellow = None
                        continue
                    if pending_yellow is None:
                        pending_yellow = selected
                        print("yellow pose frozen; press S again to confirm")
                        continue
                    selected = pending_yellow
                pending_yellow = None
                pose_id = _next_pose_id(store, args.first_pose_id)
                _wait_for_recording_window(
                    rclpy,
                    node,
                    states,
                    selected[0].timing.receipt_monotonic_s,
                    recording.stationary_duration_s,
                )
                preview_directory.mkdir(parents=True, exist_ok=True)
                preview_path = preview_directory / f"{pose_id}.png"
                if preview_path.exists():
                    raise FileExistsError(
                        f"pose preview already exists: {preview_path}"
                    )
                request = PoseRecordingRequest(
                    pose_id=pose_id,
                    group=args.group,
                    image_timing=selected[0].timing,
                    visual_report=selected[1],
                    head_witness_ack=True,
                    anchor=anchor_next,
                    preview_path=preview_path,
                    yellow_override_reason=(
                        args.yellow_override_reason
                        if selected[1].grade is QualityGrade.YELLOW
                        else None
                    ),
                )
                now = states.latest.receipt_monotonic_s
                assessment = recorder.assess(request, now_monotonic_s=now)
                if not assessment.allowed:
                    print("pose not saved: " + "; ".join(assessment.failures))
                    continue
                if not cv2.imwrite(str(preview_path), selected[2]):
                    raise RuntimeError(f"failed to write pose preview: {preview_path}")
                updated = recorder.record(request, now_monotonic_s=now)
                if selected[1].signature is not None:
                    history.append(selected[1].signature)
                anchor_next = False
                print(
                    f"saved {pose_id}; total={len(updated.poses)}; "
                    f"hash={updated.content_sha256}"
                )
        return 0
    finally:
        if observer is not None:
            observer.close()
        if camera is not None:
            camera.close()
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


def run_commission_hold(args: argparse.Namespace) -> int:
    _require_ack(args.confirm, MOTION_ACK)
    if args.duration_s <= 0:
        raise ValueError("--duration-s must be positive")
    pose_set = PoseStore(args.pose_set).load()
    _validate_calibration_arm(args.hardware_config, pose_set)
    _validate_collision_preflight(args.collision_config)
    config, rate_hz = _executor_config(args.hardware_config)
    clock = SystemClock()
    with CommandOwnerLock(args.lock_file):
        transport = UnitreeArmSDKTransport(_transport_config(args), clock=clock)
        executor = PoseExecutor(
            transport=transport,
            clock=clock,
            pose_set=pose_set,
            approved_validation_report_sha256="0" * 64,
            config=config,
        )
        try:
            _wait_for_state(transport, 5.0)
            executor.acquire(operator_confirmed=True)
            _drive_direct(executor, ExecutorState.HOLDING, rate_hz=rate_hz)
            deadline = time.monotonic() + args.duration_s
            while time.monotonic() < deadline:
                time.sleep(1.0 / rate_hz)
                executor.tick()
            executor.emergency_stop("commissioning stationary hold complete")
            _drive_direct(executor, ExecutorState.STOPPED, rate_hz=rate_hz)
        except (KeyboardInterrupt, RuntimeError, TypeError, ValueError):
            _drain_emergency(executor, rate_hz=rate_hz)
            raise
    print("measured-pose acquisition/hold completed with terminal weight zero")
    return 0


def run_commission_pose(args: argparse.Namespace) -> int:
    _require_ack(args.confirm, MOTION_ACK)
    if args.home_pose == args.target_pose:
        raise ValueError("home and target pose must differ")
    pose_set = PoseStore(args.pose_set).load()
    _validate_calibration_arm(args.hardware_config, pose_set)
    report = _load_validation_report(args.validation_report)
    _validate_report_binding(pose_set, report)
    _validate_collision_preflight(args.collision_config, report=report)
    # Resolve both directions before any command publisher exists.
    outward = report.approval(args.home_pose, args.target_pose)
    returning = report.approval(args.target_pose, args.home_pose)
    if not outward.passed or not returning.passed:
        raise ValueError("single-pose commissioning requires both edges to pass")
    config, rate_hz = _executor_config(args.hardware_config)
    clock = SystemClock()
    with CommandOwnerLock(args.lock_file):
        transport = UnitreeArmSDKTransport(_transport_config(args), clock=clock)
        executor = PoseExecutor(
            transport=transport,
            clock=clock,
            pose_set=pose_set,
            approved_validation_report_sha256=report.content_sha256,
            config=config,
        )
        try:
            _wait_for_state(transport, 5.0)
            executor.acquire(operator_confirmed=True, initial_pose_id=args.home_pose)
            _drive_direct(executor, ExecutorState.READY, rate_hz=rate_hz)
            executor.start_pose(
                args.target_pose, approval=outward, operator_confirmed=True
            )
            _drive_direct(executor, ExecutorState.READY, rate_hz=rate_hz)
            executor.start_pose(
                args.home_pose, approval=returning, operator_confirmed=True
            )
            _drive_direct(executor, ExecutorState.READY, rate_hz=rate_hz)
            executor.begin_clean_release(
                approved_home_pose_id=args.home_pose,
                operator_confirmed=True,
            )
            _drive_direct(executor, ExecutorState.STOPPED, rate_hz=rate_hz)
        except (KeyboardInterrupt, RuntimeError, TypeError, ValueError):
            _drain_emergency(executor, rate_hz=rate_hz)
            raise
    print("single-pose round trip passed with terminal weight zero")
    return 0


def run_collect_session(args: argparse.Namespace) -> int:
    _require_ack(args.confirm, MOTION_ACK)
    pose_set = PoseStore(args.pose_set).load()
    report = _load_validation_report(args.validation_report)
    _validate_report_binding(pose_set, report)
    _validate_collision_preflight(args.collision_config, report=report)
    plan = _load_session_plan(args.plan_yaml)
    _preflight_plan(plan, report)
    hardware_bytes = args.hardware_config.read_bytes()
    target_bytes = args.target_config.read_bytes()
    validation_bytes = args.validation_report.read_bytes()
    collision_bytes = args.collision_config.read_bytes()
    pose_bytes = args.pose_set.read_bytes()
    recording, pairing, executor_config, rate_hz = _runtime_configs(
        args.hardware_config
    )
    _validate_hardware_preflight(hardware_bytes, pose_set)
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
            transport = UnitreeArmSDKTransport(
                _transport_config(args), on_sample=states.add
            )
            _wait_for_state(transport, 5.0)
            store = SessionStore(args.session_directory)
            store.create(
                session_id=args.session_id,
                created_at_utc=utc_now_iso(),
                camera_info=camera_info,
                pose_set_content_sha256=pose_set.content_sha256,
                artifacts={
                    "pose_set.yaml": pose_bytes,
                    "hardware.yaml": hardware_bytes,
                    "target.json": target_bytes,
                    "collision_pairs.yaml": collision_bytes,
                    "validation_report.json": validation_bytes,
                },
                pairing_config=pairing,
                recording_gate_config=recording,
                provenance={
                    "command": "g1-calib collect-session",
                    "mode_machine": 5,
                    "head_witness_all_acknowledged": all(
                        pose.head_witness_ack for pose in pose_set.poses
                    ),
                    "yellow_override_reason": args.yellow_override_reason,
                    "camera_image_topic": args.image_topic,
                    "camera_info_topic": args.camera_info_topic,
                    "ros_camera_reliability": args.ros_camera_reliability,
                    "network_interface": args.network_interface,
                },
            )
            raw_executor = PoseExecutor(
                transport=transport,
                clock=SystemClock(),
                pose_set=pose_set,
                approved_validation_report_sha256=report.content_sha256,
                config=executor_config,
            )
            synchronized = SynchronizedPoseExecutor(raw_executor)
            driver = ExecutorControlDriver(synchronized, rate_hz=rate_hz)
            driver.start()
            cancelled = [False]

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
                )
                cv2.imshow("G1 calibration collection", rendered)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    cancelled[0] = True

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
                ),
                wait_once=wait_once,
                accept_yellow=lambda _frame: bool(args.yellow_override_reason),
                preview=preview,
                cancelled=lambda: cancelled[0],
            )
            runner = CaptureSessionRunner(executor=synchronized, store=store)

            def confirm_move(source_pose: str, target_pose: str) -> bool:
                if args.auto_confirm_transitions:
                    return True
                response = input(
                    f"Ready for validated move {source_pose} -> {target_pose}. "
                    "Type MOVE: "
                )
                return response == "MOVE"

            orchestrator = ApprovedSessionOrchestrator(
                executor=synchronized,
                validation_report=report,
                capture_runner=runner,
                frame_source=source,
                control_step=lambda: wait_once(1.0 / rate_hz),
                confirm_move=confirm_move,
                plan=plan,
            )
            try:
                orchestrator.run(
                    confirm_acquisition=True,
                    confirm_release=True,
                )
            except KeyboardInterrupt:
                synchronized.emergency_stop("operator interrupted collection")
                _wait_threaded_stop(synchronized, driver, wait_once)
                return 130
            driver.close()
            driver.check()
        print(f"finalized session: {args.session_directory}")
        return 0
    finally:
        if synchronized is not None and synchronized.state is not ExecutorState.STOPPED:
            try:
                synchronized.emergency_stop("collection cleanup")
                deadline = time.monotonic() + executor_config.release_ramp_s + 2.0
                while (
                    synchronized.state is not ExecutorState.STOPPED
                    and time.monotonic() < deadline
                ):
                    if driver is None or not driver.is_alive:
                        synchronized.tick()
                    time.sleep(1.0 / rate_hz)
            except (RuntimeError, TypeError, ValueError):
                pass
        if driver is not None and driver.is_alive:
            driver.close()
        if transport is not None and transport.command_count == 0:
            transport.close()
        if camera is not None:
            camera.close()
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


def _transport_config(args: argparse.Namespace) -> UnitreeTransportConfig:
    return UnitreeTransportConfig(
        network_interface=args.network_interface,
        domain_id=args.domain_id,
    )


def _require_ack(actual: str, expected: str) -> None:
    if actual != expected:
        raise ValueError(f"--confirm must exactly equal: {expected}")


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


def _drive_direct(
    executor: PoseExecutor,
    desired: ExecutorState,
    *,
    rate_hz: float,
) -> None:
    deadline = time.monotonic() + executor.config.motion_timeout_s + 5.0
    while executor.state is not desired:
        if time.monotonic() >= deadline:
            raise RuntimeError(f"timed out waiting for {desired.value}")
        if executor.state is ExecutorState.STOPPED:
            raise RuntimeError(
                "executor stopped: " + (executor.fault_reason or "unknown reason")
            )
        time.sleep(1.0 / rate_hz)
        executor.tick()


def _drain_emergency(executor: PoseExecutor, *, rate_hz: float) -> None:
    if executor.state is ExecutorState.STOPPED:
        return
    executor.emergency_stop("commissioning failed or was interrupted")
    deadline = time.monotonic() + executor.config.release_ramp_s + 2.0
    while executor.state is not ExecutorState.STOPPED and time.monotonic() < deadline:
        time.sleep(1.0 / rate_hz)
        executor.tick()


def _executor_config(path: Path) -> tuple[ExecutorConfig, float]:
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    control = data["control"]
    config = ExecutorConfig(
        maximum_joint_velocity_rad_s=float(control["max_joint_velocity_rad_s"]),
        coarse_arrival_tolerance_rad=float(control["coarse_arrival_tolerance_rad"]),
        settled_position_tolerance_rad=float(control["position_tolerance_rad"]),
        settled_velocity_tolerance_rad_s=float(control["velocity_tolerance_rad_s"]),
        settle_dwell_s=float(control["settle_dwell_s"]),
        state_freshness_timeout_s=float(control["lowstate_timeout_s"]),
        maximum_tick_gap_s=float(control["maximum_control_tick_gap_s"]),
        acquisition_ramp_s=float(control["acquisition_weight_ramp_s"]),
        release_ramp_s=float(control["release_weight_ramp_s"]),
        motion_timeout_s=float(control["motion_timeout_s"]),
    )
    return config, float(control["command_rate_hz"])


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
        maximum_calibration_velocity_rad_s=float(
            recording_data["maximum_calibration_velocity_rad_s"]
        ),
        maximum_calibration_position_spread_rad=float(
            recording_data["maximum_calibration_position_spread_rad"]
        ),
        maximum_hold_velocity_rad_s=float(
            recording_data["maximum_hold_velocity_rad_s"]
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
        "home_pose_id",
        "capture_pose_ids",
    }:
        raise ValueError("session plan fields do not match schema version 1")
    if data["schema_version"] != 1:
        raise ValueError("unsupported session plan schema version")
    return SessionExecutionPlan(
        home_pose_id=str(data["home_pose_id"]),
        capture_pose_ids=tuple(str(item) for item in data["capture_pose_ids"]),
    )


def _preflight_plan(plan: SessionExecutionPlan, report: ValidationReport) -> None:
    route = list(plan.capture_pose_ids)
    if route[0] != plan.home_pose_id:
        route.insert(0, plan.home_pose_id)
    if route[-1] != plan.home_pose_id:
        route.append(plan.home_pose_id)
    for source, target in pairwise(route):
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


def _wait_for_recording_window(
    rclpy,
    node,
    states: StateSampleBuffer,
    image_receipt_s: float,
    stationary_duration_s: float,
) -> None:
    required_receipt = image_receipt_s + stationary_duration_s / 2.0
    deadline = time.monotonic() + stationary_duration_s + 2.0
    while states.latest is None or states.latest.receipt_monotonic_s < required_receipt:
        if time.monotonic() >= deadline:
            raise RuntimeError("timed out waiting for post-image stationary state")
        rclpy.spin_once(node, timeout_sec=0.01)


def _next_pose_id(store: PoseStore, first_pose_id: str) -> str:
    pose_set = store.load()
    if not pose_set.poses:
        return first_pose_id
    used = {pose.id for pose in pose_set.poses}
    index = 1
    while f"pose_{index:03d}" in used:
        index += 1
    return f"pose_{index:03d}"


def _validate_hardware_preflight(hardware_bytes: bytes, pose_set) -> None:
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
    if not mount["mechanically_fixed"] or not mount["witness_marked"]:
        raise ValueError("camera pitch must be mechanically fixed and witness marked")
    if not pose_set.poses:
        raise ValueError("collection requires a non-empty pose set")
    unacknowledged = [pose.id for pose in pose_set.poses if not pose.head_witness_ack]
    if unacknowledged:
        raise ValueError(
            "head witness was not acknowledged for poses: " + ", ".join(unacknowledged)
        )


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


def _wait_threaded_stop(executor, driver, wait_once) -> None:
    deadline = time.monotonic() + 3.0
    while executor.state is not ExecutorState.STOPPED and time.monotonic() < deadline:
        wait_once(0.01)
        driver.check()
