"""Explicitly armed G1 commissioning and rectified-session collection commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from itertools import pairwise
from pathlib import Path

import cv2
import yaml

from aprilcube import CorrespondenceDetector
from g1_aprilcube_calibration.activation_handoff import (
    ActivationHandoff,
    build_activation_handoff,
)
from g1_aprilcube_calibration.clock import SystemClock
from g1_aprilcube_calibration.collision import (
    CollisionConfig,
    FCLCollisionChecker,
)
from g1_aprilcube_calibration.config import QualityThresholds
from g1_aprilcube_calibration.dataset_builder import DatasetBuilder
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
from g1_aprilcube_calibration.pc2_safety import (
    PC2DampingWatchdog,
    PC2SafetyConfig,
)
from g1_aprilcube_calibration.pose_recorder import (
    PoseRecorder,
    PoseRecordingRequest,
)
from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID, PoseSet
from g1_aprilcube_calibration.pose_store import PoseStore, TransientPoseStore
from g1_aprilcube_calibration.pose_validator import (
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
from g1_aprilcube_calibration.urdf_model import URDFModel

WRITE_ACK = "I UNDERSTAND THIS WRITES RT/ARM_SDK"
DAMP_ACK = "DAMP THE LOAD-BEARING-HARNESSED G1 NOW"
MOTION_ACK = (
    "I CONFIRM THE G1 IS SECURED BY THE LOAD-BEARING HARNESS "
    "AND THE WORKSPACE IS CLEAR"
)


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

    teach = subparsers.add_parser(
        "teach-poses",
        help=(
            "record measured poses and synchronized lossless calibration bursts "
            "without commanding the robot"
        ),
    )
    _add_network_arguments(teach)
    teach.add_argument("--session-directory", type=Path, required=True)
    teach.add_argument("--session-id", required=True)
    _add_camera_arguments(teach)
    teach.add_argument("--hardware-config", type=Path, default=default_hardware)
    teach.add_argument("--collision-config", type=Path, default=default_collision)
    teach.add_argument("--target-config", type=Path, default=default_target)
    teach.add_argument("--quality-config", type=Path, default=default_quality)
    teach.add_argument("--camera-timeout-s", type=float, default=10.0)
    teach.add_argument("--burst-timeout-s", type=float, default=15.0)
    teach.add_argument("--dataset-output", type=Path)
    teach.add_argument("--group", default="calibration")
    teach.add_argument("--first-pose-id", default="pose_001")
    teach.add_argument("--preview-directory", type=Path)
    teach.add_argument("--yellow-override-reason")
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
    collect.add_argument("--burst-timeout-s", type=float, default=15.0)
    collect.add_argument("--yellow-override-reason")
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
    watchdog = _pc2_damping_watchdog(args, args.hardware_config)
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


def run_teach_poses(args: argparse.Namespace) -> int:
    hardware_bytes = args.hardware_config.read_bytes()
    target_bytes = args.target_config.read_bytes()
    collision_bytes = args.collision_config.read_bytes()
    quality_bytes = args.quality_config.read_bytes()
    pose_set = _manual_pose_set_for_session(args)
    _validate_calibration_arm(args.hardware_config, pose_set)
    _validate_collision_preflight(args.collision_config)
    _validate_hardware_preflight(hardware_bytes, pose_set, require_poses=False)
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
        _wait_for_camera(rclpy, node, camera, args.camera_timeout_s)
        camera_info = camera.frames.latest.camera_info
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
        pose_store = TransientPoseStore(
            PoseStore(args.session_directory / "pose_set.yaml").load()
        )
        detector = CorrespondenceDetector(args.target_config)
        evaluator = PoseQualityEvaluator(thresholds)
        recorder = PoseRecorder(
            store=pose_store,
            state_buffer=states,
            gate_config=recording,
            pairing_config=pairing,
        )
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
                maximum_inter_frame_gap_s=(
                    thresholds.stationary_burst_maximum_inter_frame_gap_s
                ),
                maximum_duration_s=(
                    thresholds.stationary_burst_maximum_duration_s
                ),
            ),
            wait_once=lambda duration: _spin_and_wait(rclpy, node, duration),
            accept_yellow=lambda _frame: bool(args.yellow_override_reason),
            history=tuple(history),
        )
        anchor_next = not pose_set.poses
        latest_evaluation = None
        last_frame_key = None
        finalize_requested = False
        print(
            "manual calibration capture is read-only: S=save burst, "
            "A=toggle anchor, U=undo, P/Esc=pause, Q=finalize"
        )
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
                        f"next={_next_pose_id(pose_store, args.first_pose_id)} "
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
            if key in (ord("q"), ord("Q")):
                if not pose_store.load().poses:
                    print("cannot finalize an empty manual calibration session")
                    continue
                finalize_requested = True
                break
            if key in (ord("p"), ord("P"), 27):
                break
            if key in (ord("a"), ord("A")):
                anchor_next = not anchor_next
                print(f"next pose anchor={anchor_next}")
            elif key in (ord("u"), ord("U")):
                current = pose_store.load()
                if not current.poses:
                    print("nothing to undo")
                    continue
                pose_id = current.poses[-1].id
                capture = _latest_accepted_capture_for_pose(
                    session_store, pose_id
                )
                updated = pose_store.undo_last(reason="operator live undo")
                try:
                    session_store.undo_manual_capture(
                        capture_id=capture.capture_id,
                        updated_pose_set=updated,
                        reason="operator live undo",
                    )
                except Exception:
                    pose_store.initialize(current, overwrite=True)
                    raise
                if history:
                    history.pop()
                    capture_source.undo_last_signature()
                print(f"undid final pose; {len(updated.poses)} remain")
            elif key in (ord("s"), ord("S")) and latest_evaluation is not None:
                quality = latest_evaluation[1]
                if quality.grade is QualityGrade.RED:
                    print("pose not saved: visual quality is red")
                    continue
                if (
                    quality.grade is QualityGrade.YELLOW
                    and not args.yellow_override_reason
                ):
                    print("pose not saved: yellow requires --yellow-override-reason")
                    continue
                pose_id = _next_pose_id(pose_store, args.first_pose_id)
                capture_id = _next_manual_capture_id(session_store)
                print(
                    f"collecting {thresholds.stationary_burst_frames} "
                    f"lossless stationary frames for {pose_id}..."
                )
                try:
                    burst = capture_source.capture_burst(
                        pose_id=pose_id,
                        capture_id=capture_id,
                    )
                except RuntimeError as error:
                    print(f"pose not saved: {error}")
                    continue
                selected = SessionStore.select_medoid_frame(burst)
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
                        args.yellow_override_reason
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
                if selected.quality.signature is not None:
                    history.append(selected.quality.signature)
                anchor_next = False
                print(
                    f"saved {pose_id} with {len(burst)} raw frames; "
                    f"total={len(updated.poses)}; hash={updated.content_sha256}"
                )
        if finalize_requested:
            session_store.validate_manual_alignment(pose_store.load())
            manifest = session_store.finalize()
            dataset_path = args.dataset_output or (
                args.session_directory / "dataset.json"
            )
            dataset = DatasetBuilder(args.session_directory).build(
                output_path=dataset_path
            )
            print(
                f"finalized manual session {manifest.session_id}: "
                f"{len(dataset.samples)} calibration samples; "
                f"dataset={dataset_path}"
            )
        else:
            print(
                f"paused unfinalized manual session: {args.session_directory}; "
                "rerun the same command to resume"
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
                    "yellow_override_reason": args.yellow_override_reason,
                    "camera_image_topic": args.image_topic,
                    "camera_info_topic": args.camera_info_topic,
                    "ros_camera_reliability": args.ros_camera_reliability,
                    "network_interface": args.network_interface,
                    "source_pose_set_sha256": (
                        activation.source_pose_set_sha256
                    ),
                    "policy_validation_report_sha256": (
                        policy_report.content_sha256
                    ),
                    "dynamic_handoff_state": (
                        activation.reference_state.to_dict()
                    ),
                    "dynamic_handoff_readiness": (
                        activation.readiness.to_dict()
                    ),
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
            )
            synchronized = SynchronizedPoseExecutor(raw_executor)
            driver = ExecutorControlDriver(
                synchronized,
                rate_hz=rate_hz,
                safety_heartbeat=watchdog.pulse,
            )
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
                    maximum_inter_frame_gap_s=(
                        thresholds.stationary_burst_maximum_inter_frame_gap_s
                    ),
                    maximum_duration_s=(
                        thresholds.stationary_burst_maximum_duration_s
                    ),
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


def _wait_for_activation_handoff(
    observer: UnitreeLowStateObserver,
    states: StateSampleBuffer,
    pose_set,
    recording: RecordingGateConfig,
    *,
    timeout_s: float = 5.0,
) -> ActivationHandoff:
    """Select a stationary measured handoff while no command publisher exists."""

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


def _unique_route_edges(route: list[str] | tuple[str, ...]) -> tuple[tuple[str, str], ...]:
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
    configured = Path(data["robot"]["urdf"])
    if configured.is_absolute():
        return configured
    return hardware_config.resolve().parent.parent / configured


def _runtime_validation_report(
    *,
    pose_set,
    reference_full_q,
    directed_edges: tuple[tuple[str, str], ...],
    hardware_config: Path,
    collision_config: CollisionConfig,
    policy_config: dict | None = None,
) -> ValidationReport:
    if not directed_edges:
        raise ValueError("runtime validation requires at least one directed edge")
    model = URDFModel(_configured_urdf(hardware_config))
    validator = PosePathValidator(
        model=model,
        collision_checker=FCLCollisionChecker(model, collision_config),
        config=(
            PathValidationConfig()
            if policy_config is None
            else PathValidationConfig(**policy_config)
        ),
    )
    report = validator.validate(
        pose_set,
        directed_edges=directed_edges,
        reference_full_q=reference_full_q,
    )
    if not report.passed:
        failures = [
            f"{edge.from_pose_id}->{edge.to_pose_id}: "
            + "; ".join(edge.failures)
            for edge in report.edges
            if not edge.passed
        ]
        raise ValueError("dynamic route validation failed: " + " | ".join(failures))
    return report


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
                "executor faulted: "
                f"{executor.fault_reason or 'unknown reason'}; "
                "PC2 heartbeat intentionally stopped"
            )
        if state in {ExecutorState.MOVING, ExecutorState.SETTLING}:
            now = time.monotonic()
            if now - last_motion_status_s >= 1.0:
                print(executor.motion_diagnostic())
                last_motion_status_s = now
        if safety_heartbeat is not None:
            safety_heartbeat()


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


def _pc2_damping_watchdog(
    args: argparse.Namespace,
    hardware_config: Path,
) -> PC2DampingWatchdog:
    with hardware_config.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    safety = data.get("safety")
    if not isinstance(safety, dict):
        raise TypeError("hardware config is missing the PC2 damping safety section")
    if safety.get("emergency_policy") != "pc2_g1_loco_damp":
        raise ValueError("hardware emergency policy must be pc2_g1_loco_damp")
    if safety.get("physical_support") != "load_bearing_harness":
        raise ValueError("whole-body damping requires the load-bearing harness")
    return PC2DampingWatchdog(
        PC2SafetyConfig(
            host=args.pc2_host,
            ssh_identity=args.pc2_ssh_identity,
            heartbeat_interval_s=float(safety["heartbeat_interval_s"]),
            heartbeat_timeout_s=float(safety["heartbeat_timeout_s"]),
            connect_timeout_s=float(safety["connect_timeout_s"]),
            client_timeout_s=float(safety["loco_client_timeout_s"]),
            remote_ros_setup=Path(safety["pc2_ros_setup"]),
            remote_cyclonedds_setup=Path(safety["pc2_cyclonedds_setup"]),
            remote_unitree_setup=Path(safety["pc2_unitree_ros2_setup"]),
            remote_cyclonedds_uri=Path(safety["pc2_cyclonedds_uri"]),
            remote_damp_executable=Path(safety["pc2_g1_loco_client"]),
        )
    )


def _executor_config(path: Path) -> tuple[ExecutorConfig, float]:
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    control = data["control"]
    config = ExecutorConfig(
        maximum_joint_velocity_rad_s=float(control["max_joint_velocity_rad_s"]),
        coarse_arrival_tolerance_rad=float(control["coarse_arrival_tolerance_rad"]),
        target_position_tolerance_rad=float(
            control["target_position_tolerance_rad"]
        ),
        activation_position_tolerance_rad=float(
            control["activation_position_tolerance_rad"]
        ),
        held_arm_position_tolerance_rad=float(
            control["held_arm_position_tolerance_rad"]
        ),
        settled_position_spread_rad=float(
            control["settled_position_spread_rad"]
        ),
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


def _latest_accepted_capture_for_pose(store: SessionStore, pose_id: str):
    matches = [
        capture
        for capture in store.load().captures
        if capture.pose_id == pose_id and capture.outcome == "accepted"
    ]
    if not matches:
        raise ValueError(f"pose {pose_id} has no accepted manual capture to undo")
    return matches[-1]


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


def _spin_and_wait(rclpy, node, duration_s: float) -> None:
    rclpy.spin_once(node, timeout_sec=0.0)
    time.sleep(duration_s)


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
                "yellow_override_reason": args.yellow_override_reason,
                "arm_command_publisher_created": False,
            },
            collection_method="manual_teaching",
        )
        return store

    if not store.manifest_path.is_file():
        raise ValueError(
            "session directory exists without a manifest: "
            f"{args.session_directory}"
        )
    manifest = store.load()
    if manifest.finalized:
        raise RuntimeError("manual calibration session is already finalized")
    if manifest.session_id != args.session_id:
        raise ValueError("--session-id does not match the resumable session")
    if manifest.provenance.get("collection_method") != "manual_teaching":
        raise ValueError("existing session was not created by manual teaching")
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
                "session directory exists without a manifest: "
                f"{session_directory}"
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
    if require_poses and not pose_set.poses:
        raise ValueError("collection requires a non-empty pose set")


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
