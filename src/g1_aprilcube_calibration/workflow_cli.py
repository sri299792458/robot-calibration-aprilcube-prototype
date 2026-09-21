"""Offline workflow and subscriber-only hardware inspection CLI commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import yaml

from aprilcube import CorrespondenceDetector
from g1_aprilcube_calibration.anchor_stability import (
    AnchorStabilityConfig,
    analyze_dataset_anchors,
)
from g1_aprilcube_calibration.authored_collection import (
    modeled_hand_T_target_from_hardware,
    validate_hardware_target_profile,
)
from g1_aprilcube_calibration.auto_collection_designer import (
    AutoCollectionDesignConfig,
    design_auto_collection,
    load_design_inputs,
)
from g1_aprilcube_calibration.calibration_bundle import CalibrationBundle
from g1_aprilcube_calibration.calibration_pipeline import (
    CalibrationPipeline,
    PipelineConfig,
)
from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.collision import CollisionConfig, FCLCollisionChecker
from g1_aprilcube_calibration.dataset_builder import CalibrationDataset, DatasetBuilder
from g1_aprilcube_calibration.joint_map import arm_joint_names
from g1_aprilcube_calibration.models import RobotStateSample
from g1_aprilcube_calibration.pose_store import PoseStore
from g1_aprilcube_calibration.pose_validator import (
    PathValidationConfig,
    PosePathValidator,
)
from g1_aprilcube_calibration.replay_targets import back_off_replay_targets
from g1_aprilcube_calibration.residual_report import (
    load_exported_result,
    validate_exported_result,
)
from g1_aprilcube_calibration.session_store import SessionStore
from g1_aprilcube_calibration.synthetic import (
    make_synthetic_dataset,
    perturbed_initial_transforms,
)
from g1_aprilcube_calibration.table_accuracy import (
    CharucoBoardPoseDetector,
    CharucoBoardSpec,
    camera_info_from_dataset,
    create_evaluation_document,
    create_plan_document,
    load_plan_document,
    observe_burst,
)
from g1_aprilcube_calibration.transforms import transform_to_pose_vector
from g1_aprilcube_calibration.transports.unitree_arm_sdk import (
    UnitreeLowStateObserver,
    UnitreeTransportConfig,
)
from g1_aprilcube_calibration.urdf_model import URDFModel


def add_workflow_subparsers(
    subparsers: argparse._SubParsersAction,
    *,
    workspace_root: Path,
    default_target: Path,
) -> None:
    default_urdf = (
        workspace_root / "config" / "urdf" / "g1_29dof_rev_1_0_g1pilot_collision.urdf"
    )
    default_collision = workspace_root / "config" / "collision_pairs_dex3_aruco.yaml"
    default_hardware = workspace_root / "config" / "hardware_dex3_aruco.yaml"

    inspect = subparsers.add_parser(
        "inspect-artifacts",
        help="verify pinned URDF, hand target, and hardware readiness offline",
    )
    inspect.add_argument("--urdf", type=Path, default=default_urdf)
    inspect.add_argument("--target-config", type=Path, default=default_target)
    inspect.add_argument("--collision-config", type=Path, default=default_collision)
    inspect.set_defaults(handler=run_inspect_artifacts)

    bundle = subparsers.add_parser(
        "inspect-calibration-bundle",
        help=(
            "validate a versioned calibration overlay and optionally materialize "
            "its calibrated URDF without changing the base model"
        ),
    )
    bundle.add_argument("--calibration-bundle", type=Path, required=True)
    bundle.add_argument("--urdf", type=Path, default=default_urdf)
    bundle.add_argument(
        "--output-urdf",
        type=Path,
        help="optional new path for the materialized calibrated URDF",
    )
    bundle.set_defaults(handler=run_inspect_calibration_bundle)

    hardware = subparsers.add_parser(
        "inspect-hardware",
        help="subscribe to rt/lowstate without creating a command publisher",
    )
    hardware.add_argument("--network-interface", required=True)
    hardware.add_argument("--domain-id", type=int, default=0)
    hardware.add_argument("--timeout-s", type=float, default=5.0)
    hardware.add_argument("--state-json", type=Path)
    hardware.set_defaults(handler=run_inspect_hardware)

    summary = subparsers.add_parser(
        "pose-summary", help="validate and summarize a content-hashed pose set"
    )
    summary.add_argument("--pose-set", type=Path, required=True)
    summary.set_defaults(handler=run_pose_summary)

    undo = subparsers.add_parser(
        "undo-pose", help="atomically undo the final manually recorded pose"
    )
    undo.add_argument("--pose-set", type=Path, required=True)
    undo.add_argument("--reason", required=True)
    undo.set_defaults(handler=run_undo_pose)

    prepare_replay = subparsers.add_parser(
        "prepare-replay",
        help="derive joint-limit-safe replay targets and a sequential pose route",
    )
    prepare_replay.add_argument("--pose-set", type=Path, required=True)
    prepare_replay.add_argument("--output-directory", type=Path, required=True)
    prepare_replay.add_argument("--urdf", type=Path, default=default_urdf)
    prepare_replay.add_argument(
        "--joint-limit-margin-rad",
        type=float,
        default=PathValidationConfig().joint_limit_margin_rad,
    )
    prepare_replay.set_defaults(handler=run_prepare_replay)

    validate = subparsers.add_parser(
        "validate-poses",
        help="validate explicit directed pose transitions against URDF and FCL",
    )
    validate.add_argument("--pose-set", type=Path, required=True)
    validate.add_argument("--reference-state-json", type=Path, required=True)
    validate.add_argument("--edges-yaml", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--urdf", type=Path, default=default_urdf)
    validate.add_argument("--collision-config", type=Path, default=default_collision)
    validate.add_argument(
        "--joint-limit-margin-rad",
        type=float,
        default=PathValidationConfig().joint_limit_margin_rad,
    )
    validate.set_defaults(handler=run_validate_poses)

    auto_design = subparsers.add_parser(
        "design-auto-collection",
        help=(
            "generate camera-frustum hand-target poses, solve IK, and build a "
            "reversible collision-validated standing route"
        ),
    )
    auto_design.add_argument("--reference-state-json", type=Path, required=True)
    auto_design.add_argument("--output-directory", type=Path, required=True)
    auto_design.add_argument(
        "--calibration-result",
        type=Path,
        required=True,
        help="validated calibration result supplying the camera-frustum pose",
    )
    auto_design.add_argument("--hardware-config", type=Path, default=default_hardware)
    auto_design.add_argument("--target-config", type=Path, default=default_target)
    auto_design.add_argument("--urdf", type=Path, default=default_urdf)
    auto_design.add_argument("--collision-config", type=Path, default=default_collision)
    auto_design.add_argument("--target-count", type=int, default=80)
    auto_design.add_argument("--candidate-count", type=int, default=1600)
    auto_design.add_argument(
        "--workers",
        type=int,
        help="IK/FCL worker processes; default: auto, capped at 8",
    )
    auto_design.add_argument("--minimum-depth-m", type=float, default=0.25)
    auto_design.add_argument("--maximum-depth-m", type=float, default=0.60)
    auto_design.add_argument("--image-margin-px", type=float, default=60.0)
    auto_design.add_argument(
        "--minimum-projected-target-span-px", type=float, default=45.0
    )
    auto_design.add_argument("--maximum-view-obliquity-deg", type=float, default=55.0)
    auto_design.add_argument("--maximum-view-roll-deg", type=float, default=35.0)
    auto_design.add_argument("--ik-restarts", type=int, default=4)
    auto_design.add_argument("--seed", type=int, default=17)
    auto_design.add_argument(
        "--joint-limit-margin-rad",
        type=float,
        default=PathValidationConfig().joint_limit_margin_rad,
    )
    auto_design.set_defaults(handler=run_design_auto_collection)

    session = subparsers.add_parser(
        "session-summary", help="verify and summarize an immutable raw session"
    )
    session.add_argument("--session", type=Path, required=True)
    session.set_defaults(handler=run_session_summary)

    dataset = subparsers.add_parser(
        "build-dataset",
        help="verify a finalized raw session and build solver input",
    )
    dataset.add_argument("--session", type=Path, required=True)
    dataset.add_argument("--output", type=Path, required=True)
    dataset.add_argument(
        "--observation-phase",
        choices=("held", "supported"),
        default="held",
    )
    dataset.add_argument("--allow-unfinalized", action="store_true")
    dataset.set_defaults(handler=run_build_dataset)

    solve = subparsers.add_parser(
        "solve", help="solve camera extrinsics and export residual reports"
    )
    solve.add_argument("--dataset", type=Path, required=True)
    solve.add_argument("--output-directory", type=Path, required=True)
    solve.add_argument("--urdf", type=Path, default=default_urdf)
    solve.add_argument("--hardware-config", type=Path, default=default_hardware)
    solve.add_argument("--target-config", type=Path, default=default_target)
    solve.add_argument("--holdout-fraction", type=float, default=0.2)
    solve.add_argument("--bootstrap-trials", type=int, default=50)
    solve.add_argument("--bootstrap-seed", type=int, default=17)
    solve.add_argument(
        "--target-transform-mode",
        choices=("fixed", "optimize"),
        default="fixed",
        help=(
            "hold the configured palm-to-marker transform fixed, or jointly "
            "optimize it with the camera transform"
        ),
    )
    solve.add_argument(
        "--free-joint-offset",
        action="append",
        default=[],
        help="arm joint offset to fit; repeat for a staged kinematic model",
    )
    solve.add_argument(
        "--all-arm-joint-offsets",
        action="store_true",
        help="fit all seven calibration-arm offsets with regularized priors",
    )
    solve.add_argument("--joint-offset-prior-sigma-deg", type=float, default=5.0)
    solve.add_argument(
        "--robot-calibration-directory",
        type=Path,
        default=workspace_root / "robot_calibration",
    )
    solve.add_argument(
        "--native-runner",
        type=Path,
        default=workspace_root / "tools" / "g1_robot_calibration.sh",
    )
    solve.add_argument("--native-timeout-s", type=float, default=300.0)
    solve.set_defaults(handler=run_solve)

    synthetic = subparsers.add_parser(
        "synthetic-check",
        help="generate, solve, and report a known-truth calibration without hardware",
    )
    synthetic.add_argument("--output-directory", type=Path, required=True)
    synthetic.add_argument("--urdf", type=Path, default=default_urdf)
    synthetic.add_argument("--target-config", type=Path, default=default_target)
    synthetic.add_argument("--pose-count", type=int, default=40)
    synthetic.add_argument("--pixel-noise", type=float, default=0.3)
    synthetic.add_argument("--seed", type=int, default=7)
    synthetic.add_argument("--bootstrap-trials", type=int, default=10)
    synthetic.add_argument(
        "--calibration-arm", choices=("left", "right"), default="left"
    )
    synthetic.add_argument(
        "--robot-calibration-directory",
        type=Path,
        default=workspace_root / "robot_calibration",
    )
    synthetic.add_argument(
        "--native-runner",
        type=Path,
        default=workspace_root / "tools" / "g1_robot_calibration.sh",
    )
    synthetic.add_argument("--native-timeout-s", type=float, default=300.0)
    synthetic.set_defaults(handler=run_synthetic_check)

    anchor = subparsers.add_parser(
        "anchor-stability",
        help="test repeated anchor captures after compensating measured joint FK",
    )
    anchor.add_argument("--dataset", type=Path, required=True)
    anchor.add_argument("--result-json", type=Path, required=True)
    anchor.add_argument("--urdf", type=Path, default=default_urdf)
    anchor.add_argument("--pose-id", action="append", default=[])
    anchor.add_argument("--minimum-captures", type=int, default=3)
    anchor.add_argument("--maximum-translation-mm", type=float, default=2.0)
    anchor.add_argument("--maximum-rotation-deg", type=float, default=0.5)
    anchor.add_argument("--output", type=Path)
    anchor.set_defaults(handler=run_anchor_stability)

    table_plan = subparsers.add_parser(
        "plan-table-accuracy",
        help="derive an offline hand target above a fixed ChArUco table board",
    )
    table_plan.add_argument("--dataset", type=Path, required=True)
    table_plan.add_argument("--result-json", type=Path, required=True)
    table_plan_images = table_plan.add_mutually_exclusive_group(required=True)
    table_plan_images.add_argument(
        "--image",
        type=Path,
        action="append",
        help="planning burst image; repeat for at least three distinct frames",
    )
    table_plan_images.add_argument(
        "--image-directory",
        type=Path,
        help="directory produced by capture-table-images",
    )
    table_plan.add_argument("--output", type=Path, required=True)
    table_plan.add_argument("--hand-cube-config", type=Path, default=default_target)
    table_plan.add_argument(
        "--lift-mm",
        type=float,
        default=100.0,
        help="vertical lift relative to the initially observed supported cube pose",
    )
    table_plan.add_argument("--board-x-mm", type=float)
    table_plan.add_argument("--board-y-mm", type=float)
    table_plan.add_argument("--minimum-frames", type=int, default=3)
    table_plan.add_argument(
        "--legacy-charuco-pattern",
        action="store_true",
        help="only for a board printed with OpenCV's pre-4.6 legacy layout",
    )
    table_plan.set_defaults(handler=run_plan_table_accuracy)

    table_evaluate = subparsers.add_parser(
        "evaluate-table-accuracy",
        help="measure achieved hand-cube error relative to the table board",
    )
    table_evaluate.add_argument("--plan", type=Path, required=True)
    table_evaluate_images = table_evaluate.add_mutually_exclusive_group(required=True)
    table_evaluate_images.add_argument(
        "--image",
        type=Path,
        action="append",
        help="achieved burst image; repeat for at least three distinct frames",
    )
    table_evaluate_images.add_argument(
        "--image-directory",
        type=Path,
        help="directory produced by capture-table-images",
    )
    table_evaluate.add_argument("--output", type=Path, required=True)
    table_evaluate.add_argument("--minimum-frames", type=int, default=3)
    table_evaluate.set_defaults(handler=run_evaluate_table_accuracy)


def run_inspect_artifacts(args: argparse.Namespace) -> int:
    model = URDFModel(args.urdf)
    detector = CorrespondenceDetector(args.target_config)
    collision = CollisionConfig.from_yaml(args.collision_config)
    _print_json(
        {
            "urdf": str(model.path),
            "urdf_sha256": model.sha256,
            "robot_name": model.name,
            "target_config": str(args.target_config.resolve()),
            "target_tag_ids": sorted(detector.tag_corner_map),
            "collision_config_sha256": collision.content_sha256,
            "hardware_ready": collision.hardware_ready,
            "blocking_reasons": list(collision.blocking_reasons),
        }
    )
    return 0 if collision.hardware_ready else 1


def run_inspect_calibration_bundle(args: argparse.Namespace) -> int:
    bundle_path = args.calibration_bundle.resolve()
    bundle = CalibrationBundle.load(bundle_path)
    actual_urdf_sha256 = hashlib.sha256(args.urdf.read_bytes()).hexdigest()
    if actual_urdf_sha256 != bundle.base_urdf_sha256:
        raise ValueError(
            "calibration bundle belongs to a different base URDF: "
            f"expected={bundle.base_urdf_sha256}, actual={actual_urdf_sha256}"
        )
    output = None
    output_sha256 = None
    if args.output_urdf is not None:
        output = args.output_urdf.resolve()
        if output.exists():
            raise FileExistsError(f"calibrated URDF already exists: {output}")
        bundle.materialize_urdf(args.urdf, output)
        output_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
        URDFModel(output)
    _print_json(
        {
            "bundle_path": str(bundle_path),
            "bundle_id": bundle.bundle_id,
            "bundle_sha256": bundle.content_sha256,
            "base_urdf_sha256": bundle.base_urdf_sha256,
            "camera_parent": "torso_link",
            "camera_child": "camera_color_optical_frame",
            "torso_T_camera": bundle.torso_T_camera.tolist(),
            "joint_position_offsets_rad": bundle.joint_position_offsets_rad,
            "target_sides": sorted(bundle.targets),
            "materialized_urdf": None if output is None else str(output),
            "materialized_urdf_sha256": output_sha256,
            "commands_robot": False,
        }
    )
    return 0


def run_inspect_hardware(args: argparse.Namespace) -> int:
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be positive")
    observer = UnitreeLowStateObserver(
        UnitreeTransportConfig(
            network_interface=args.network_interface,
            domain_id=args.domain_id,
        )
    )
    try:
        deadline = time.monotonic() + args.timeout_s
        while True:
            try:
                sample = observer.observe()
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "timed out waiting for a valid complete rt/lowstate sample"
                    ) from None
                time.sleep(0.02)
        result = sample.to_dict()
        result["is_mode5"] = sample.is_mode5
        result["left_q"] = sample.left_q.tolist()
        result["right_q"] = sample.right_q.tolist()
        _print_json(result)
        if args.state_json is not None:
            _write_json(args.state_json, sample.to_dict())
        return 0 if sample.is_mode5 else 1
    finally:
        observer.close()


def run_pose_summary(args: argparse.Namespace) -> int:
    pose_set = PoseStore(args.pose_set).load()
    _print_json(
        {
            "path": str(args.pose_set.resolve()),
            "pose_count": len(pose_set.poses),
            "pose_ids": [pose.id for pose in pose_set.poses],
            "anchor_pose_ids": [pose.id for pose in pose_set.poses if pose.anchor],
            "content_sha256": pose_set.content_sha256,
            "urdf_sha256": pose_set.urdf_sha256,
            "calibration_arm": pose_set.calibration_arm,
        }
    )
    return 0


def run_undo_pose(args: argparse.Namespace) -> int:
    updated = PoseStore(args.pose_set).undo_last(reason=args.reason)
    _print_json(
        {
            "pose_count": len(updated.poses),
            "pose_ids": [pose.id for pose in updated.poses],
            "content_sha256": updated.content_sha256,
        }
    )
    return 0


def run_prepare_replay(args: argparse.Namespace) -> int:
    source_path = args.pose_set.resolve()
    source = PoseStore(source_path).load()
    model = URDFModel(args.urdf)
    derived, adjustments = back_off_replay_targets(
        source,
        model,
        joint_limit_margin_rad=args.joint_limit_margin_rad,
    )
    if not derived.poses:
        raise ValueError("cannot prepare replay artifacts from an empty pose set")

    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pose_path = output / "pose_set.yaml"
    edges_path = output / "edges.yaml"
    plan_path = output / "session_plan.yaml"
    report_path = output / "adjustments.json"
    pose_ids = [pose.id for pose in derived.poses]
    edges = [list(pair) for pair in pairwise(pose_ids)]
    PoseStore(pose_path).initialize(derived)
    _write_yaml(edges_path, {"schema_version": 1, "edges": edges})
    _write_yaml(
        plan_path,
        {"schema_version": 2, "capture_pose_ids": pose_ids},
    )
    adjusted_pose_ids = tuple(dict.fromkeys(item.pose_id for item in adjustments))
    maximum_delta = max((abs(item.delta_rad) for item in adjustments), default=0.0)
    report = {
        "schema_version": 1,
        "source_pose_set": str(source_path),
        "source_pose_set_sha256": source.content_sha256,
        "derived_pose_set_sha256": derived.content_sha256,
        "urdf_sha256": model.sha256,
        "joint_limit_margin_rad": args.joint_limit_margin_rad,
        "pose_count": len(derived.poses),
        "edge_count": len(edges),
        "adjusted_pose_count": len(adjusted_pose_ids),
        "adjusted_joint_target_count": len(adjustments),
        "maximum_absolute_delta_rad": maximum_delta,
        "maximum_absolute_delta_deg": float(np.degrees(maximum_delta)),
        "adjustments": [item.to_dict() for item in adjustments],
        "outputs": {
            "pose_set": str(pose_path),
            "edges": str(edges_path),
            "session_plan": str(plan_path),
            "adjustments": str(report_path),
        },
    }
    _write_json(report_path, report)
    _print_json(report)
    return 0


def run_validate_poses(args: argparse.Namespace) -> int:
    pose_set = PoseStore(args.pose_set).load()
    state = _load_state(args.reference_state_json)
    edges = _load_edges(args.edges_yaml)
    model = URDFModel(args.urdf)
    collision = FCLCollisionChecker(
        model, CollisionConfig.from_yaml(args.collision_config)
    )
    report = PosePathValidator(
        model=model,
        collision_checker=collision,
        config=PathValidationConfig(
            joint_limit_margin_rad=args.joint_limit_margin_rad,
        ),
    ).validate(
        pose_set,
        directed_edges=edges,
        reference_full_q=state.position,
    )
    report.write_json(args.output)
    _print_json(
        {
            "passed": report.passed,
            "edge_count": len(report.edges),
            "failed_edges": [
                f"{edge.from_pose_id}->{edge.to_pose_id}"
                for edge in report.edges
                if not edge.passed
            ],
            "content_sha256": report.content_sha256,
            "output": str(args.output.resolve()),
        }
    )
    return 0 if report.passed else 1


def run_design_auto_collection(args: argparse.Namespace) -> int:
    output = args.output_directory.resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    hardware, target, reference_state = load_design_inputs(
        hardware_path=args.hardware_config,
        target_path=args.target_config,
        state_path=args.reference_state_json,
    )
    model = URDFModel(args.urdf)
    calibration_path = args.calibration_result.resolve()
    calibration_result = load_exported_result(calibration_path)
    torso_T_camera = np.asarray(
        calibration_result["solution"]["torso_T_camera"], dtype=np.float64
    )
    torso_T_camera_source = {
        "kind": "calibration_result",
        "path": str(calibration_path),
        "content_sha256": calibration_result["content_sha256"],
        "dataset_sha256": calibration_result["dataset_sha256"],
    }
    collision = CollisionConfig.from_yaml(args.collision_config)
    if not collision.hardware_ready:
        raise ValueError(
            "collision configuration is not hardware-ready: "
            + ", ".join(collision.blocking_reasons)
        )
    design = AutoCollectionDesignConfig(
        target_count=args.target_count,
        candidate_count=args.candidate_count,
        minimum_depth_m=args.minimum_depth_m,
        maximum_depth_m=args.maximum_depth_m,
        image_margin_px=args.image_margin_px,
        minimum_projected_target_span_px=args.minimum_projected_target_span_px,
        maximum_view_obliquity_deg=args.maximum_view_obliquity_deg,
        maximum_view_roll_deg=args.maximum_view_roll_deg,
        seed=args.seed,
        ik_restart_count=args.ik_restarts,
    )
    path = PathValidationConfig(
        joint_limit_margin_rad=args.joint_limit_margin_rad,
        # Match the pinned G1Pilot controller's selected-pair threshold.
        minimum_collision_clearance_m=0.01,
    )
    result = design_auto_collection(
        model=model,
        collision_config=collision,
        hardware=hardware,
        target_config=target,
        reference_state=reference_state,
        torso_T_camera=torso_T_camera,
        torso_T_camera_source=torso_T_camera_source,
        design_config=design,
        path_config=path,
        progress=lambda message: print(message, flush=True),
        parallel_workers=args.workers,
    )
    output.mkdir(parents=True)
    _write_json(output / "authored_plan.json", result.plan.to_dict())
    _write_json(output / "validation_report.json", result.validation_report.to_dict())
    selection = dict(result.plan.generation_config["selection"])
    selection.pop("selection_steps", None)
    summary = {
        "schema_version": 1,
        "plan_content_sha256": result.plan.content_sha256,
        "validation_report_sha256": result.validation_report.content_sha256,
        "target_count": len(result.plan.targets),
        "route_step_count": len(result.plan.route_pose_ids) - 1,
        "validated_directed_edge_count": len(result.validation_report.edges),
        "attempted_camera_targets": result.attempted_camera_targets,
        "ik_failures": result.ik_failures,
        "endpoint_failures": result.endpoint_failures,
        "parallel_worker_count": result.parallel_worker_count,
        "calibration_result_sha256": calibration_result["content_sha256"],
        "selection": selection,
    }
    _write_json(output / "design_summary.json", summary)
    _print_json({"output_directory": str(output), **summary})
    return 0


def run_session_summary(args: argparse.Namespace) -> int:
    store = SessionStore(args.session)
    manifest = store.load()
    store.verify_artifacts()
    orphans = store.find_orphans()
    _print_json(
        {
            "session_id": manifest.session_id,
            "finalized": manifest.finalized,
            "capture_count": len(manifest.captures),
            "accepted_count": sum(
                capture.outcome == "accepted" for capture in manifest.captures
            ),
            "outcomes": {
                outcome: sum(
                    capture.outcome == outcome for capture in manifest.captures
                )
                for outcome in sorted(
                    {capture.outcome for capture in manifest.captures}
                )
            },
            "orphans": list(orphans),
            "content_sha256": manifest.content_sha256,
        }
    )
    return 0 if not orphans else 1


def run_build_dataset(args: argparse.Namespace) -> int:
    dataset = DatasetBuilder(args.session).build(
        output_path=args.output,
        require_finalized=not args.allow_unfinalized,
        observation_phase=args.observation_phase,
    )
    _print_json(
        {
            "session_id": dataset.session_id,
            "sample_count": len(dataset.samples),
            "observation_phase": dataset.observation_phase,
            "content_sha256": dataset.content_sha256,
            "output": str(args.output.resolve()),
        }
    )
    return 0


def run_solve(args: argparse.Namespace) -> int:
    dataset = CalibrationDataset.from_json(args.dataset)
    hardware, initial_target = _validated_modeled_target_initialization(
        dataset,
        hardware_path=args.hardware_config,
        target_path=args.target_config,
    )
    if args.all_arm_joint_offsets and args.free_joint_offset:
        raise ValueError(
            "use either --all-arm-joint-offsets or repeated --free-joint-offset"
        )
    free_joint_offsets = (
        arm_joint_names(dataset.calibration_arm)
        if args.all_arm_joint_offsets
        else tuple(args.free_joint_offset)
    )
    model = URDFModel(args.urdf)
    pipeline = CalibrationPipeline(
        model,
        calibration_arm=dataset.calibration_arm,
        robot_calibration_directory=args.robot_calibration_directory,
        runner_path=args.native_runner,
        config=PipelineConfig(
            holdout_fraction=args.holdout_fraction,
            bootstrap_trials=args.bootstrap_trials,
            bootstrap_seed=args.bootstrap_seed,
            optimize_hand_target=args.target_transform_mode == "optimize",
            free_joint_offsets=free_joint_offsets,
            joint_offset_prior_sigma_deg=args.joint_offset_prior_sigma_deg,
            native_timeout_s=args.native_timeout_s,
        ),
    )
    result = pipeline.run(
        dataset,
        initial_hand_T_target=initial_target,
        output_directory=args.output_directory,
        provenance={
            "command": "g1-calib solve",
            "optimizer_backend": "mikeferguson/robot_calibration:Ceres",
            "robot_calibration_revision": ("db991b040d1dc28af09d8865fc72f09720e12b73"),
            "dataset_path": str(args.dataset.resolve()),
            "camera_initialization": "official_urdf_nominal_d435_optical",
            "initial_target": "configured_nominal_palm_T_marker",
            "target_transform_mode": args.target_transform_mode,
            "free_joint_offsets": list(free_joint_offsets),
            "joint_offset_prior_sigma_deg": args.joint_offset_prior_sigma_deg,
            "initial_hand_T_target": initial_target.tolist(),
            "hardware_config_path": str(args.hardware_config.resolve()),
            "hardware_config_sha256": hashlib.sha256(
                args.hardware_config.read_bytes()
            ).hexdigest(),
            "target_config_path": str(args.target_config.resolve()),
            "target_config_sha256": dataset.target_artifact_sha256,
            "target_mount": hardware["robot"]["calibration_target_mount"],
            "calibration_arm": dataset.calibration_arm,
        },
    )
    _print_pipeline_result(result, args.output_directory)
    return 0


def _validated_modeled_target_initialization(
    dataset: CalibrationDataset,
    *,
    hardware_path: Path,
    target_path: Path,
) -> tuple[dict, np.ndarray]:
    """Bind a dataset to the nominal palm/marker transform used to initialize it."""

    with hardware_path.open(encoding="utf-8") as stream:
        hardware = yaml.safe_load(stream)
    if not isinstance(hardware, dict):
        raise TypeError("hardware configuration must contain a mapping")
    configured_arm = str(hardware["robot"]["calibration_arm"])
    if configured_arm != str(hardware["control"]["calibration_arm"]):
        raise ValueError("hardware robot/control calibration arms disagree")
    if configured_arm != dataset.calibration_arm:
        raise ValueError(
            "dataset calibration arm differs from the hardware target profile"
        )
    target_sha256 = hashlib.sha256(target_path.read_bytes()).hexdigest()
    if target_sha256 != dataset.target_artifact_sha256:
        raise ValueError("dataset target differs from --target-config")
    with target_path.open(encoding="utf-8") as stream:
        target = json.load(stream)
    validate_hardware_target_profile(hardware, target)
    return hardware, modeled_hand_T_target_from_hardware(hardware)


def run_synthetic_check(args: argparse.Namespace) -> int:
    model = URDFModel(args.urdf)
    dataset, truth = make_synthetic_dataset(
        model,
        args.target_config,
        pose_count=args.pose_count,
        pixel_noise_stddev=args.pixel_noise,
        seed=args.seed,
        calibration_arm=args.calibration_arm,
    )
    _, initial_target = perturbed_initial_transforms(truth)
    result = CalibrationPipeline(
        model,
        calibration_arm=dataset.calibration_arm,
        robot_calibration_directory=args.robot_calibration_directory,
        runner_path=args.native_runner,
        config=PipelineConfig(
            holdout_fraction=0.2,
            bootstrap_trials=args.bootstrap_trials,
            bootstrap_seed=args.seed + 1,
            optimize_hand_target=True,
            native_timeout_s=args.native_timeout_s,
        ),
    ).run(
        dataset,
        initial_hand_T_target=initial_target,
        output_directory=args.output_directory,
        provenance={
            "command": "g1-calib synthetic-check",
            "optimizer_backend": "mikeferguson/robot_calibration:Ceres",
            "known_truth": {
                "torso_T_camera": truth.torso_T_camera.tolist(),
                "hand_T_target": truth.hand_T_target.tolist(),
            },
            "seed": args.seed,
            "pixel_noise_stddev": args.pixel_noise,
        },
    )
    camera_error = transform_to_pose_vector(
        np.linalg.inv(truth.torso_T_camera) @ result.solution.torso_T_camera
    )
    target_error = transform_to_pose_vector(
        np.linalg.inv(truth.hand_T_target) @ result.solution.hand_T_target
    )
    summary = _pipeline_summary(result, args.output_directory)
    summary["truth_error"] = {
        "camera_translation_m": float(np.linalg.norm(camera_error[:3])),
        "camera_rotation_deg": float(np.degrees(np.linalg.norm(camera_error[3:]))),
        "target_translation_m": float(np.linalg.norm(target_error[:3])),
        "target_rotation_deg": float(np.degrees(np.linalg.norm(target_error[3:]))),
    }
    _print_json(summary)
    return 0


def run_anchor_stability(args: argparse.Namespace) -> int:
    dataset = CalibrationDataset.from_json(args.dataset)
    with args.result_json.open(encoding="utf-8") as stream:
        result = json.load(stream)
    validate_exported_result(result)
    if result["dataset_sha256"] != dataset.content_sha256:
        raise ValueError("calibration result belongs to a different dataset")
    report = analyze_dataset_anchors(
        dataset,
        URDFModel(args.urdf),
        torso_T_camera=np.asarray(
            result["solution"]["torso_T_camera"], dtype=np.float64
        ),
        pose_ids=tuple(args.pose_id),
        config=AnchorStabilityConfig(
            minimum_captures=args.minimum_captures,
            maximum_pairwise_translation_m=args.maximum_translation_mm / 1000.0,
            maximum_pairwise_rotation_deg=args.maximum_rotation_deg,
        ),
    )
    document = report.to_dict()
    _print_json(document)
    if args.output is not None:
        _write_json(args.output, document)
    return 0 if report.passed else 1


def run_plan_table_accuracy(args: argparse.Namespace) -> int:
    if (args.board_x_mm is None) != (args.board_y_mm is None):
        raise ValueError("--board-x-mm and --board-y-mm must be supplied together")
    if args.output.exists():
        raise FileExistsError(f"table-accuracy plan already exists: {args.output}")
    dataset = CalibrationDataset.from_json(args.dataset)
    camera_info = camera_info_from_dataset(dataset)
    board_spec = CharucoBoardSpec(legacy_pattern=args.legacy_charuco_pattern)
    burst = observe_burst(
        _table_image_paths(
            args, expected_camera_profile_sha256=camera_info.profile_sha256
        ),
        camera_info=camera_info,
        hand_cube_detector=CorrespondenceDetector(args.hand_cube_config),
        board_detector=CharucoBoardPoseDetector(board_spec),
        minimum_accepted_frames=args.minimum_frames,
    )
    board_xy = None if args.board_x_mm is None else (args.board_x_mm, args.board_y_mm)
    document = create_plan_document(
        dataset_path=args.dataset,
        result_path=args.result_json,
        hand_cube_config_path=args.hand_cube_config,
        planning_burst=burst,
        board_spec=board_spec,
        lift_mm=args.lift_mm,
        board_xy_mm=board_xy,
    )
    _write_json(args.output, document)
    _print_json(
        {
            "output": str(args.output.resolve()),
            "content_sha256": document["content_sha256"],
            "accepted_frame_count": burst["accepted_frame_count"],
            "rejected_frame_count": len(burst["rejected_frames"]),
            "hand_link": document["hand_link"],
            "board_xy_mm": document["target"]["board_xy_mm"],
            "desired_torso_T_hand": document["target"]["desired_torso_T_hand"],
            "lift_above_initial_cube_mm": document["target"][
                "lift_above_initial_cube_mm"
            ],
            "commands_robot": False,
        }
    )
    return 0


def run_evaluate_table_accuracy(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(
            f"table-accuracy evaluation already exists: {args.output}"
        )
    plan = load_plan_document(args.plan)
    cube_path = Path(plan["sources"]["hand_cube_config_path"])
    if not cube_path.is_file():
        raise FileNotFoundError(f"hand-cube config from plan is missing: {cube_path}")
    actual_cube_hash = hashlib.sha256(cube_path.read_bytes()).hexdigest()
    if actual_cube_hash != plan["sources"]["hand_cube_config_sha256"]:
        raise ValueError("hand-cube config has changed since the plan was created")
    camera_info = RectifiedCameraInfo.from_dict(plan["camera_info"])
    board_spec = CharucoBoardSpec.from_dict(plan["board_spec"])
    burst = observe_burst(
        _table_image_paths(
            args, expected_camera_profile_sha256=camera_info.profile_sha256
        ),
        camera_info=camera_info,
        hand_cube_detector=CorrespondenceDetector(cube_path),
        board_detector=CharucoBoardPoseDetector(board_spec),
        minimum_accepted_frames=args.minimum_frames,
        reject_unstable=False,
    )
    document = create_evaluation_document(plan=plan, achieved_burst=burst)
    _write_json(args.output, document)
    _print_json(
        {
            "output": str(args.output.resolve()),
            "content_sha256": document["content_sha256"],
            "plan_sha256": document["plan_sha256"],
            "accepted_frame_count": burst["accepted_frame_count"],
            "rejected_frame_count": len(burst["rejected_frames"]),
            **document["task_error"],
            "measurement_quality": document["measurement_quality"],
            "camera_motion_between_bursts": document["camera_motion_between_bursts"],
        }
    )
    return 0


def _table_image_paths(
    args: argparse.Namespace, *, expected_camera_profile_sha256: str
) -> list[Path]:
    if args.image_directory is None:
        return list(args.image)
    directory = args.image_directory
    if not directory.is_dir():
        raise NotADirectoryError(f"table image directory does not exist: {directory}")
    images = sorted(directory.glob("frame_*.png"))
    if not images:
        raise ValueError(f"table image directory contains no frame_*.png: {directory}")
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"table image directory has no manifest.json: {directory}")
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise TypeError("table image manifest must contain a JSON object")
    canonical = dict(manifest)
    expected_manifest_hash = canonical.pop("content_sha256", None)
    actual_manifest_hash = hashlib.sha256(
        json.dumps(
            canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    if expected_manifest_hash != actual_manifest_hash:
        raise ValueError("table image manifest content SHA-256 does not match")
    if manifest.get("kind") != "g1_table_accuracy_image_burst":
        raise ValueError("image directory is not a table-accuracy capture burst")
    captured_info = RectifiedCameraInfo.from_dict(manifest["camera_info"])
    if captured_info.profile_sha256 != manifest.get("camera_profile_sha256"):
        raise ValueError("table image manifest camera profile hash does not match")
    if captured_info.profile_sha256 != expected_camera_profile_sha256:
        raise ValueError(
            "table image camera profile differs from the calibration dataset"
        )
    expected_names = [item["file"] for item in manifest.get("frames", [])]
    if expected_names != [item.name for item in images]:
        raise ValueError("table image directory does not match its capture manifest")
    for path, item in zip(images, manifest["frames"], strict=True):
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"table image hash does not match manifest: {path}")
    return images


def _print_pipeline_result(result, output_directory: Path) -> None:
    _print_json(_pipeline_summary(result, output_directory))


def _pipeline_summary(result, output_directory: Path) -> dict:
    return {
        "output_directory": str(output_directory.resolve()),
        "optimization_rms_px": result.solution.optimization_rms_px,
        "training_radial_rms_px": result.residuals.training.rms_px,
        "holdout_radial_rms_px": result.residuals.holdout.rms_px,
        "jacobian_rank": result.solution.observability.rank,
        "jacobian_condition_number": result.solution.observability.condition_number,
        "observable": result.solution.observability.observable,
        "bootstrap_successful": result.bootstrap.successful_trials,
        "bootstrap_failed": result.bootstrap.failed_trials,
        "optimizer_backend": result.solution.backend,
        "free_parameters": list(result.solution.parameter_names),
    }


def _load_state(path: Path) -> RobotStateSample:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise TypeError("state JSON must contain an object")
    return RobotStateSample.from_dict(data)


def _load_edges(path: Path) -> tuple[tuple[str, str], ...]:
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if isinstance(data, dict):
        data = data.get("edges")
    if not isinstance(data, list) or not data:
        raise ValueError("edges YAML must contain a non-empty edges list")
    edges: list[tuple[str, str]] = []
    for item in data:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("each directed edge must be [from_pose, to_pose]")
        edges.append((str(item[0]), str(item[1])))
    return tuple(edges)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _print_json(data: dict) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, allow_nan=False))
