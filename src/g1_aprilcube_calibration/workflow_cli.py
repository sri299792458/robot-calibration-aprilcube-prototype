"""Offline workflow and subscriber-only hardware inspection CLI commands."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from aprilcube import CorrespondenceDetector
from g1_aprilcube_calibration.anchor_stability import (
    AnchorStabilityConfig,
    analyze_dataset_anchors,
)
from g1_aprilcube_calibration.calibration_pipeline import (
    CalibrationPipeline,
    PipelineConfig,
)
from g1_aprilcube_calibration.calibration_solver import ExtrinsicsSolver
from g1_aprilcube_calibration.camera_initialization import (
    estimate_hand_T_target_from_sample,
    nominal_torso_T_color_optical,
)
from g1_aprilcube_calibration.collision import CollisionConfig, FCLCollisionChecker
from g1_aprilcube_calibration.dataset_builder import CalibrationDataset, DatasetBuilder
from g1_aprilcube_calibration.joint_map import arm_indices, opposite_arm
from g1_aprilcube_calibration.models import RobotStateSample
from g1_aprilcube_calibration.pose_schema import PoseSet
from g1_aprilcube_calibration.pose_store import PoseStore
from g1_aprilcube_calibration.pose_validator import PosePathValidator
from g1_aprilcube_calibration.residual_report import validate_exported_result
from g1_aprilcube_calibration.session_store import SessionStore
from g1_aprilcube_calibration.synthetic import (
    make_synthetic_dataset,
    perturbed_initial_transforms,
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
        workspace_root
        / "unitree_ros"
        / "robots"
        / "g1_description"
        / "g1_29dof_rev_1_0.urdf"
    )
    default_collision = workspace_root / "config" / "collision_pairs.yaml"

    inspect = subparsers.add_parser(
        "inspect-artifacts",
        help="verify pinned URDF, AprilCube target, and hardware readiness offline",
    )
    inspect.add_argument("--urdf", type=Path, default=default_urdf)
    inspect.add_argument("--target-config", type=Path, default=default_target)
    inspect.add_argument("--collision-config", type=Path, default=default_collision)
    inspect.set_defaults(handler=run_inspect_artifacts)

    hardware = subparsers.add_parser(
        "inspect-hardware",
        help="subscribe to rt/lowstate without creating a command publisher",
    )
    hardware.add_argument("--network-interface", required=True)
    hardware.add_argument("--domain-id", type=int, default=0)
    hardware.add_argument("--timeout-s", type=float, default=5.0)
    hardware.add_argument("--state-json", type=Path)
    hardware.set_defaults(handler=run_inspect_hardware)

    initialize = subparsers.add_parser(
        "init-pose-set",
        help="initialize measured handoff and an empty taught-pose set",
    )
    initialize.add_argument("--state-json", type=Path, required=True)
    initialize.add_argument("--output", type=Path, required=True)
    initialize.add_argument("--urdf", type=Path, default=default_urdf)
    initialize.add_argument(
        "--calibration-arm", choices=("left", "right"), default="left"
    )
    initialize.set_defaults(handler=run_init_pose_set)

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
    validate.set_defaults(handler=run_validate_poses)

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
    dataset.add_argument("--allow-unfinalized", action="store_true")
    dataset.set_defaults(handler=run_build_dataset)

    solve = subparsers.add_parser(
        "solve", help="solve twelve extrinsic parameters and export residual reports"
    )
    solve.add_argument("--dataset", type=Path, required=True)
    solve.add_argument("--output-directory", type=Path, required=True)
    solve.add_argument("--urdf", type=Path, default=default_urdf)
    solve.add_argument("--pnp-sample-index", type=int, default=0)
    solve.add_argument("--holdout-fraction", type=float, default=0.2)
    solve.add_argument("--bootstrap-trials", type=int, default=50)
    solve.add_argument("--bootstrap-seed", type=int, default=17)
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


def run_init_pose_set(args: argparse.Namespace) -> int:
    state = _load_state(args.state_json)
    if not state.is_mode5:
        raise ValueError("state JSON is not mode_machine=5")
    maximum_arm_velocity = float(np.max(np.abs(state.velocity[15:29])))
    model = URDFModel(args.urdf)
    pose_set = PoseSet(
        robot_model="g1_29dof_rev_1_0",
        mode_machine=5,
        urdf_sha256=model.sha256,
        calibration_arm=args.calibration_arm,
        handoff_q=tuple(
            state.position[np.asarray(arm_indices(args.calibration_arm))]
        ),
        hold_q=tuple(
            state.position[np.asarray(arm_indices(opposite_arm(args.calibration_arm)))]
        ),
    )
    PoseStore(args.output).initialize(pose_set)
    _print_json(
        {
            "path": str(args.output.resolve()),
            "pose_count": 0,
            "content_sha256": pose_set.content_sha256,
            "urdf_sha256": model.sha256,
            "calibration_arm": pose_set.calibration_arm,
            "handoff_q": list(pose_set.handoff_q),
            "hold_q": list(pose_set.hold_q),
            "maximum_measured_arm_velocity_rad_s": maximum_arm_velocity,
        }
    )
    return 0


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
            "handoff_q": list(pose_set.handoff_q),
            "hold_q": list(pose_set.hold_q),
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


def run_validate_poses(args: argparse.Namespace) -> int:
    pose_set = PoseStore(args.pose_set).load()
    state = _load_state(args.reference_state_json)
    edges = _load_edges(args.edges_yaml)
    model = URDFModel(args.urdf)
    collision = FCLCollisionChecker(
        model, CollisionConfig.from_yaml(args.collision_config)
    )
    report = PosePathValidator(model=model, collision_checker=collision).validate(
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
    )
    _print_json(
        {
            "session_id": dataset.session_id,
            "sample_count": len(dataset.samples),
            "content_sha256": dataset.content_sha256,
            "output": str(args.output.resolve()),
        }
    )
    return 0


def run_solve(args: argparse.Namespace) -> int:
    dataset = CalibrationDataset.from_json(args.dataset)
    if not 0 <= args.pnp_sample_index < len(dataset.samples):
        raise ValueError("--pnp-sample-index is outside the dataset")
    model = URDFModel(args.urdf)
    initial_camera = nominal_torso_T_color_optical(model)
    initial_target = estimate_hand_T_target_from_sample(
        model,
        dataset.samples[args.pnp_sample_index],
        initial_torso_T_camera=initial_camera,
        calibration_arm=dataset.calibration_arm,
    )
    pipeline = CalibrationPipeline(
        ExtrinsicsSolver(model, calibration_arm=dataset.calibration_arm),
        config=PipelineConfig(
            holdout_fraction=args.holdout_fraction,
            bootstrap_trials=args.bootstrap_trials,
            bootstrap_seed=args.bootstrap_seed,
        ),
    )
    result = pipeline.run(
        dataset,
        initial_torso_T_camera=initial_camera,
        initial_hand_T_target=initial_target,
        output_directory=args.output_directory,
        provenance={
            "command": "g1-calib solve",
            "dataset_path": str(args.dataset.resolve()),
            "initial_camera": "official_urdf_nominal_d435_optical",
            "initial_target": f"rectified_pnp_sample_{args.pnp_sample_index}",
            "calibration_arm": dataset.calibration_arm,
        },
    )
    _print_pipeline_result(result, args.output_directory)
    return 0


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
    initial_camera, initial_target = perturbed_initial_transforms(truth)
    result = CalibrationPipeline(
        ExtrinsicsSolver(model, calibration_arm=dataset.calibration_arm),
        config=PipelineConfig(
            holdout_fraction=0.2,
            bootstrap_trials=args.bootstrap_trials,
            bootstrap_seed=args.seed + 1,
        ),
    ).run(
        dataset,
        initial_torso_T_camera=initial_camera,
        initial_hand_T_target=initial_target,
        output_directory=args.output_directory,
        provenance={
            "command": "g1-calib synthetic-check",
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
        "bootstrap_successful": result.bootstrap.successful_trials,
        "bootstrap_degenerate": result.bootstrap.degenerate_trials,
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


def _print_json(data: dict) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, allow_nan=False))
