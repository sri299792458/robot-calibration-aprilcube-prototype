"""Design collision-checked G1 calibration targets from the camera frustum."""

from __future__ import annotations

import hashlib
import heapq
import json
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import pairwise
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import qmc

from g1_aprilcube_calibration.authored_collection import (
    AuthoredCollectionPlan,
    AuthoredPoseTarget,
    exposed_target_normal_from_hardware,
    modeled_hand_T_target_from_hardware,
    validate_exposed_camera_views,
    validate_hardware_target_profile,
)
from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.collision import CollisionConfig, FCLCollisionChecker
from g1_aprilcube_calibration.inverse_kinematics import (
    IKConfig,
    solve_arm_ik_candidates,
)
from g1_aprilcube_calibration.joint_map import (
    G1_29_JOINT_NAMES,
    G1_MODE_MACHINE,
    arm_hand_link,
    arm_indices,
    validate_arm_side,
    validate_full_joint_vector,
)
from g1_aprilcube_calibration.models import RobotStateSample
from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID
from g1_aprilcube_calibration.pose_validator import (
    DirectedEdgeResult,
    PathValidationConfig,
    PosePathValidator,
    ValidationReport,
)
from g1_aprilcube_calibration.transforms import (
    invert_transform,
    pose_vector_to_transform,
    transform_points,
    transform_to_pose_vector,
    validate_transform,
)
from g1_aprilcube_calibration.urdf_model import URDFModel

_MAX_AUTO_DESIGN_WORKERS = 8
_auto_candidate_worker: _CandidateWorkerContext | None = None
_auto_route_worker: _RouteWorkerContext | None = None


@dataclass(frozen=True, slots=True)
class AutoCollectionDesignConfig:
    target_count: int = 80
    candidate_count: int = 1600
    minimum_depth_m: float = 0.25
    maximum_depth_m: float = 0.60
    image_margin_px: float = 60.0
    minimum_projected_target_span_px: float = 45.0
    maximum_view_obliquity_deg: float = 55.0
    maximum_view_roll_deg: float = 35.0
    image_grid_columns: int = 3
    image_grid_rows: int = 3
    depth_bins: int = 3
    obliquity_bins: int = 3
    azimuth_bins: int = 6
    in_plane_rotation_bins: int = 6
    coverage_score_weight: float = 0.35
    information_translation_scale_m: float = 0.01
    information_rotation_scale_deg: float = 5.0
    information_ridge: float = 1e-6
    seed: int = 17
    ik_restart_count: int = 4

    def __post_init__(self) -> None:
        if self.target_count < 1:
            raise ValueError("target_count must be positive")
        if self.candidate_count < self.target_count:
            raise ValueError("candidate_count must be at least target_count")
        if not 0 < self.minimum_depth_m < self.maximum_depth_m:
            raise ValueError("camera depth bounds are invalid")
        for name in (
            "image_margin_px",
            "minimum_projected_target_span_px",
            "maximum_view_obliquity_deg",
            "maximum_view_roll_deg",
            "information_translation_scale_m",
            "information_rotation_scale_deg",
            "information_ridge",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "image_grid_columns",
            "image_grid_rows",
            "depth_bins",
            "obliquity_bins",
            "azimuth_bins",
            "in_plane_rotation_bins",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.coverage_score_weight) or not (
            0 <= self.coverage_score_weight <= 1
        ):
            raise ValueError("coverage_score_weight must lie in [0, 1]")
        if self.maximum_view_obliquity_deg >= 90:
            raise ValueError("maximum target obliquity must be below 90 degrees")
        if self.ik_restart_count < 1:
            raise ValueError("ik_restart_count must be positive")

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class AutoCollectionDesignResult:
    plan: AuthoredCollectionPlan
    validation_report: ValidationReport
    attempted_camera_targets: int
    ik_failures: int
    endpoint_failures: int
    parallel_worker_count: int


@dataclass(frozen=True, slots=True)
class _CommandSet:
    robot_model: str
    mode_machine: int
    urdf_sha256: str
    calibration_arm: str
    poses: tuple[AuthoredPoseTarget, ...]
    content_sha256: str


@dataclass(frozen=True, slots=True)
class _CoverageSignature:
    image_cell: tuple[int, int]
    depth_bin: int
    obliquity_bin: int
    azimuth_bin: int
    in_plane_rotation_bin: int
    normalized_centroid: tuple[float, float]
    depth_m: float
    obliquity_deg: float
    azimuth_deg: float
    in_plane_rotation_deg: float

    def to_dict(self) -> dict:
        return {
            "image_cell": list(self.image_cell),
            "depth_bin": self.depth_bin,
            "obliquity_bin": self.obliquity_bin,
            "azimuth_bin": self.azimuth_bin,
            "in_plane_rotation_bin": self.in_plane_rotation_bin,
            "normalized_centroid": list(self.normalized_centroid),
            "depth_m": self.depth_m,
            "obliquity_deg": self.obliquity_deg,
            "azimuth_deg": self.azimuth_deg,
            "in_plane_rotation_deg": self.in_plane_rotation_deg,
        }


@dataclass(frozen=True, slots=True)
class _FeasibleCandidate:
    target: AuthoredPoseTarget
    information_matrix: np.ndarray
    coverage: _CoverageSignature


@dataclass(frozen=True, slots=True)
class _CandidateTask:
    sample_index: int
    camera_T_target: np.ndarray


@dataclass(frozen=True, slots=True)
class _CandidateEvaluation:
    sample_index: int
    outcome: str
    candidate: _FeasibleCandidate | None = None


@dataclass(slots=True)
class _CandidateWorkerContext:
    model: URDFModel
    validator: PosePathValidator
    reference: np.ndarray
    arm: str
    ik_config: IKConfig
    camera_info: RectifiedCameraInfo
    target_object_points_m: np.ndarray
    torso_T_camera: np.ndarray
    hand_T_target: np.ndarray
    exposed_target_normal: np.ndarray
    design: AutoCollectionDesignConfig
    command_hash: str


@dataclass(slots=True)
class _RouteWorkerContext:
    validator: PosePathValidator
    plan: AuthoredCollectionPlan
    reference: np.ndarray


def camera_info_from_hardware(hardware: dict) -> RectifiedCameraInfo:
    camera = hardware["camera"]
    profile = camera["color_profile"]
    return RectifiedCameraInfo(
        width=int(profile["width"]),
        height=int(profile["height"]),
        frame_id="camera_color_optical_frame",
        camera_name=str(camera["name"]),
        serial_number=str(camera["serial_number"]),
        distortion_model=str(profile["distortion_model"]),
        d=tuple(profile["d"]),
        k=tuple(profile["k"]),
        r=tuple(profile["r"]),
        p=tuple(profile["p"]),
    )


def _auto_design_worker_count(*, requested: int | None, task_count: int) -> int:
    if task_count < 1:
        raise ValueError("automatic design requires at least one candidate task")
    if requested is not None:
        if isinstance(requested, bool) or not isinstance(requested, int):
            raise TypeError("parallel worker count must be an integer")
        if not 1 <= requested <= _MAX_AUTO_DESIGN_WORKERS:
            raise ValueError(
                "parallel worker count must lie within "
                f"[1, {_MAX_AUTO_DESIGN_WORKERS}]"
            )
        return min(requested, task_count)
    return min(_MAX_AUTO_DESIGN_WORKERS, os.cpu_count() or 1, task_count)


def _initialize_auto_candidate_worker(
    urdf_path: str,
    collision_config: CollisionConfig,
    path_config: PathValidationConfig,
    reference: np.ndarray,
    arm: str,
    ik_config: IKConfig,
    camera_info: RectifiedCameraInfo,
    target_object_points_m: np.ndarray,
    torso_T_camera: np.ndarray,
    hand_T_target: np.ndarray,
    exposed_target_normal: np.ndarray,
    design: AutoCollectionDesignConfig,
    command_hash: str,
) -> None:
    global _auto_candidate_worker
    model = URDFModel(urdf_path)
    _auto_candidate_worker = _CandidateWorkerContext(
        model=model,
        validator=PosePathValidator(
            model=model,
            collision_checker=FCLCollisionChecker(model, collision_config),
            config=path_config,
        ),
        reference=validate_full_joint_vector(reference),
        arm=validate_arm_side(arm),
        ik_config=ik_config,
        camera_info=camera_info,
        target_object_points_m=np.asarray(
            target_object_points_m, dtype=np.float64
        ),
        torso_T_camera=validate_transform(torso_T_camera),
        hand_T_target=validate_transform(hand_T_target),
        exposed_target_normal=np.asarray(exposed_target_normal, dtype=np.float64),
        design=design,
        command_hash=command_hash,
    )


def _evaluate_auto_candidate_worker(task: _CandidateTask) -> _CandidateEvaluation:
    context = _auto_candidate_worker
    if context is None:
        raise RuntimeError("automatic candidate worker was not initialized")
    camera_T_target = validate_transform(task.camera_T_target)
    torso_T_hand = validate_transform(
        context.torso_T_camera
        @ camera_T_target
        @ invert_transform(context.hand_T_target)
    )
    try:
        solutions = solve_arm_ik_candidates(
            context.model,
            desired_torso_T_hand=torso_T_hand,
            reference_full_q=context.reference,
            calibration_arm=context.arm,
            config=context.ik_config,
        )
    except ValueError:
        return _CandidateEvaluation(task.sample_index, "ik_failure")

    selected = None
    for solution in solutions:
        candidate = AuthoredPoseTarget(
            id=f"candidate_{task.sample_index + 1:04d}",
            authored_calibration_q=solution.calibration_q,
            desired_camera_T_cube=camera_T_target,
            ik_diagnostics=solution.to_dict(),
        )
        command_set = _CommandSet(
            robot_model=context.model.name,
            mode_machine=G1_MODE_MACHINE,
            urdf_sha256=context.model.sha256,
            calibration_arm=context.arm,
            poses=(candidate,),
            content_sha256=context.command_hash,
        )
        endpoint = context.validator.validate(
            command_set,
            directed_edges=((candidate.id, candidate.id),),
            reference_full_q=context.reference,
        ).edges[0]
        if endpoint.passed:
            selected = candidate
            break
    if selected is None:
        return _CandidateEvaluation(task.sample_index, "endpoint_failure")

    information = _candidate_information_matrix(
        model=context.model,
        reference_full_q=context.reference,
        calibration_arm=context.arm,
        command_calibration_q=np.asarray(selected.command_calibration_q),
        camera_info=context.camera_info,
        object_points_m=context.target_object_points_m,
        torso_T_camera=context.torso_T_camera,
        hand_T_target=context.hand_T_target,
        translation_scale_m=context.design.information_translation_scale_m,
        rotation_scale_rad=np.deg2rad(
            context.design.information_rotation_scale_deg
        ),
    )
    coverage = _coverage_signature(
        camera_T_target,
        camera_info=context.camera_info,
        exposed_target_normal=context.exposed_target_normal,
        config=context.design,
    )
    return _CandidateEvaluation(
        task.sample_index,
        "feasible",
        _FeasibleCandidate(selected, information, coverage),
    )


def _evaluate_candidate_tasks(
    *,
    tasks: list[_CandidateTask],
    worker_count: int,
    urdf_path: Path,
    collision_config: CollisionConfig,
    path_config: PathValidationConfig,
    reference: np.ndarray,
    arm: str,
    ik_config: IKConfig,
    camera_info: RectifiedCameraInfo,
    target_object_points_m: np.ndarray,
    torso_T_camera: np.ndarray,
    hand_T_target: np.ndarray,
    exposed_target_normal: np.ndarray,
    design: AutoCollectionDesignConfig,
    command_hash: str,
    progress: Callable[[str], None],
) -> tuple[_CandidateEvaluation, ...]:
    initializer_args = (
        str(urdf_path),
        collision_config,
        path_config,
        reference,
        arm,
        ik_config,
        camera_info,
        target_object_points_m,
        torso_T_camera,
        hand_T_target,
        exposed_target_normal,
        design,
        command_hash,
    )
    completed_results: list[_CandidateEvaluation] = []

    def record(result: _CandidateEvaluation) -> None:
        results_by_index[result.sample_index] = result
        completed_results.append(result)
        completed = len(completed_results)
        if completed == 1 or completed % 50 == 0 or completed == len(tasks):
            progress(
                "parallel camera/IK/FCL progress: "
                f"{completed}/{len(tasks)} evaluated; "
                f"{sum(item.outcome == 'feasible' for item in completed_results)} "
                "feasible"
            )

    results_by_index: dict[int, _CandidateEvaluation] = {}
    if worker_count == 1:
        _initialize_auto_candidate_worker(*initializer_args)
        for task in tasks:
            record(_evaluate_auto_candidate_worker(task))
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=get_context("spawn"),
            initializer=_initialize_auto_candidate_worker,
            initargs=initializer_args,
        ) as executor:
            futures = {
                executor.submit(_evaluate_auto_candidate_worker, task): task
                for task in tasks
            }
            try:
                for future in as_completed(futures):
                    record(future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    ordered = tuple(results_by_index[task.sample_index] for task in tasks)
    if len(ordered) != len(tasks):
        raise RuntimeError("parallel candidate evaluation lost a result")
    return ordered


def _initialize_auto_route_worker(
    urdf_path: str,
    collision_config: CollisionConfig,
    path_config: PathValidationConfig,
    plan: AuthoredCollectionPlan,
    reference: np.ndarray,
) -> None:
    global _auto_route_worker
    model = URDFModel(urdf_path)
    _auto_route_worker = _RouteWorkerContext(
        validator=PosePathValidator(
            model=model,
            collision_checker=FCLCollisionChecker(model, collision_config),
            config=path_config,
        ),
        plan=plan,
        reference=validate_full_joint_vector(reference),
    )


def _validate_auto_route_edge_worker(
    edge: tuple[str, str],
) -> DirectedEdgeResult:
    context = _auto_route_worker
    if context is None:
        raise RuntimeError("automatic route worker was not initialized")
    return context.validator.validate(
        context.plan,
        directed_edges=(edge,),
        reference_full_q=context.reference,
    ).edges[0]


def _validate_final_route(
    *,
    model: URDFModel,
    collision_config: CollisionConfig,
    path_config: PathValidationConfig,
    validator: PosePathValidator,
    plan: AuthoredCollectionPlan,
    directed_edges: tuple[tuple[str, str], ...],
    reference: np.ndarray,
    worker_count: int,
    progress: Callable[[int, int, str, str], None] | None = None,
) -> ValidationReport:
    """Validate every final directed edge, preserving deterministic order."""

    if worker_count <= 1:
        return validator.validate(
            plan,
            directed_edges=directed_edges,
            reference_full_q=reference,
            progress=progress,
        )
    results = [None] * len(directed_edges)
    with ProcessPoolExecutor(
        max_workers=min(worker_count, len(directed_edges)),
        mp_context=get_context("spawn"),
        initializer=_initialize_auto_route_worker,
        initargs=(
            str(model.path),
            collision_config,
            path_config,
            plan,
            reference,
        ),
    ) as executor:
        futures = {
            executor.submit(_validate_auto_route_edge_worker, edge): index
            for index, edge in enumerate(directed_edges)
        }
        try:
            for completed, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                results[index] = future.result()
                if progress is not None:
                    progress(
                        completed,
                        len(directed_edges),
                        *directed_edges[index],
                    )
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    if any(result is None for result in results):
        raise RuntimeError("parallel route validation lost an edge result")
    return ValidationReport(
        pose_set_sha256=plan.content_sha256,
        urdf_sha256=model.sha256,
        collision_config_sha256=collision_config.content_sha256,
        reference_full_q_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
        config={
            name: getattr(path_config, name)
            for name in path_config.__dataclass_fields__
        },
        edges=tuple(result for result in results if result is not None),
    )


def design_auto_collection(
    *,
    model: URDFModel,
    collision_config: CollisionConfig,
    hardware: dict,
    target_config: dict,
    reference_state: RobotStateSample,
    torso_T_camera: np.ndarray,
    torso_T_camera_source: dict,
    design_config: AutoCollectionDesignConfig | None = None,
    path_config: PathValidationConfig | None = None,
    progress: Callable[[str], None] | None = None,
    parallel_workers: int | None = None,
) -> AutoCollectionDesignResult:
    """Generate camera-visible IK targets and a reversible validated tree route."""

    design = design_config or AutoCollectionDesignConfig()
    path = path_config or PathValidationConfig()
    report_progress = progress or (lambda _message: None)
    if reference_state.mode_machine != G1_MODE_MACHINE:
        raise ValueError(
            f"reference state is mode_machine={reference_state.mode_machine}; "
            f"expected {G1_MODE_MACHINE}"
        )
    reference = validate_full_joint_vector(reference_state.position)
    robot = hardware["robot"]
    control = hardware["control"]
    arm = validate_arm_side(str(control["calibration_arm"]))
    if str(robot["calibration_arm"]) != arm:
        raise ValueError("hardware robot/control calibration arms disagree")
    camera_info = camera_info_from_hardware(hardware)
    target_dimensions_m = _target_dimensions_m(target_config)
    target_object_points_m = _target_object_points_m(target_config)
    hand_T_target = modeled_hand_T_target_from_hardware(hardware)
    exposed_target_normal = exposed_target_normal_from_hardware(hardware)
    torso_T_camera = validate_transform(torso_T_camera)
    if not isinstance(torso_T_camera_source, dict) or not torso_T_camera_source:
        raise ValueError("torso_T_camera_source must be a non-empty mapping")
    checker = FCLCollisionChecker(model, collision_config)
    validator = PosePathValidator(
        model=model,
        collision_checker=checker,
        config=path,
    )
    command_hash = hashlib.sha256(
        json.dumps(
            {
                "reference": reference.tolist(),
                "design": design.to_dict(),
                "urdf": model.sha256,
                "torso_T_camera": torso_T_camera.tolist(),
                "torso_T_camera_source": torso_T_camera_source,
                "hand_T_target": hand_T_target.tolist(),
                "target_config": target_config,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    samples = qmc.Halton(d=6, scramble=True, seed=design.seed).random(
        design.candidate_count
    )
    tasks: list[_CandidateTask] = []
    for sample_index, sample in enumerate(samples):
        camera_T_target = _sample_camera_T_target(
            sample,
            camera_info=camera_info,
            target_dimensions_m=target_dimensions_m,
            exposed_target_normal=exposed_target_normal,
            config=design,
        )
        if camera_T_target is not None:
            tasks.append(_CandidateTask(sample_index, camera_T_target))
    if not tasks:
        raise ValueError("camera sampling produced no geometrically valid targets")
    ik_config = IKConfig(
        joint_limit_margin_rad=path.joint_limit_margin_rad,
        restart_count=design.ik_restart_count,
    )
    worker_count = _auto_design_worker_count(
        requested=parallel_workers,
        task_count=len(tasks),
    )
    report_progress(
        f"camera sampling retained {len(tasks)}/{design.candidate_count} "
        f"geometrically valid targets; evaluating IK/FCL with "
        f"{worker_count} worker{'s' if worker_count != 1 else ''}"
    )
    evaluations = _evaluate_candidate_tasks(
        tasks=tasks,
        worker_count=worker_count,
        urdf_path=model.path,
        collision_config=collision_config,
        path_config=path,
        reference=reference,
        arm=arm,
        ik_config=ik_config,
        camera_info=camera_info,
        target_object_points_m=target_object_points_m,
        torso_T_camera=torso_T_camera,
        hand_T_target=hand_T_target,
        exposed_target_normal=exposed_target_normal,
        design=design,
        command_hash=command_hash,
        progress=report_progress,
    )
    candidates = [
        result.candidate
        for result in evaluations
        if result.candidate is not None
    ]
    ik_failures = sum(result.outcome == "ik_failure" for result in evaluations)
    endpoint_failures = sum(
        result.outcome == "endpoint_failure" for result in evaluations
    )
    attempted = design.candidate_count

    if len(candidates) < design.target_count:
        raise ValueError(
            "camera/IK/collision filtering produced only "
            f"{len(candidates)} targets; requested {design.target_count} from "
            f"{attempted} attempted camera poses"
        )

    selected_candidates, selection_steps = _select_information_targets(
        candidates,
        count=design.target_count,
        config=design,
        role="selected",
    )
    targets = [item.target for item in selected_candidates]
    selection_diagnostics = _selection_diagnostics(
        candidates=candidates,
        selected=selected_candidates,
        selection_steps=selection_steps,
        config=design,
    )
    report_progress(
        f"selected {design.target_count} coverage/information poses from "
        f"{len(candidates)} feasible poses"
    )

    provisional = _CommandSet(
        robot_model=model.name,
        mode_machine=G1_MODE_MACHINE,
        urdf_sha256=model.sha256,
        calibration_arm=arm,
        poses=tuple(targets),
        content_sha256=command_hash,
    )
    selected_targets, parents = _connect_targets(
        targets,
        target_count=len(targets),
        reference=reference,
        calibration_arm=arm,
        validator=validator,
        command_set=provisional,
        progress=report_progress,
    )
    report_progress(
        f"connected {len(selected_targets)} targets to the measured handoff"
    )
    route = _tree_route(selected_targets, parents)
    capture_ids = tuple(
        dict.fromkeys(item for item in route if item != HANDOFF_POSE_ID)
    )
    if set(capture_ids) != {target.id for target in selected_targets}:
        raise RuntimeError("route does not visit every selected target exactly once")
    plan = AuthoredCollectionPlan(
        robot_model=model.name,
        mode_machine=G1_MODE_MACHINE,
        urdf_sha256=model.sha256,
        calibration_arm=arm,
        camera_profile_sha256=camera_info.profile_sha256,
        reference_full_q_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
        targets=tuple(selected_targets),
        route_pose_ids=route,
        capture_pose_ids=capture_ids,
        generation_config={
            **design.to_dict(),
            "path_validation": {
                name: getattr(path, name) for name in path.__dataclass_fields__
            },
            "torso_T_camera": torso_T_camera.tolist(),
            "torso_T_camera_source": torso_T_camera_source,
            "modeled_hand_T_target": hand_T_target.tolist(),
            "exposed_target_normal": exposed_target_normal.tolist(),
            "target_dimensions_m": target_dimensions_m.tolist(),
            "selection": selection_diagnostics,
        },
    )
    validate_exposed_camera_views(
        plan,
        exposed_target_normal=exposed_target_normal,
    )
    directed_edges = _unique_edges(plan.route_pose_ids)
    def report_final_validation(
        completed: int,
        total: int,
        source_pose: str,
        target_pose: str,
    ) -> None:
        if completed == 1 or completed % 10 == 0 or completed == total:
            report_progress(
                "final directed-route validation: "
                f"{completed}/{total}; last={source_pose}->{target_pose}"
            )

    report = _validate_final_route(
        model=model,
        collision_config=collision_config,
        path_config=path,
        validator=validator,
        plan=plan,
        directed_edges=directed_edges,
        reference=reference,
        worker_count=worker_count,
        progress=report_final_validation,
    )
    if not report.passed:
        failures = [
            f"{edge.from_pose_id}->{edge.to_pose_id}: {'; '.join(edge.failures)}"
            for edge in report.edges
            if not edge.passed
        ]
        raise ValueError(
            "final authored route validation failed: " + " | ".join(failures)
        )
    report_progress(
        f"validated {len(report.edges)} directed route edges against the full state"
    )
    return AutoCollectionDesignResult(
        plan=plan,
        validation_report=report,
        attempted_camera_targets=attempted,
        ik_failures=ik_failures,
        endpoint_failures=endpoint_failures,
        parallel_worker_count=worker_count,
    )


def _connect_targets(
    targets: list[AuthoredPoseTarget],
    *,
    target_count: int,
    reference: np.ndarray,
    calibration_arm: str,
    validator: PosePathValidator,
    command_set: _CommandSet,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[AuthoredPoseTarget], dict[str, str]]:
    """Grow a shortest-valid-edge tree from the measured handoff."""

    handoff_q = reference[np.asarray(arm_indices(calibration_arm))]
    q_by_id = {
        HANDOFF_POSE_ID: handoff_q,
        **{
            target.id: np.asarray(target.command_calibration_q, dtype=np.float64)
            for target in targets
        },
    }
    if not 0 <= target_count <= len(targets):
        raise ValueError("route target_count is outside the supplied targets")
    parents: dict[str, str] = {}
    connected = {HANDOFF_POSE_ID}
    report_progress = progress or (lambda _message: None)
    heap: list[tuple[float, str, str]] = []
    for source_id in connected:
        for target in targets:
            if target.id in connected:
                continue
            heapq.heappush(
                heap,
                (
                    float(
                        np.linalg.norm(q_by_id[target.id] - q_by_id[source_id])
                    ),
                    source_id,
                    target.id,
                ),
            )
    while heap and len(parents) < target_count:
        _distance, parent, target_id = heapq.heappop(heap)
        if target_id in connected or parent not in connected:
            continue
        # Joint interpolation, limits, FK, and FCL are static and therefore
        # invariant to traversal direction. Tree construction only needs this
        # undirected feasibility result; the completed route is still checked
        # independently in every direction that will actually be executed.
        passed = validator.validate(
            command_set,
            directed_edges=((parent, target_id),),
            reference_full_q=reference,
        ).passed
        if not passed:
            continue
        parents[target_id] = parent
        connected.add(target_id)
        if (
            len(parents) == target_count
            or len(parents) == 1
            or len(parents) % 10 == 0
        ):
            report_progress(
                "route collision-tree progress: "
                f"{len(parents)}/{target_count} targets connected"
            )
        for remaining in targets:
            if remaining.id in connected:
                continue
            heapq.heappush(
                heap,
                (
                    float(np.linalg.norm(q_by_id[remaining.id] - q_by_id[target_id])),
                    target_id,
                    remaining.id,
                ),
            )
    if len(parents) < target_count:
        raise ValueError(
            "only "
            f"{len(parents)} collision-connected targets could be rooted at the "
            f"handoff; requested {target_count}"
        )
    selected = [target for target in targets if target.id in parents]
    return selected, parents


def _tree_route(
    targets: list[AuthoredPoseTarget], parents: dict[str, str]
) -> tuple[str, ...]:
    order = {target.id: index for index, target in enumerate(targets)}
    children: dict[str, list[str]] = {HANDOFF_POSE_ID: []}
    for target in targets:
        children.setdefault(target.id, [])
    for child, parent in parents.items():
        children.setdefault(parent, []).append(child)
    for values in children.values():
        values.sort(key=order.get)

    route = [HANDOFF_POSE_ID]

    def visit(parent: str) -> None:
        for child in children[parent]:
            route.append(child)
            visit(child)
            route.append(parent)

    visit(HANDOFF_POSE_ID)
    return tuple(route)


def _select_information_targets(
    candidates: list[_FeasibleCandidate],
    *,
    count: int,
    config: AutoCollectionDesignConfig,
    role: str,
    initial_information: np.ndarray | None = None,
) -> tuple[list[_FeasibleCandidate], list[dict]]:
    """Balance reachable image cells, then maximize scaled pixel information."""

    if count < 0 or count > len(candidates):
        raise ValueError(f"cannot select {count} {role} poses from {len(candidates)}")
    if count == 0:
        return [], []
    information = np.eye(12, dtype=np.float64) * config.information_ridge
    if initial_information is not None:
        initial = np.asarray(initial_information, dtype=np.float64)
        if initial.shape != (12, 12) or not np.all(np.isfinite(initial)):
            raise ValueError("initial information must be a finite 12x12 matrix")
        information += initial
    capacities: dict[tuple[int, int], int] = {}
    for candidate in candidates:
        cell = candidate.coverage.image_cell
        capacities[cell] = capacities.get(cell, 0) + 1
    quotas = _balanced_image_quotas(capacities, count)
    coverage_counts = {
        "depth": [0] * config.depth_bins,
        "obliquity": [0] * config.obliquity_bins,
        "azimuth": [0] * config.azimuth_bins,
        "in_plane_rotation": [0] * config.in_plane_rotation_bins,
    }
    remaining = list(candidates)
    selected: list[_FeasibleCandidate] = []
    steps: list[dict] = []
    for selection_index in range(count):
        eligible = [
            item for item in remaining if quotas[item.coverage.image_cell] > 0
        ]
        if not eligible:
            raise RuntimeError(f"{role} image-cell quotas became infeasible")
        base_logdet = _information_logdet(information)
        information_gains = np.asarray(
            [
                _information_logdet(information + item.information_matrix)
                - base_logdet
                for item in eligible
            ],
            dtype=np.float64,
        )
        coverage_gains = np.asarray(
            [
                _marginal_coverage_gain(item.coverage, coverage_counts)
                for item in eligible
            ],
            dtype=np.float64,
        )
        normalized_information = _normalize_scores(information_gains)
        normalized_coverage = _normalize_scores(coverage_gains)
        combined = (
            (1.0 - config.coverage_score_weight) * normalized_information
            + config.coverage_score_weight * normalized_coverage
        )
        chosen_index = max(
            range(len(eligible)),
            key=lambda index: (
                float(combined[index]),
                float(information_gains[index]),
                eligible[index].target.id,
            ),
        )
        chosen = eligible[chosen_index]
        selected.append(chosen)
        remaining = [item for item in remaining if item is not chosen]
        quotas[chosen.coverage.image_cell] -= 1
        _increment_coverage_counts(chosen.coverage, coverage_counts)
        information += chosen.information_matrix
        steps.append(
            {
                "selection_index": selection_index + 1,
                "role": role,
                "target_id": chosen.target.id,
                "information_gain_logdet": float(information_gains[chosen_index]),
                "coverage_gain": float(coverage_gains[chosen_index]),
                "combined_score": float(combined[chosen_index]),
                "coverage": chosen.coverage.to_dict(),
            }
        )
    return selected, steps


def _balanced_image_quotas(
    capacities: dict[tuple[int, int], int], count: int
) -> dict[tuple[int, int], int]:
    if count < 0 or count > sum(capacities.values()):
        raise ValueError("image-cell quota count exceeds candidate capacity")
    quotas = {cell: 0 for cell in sorted(capacities)}
    for _ in range(count):
        available = [
            cell for cell in sorted(capacities) if quotas[cell] < capacities[cell]
        ]
        if not available:
            raise RuntimeError("image-cell capacities were exhausted")
        cell = min(available, key=lambda item: (quotas[item], item))
        quotas[cell] += 1
    return quotas


def _marginal_coverage_gain(
    signature: _CoverageSignature, counts: dict[str, list[int]]
) -> float:
    indices = {
        "depth": signature.depth_bin,
        "obliquity": signature.obliquity_bin,
        "azimuth": signature.azimuth_bin,
        "in_plane_rotation": signature.in_plane_rotation_bin,
    }
    return float(
        sum(1.0 / (1.0 + counts[name][index]) for name, index in indices.items())
    )


def _increment_coverage_counts(
    signature: _CoverageSignature, counts: dict[str, list[int]]
) -> None:
    counts["depth"][signature.depth_bin] += 1
    counts["obliquity"][signature.obliquity_bin] += 1
    counts["azimuth"][signature.azimuth_bin] += 1
    counts["in_plane_rotation"][signature.in_plane_rotation_bin] += 1


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("selection scores must be finite")
    lower = float(np.min(scores))
    upper = float(np.max(scores))
    if upper - lower <= 1e-12:
        return np.ones_like(scores)
    return (scores - lower) / (upper - lower)


def _information_logdet(information: np.ndarray) -> float:
    matrix = np.asarray(information, dtype=np.float64)
    sign, value = np.linalg.slogdet((matrix + matrix.T) / 2.0)
    if sign <= 0 or not np.isfinite(value):
        raise ValueError("information matrix is not positive definite")
    return float(value)


def _selection_diagnostics(
    *,
    candidates: list[_FeasibleCandidate],
    selected: list[_FeasibleCandidate],
    selection_steps: list[dict],
    config: AutoCollectionDesignConfig,
) -> dict:
    return {
        "method": "reachable_image_balanced_scaled_d_optimal",
        "parameterization": "12_parameter_camera_and_palm_target_extrinsics",
        "parameter_scales": {
            "translation_m": config.information_translation_scale_m,
            "rotation_deg": config.information_rotation_scale_deg,
        },
        "feasible_candidate_count": len(candidates),
        "selected_count": len(selected),
        "selected_information": _information_summary(selected, config),
        "coverage": {
            "feasible": _coverage_summary(candidates, config),
            "selected": _coverage_summary(selected, config),
        },
        "selection_steps": selection_steps,
    }


def _information_summary(
    candidates: list[_FeasibleCandidate], config: AutoCollectionDesignConfig
) -> dict:
    information = np.eye(12, dtype=np.float64) * config.information_ridge
    for candidate in candidates:
        information += candidate.information_matrix
    eigenvalues = np.linalg.eigvalsh((information + information.T) / 2.0)
    largest = float(eigenvalues[-1])
    smallest = float(eigenvalues[0])
    if smallest <= 0:
        raise ValueError("regularized information matrix is not positive definite")
    condition = float(np.sqrt(largest / smallest))
    return {
        "scaled_information_logdet": _information_logdet(information),
        "scaled_jacobian_condition_number": condition,
        "scaled_information_eigenvalues": [float(value) for value in eigenvalues],
    }


def _coverage_summary(
    candidates: list[_FeasibleCandidate], config: AutoCollectionDesignConfig
) -> dict:
    image = [
        [0 for _ in range(config.image_grid_columns)]
        for _ in range(config.image_grid_rows)
    ]
    counts = {
        "depth": [0] * config.depth_bins,
        "obliquity": [0] * config.obliquity_bins,
        "azimuth": [0] * config.azimuth_bins,
        "in_plane_rotation": [0] * config.in_plane_rotation_bins,
    }
    for candidate in candidates:
        col, row = candidate.coverage.image_cell
        image[row][col] += 1
        _increment_coverage_counts(candidate.coverage, counts)
    return {
        "image_cells": image,
        "depth_bins": counts["depth"],
        "obliquity_bins": counts["obliquity"],
        "azimuth_bins": counts["azimuth"],
        "in_plane_rotation_bins": counts["in_plane_rotation"],
    }


def _sample_camera_T_target(
    sample: np.ndarray,
    *,
    camera_info: RectifiedCameraInfo,
    target_dimensions_m: np.ndarray,
    exposed_target_normal: np.ndarray,
    config: AutoCollectionDesignConfig,
) -> np.ndarray | None:
    sample = np.asarray(sample, dtype=np.float64).reshape(-1)
    if sample.shape != (6,) or not np.all(np.isfinite(sample)):
        raise ValueError("camera target sample must contain six finite values")
    dimensions = np.asarray(target_dimensions_m, dtype=np.float64).reshape(-1)
    if dimensions.shape != (3,) or np.any(dimensions <= 0):
        raise ValueError("target dimensions must contain three positive values")
    margin = config.image_margin_px
    if 2 * margin >= camera_info.width or 2 * margin >= camera_info.height:
        raise ValueError("image margin removes the complete camera frame")
    depth = config.minimum_depth_m + sample[2] * (
        config.maximum_depth_m - config.minimum_depth_m
    )
    matrix = camera_info.rectified_camera_matrix
    fx, fy = matrix[0, 0], matrix[1, 1]
    cx, cy = matrix[0, 2], matrix[1, 2]
    bounding_radius = float(np.linalg.norm(dimensions) / 2.0)
    nearest_target_depth = depth - bounding_radius
    if nearest_target_depth <= 0:
        return None
    half_width_px = fx * bounding_radius / nearest_target_depth
    half_height_px = fy * bounding_radius / nearest_target_depth
    lower_u, upper_u = (
        margin + half_width_px,
        camera_info.width - margin - half_width_px,
    )
    lower_v, upper_v = (
        margin + half_height_px,
        camera_info.height - margin - half_height_px,
    )
    if lower_u >= upper_u or lower_v >= upper_v:
        return None
    center_u = lower_u + sample[0] * (upper_u - lower_u)
    center_v = lower_v + sample[1] * (upper_v - lower_v)
    center = np.asarray(
        [(center_u - cx) * depth / fx, (center_v - cy) * depth / fy, depth]
    )
    exposed_normal = _unit(exposed_target_normal)
    reference = np.asarray([1.0, 0.0, 0.0])
    if abs(float(reference @ exposed_normal)) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0])
    tangent_x = _unit(np.cross(exposed_normal, reference))
    tangent_y = np.cross(exposed_normal, tangent_x)
    azimuth = 2.0 * np.pi * sample[3]
    minimum_cosine = np.cos(np.deg2rad(config.maximum_view_obliquity_deg))
    cosine = 1.0 - sample[4] * (1.0 - minimum_cosine)
    sine = np.sqrt(max(0.0, 1.0 - cosine * cosine))
    target_to_camera = (
        cosine * exposed_normal
        + sine * (np.cos(azimuth) * tangent_x + np.sin(azimuth) * tangent_y)
    )
    roll = np.deg2rad((2.0 * sample[5] - 1.0) * config.maximum_view_roll_deg)
    rotation = _view_rotation(
        target_to_camera_in_target=target_to_camera,
        target_to_camera_in_camera=-center / np.linalg.norm(center),
        roll_rad=roll,
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = center
    if not _target_projects_inside(
        transform,
        camera_info=camera_info,
        target_dimensions_m=dimensions,
        margin_px=margin,
        minimum_span_px=config.minimum_projected_target_span_px,
    ):
        return None
    return validate_transform(transform)


def _view_rotation(
    *,
    target_to_camera_in_target: np.ndarray,
    target_to_camera_in_camera: np.ndarray,
    roll_rad: float,
) -> np.ndarray:
    source_z = _unit(target_to_camera_in_target)
    source_reference = np.asarray([0.0, 0.0, 1.0])
    if abs(float(source_reference @ source_z)) > 0.9:
        source_reference = np.asarray([0.0, 1.0, 0.0])
    source_x = _unit(np.cross(source_reference, source_z))
    source_y = np.cross(source_z, source_x)
    source_basis = np.column_stack((source_x, source_y, source_z))

    target_z = _unit(target_to_camera_in_camera)
    image_up = np.asarray([0.0, -1.0, 0.0])
    if abs(float(image_up @ target_z)) > 0.9:
        image_up = np.asarray([1.0, 0.0, 0.0])
    target_x = _unit(np.cross(image_up, target_z))
    target_y = np.cross(target_z, target_x)
    cosine, sine = np.cos(roll_rad), np.sin(roll_rad)
    rolled_x = cosine * target_x + sine * target_y
    rolled_y = -sine * target_x + cosine * target_y
    target_basis = np.column_stack((rolled_x, rolled_y, target_z))
    return target_basis @ source_basis.T


def _target_projects_inside(
    camera_T_target: np.ndarray,
    *,
    camera_info: RectifiedCameraInfo,
    target_dimensions_m: np.ndarray,
    margin_px: float,
    minimum_span_px: float,
) -> bool:
    half = np.asarray(target_dimensions_m, dtype=np.float64) / 2.0
    corners = np.asarray(
        [
            [x, y, z]
            for x in (-half[0], half[0])
            for y in (-half[1], half[1])
            for z in (-half[2], half[2])
        ]
    )
    camera_points = (
        camera_T_target[:3, :3] @ corners.T
    ).T + camera_T_target[:3, 3]
    if np.any(camera_points[:, 2] <= 0):
        return False
    projected = (camera_info.rectified_camera_matrix @ camera_points.T).T
    pixels = projected[:, :2] / projected[:, 2, None]
    if (
        np.min(pixels[:, 0]) < margin_px
        or np.max(pixels[:, 0]) > camera_info.width - margin_px
        or np.min(pixels[:, 1]) < margin_px
        or np.max(pixels[:, 1]) > camera_info.height - margin_px
    ):
        return False
    face_corners = np.asarray(
        [
            [-half[0], half[1], 0.0],
            [half[0], half[1], 0.0],
            [half[0], -half[1], 0.0],
            [-half[0], -half[1], 0.0],
        ]
    )
    camera_face = (
        camera_T_target[:3, :3] @ face_corners.T
    ).T + camera_T_target[:3, 3]
    projected_face = (camera_info.rectified_camera_matrix @ camera_face.T).T
    face_pixels = projected_face[:, :2] / projected_face[:, 2, None]
    edge_lengths = np.linalg.norm(
        face_pixels - np.roll(face_pixels, shift=-1, axis=0), axis=1
    )
    return float(np.min(edge_lengths)) >= minimum_span_px


def _target_dimensions_m(target: dict) -> np.ndarray:
    dimensions = np.asarray(target["box_dims"], dtype=np.float64).reshape(-1)
    if dimensions.shape != (3,) or np.any(dimensions <= 0):
        raise ValueError("target box_dims must contain three positive millimetres")
    return dimensions / 1000.0


def _target_object_points_m(target: dict) -> np.ndarray:
    markers = target.get("markers")
    if not isinstance(markers, list) or not markers:
        raise ValueError("target markers must be a non-empty list")
    points: list[np.ndarray] = []
    for marker in markers:
        corners = np.asarray(marker["corners_mm"], dtype=np.float64)
        if corners.shape != (4, 3) or not np.all(np.isfinite(corners)):
            raise ValueError("every target marker must contain four finite 3D corners")
        points.append(corners / 1000.0)
    return np.vstack(points)


def _candidate_information_matrix(
    *,
    model: URDFModel,
    reference_full_q: np.ndarray,
    calibration_arm: str,
    command_calibration_q: np.ndarray,
    camera_info: RectifiedCameraInfo,
    object_points_m: np.ndarray,
    torso_T_camera: np.ndarray,
    hand_T_target: np.ndarray,
    translation_scale_m: float,
    rotation_scale_rad: float,
) -> np.ndarray:
    full_q = np.asarray(reference_full_q, dtype=np.float64).copy()
    full_q[np.asarray(arm_indices(calibration_arm))] = np.asarray(
        command_calibration_q, dtype=np.float64
    )
    positions = dict(zip(G1_29_JOINT_NAMES, full_q, strict=True))
    torso_T_hand = model.transform(
        "torso_link", arm_hand_link(calibration_arm), positions
    )
    parameters = np.concatenate(
        (
            transform_to_pose_vector(torso_T_camera),
            transform_to_pose_vector(hand_T_target),
        )
    )
    scales = np.asarray(
        [translation_scale_m] * 3
        + [rotation_scale_rad] * 3
        + [translation_scale_m] * 3
        + [rotation_scale_rad] * 3,
        dtype=np.float64,
    )
    normalized_step = 1e-4
    jacobian = np.empty((2 * len(object_points_m), 12), dtype=np.float64)
    for index, scale in enumerate(scales):
        delta = np.zeros(12, dtype=np.float64)
        delta[index] = normalized_step * scale
        plus = _project_parameterized_target(
            parameters + delta,
            torso_T_hand=torso_T_hand,
            object_points_m=object_points_m,
            camera_info=camera_info,
        )
        minus = _project_parameterized_target(
            parameters - delta,
            torso_T_hand=torso_T_hand,
            object_points_m=object_points_m,
            camera_info=camera_info,
        )
        jacobian[:, index] = ((plus - minus) / (2.0 * normalized_step)).reshape(-1)
    information = jacobian.T @ jacobian
    return (information + information.T) / 2.0


def _project_parameterized_target(
    parameters: np.ndarray,
    *,
    torso_T_hand: np.ndarray,
    object_points_m: np.ndarray,
    camera_info: RectifiedCameraInfo,
) -> np.ndarray:
    vector = np.asarray(parameters, dtype=np.float64)
    torso_T_camera = pose_vector_to_transform(vector[:6])
    hand_T_target = pose_vector_to_transform(vector[6:])
    camera_points = transform_points(
        invert_transform(torso_T_camera) @ torso_T_hand @ hand_T_target,
        object_points_m,
    )
    if np.any(camera_points[:, 2] <= 0):
        raise ValueError("predicted calibration target lies behind the camera")
    projected = (camera_info.rectified_camera_matrix @ camera_points.T).T
    return projected[:, :2] / projected[:, 2, None]


def _coverage_signature(
    camera_T_target: np.ndarray,
    *,
    camera_info: RectifiedCameraInfo,
    exposed_target_normal: np.ndarray,
    config: AutoCollectionDesignConfig,
) -> _CoverageSignature:
    transform = validate_transform(camera_T_target)
    center = transform[:3, 3]
    if center[2] <= 0:
        raise ValueError("target center must lie in front of the camera")
    matrix = camera_info.rectified_camera_matrix
    center_pixel = matrix @ center
    center_pixel = center_pixel[:2] / center_pixel[2]
    normalized = (
        float(np.clip(center_pixel[0] / camera_info.width, 0.0, 1.0)),
        float(np.clip(center_pixel[1] / camera_info.height, 0.0, 1.0)),
    )
    image_cell = (
        min(int(normalized[0] * config.image_grid_columns), config.image_grid_columns - 1),
        min(int(normalized[1] * config.image_grid_rows), config.image_grid_rows - 1),
    )
    normal = _unit(exposed_target_normal)
    target_to_camera = _unit(transform[:3, :3].T @ -center)
    cosine = float(np.clip(target_to_camera @ normal, -1.0, 1.0))
    obliquity_deg = float(np.degrees(np.arccos(cosine)))
    reference = np.asarray([1.0, 0.0, 0.0])
    if abs(float(reference @ normal)) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0])
    tangent_x = _unit(np.cross(normal, reference))
    tangent_y = np.cross(normal, tangent_x)
    tangent = target_to_camera - cosine * normal
    if np.linalg.norm(tangent) <= 1e-12:
        azimuth_deg = 0.0
    else:
        tangent = _unit(tangent)
        azimuth_deg = float(
            np.degrees(
                np.arctan2(float(tangent @ tangent_y), float(tangent @ tangent_x))
            )
            % 360.0
        )
    x_endpoint = center + 0.01 * transform[:3, 0]
    endpoint_pixel = matrix @ x_endpoint
    endpoint_pixel = endpoint_pixel[:2] / endpoint_pixel[2]
    image_axis = endpoint_pixel - center_pixel
    in_plane_rotation_deg = float(
        np.degrees(np.arctan2(image_axis[1], image_axis[0])) % 360.0
    )
    return _CoverageSignature(
        image_cell=image_cell,
        depth_bin=_bounded_bin(
            float(center[2]),
            lower=config.minimum_depth_m,
            upper=config.maximum_depth_m,
            count=config.depth_bins,
        ),
        obliquity_bin=_bounded_bin(
            obliquity_deg,
            lower=0.0,
            upper=config.maximum_view_obliquity_deg,
            count=config.obliquity_bins,
        ),
        azimuth_bin=_cyclic_bin(azimuth_deg, config.azimuth_bins),
        in_plane_rotation_bin=_cyclic_bin(
            in_plane_rotation_deg, config.in_plane_rotation_bins
        ),
        normalized_centroid=normalized,
        depth_m=float(center[2]),
        obliquity_deg=obliquity_deg,
        azimuth_deg=azimuth_deg,
        in_plane_rotation_deg=in_plane_rotation_deg,
    )


def _bounded_bin(value: float, *, lower: float, upper: float, count: int) -> int:
    if not lower < upper or count < 1:
        raise ValueError("bounded-bin configuration is invalid")
    fraction = float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))
    return min(int(fraction * count), count - 1)


def _cyclic_bin(angle_deg: float, count: int) -> int:
    if count < 1:
        raise ValueError("cyclic-bin count must be positive")
    return min(int((angle_deg % 360.0) / 360.0 * count), count - 1)


def _unit(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("cannot normalize a zero/non-finite vector")
    return array / norm


def _unique_edges(route: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(dict.fromkeys(pairwise(route)))


def load_design_inputs(
    *, hardware_path: str | Path, target_path: str | Path, state_path: str | Path
) -> tuple[dict, dict, RobotStateSample]:
    with Path(hardware_path).open(encoding="utf-8") as stream:
        hardware = yaml.safe_load(stream)
    with Path(target_path).open(encoding="utf-8") as stream:
        target = json.load(stream)
    with Path(state_path).open(encoding="utf-8") as stream:
        state = RobotStateSample.from_dict(json.load(stream))
    if not isinstance(hardware, dict) or not isinstance(target, dict):
        raise TypeError("hardware and target configurations must be mappings")
    validate_hardware_target_profile(hardware, target)
    return hardware, target, state
