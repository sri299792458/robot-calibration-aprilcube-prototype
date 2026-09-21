"""Dynamic IK selection and FCL approval for the tabletop accuracy move."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from g1_aprilcube_calibration.collision import CollisionConfig, FCLCollisionChecker
from g1_aprilcube_calibration.inverse_kinematics import (
    IKConfig,
    IKSolution,
    solve_arm_ik_candidates,
)
from g1_aprilcube_calibration.joint_map import (
    G1_29_JOINT_NAMES,
    arm_hand_link,
    arm_indices,
    arm_joint_names,
    validate_arm_side,
    validate_full_joint_vector,
)
from g1_aprilcube_calibration.pose_schema import (
    HANDOFF_POSE_ID,
    PoseAuditEvent,
    PoseRecord,
    PoseSet,
)
from g1_aprilcube_calibration.pose_validator import (
    PathValidationConfig,
    PosePathValidator,
    ValidationReport,
)
from g1_aprilcube_calibration.transforms import invert_transform, validate_transform
from g1_aprilcube_calibration.urdf_model import (
    URDFModel,
    transform_from_xyz_rpy,
)

TABLE_TARGET_POSE_ID = "table_accuracy_target"
TABLE_ESCAPE_POSE_ID = "table_lifted_start"


@dataclass(frozen=True, slots=True)
class TableMotionPreflight:
    pose_set: PoseSet
    validation_report: ValidationReport
    escape_ik_solution: IKSolution
    target_ik_solution: IKSolution
    escape_clearance: TablePlaneEscapeClearance
    elevated_route_clearance: TablePlaneClearance
    escape_ik_candidate_count: int
    target_ik_candidate_count: int
    rejected_candidates: tuple[dict, ...]

    def to_dict(self) -> dict:
        return {
            "pose_set_sha256": self.pose_set.content_sha256,
            "validation_report_sha256": self.validation_report.content_sha256,
            "escape_ik_solution": self.escape_ik_solution.to_dict(),
            "target_ik_solution": self.target_ik_solution.to_dict(),
            "escape_clearance": self.escape_clearance.to_dict(),
            "elevated_route_clearance": self.elevated_route_clearance.to_dict(),
            "escape_ik_candidate_count": self.escape_ik_candidate_count,
            "target_ik_candidate_count": self.target_ik_candidate_count,
            "rejected_candidates": list(self.rejected_candidates),
        }


@dataclass(frozen=True, slots=True)
class TablePlaneConfig:
    board_width_m: float
    board_height_m: float
    table_margin_m: float = 0.2
    maximum_joint_increment_rad: float = 0.005
    minimum_clearance_m: float = 0.01

    def __post_init__(self) -> None:
        if self.board_width_m <= 0 or self.board_height_m <= 0:
            raise ValueError("table board dimensions must be positive")
        if self.table_margin_m < 0:
            raise ValueError("table margin cannot be negative")
        if self.maximum_joint_increment_rad <= 0:
            raise ValueError("table-plane joint increment must be positive")
        if self.minimum_clearance_m <= 0:
            raise ValueError("table-plane clearance must be positive")


@dataclass(frozen=True, slots=True)
class TablePlaneClearance:
    minimum_clearance_m: float
    minimum_link: str
    minimum_sample_index: int
    sample_count: int
    required_clearance_m: float
    modeled_table_xy_bounds_m: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not np.isfinite(self.minimum_clearance_m):
            raise ValueError("table-plane clearance must be finite")
        if not self.minimum_link:
            raise ValueError("table-plane minimum link must be non-empty")
        if self.minimum_sample_index < 0 or self.sample_count < 2:
            raise ValueError("table-plane sample indices are invalid")
        if self.minimum_sample_index >= self.sample_count:
            raise ValueError("table-plane minimum sample is outside the path")
        if self.required_clearance_m <= 0:
            raise ValueError("required table-plane clearance must be positive")
        bounds = np.asarray(self.modeled_table_xy_bounds_m, dtype=np.float64)
        if bounds.shape != (4,) or not np.all(np.isfinite(bounds)):
            raise ValueError("modeled table bounds must contain four finite values")
        if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            raise ValueError("modeled table bounds are invalid")

    @property
    def passed(self) -> bool:
        return self.minimum_clearance_m >= self.required_clearance_m

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "minimum_clearance_m": self.minimum_clearance_m,
            "minimum_link": self.minimum_link,
            "minimum_sample_index": self.minimum_sample_index,
            "sample_count": self.sample_count,
            "required_clearance_m": self.required_clearance_m,
            "modeled_table_xy_bounds_m": list(self.modeled_table_xy_bounds_m),
            "plane_model": "board_rectangle_expanded_by_table_margin",
        }


@dataclass(frozen=True, slots=True)
class TablePlaneEscapeClearance:
    """Plane evidence for leaving an initially table-supported configuration."""

    source_clearance_m: float | None
    destination_clearance_m: float | None
    minimum_clearance_m: float | None
    minimum_link: str | None
    minimum_sample_index: int | None
    sample_count: int
    required_destination_clearance_m: float
    minimum_allowed_path_clearance_m: float
    maximum_additional_penetration_m: float
    modeled_table_xy_bounds_m: tuple[float, float, float, float]
    numerical_tolerance_m: float = 1e-6

    def __post_init__(self) -> None:
        for name in (
            "source_clearance_m",
            "destination_clearance_m",
            "minimum_clearance_m",
        ):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite when present")
        if self.minimum_clearance_m is None:
            if self.minimum_link is not None or self.minimum_sample_index is not None:
                raise ValueError("an empty escape-plane result cannot name a minimum")
        elif not self.minimum_link or self.minimum_sample_index is None:
            raise ValueError(
                "an escape-plane minimum must identify its link and sample"
            )
        if self.minimum_sample_index is not None and not (
            0 <= self.minimum_sample_index < self.sample_count
        ):
            raise ValueError("escape-plane minimum sample is outside the path")
        if self.sample_count < 2:
            raise ValueError("escape-plane path must contain at least two samples")
        if self.required_destination_clearance_m <= 0:
            raise ValueError("required escape destination clearance must be positive")
        if not np.isfinite(self.minimum_allowed_path_clearance_m):
            raise ValueError("minimum allowed escape clearance must be finite")
        if self.maximum_additional_penetration_m < 0:
            raise ValueError("additional table penetration cannot be negative")
        if self.numerical_tolerance_m <= 0:
            raise ValueError("escape-plane numerical tolerance must be positive")
        bounds = np.asarray(self.modeled_table_xy_bounds_m, dtype=np.float64)
        if bounds.shape != (4,) or not np.all(np.isfinite(bounds)):
            raise ValueError("modeled escape bounds must contain four finite values")
        if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            raise ValueError("modeled escape bounds are invalid")

    @property
    def passed(self) -> bool:
        path_passed = (
            self.minimum_clearance_m is None
            or self.minimum_clearance_m
            >= self.minimum_allowed_path_clearance_m - self.numerical_tolerance_m
        )
        destination_passed = (
            self.destination_clearance_m is None
            or self.destination_clearance_m >= self.required_destination_clearance_m
        )
        return path_passed and destination_passed

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "source_clearance_m": self.source_clearance_m,
            "destination_clearance_m": self.destination_clearance_m,
            "minimum_clearance_m": self.minimum_clearance_m,
            "minimum_link": self.minimum_link,
            "minimum_sample_index": self.minimum_sample_index,
            "sample_count": self.sample_count,
            "required_destination_clearance_m": (self.required_destination_clearance_m),
            "minimum_allowed_path_clearance_m": (self.minimum_allowed_path_clearance_m),
            "maximum_additional_penetration_m": (self.maximum_additional_penetration_m),
            "modeled_table_xy_bounds_m": list(self.modeled_table_xy_bounds_m),
            "plane_model": (
                "supported_start_may_touch_modeled_table_rectangle; "
                "path_must_not_deepen_contact; destination_must_clear"
            ),
        }


def build_lifted_start_cube_target(
    *,
    current_board_T_cube: np.ndarray,
    desired_board_T_cube: np.ndarray,
) -> np.ndarray:
    """Lift the live cube vertically to the plan's relative-lift target Z."""

    current = validate_transform(current_board_T_cube)
    desired = validate_transform(desired_board_T_cube)
    if desired[2, 3] >= current[2, 3]:
        raise ValueError(
            "planned escape target does not lift the live cube above its "
            "supported position"
        )
    escape = current.copy()
    # Board +Z points into the table.  The plan target was constructed exactly
    # lift_mm above its observed supported pose, hence its more-negative Z is
    # also the height for the horizontal corner move.
    escape[2, 3] = float(desired[2, 3])
    return validate_transform(escape)


def build_table_motion_preflight(
    *,
    model: URDFModel,
    collision_config: CollisionConfig,
    desired_torso_T_escape: np.ndarray,
    desired_torso_T_hand: np.ndarray,
    torso_T_board: np.ndarray,
    reference_full_q: np.ndarray,
    calibration_arm: str,
    plan_sha256: str,
    recorded_at_utc: str,
    recorded_monotonic_s: float,
    ik_config: IKConfig | None = None,
    path_config: PathValidationConfig | None = None,
    table_plane_config: TablePlaneConfig,
) -> TableMotionPreflight:
    """Select a lifted start, elevated corner target, and exact reverse route."""
    calibration_arm = validate_arm_side(calibration_arm)
    if len(plan_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in plan_sha256
    ):
        raise ValueError("table plan hash must be lowercase SHA-256")
    try:
        escape_candidates = solve_arm_ik_candidates(
            model,
            desired_torso_T_hand=desired_torso_T_escape,
            reference_full_q=reference_full_q,
            calibration_arm=calibration_arm,
            config=ik_config,
        )
    except ValueError as error:
        raise ValueError(f"lifted-start escape waypoint IK failed: {error}") from error
    try:
        target_candidates = solve_arm_ik_candidates(
            model,
            desired_torso_T_hand=desired_torso_T_hand,
            reference_full_q=reference_full_q,
            calibration_arm=calibration_arm,
            config=ik_config,
        )
    except ValueError as error:
        raise ValueError(f"table target waypoint IK failed: {error}") from error
    validator = PosePathValidator(
        model=model,
        collision_checker=FCLCollisionChecker(model, collision_config),
        config=path_config,
    )
    reference = validate_full_joint_vector(reference_full_q)
    indices = np.asarray(arm_indices(calibration_arm))
    source_q = reference[indices]
    candidate_pairs = [
        (escape_index, target_index, escape, target)
        for escape_index, escape in enumerate(escape_candidates)
        for target_index, target in enumerate(target_candidates)
    ]
    candidate_pairs.sort(
        key=lambda item: (
            float(np.linalg.norm(np.asarray(item[2].calibration_q) - source_q))
            + 2.0
            * float(
                np.linalg.norm(
                    np.asarray(item[3].calibration_q)
                    - np.asarray(item[2].calibration_q)
                )
            ),
            item[0],
            item[1],
        )
    )
    rejected: list[dict] = []
    for escape_index, target_index, escape, target in candidate_pairs:
        pose_set = _motion_pose_set(
            model=model,
            reference_full_q=reference_full_q,
            calibration_arm=calibration_arm,
            escape_q=np.asarray(escape.calibration_q),
            target_q=np.asarray(target.calibration_q),
            plan_sha256=plan_sha256,
            recorded_at_utc=recorded_at_utc,
            recorded_monotonic_s=recorded_monotonic_s,
            escape_solution=escape,
            target_solution=target,
        )
        report = validator.validate(
            pose_set,
            directed_edges=(
                (HANDOFF_POSE_ID, TABLE_ESCAPE_POSE_ID),
                (TABLE_ESCAPE_POSE_ID, TABLE_TARGET_POSE_ID),
                (TABLE_TARGET_POSE_ID, TABLE_ESCAPE_POSE_ID),
                (TABLE_ESCAPE_POSE_ID, HANDOFF_POSE_ID),
            ),
            reference_full_q=reference_full_q,
        )
        escape_clearance = validate_table_plane_escape_path(
            model=model,
            collision_config=collision_config,
            torso_T_board=torso_T_board,
            reference_full_q=reference_full_q,
            calibration_arm=calibration_arm,
            target_calibration_q=np.asarray(escape.calibration_q),
            config=table_plane_config,
        )
        elevated_route_clearance = validate_table_plane_path(
            model=model,
            collision_config=collision_config,
            torso_T_board=torso_T_board,
            reference_full_q=reference_full_q,
            calibration_arm=calibration_arm,
            source_calibration_q=np.asarray(escape.calibration_q),
            target_calibration_q=np.asarray(target.calibration_q),
            config=table_plane_config,
        )
        if (
            report.passed
            and escape_clearance.passed
            and elevated_route_clearance.passed
        ):
            return TableMotionPreflight(
                pose_set=pose_set,
                validation_report=report,
                escape_ik_solution=escape,
                target_ik_solution=target,
                escape_clearance=escape_clearance,
                elevated_route_clearance=elevated_route_clearance,
                escape_ik_candidate_count=len(escape_candidates),
                target_ik_candidate_count=len(target_candidates),
                rejected_candidates=tuple(rejected),
            )
        table_failures = []
        if not escape_clearance.passed:
            table_failures.append(
                "supported-start escape either deepens table contact or does not "
                "finish with the required clearance"
            )
        if not elevated_route_clearance.passed:
            table_failures.append(
                f"sample {elevated_route_clearance.minimum_sample_index}: "
                f"{elevated_route_clearance.minimum_link} table clearance "
                f"{elevated_route_clearance.minimum_clearance_m:.4f}m is below "
                f"{elevated_route_clearance.required_clearance_m:.4f}m"
            )
        rejected.append(
            {
                "escape_candidate_index": escape_index,
                "target_candidate_index": target_index,
                "escape_ik_solution": escape.to_dict(),
                "target_ik_solution": target.to_dict(),
                "path_failures": [
                    {
                        "edge": f"{edge.from_pose_id}->{edge.to_pose_id}",
                        "failures": list(edge.failures),
                    }
                    for edge in report.edges
                    if not edge.passed
                ],
                "escape_table_plane": escape_clearance.to_dict(),
                "elevated_route_table_plane": elevated_route_clearance.to_dict(),
                "table_failures": table_failures,
            }
        )
    summary = " | ".join(
        f"escape {item['escape_candidate_index']}/target "
        f"{item['target_candidate_index']}: "
        + "; ".join(
            [failure for edge in item["path_failures"] for failure in edge["failures"]]
            + item["table_failures"]
        )
        for item in rejected
    )
    raise ValueError(
        "no escape/target IK pair passed the FCL and table-plane route: " + summary
    )


def validate_table_plane_path(
    *,
    model: URDFModel,
    collision_config: CollisionConfig,
    torso_T_board: np.ndarray,
    reference_full_q: np.ndarray,
    calibration_arm: str,
    source_calibration_q: np.ndarray | None = None,
    target_calibration_q: np.ndarray,
    config: TablePlaneConfig,
) -> TablePlaneClearance:
    """Keep all moving geometry above the board-defined tabletop plane."""

    samples, bounds = _table_plane_path_samples(
        model=model,
        collision_config=collision_config,
        torso_T_board=torso_T_board,
        reference_full_q=reference_full_q,
        calibration_arm=calibration_arm,
        source_calibration_q=source_calibration_q,
        target_calibration_q=target_calibration_q,
        config=config,
    )
    finite = [item for item in samples if item[0] is not None]
    if not finite:
        raise ValueError("motion path never overlaps the modeled tabletop footprint")
    minimum_clearance, minimum_link, minimum_sample_index = min(
        finite, key=lambda item: item[0]
    )
    assert minimum_clearance is not None and minimum_link is not None
    return TablePlaneClearance(
        minimum_clearance_m=minimum_clearance,
        minimum_link=minimum_link,
        minimum_sample_index=minimum_sample_index,
        sample_count=len(samples),
        required_clearance_m=config.minimum_clearance_m,
        modeled_table_xy_bounds_m=bounds,
    )


def validate_table_plane_escape_path(
    *,
    model: URDFModel,
    collision_config: CollisionConfig,
    torso_T_board: np.ndarray,
    reference_full_q: np.ndarray,
    calibration_arm: str,
    target_calibration_q: np.ndarray,
    config: TablePlaneConfig,
) -> TablePlaneEscapeClearance:
    """Allow existing support contact while forbidding deeper commanded contact."""

    samples, bounds = _table_plane_path_samples(
        model=model,
        collision_config=collision_config,
        torso_T_board=torso_T_board,
        reference_full_q=reference_full_q,
        calibration_arm=calibration_arm,
        source_calibration_q=None,
        target_calibration_q=target_calibration_q,
        config=config,
    )
    source_clearance = samples[0][0]
    destination_clearance = samples[-1][0]
    finite = [item for item in samples if item[0] is not None]
    if finite:
        minimum_clearance, minimum_link, minimum_sample_index = min(
            finite, key=lambda item: item[0]
        )
    else:
        minimum_clearance = None
        minimum_link = None
        minimum_sample_index = None
    minimum_allowed = (
        config.minimum_clearance_m
        if source_clearance is None
        else min(source_clearance, config.minimum_clearance_m)
    )
    additional_penetration = (
        0.0
        if minimum_clearance is None
        else max(minimum_allowed - minimum_clearance, 0.0)
    )
    return TablePlaneEscapeClearance(
        source_clearance_m=source_clearance,
        destination_clearance_m=destination_clearance,
        minimum_clearance_m=minimum_clearance,
        minimum_link=minimum_link,
        minimum_sample_index=minimum_sample_index,
        sample_count=len(samples),
        required_destination_clearance_m=config.minimum_clearance_m,
        minimum_allowed_path_clearance_m=minimum_allowed,
        maximum_additional_penetration_m=additional_penetration,
        modeled_table_xy_bounds_m=bounds,
    )


def _table_plane_path_samples(
    *,
    model: URDFModel,
    collision_config: CollisionConfig,
    torso_T_board: np.ndarray,
    reference_full_q: np.ndarray,
    calibration_arm: str,
    source_calibration_q: np.ndarray | None,
    target_calibration_q: np.ndarray,
    config: TablePlaneConfig,
) -> tuple[
    tuple[tuple[float | None, str | None, int], ...],
    tuple[float, float, float, float],
]:
    calibration_arm = validate_arm_side(calibration_arm)
    reference = validate_full_joint_vector(reference_full_q)
    target = np.asarray(target_calibration_q, dtype=np.float64).reshape(-1)
    if target.shape != (7,) or not np.all(np.isfinite(target)):
        raise ValueError("table-plane target must contain seven finite joints")
    indices = np.asarray(arm_indices(calibration_arm))
    if source_calibration_q is None:
        source = reference[indices]
    else:
        source = np.asarray(source_calibration_q, dtype=np.float64).reshape(-1)
        if source.shape != (7,) or not np.all(np.isfinite(source)):
            raise ValueError("table-plane source must contain seven finite joints")
    board_T_torso = invert_transform(validate_transform(torso_T_board))
    maximum_delta = float(np.max(np.abs(target - source)))
    intervals = max(
        int(np.ceil(maximum_delta / config.maximum_joint_increment_rad)),
        1,
    )
    links = tuple(
        [model.joints[name].child for name in arm_joint_names(calibration_arm)]
        + [arm_hand_link(calibration_arm)]
    )
    geometry_points: dict[
        str, tuple[str, tuple[tuple[np.ndarray, np.ndarray], ...]]
    ] = {}
    for link in links:
        geometries = model.link_geometries(
            link,
            visual_fallback=link == arm_hand_link(calibration_arm),
        )
        if not geometries:
            raise ValueError(f"moving arm link has no table-check geometry: {link}")
        geometry_points[link] = (
            link,
            tuple(
                (item.local_transform, _bounds_corners(item.mesh.bounds))
                for item in geometries
            ),
        )
    for attached in collision_config.attached_boxes:
        if attached.parent_link not in links:
            continue
        local = transform_from_xyz_rpy(
            np.asarray(attached.xyz_m, dtype=np.float64),
            np.asarray(attached.rpy_rad, dtype=np.float64),
        )
        half = np.asarray(attached.size_m, dtype=np.float64) / 2.0
        corners = _bounds_corners(np.vstack((-half, half)))
        geometry_points[attached.name] = (
            attached.parent_link,
            ((local, corners),),
        )

    bounds = (
        -config.table_margin_m,
        config.board_width_m + config.table_margin_m,
        -config.table_margin_m,
        config.board_height_m + config.table_margin_m,
    )
    samples: list[tuple[float | None, str | None, int]] = []
    for sample_index, alpha in enumerate(np.linspace(0.0, 1.0, intervals + 1)):
        full = reference.copy()
        full[indices] = source + alpha * (target - source)
        positions = dict(zip(G1_29_JOINT_NAMES, full, strict=True))
        transforms = model.forward_kinematics(positions, root_link="torso_link")
        sample_minimum: float | None = None
        sample_link: str | None = None
        for name, (parent, parts) in geometry_points.items():
            torso_T_parent = transforms.get(parent)
            if torso_T_parent is None:
                raise ValueError(f"table check is missing FK for {parent}")
            for local_transform, points in parts:
                torso_points = _transform_points(
                    torso_T_parent @ local_transform,
                    points,
                )
                board_points = _transform_points(board_T_torso, torso_points)
                x_min, y_min = np.min(board_points[:, :2], axis=0)
                x_max, y_max = np.max(board_points[:, :2], axis=0)
                overlaps_table = not (
                    x_max < bounds[0]
                    or x_min > bounds[1]
                    or y_max < bounds[2]
                    or y_min > bounds[3]
                )
                if not overlaps_table:
                    continue
                # The printed board defines +Z into the backing/table, so a
                # negative board Z is free space above the tabletop.
                clearance = -float(np.max(board_points[:, 2]))
                if sample_minimum is None or clearance < sample_minimum:
                    sample_minimum = clearance
                    sample_link = name
        samples.append((sample_minimum, sample_link, sample_index))
    return tuple(samples), bounds


def _bounds_corners(bounds: np.ndarray) -> np.ndarray:
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        raise ValueError("geometry bounds must be finite 2x3")
    lower, upper = bounds
    return np.asarray(
        [
            (x, y, z)
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=np.float64,
    )


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    transform = validate_transform(transform)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("geometry points must be a finite Nx3 array")
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def _motion_pose_set(
    *,
    model: URDFModel,
    reference_full_q: np.ndarray,
    calibration_arm: str,
    escape_q: np.ndarray,
    target_q: np.ndarray,
    plan_sha256: str,
    recorded_at_utc: str,
    recorded_monotonic_s: float,
    escape_solution: IKSolution,
    target_solution: IKSolution,
) -> PoseSet:
    reference = np.asarray(reference_full_q, dtype=np.float64).reshape(-1)
    if reference.shape != (29,) or not np.all(np.isfinite(reference)):
        raise ValueError("table motion reference must contain 29 finite joints")
    indices = np.asarray(arm_indices(calibration_arm))
    empty = PoseSet(
        robot_model=model.name,
        mode_machine=5,
        urdf_sha256=model.sha256,
        calibration_arm=calibration_arm,
    )
    escape_pose = PoseRecord(
        id=TABLE_ESCAPE_POSE_ID,
        group="table_accuracy",
        measured_calibration_q=tuple(reference[indices]),
        measured_full_q=tuple(reference),
        calibration_q_spread=(0.0,) * 7,
        recorded_at_utc=recorded_at_utc,
        recorded_monotonic_s=recorded_monotonic_s,
        replay_calibration_q=tuple(escape_q),
        source="table_accuracy_escape_ik",
        visual_quality={
            "table_accuracy_plan_sha256": plan_sha256,
            "ik_solution": escape_solution.to_dict(),
        },
    )
    target_pose = PoseRecord(
        id=TABLE_TARGET_POSE_ID,
        group="table_accuracy",
        measured_calibration_q=tuple(reference[indices]),
        measured_full_q=tuple(reference),
        calibration_q_spread=(0.0,) * 7,
        recorded_at_utc=recorded_at_utc,
        recorded_monotonic_s=recorded_monotonic_s,
        replay_calibration_q=tuple(target_q),
        source="table_accuracy_target_ik",
        visual_quality={
            "table_accuracy_plan_sha256": plan_sha256,
            "ik_solution": target_solution.to_dict(),
        },
    )
    with_escape = empty.with_pose(
        escape_pose,
        PoseAuditEvent(
            action="add",
            pose_id=escape_pose.id,
            occurred_at_utc=recorded_at_utc,
            details={"source": "dynamic_table_accuracy_escape_ik"},
        ),
    )
    return with_escape.with_pose(
        target_pose,
        PoseAuditEvent(
            action="add",
            pose_id=target_pose.id,
            occurred_at_utc=recorded_at_utc,
            details={"source": "dynamic_table_accuracy_target_ik"},
        ),
    )
