"""Validated bilateral shoulder clearance before commanding Dex3 fingers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product

import numpy as np

from g1_aprilcube_calibration.clock import MonotonicClock
from g1_aprilcube_calibration.collision import CollisionConfig, FCLCollisionChecker
from g1_aprilcube_calibration.executor_state_machine import (
    ExecutorConfig,
    ExecutorEvent,
    ExecutorState,
)
from g1_aprilcube_calibration.gravity_compensation import ArmGravityFeedforward
from g1_aprilcube_calibration.joint_map import (
    DUAL_ARM_DOF,
    G1_29_JOINT_NAMES,
    LEFT_ARM_INDICES,
    RIGHT_ARM_INDICES,
    dual_arm_vector,
    validate_arm_vector,
    validate_full_joint_vector,
)
from g1_aprilcube_calibration.motion_profile import velocity_limited_step
from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID
from g1_aprilcube_calibration.pose_validator import (
    PathValidationConfig,
    PosePathValidator,
    ValidationReport,
)
from g1_aprilcube_calibration.transports.base import ArmCommand, ArmTransport
from g1_aprilcube_calibration.transports.unitree_dex3 import (
    DEX3_MOTOR_JOINT_SUFFIXES,
)
from g1_aprilcube_calibration.urdf_model import URDFModel

RIGHT_CLEARANCE_POSE_ID = "right_shoulder_clearance"
DUAL_CLEARANCE_POSE_ID = "dual_shoulder_clearance"
NVIDIA_G1_ARM_HOME_REFERENCE = (
    "unitreerobotics/xr_teleoperate G1_29_ArmController shoulder-roll convention"
)
_COMMAND_COMPLETION_EPSILON_RAD = 1e-9
DEX3_CLEARANCE_INITIAL_OFFSET_RAD = 0.08


@dataclass(frozen=True, slots=True)
class _ClearanceTarget:
    id: str
    command_calibration_q: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _ClearanceCommandSet:
    urdf_sha256: str
    calibration_arm: str
    poses: tuple[_ClearanceTarget, ...]
    content_sha256: str


@dataclass(frozen=True, slots=True)
class DualArmClearancePlan:
    """Live-relative, collision-validated right-then-left shoulder route."""

    source_full_q: tuple[float, ...]
    right_clearance_q14: tuple[float, ...]
    dual_clearance_q14: tuple[float, ...]
    shoulder_roll_offset_rad: float
    initial_left_hand_q_rad: tuple[float, ...]
    initial_right_hand_q_rad: tuple[float, ...]
    target_left_hand_q_rad: tuple[float, ...]
    target_right_hand_q_rad: tuple[float, ...]
    finger_sweep_sample_count: int
    finger_sweep_minimum_clearance_m: float
    finger_sweep_minimum_pair: tuple[str, str] | None
    finger_sweep_recovered_start_limit_joints: tuple[str, ...]
    right_validation: ValidationReport
    left_validation: ValidationReport
    urdf_sha256: str
    collision_config_sha256: str

    def __post_init__(self) -> None:
        source = validate_full_joint_vector(
            self.source_full_q, name="clearance source full q"
        )
        right = _validate_q14(self.right_clearance_q14, "right-clearance q14")
        dual = _validate_q14(self.dual_clearance_q14, "dual-clearance q14")
        if not np.isfinite(self.shoulder_roll_offset_rad) or (
            self.shoulder_roll_offset_rad <= 0.0
        ):
            raise ValueError("shoulder-roll clearance offset must be positive")
        for name in (
            "initial_left_hand_q_rad",
            "initial_right_hand_q_rad",
            "target_left_hand_q_rad",
            "target_right_hand_q_rad",
        ):
            values = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if values.shape != (7,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain seven finite values")
            object.__setattr__(self, name, tuple(float(value) for value in values))
        if self.finger_sweep_sample_count < 2:
            raise ValueError("finger sweep must contain at least two samples")
        if any(
            not isinstance(name, str) or not name.strip()
            for name in self.finger_sweep_recovered_start_limit_joints
        ) or len(set(self.finger_sweep_recovered_start_limit_joints)) != len(
            self.finger_sweep_recovered_start_limit_joints
        ):
            raise ValueError(
                "recovered finger start-limit joints must be unique non-empty names"
            )
        if (
            not np.isfinite(self.finger_sweep_minimum_clearance_m)
            or self.finger_sweep_minimum_clearance_m < 0.0
        ):
            raise ValueError("finger-sweep clearance must be finite and non-negative")
        if not self.right_validation.passed or not self.left_validation.passed:
            raise ValueError("dual-arm clearance plan contains a failed route")
        if self.right_validation.urdf_sha256 != self.urdf_sha256 or (
            self.left_validation.urdf_sha256 != self.urdf_sha256
        ):
            raise ValueError("clearance validation belongs to a different URDF")
        if (
            self.right_validation.collision_config_sha256
            != self.collision_config_sha256
            or self.left_validation.collision_config_sha256
            != self.collision_config_sha256
        ):
            raise ValueError(
                "clearance validation belongs to a different collision profile"
            )
        object.__setattr__(self, "source_full_q", tuple(float(v) for v in source))
        object.__setattr__(self, "right_clearance_q14", tuple(float(v) for v in right))
        object.__setattr__(self, "dual_clearance_q14", tuple(float(v) for v in dual))

    @property
    def source_q14(self) -> tuple[float, ...]:
        full = np.asarray(self.source_full_q, dtype=np.float64)
        return tuple(
            dual_arm_vector(
                full[np.asarray(LEFT_ARM_INDICES)],
                full[np.asarray(RIGHT_ARM_INDICES)],
            )
        )

    @property
    def minimum_clearance_m(self) -> float:
        values = [
            edge.minimum_clearance_m
            for report in (self.right_validation, self.left_validation)
            for edge in report.edges
            if edge.minimum_clearance_m is not None
        ]
        if not values:
            raise RuntimeError("clearance validation reported no collision distance")
        return float(min(values))

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "source_full_q": self.source_full_q,
                    "right_clearance_q14": self.right_clearance_q14,
                    "dual_clearance_q14": self.dual_clearance_q14,
                    "shoulder_roll_offset_rad": self.shoulder_roll_offset_rad,
                    "initial_left_hand_q_rad": self.initial_left_hand_q_rad,
                    "initial_right_hand_q_rad": self.initial_right_hand_q_rad,
                    "target_left_hand_q_rad": self.target_left_hand_q_rad,
                    "target_right_hand_q_rad": self.target_right_hand_q_rad,
                    "finger_sweep_sample_count": self.finger_sweep_sample_count,
                    "finger_sweep_minimum_clearance_m": (
                        self.finger_sweep_minimum_clearance_m
                    ),
                    "finger_sweep_minimum_pair": self.finger_sweep_minimum_pair,
                    "finger_sweep_recovered_start_limit_joints": (
                        self.finger_sweep_recovered_start_limit_joints
                    ),
                    "right_validation": self.right_validation.content_sha256,
                    "left_validation": self.left_validation.content_sha256,
                    "urdf_sha256": self.urdf_sha256,
                    "collision_config_sha256": self.collision_config_sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def target(self, pose_id: str) -> np.ndarray:
        values = {
            HANDOFF_POSE_ID: self.source_q14,
            RIGHT_CLEARANCE_POSE_ID: self.right_clearance_q14,
            DUAL_CLEARANCE_POSE_ID: self.dual_clearance_q14,
        }
        if pose_id not in values:
            raise ValueError(f"unknown shoulder-clearance pose: {pose_id}")
        return np.asarray(values[pose_id], dtype=np.float64)

    def transition_is_validated(self, source: str, target: str) -> bool:
        return (source, target) in {
            (HANDOFF_POSE_ID, RIGHT_CLEARANCE_POSE_ID),
            (RIGHT_CLEARANCE_POSE_ID, DUAL_CLEARANCE_POSE_ID),
            (DUAL_CLEARANCE_POSE_ID, RIGHT_CLEARANCE_POSE_ID),
            (RIGHT_CLEARANCE_POSE_ID, HANDOFF_POSE_ID),
        }


def plan_dual_arm_shoulder_clearance(
    *,
    model: URDFModel,
    collision_checker: FCLCollisionChecker,
    dex3_model: URDFModel,
    dex3_collision_checker: FCLCollisionChecker,
    reference_full_q,
    initial_left_hand_q_rad,
    initial_right_hand_q_rad,
    target_left_hand_q_rad,
    target_right_hand_q_rad,
    path_config: PathValidationConfig,
    progress: Callable[[int, int, float], None] | None = None,
    rejection: Callable[[float, str], None] | None = None,
) -> DualArmClearancePlan:
    """Find the first valid live-relative offset starting at 0.08 rad."""

    reference = validate_full_joint_vector(
        reference_full_q, name="clearance reference full q"
    )
    initial_left = _validate_hand_q(initial_left_hand_q_rad, "initial left hand")
    initial_right = _validate_hand_q(initial_right_hand_q_rad, "initial right hand")
    target_left = _validate_hand_q(target_left_hand_q_rad, "target left hand")
    target_right = _validate_hand_q(target_right_hand_q_rad, "target right hand")
    source_q14 = dual_arm_vector(
        reference[np.asarray(LEFT_ARM_INDICES)],
        reference[np.asarray(RIGHT_ARM_INDICES)],
    )
    validator = PosePathValidator(
        model=model,
        collision_checker=collision_checker,
        config=path_config,
    )
    limits = model.joint_limits(G1_29_JOINT_NAMES)
    left_limit = limits[16]
    right_limit = limits[23]
    source = np.asarray(source_q14, dtype=np.float64)
    maximum_offset = min(
        left_limit.upper - path_config.joint_limit_margin_rad - source[1],
        source[8] - (right_limit.lower + path_config.joint_limit_margin_rad),
    )
    step = path_config.maximum_joint_increment_rad
    candidate_count = (
        int(
            np.floor(
                (maximum_offset - DEX3_CLEARANCE_INITIAL_OFFSET_RAD) / step + 1e-12
            )
        )
        + 1
    )
    if candidate_count < 1:
        raise ValueError(
            "live Ready shoulders cannot reach the 0.0800rad Dex3 clearance "
            "search start within the configured joint-limit margin"
        )
    failures: list[str] = []
    for candidate_index in range(candidate_count):
        offset = DEX3_CLEARANCE_INITIAL_OFFSET_RAD + candidate_index * step
        if progress is not None:
            progress(candidate_index + 1, candidate_count, offset)
        right_q14 = source.copy()
        right_q14[8] -= offset
        dual_q14 = right_q14.copy()
        dual_q14[1] += offset
        sweep = _validate_finger_sweep(
            model=dex3_model,
            collision_checker=dex3_collision_checker,
            body_q=reference,
            arm_q14=dual_q14,
            initial_left_q=initial_left,
            initial_right_q=initial_right,
            target_left_q=target_left,
            target_right_q=target_right,
            config=path_config,
        )
        if not sweep.passed:
            failure = f"finger sweep: {sweep.failure}"
            failures.append(f"offset {offset:.4f}rad {failure}")
            if rejection is not None:
                rejection(offset, failure)
            continue
        right_validation = _validate_side_clearance(
            validator=validator,
            model=model,
            reference=reference,
            side="right",
            target_q=right_q14[7:],
            target_id=RIGHT_CLEARANCE_POSE_ID,
        )
        if not right_validation.passed:
            failure = "right route: " + _failed_clearance_text(right_validation)
            failures.append(f"offset {offset:.4f}rad {failure}")
            if rejection is not None:
                rejection(offset, failure)
            continue
        reference_after_right = reference.copy()
        reference_after_right[np.asarray(RIGHT_ARM_INDICES)] = right_q14[7:]
        left_validation = _validate_side_clearance(
            validator=validator,
            model=model,
            reference=reference_after_right,
            side="left",
            target_q=dual_q14[:7],
            target_id=DUAL_CLEARANCE_POSE_ID,
        )
        if not left_validation.passed:
            failure = "left route: " + _failed_clearance_text(left_validation)
            failures.append(f"offset {offset:.4f}rad {failure}")
            if rejection is not None:
                rejection(offset, failure)
            continue
        return DualArmClearancePlan(
            source_full_q=tuple(reference),
            right_clearance_q14=tuple(right_q14),
            dual_clearance_q14=tuple(dual_q14),
            shoulder_roll_offset_rad=float(offset),
            initial_left_hand_q_rad=tuple(initial_left),
            initial_right_hand_q_rad=tuple(initial_right),
            target_left_hand_q_rad=tuple(target_left),
            target_right_hand_q_rad=tuple(target_right),
            finger_sweep_sample_count=sweep.sample_count,
            finger_sweep_minimum_clearance_m=sweep.minimum_clearance_m,
            finger_sweep_minimum_pair=sweep.minimum_pair,
            finger_sweep_recovered_start_limit_joints=(
                sweep.recovered_start_limit_joints
            ),
            right_validation=right_validation,
            left_validation=left_validation,
            urdf_sha256=model.sha256,
            collision_config_sha256=collision_checker.config.content_sha256,
        )
    detail = " | ".join(failures[-8:])
    raise ValueError(
        "no outward shoulder-roll candidate at or above 0.0800rad produced a "
        f"valid arm route and Dex3 finger sweep across {candidate_count} "
        f"candidates: {detail}"
    )


@dataclass(frozen=True, slots=True)
class FingerSweepResult:
    passed: bool
    sample_count: int
    minimum_clearance_m: float
    minimum_pair: tuple[str, str] | None
    failure: str | None
    recovered_start_limit_joints: tuple[str, ...] = ()


def live_dex3_collision_config(base: CollisionConfig) -> CollisionConfig:
    """Replace fixed finger boxes with the full URDF's articulated links."""

    expansions: dict[str, tuple[str, ...]] = {}
    for side in ("left", "right"):
        expansions[f"{side}_dex3_marker_palm"] = (f"{side}_dex3_marker_palm",)
        expansions[f"{side}_dex3_thumb"] = tuple(
            f"{side}_hand_thumb_{index}_link" for index in range(3)
        )
        for finger in ("middle", "index"):
            expansions[f"{side}_dex3_{finger}"] = tuple(
                f"{side}_hand_{finger}_{index}_link" for index in range(2)
            )
    pairs: list[list[str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in base.pairs:
        for first, second in product(
            expansions.get(pair.first, (pair.first,)),
            expansions.get(pair.second, (pair.second,)),
        ):
            key = tuple(sorted((first, second)))
            if first != second and key not in seen:
                seen.add(key)
                pairs.append([first, second])
    palm_boxes = [
        item for item in base.attached_boxes if item.name.endswith("marker_palm")
    ]
    if {item.name for item in palm_boxes} != {
        "left_dex3_marker_palm",
        "right_dex3_marker_palm",
    }:
        raise ValueError("Dex3 collision profile lacks both marker/palm boxes")
    return CollisionConfig.from_mapping(
        {
            "schema_version": 1,
            "hardware_ready": base.hardware_ready,
            "blocking_reasons": list(base.blocking_reasons),
            "visual_fallback_links": list(base.visual_fallback_links),
            "required_attached_boxes": [item.name for item in palm_boxes],
            "attached_boxes": [
                {
                    "name": item.name,
                    "parent_link": item.parent_link.replace(
                        "rubber_hand", "hand_palm_link"
                    ),
                    "size_m": list(item.size_m),
                    "xyz_m": list(item.xyz_m),
                    "rpy_rad": list(item.rpy_rad),
                }
                for item in palm_boxes
            ],
            "pairs": pairs,
        }
    )


def validate_dex3_finger_sweep_at_state(
    *,
    model: URDFModel,
    collision_checker: FCLCollisionChecker,
    body_q,
    initial_left_hand_q_rad,
    initial_right_hand_q_rad,
    target_left_hand_q_rad,
    target_right_hand_q_rad,
    config: PathValidationConfig,
) -> FingerSweepResult:
    """Validate the complete hand sweep at one measured whole-body state."""

    body = validate_full_joint_vector(body_q, name="live finger-sweep body q")
    return _validate_finger_sweep(
        model=model,
        collision_checker=collision_checker,
        body_q=body,
        arm_q14=dual_arm_vector(
            body[np.asarray(LEFT_ARM_INDICES)],
            body[np.asarray(RIGHT_ARM_INDICES)],
        ),
        initial_left_q=_validate_hand_q(
            initial_left_hand_q_rad, "live initial left hand"
        ),
        initial_right_q=_validate_hand_q(
            initial_right_hand_q_rad, "live initial right hand"
        ),
        target_left_q=_validate_hand_q(target_left_hand_q_rad, "live target left hand"),
        target_right_q=_validate_hand_q(
            target_right_hand_q_rad, "live target right hand"
        ),
        config=config,
    )


def _validate_finger_sweep(
    *,
    model: URDFModel,
    collision_checker: FCLCollisionChecker,
    body_q: np.ndarray,
    arm_q14: np.ndarray,
    initial_left_q: np.ndarray,
    initial_right_q: np.ndarray,
    target_left_q: np.ndarray,
    target_right_q: np.ndarray,
    config: PathValidationConfig,
) -> FingerSweepResult:
    maximum_delta = float(
        max(
            np.max(np.abs(target_left_q - initial_left_q)),
            np.max(np.abs(target_right_q - initial_right_q)),
        )
    )
    intervals = max(int(np.ceil(maximum_delta / config.maximum_joint_increment_rad)), 1)
    minimum = float("inf")
    minimum_pair = None
    full = np.asarray(body_q, dtype=np.float64).copy()
    full[np.asarray(LEFT_ARM_INDICES)] = arm_q14[:7]
    full[np.asarray(RIGHT_ARM_INDICES)] = arm_q14[7:]
    hand_joint_names = tuple(
        f"{side}_hand_{suffix}_joint"
        for side in ("left", "right")
        for suffix in DEX3_MOTOR_JOINT_SUFFIXES[side]
    )
    hand_limits = model.joint_limits(hand_joint_names)
    initial_hand_q = np.concatenate((initial_left_q, initial_right_q))
    target_hand_q = np.concatenate((target_left_q, target_right_q))
    start_limit_violations = {
        name: _joint_limit_violation(value, limit)
        for name, value, limit in zip(
            hand_joint_names, initial_hand_q, hand_limits, strict=True
        )
        if _joint_limit_violation(value, limit) > 1e-6
    }
    for name, value, limit in zip(
        hand_joint_names, target_hand_q, hand_limits, strict=True
    ):
        if _joint_limit_violation(value, limit) > 1e-6:
            return FingerSweepResult(
                False,
                intervals + 1,
                0.0,
                None,
                f"target: {name}={value:.4f}rad outside hard limits",
            )
    for index, alpha in enumerate(np.linspace(0.0, 1.0, intervals + 1)):
        left = initial_left_q + alpha * (target_left_q - initial_left_q)
        right = initial_right_q + alpha * (target_right_q - initial_right_q)
        hand_q = np.concatenate((left, right))
        for name, value, limit in zip(
            hand_joint_names, hand_q, hand_limits, strict=True
        ):
            violation = _joint_limit_violation(value, limit)
            if violation <= 1e-6:
                continue
            start_violation = start_limit_violations.get(name)
            if start_violation is None or violation > start_violation + 1e-6:
                return FingerSweepResult(
                    False,
                    intervals + 1,
                    0.0,
                    None,
                    f"sample {index}: {name}={value:.4f}rad moves farther outside "
                    "hard limits",
                )
        positions = dict(zip(G1_29_JOINT_NAMES, full, strict=True))
        positions.update(dict(zip(hand_joint_names, hand_q, strict=True)))
        collision = collision_checker.check(model.forward_kinematics(positions))
        if collision.minimum_clearance_m < minimum:
            minimum = collision.minimum_clearance_m
            if collision.minimum_pair is not None:
                minimum_pair = (
                    collision.minimum_pair.first,
                    collision.minimum_pair.second,
                )
        if collision.colliding_pairs or (
            collision.minimum_clearance_m < config.minimum_collision_clearance_m
        ):
            pair = collision.minimum_pair
            pair_text = "unknown" if pair is None else f"{pair.first}/{pair.second}"
            return FingerSweepResult(
                False,
                intervals + 1,
                max(collision.minimum_clearance_m, 0.0),
                None if pair is None else (pair.first, pair.second),
                f"sample {index}: clearance {collision.minimum_clearance_m:.4f}m "
                f"for {pair_text} is below "
                f"{config.minimum_collision_clearance_m:.4f}m",
            )
    return FingerSweepResult(
        True,
        intervals + 1,
        minimum,
        minimum_pair,
        None,
        tuple(start_limit_violations),
    )


def _joint_limit_violation(value: float, limit) -> float:
    """Return distance outside a URDF interval, or zero inside it."""

    return float(max(limit.lower - value, value - limit.upper, 0.0))


def _validate_side_clearance(
    *,
    validator: PosePathValidator,
    model: URDFModel,
    reference: np.ndarray,
    side: str,
    target_q: np.ndarray,
    target_id: str,
) -> ValidationReport:
    target = validate_arm_vector(target_q, side=side)
    content_sha256 = hashlib.sha256(
        json.dumps(
            {
                "side": side,
                "reference": reference.tolist(),
                "target": target.tolist(),
                "target_id": target_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    command_set = _ClearanceCommandSet(
        urdf_sha256=model.sha256,
        calibration_arm=side,
        poses=(_ClearanceTarget(target_id, tuple(target)),),
        content_sha256=content_sha256,
    )
    return validator.validate(
        command_set,
        directed_edges=(
            (HANDOFF_POSE_ID, target_id),
            (target_id, HANDOFF_POSE_ID),
        ),
        reference_full_q=reference,
    )


def _failed_clearance_text(report: ValidationReport) -> str:
    failures = [
        f"{edge.from_pose_id}->{edge.to_pose_id}: {'; '.join(edge.failures)}"
        for edge in report.edges
        if not edge.passed
    ]
    return " | ".join(failures)


def _validate_q14(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape != (DUAL_ARM_DOF,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain fourteen finite values")
    return array


def _validate_hand_q(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape != (7,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain seven finite values")
    return array


class DualArmClearanceExecutor:
    """Acquire, traverse the frozen shoulder route, and return before release."""

    def __init__(
        self,
        *,
        transport: ArmTransport,
        clock: MonotonicClock,
        plan: DualArmClearancePlan,
        config: ExecutorConfig,
        gravity_feedforward: ArmGravityFeedforward | None = None,
    ) -> None:
        self.transport = transport
        self.clock = clock
        self.pose_set = plan
        self.plan = plan
        self.config = config
        self.gravity_feedforward = gravity_feedforward
        self.approved_validation_report_sha256 = plan.content_sha256
        self.state = ExecutorState.OBSERVING
        self.current_pose_id: str | None = None
        self.fault_reason: str | None = None
        self.events: list[ExecutorEvent] = []
        self._command_q14: np.ndarray | None = None
        self._goal_q14: np.ndarray | None = None
        self._pending_pose_id: str | None = None
        self._weight = 0.0
        self._phase_started_s: float | None = None
        self._motion_started_s: float | None = None
        self._last_tick_s: float | None = None
        self._settle_started_s: float | None = None
        self._settle_min_q: np.ndarray | None = None
        self._settle_max_q: np.ndarray | None = None
        self._fault_initial_weight = 0.0
        self._last_motion_phase: ExecutorState | None = None
        self._last_motion_elapsed_s: float | None = None
        self._last_motion_measured_q: np.ndarray | None = None
        self._last_motion_position_errors: np.ndarray | None = None
        self._last_command_remaining_rad: float | None = None
        self._last_settle_elapsed_s: float | None = None
        self._last_settle_spread_rad: float | None = None
        self._maximum_acquisition_position_change_rad = 0.0
        self._ready_reference_q14: np.ndarray | None = None

    @property
    def maximum_acquisition_position_change_rad(self) -> float:
        return self._maximum_acquisition_position_change_rad

    def acquire(self, *, operator_confirmed: bool) -> None:
        if self.state is not ExecutorState.OBSERVING:
            raise RuntimeError("clearance control can only be acquired once")
        if not operator_confirmed:
            raise ValueError("operator confirmation is required before acquisition")
        now = self.clock.monotonic()
        sample = self.transport.observe()
        self._validate_fresh_state(sample, now)
        measured = dual_arm_vector(sample.left_q, sample.right_q)
        source = np.asarray(self.plan.source_q14, dtype=np.float64)
        error = float(np.max(np.abs(measured - source)))
        if error > self.config.activation_position_tolerance_rad:
            raise ValueError(
                "live dual-arm state differs from the validated clearance source by "
                f"{error:.4f}rad; limit is "
                f"{self.config.activation_position_tolerance_rad:.4f}rad"
            )
        self._command_q14 = np.asarray(measured, dtype=np.float64).copy()
        self._goal_q14 = self._command_q14.copy()
        if self.gravity_feedforward is not None:
            self.gravity_feedforward.seed_reference(sample.position)
        self.current_pose_id = HANDOFF_POSE_ID
        self._phase_started_s = now
        self._last_tick_s = now
        self._weight = 0.0
        self._send(now)
        acquired_at = self.clock.monotonic()
        self._phase_started_s = acquired_at
        self._last_tick_s = acquired_at
        self._transition(
            ExecutorState.ACQUIRING,
            "operator confirmed measured-state clearance acquisition",
            acquired_at,
        )

    def start_pose(self, pose_id: str, *, operator_confirmed: bool) -> None:
        if self.state not in {ExecutorState.READY, ExecutorState.HOLDING}:
            raise RuntimeError("a clearance move can only start while ready")
        if not operator_confirmed:
            raise ValueError("operator confirmation is required for every move")
        if self.current_pose_id is None or not self.plan.transition_is_validated(
            self.current_pose_id, pose_id
        ):
            raise ValueError(
                f"clearance transition is not validated: "
                f"{self.current_pose_id}->{pose_id}"
            )
        self._goal_q14 = self.plan.target(pose_id)
        self._pending_pose_id = pose_id
        self._ready_reference_q14 = None
        now = self.clock.monotonic()
        self._phase_started_s = now
        self._motion_started_s = now
        self._reset_settle_window()
        self._last_motion_phase = ExecutorState.MOVING
        self._last_motion_elapsed_s = 0.0
        self._last_motion_measured_q = None
        self._last_motion_position_errors = None
        self._last_command_remaining_rad = None
        self._last_settle_elapsed_s = 0.0
        self._last_settle_spread_rad = None
        self._transition(ExecutorState.MOVING, f"validated move to {pose_id}", now)

    def resume_owned_control(self) -> None:
        """Resume the unchanged full-weight clearance command after a handoff."""

        if self.state not in {ExecutorState.READY, ExecutorState.HOLDING}:
            raise RuntimeError("clearance control can only resume while held")
        if self.current_pose_id != DUAL_CLEARANCE_POSE_ID:
            raise RuntimeError("clearance control can only resume at dual clearance")
        if self._command_q14 is None or self._weight != 1.0:
            raise RuntimeError("clearance control is not held at full command weight")
        now = self.clock.monotonic()
        sample = self.transport.observe()
        self._validate_fresh_state(sample, now)
        self._ready_reference_q14 = dual_arm_vector(sample.left_q, sample.right_q)
        if self.gravity_feedforward is not None:
            self.gravity_feedforward.seed_reference(sample.position)
        self._last_tick_s = now
        self._send(now)
        sent_at = self.clock.monotonic()
        self._last_tick_s = sent_at
        self._transition(
            self.state,
            "resumed identical full-weight command after calibration route",
            sent_at,
        )

    def tick(self) -> ExecutorState:
        now = self.clock.monotonic()
        if self.state in {ExecutorState.STOPPED, ExecutorState.OBSERVING}:
            return self.state
        if self.state is ExecutorState.FAULT:
            self._tick_release(now, emergency=True)
            return self.state
        if self._last_tick_s is None:
            self._enter_fault("control loop has no previous tick", now)
            return self.state
        duration = now - self._last_tick_s
        if duration < 0.0:
            self._enter_fault("monotonic clock moved backwards", now)
            return self.state
        if duration > self.config.control_gap_fault_s:
            self._enter_fault(
                f"control loop gap {duration:.3f}s exceeds "
                f"hard limit {self.config.control_gap_fault_s:.3f}s",
                now,
            )
            return self.state
        self._last_tick_s = now
        try:
            sample = self.transport.observe()
            self._validate_fresh_state(sample, now)
        except (TypeError, ValueError, RuntimeError) as error:
            self._enter_fault(str(error), now)
            return self.state
        measured = dual_arm_vector(sample.left_q, sample.right_q)

        if self.state in {ExecutorState.READY, ExecutorState.HOLDING}:
            if self._ready_reference_q14 is None:
                self._enter_fault("clearance hold has no measured reference", now)
                return self.state
            drift = float(np.max(np.abs(measured - self._ready_reference_q14)))
            if drift > self.config.held_arm_position_tolerance_rad:
                self._enter_fault(
                    "clearance-held arms drifted by "
                    f"{drift:.4f}rad; limit is "
                    f"{self.config.held_arm_position_tolerance_rad:.4f}rad",
                    now,
                )
                return self.state

        if self.state is ExecutorState.ACQUIRING:
            assert self._command_q14 is not None
            acquisition_error = float(np.max(np.abs(measured - self._command_q14)))
            self._maximum_acquisition_position_change_rad = max(
                self._maximum_acquisition_position_change_rad, acquisition_error
            )
            if (
                acquisition_error
                > self.config.ownership_transition_position_tolerance_rad
            ):
                self._enter_fault(
                    "arm position changed by "
                    f"{acquisition_error:.4f}rad during ownership acquisition; "
                    "limit is "
                    f"{self.config.ownership_transition_position_tolerance_rad:.4f}rad",
                    now,
                )
                return self.state
            assert self._phase_started_s is not None
            elapsed = now - self._phase_started_s
            self._weight = min(elapsed / self.config.acquisition_ramp_s, 1.0)
            self._send(now)
            if self._weight >= 1.0:
                self._ready_reference_q14 = np.asarray(measured).copy()
                self._transition(
                    ExecutorState.READY,
                    "clearance acquisition ramp complete",
                    now,
                )
            return self.state

        if self.state is ExecutorState.RELEASING:
            self._tick_release(now, emergency=False)
            return self.state
        if self.state in {ExecutorState.HOLDING, ExecutorState.READY}:
            self._send(now)
            return self.state

        assert self._command_q14 is not None and self._goal_q14 is not None
        self._command_q14 = velocity_limited_step(
            self._command_q14,
            self._goal_q14,
            maximum_velocity_rad_s=self.config.maximum_joint_velocity_rad_s,
            duration_s=self.config.nominal_tick_period_s,
        )
        self._send(now)
        assert self._motion_started_s is not None
        position_errors = np.abs(np.asarray(measured) - self._goal_q14)
        position_error = float(np.max(position_errors))
        self._last_motion_elapsed_s = now - self._motion_started_s
        self._last_motion_measured_q = np.asarray(measured).copy()
        self._last_motion_position_errors = position_errors.copy()
        self._last_command_remaining_rad = float(
            np.max(np.abs(self._command_q14 - self._goal_q14))
        )
        if self.state is ExecutorState.MOVING:
            if self._last_command_remaining_rad <= _COMMAND_COMPLETION_EPSILON_RAD:
                self._reset_settle_window()
                self._transition(
                    ExecutorState.SETTLING,
                    "clearance command complete; waiting for measured settling",
                    now,
                )
        elif self._last_command_remaining_rad > _COMMAND_COMPLETION_EPSILON_RAD:
            self._reset_settle_window()
            self._transition(ExecutorState.MOVING, "command became incomplete", now)
        elif self._settle_started_s is None:
            self._settle_started_s = now
            self._settle_min_q = np.asarray(measured).copy()
            self._settle_max_q = np.asarray(measured).copy()
            self._last_settle_elapsed_s = 0.0
            self._last_settle_spread_rad = 0.0
        else:
            assert self._settle_min_q is not None and self._settle_max_q is not None
            self._settle_min_q = np.minimum(self._settle_min_q, measured)
            self._settle_max_q = np.maximum(self._settle_max_q, measured)
            maximum_spread = float(np.max(self._settle_max_q - self._settle_min_q))
            self._last_settle_elapsed_s = now - self._settle_started_s
            self._last_settle_spread_rad = maximum_spread
            if maximum_spread > self.config.settled_position_spread_rad:
                self._settle_started_s = now
                self._settle_min_q = np.asarray(measured).copy()
                self._settle_max_q = np.asarray(measured).copy()
                self._last_settle_elapsed_s = 0.0
            elif now - self._settle_started_s >= self.config.settle_dwell_s:
                if (
                    self.config.require_motion_endpoint_tolerance
                    and position_error > self.config.motion_position_tolerance_rad
                ):
                    self._enter_fault(
                        self.motion_diagnostic(
                            prefix="clearance motion settled outside endpoint tolerance"
                        ),
                        now,
                    )
                else:
                    self.current_pose_id = self._pending_pose_id
                    self._pending_pose_id = None
                    self._ready_reference_q14 = np.asarray(measured).copy()
                    self._transition(
                        ExecutorState.READY,
                        f"clearance settle passed; endpoint error {position_error:.4f}rad",
                        now,
                    )
        if self.state in {ExecutorState.MOVING, ExecutorState.SETTLING}:
            self._last_motion_phase = self.state
            if self._last_motion_elapsed_s > self.config.motion_timeout_s:
                self._enter_fault(
                    self.motion_diagnostic(prefix="clearance motion timed out"), now
                )
        return self.state

    def motion_diagnostic(self, *, prefix: str = "motion status") -> str:
        if (
            self._pending_pose_id is None
            or self._last_motion_phase is None
            or self._last_motion_elapsed_s is None
            or self._last_motion_measured_q is None
            or self._last_motion_position_errors is None
            or self._last_command_remaining_rad is None
        ):
            return f"{prefix}: no active measured-motion diagnostic"
        worst = int(np.argmax(self._last_motion_position_errors))
        target = self.plan.target(self._pending_pose_id)
        spread = (
            "n/a"
            if self._last_settle_spread_rad is None
            else f"{self._last_settle_spread_rad:.4f}rad"
        )
        return (
            f"{prefix} after {self._last_motion_elapsed_s:.2f}s while "
            f"{self._last_motion_phase.value} for {self._pending_pose_id}: "
            f"{G1_29_JOINT_NAMES[15 + worst]} has maximum position error "
            f"{self._last_motion_position_errors[worst]:.4f}rad "
            f"(measured={self._last_motion_measured_q[worst]:.4f}, "
            f"target={target[worst]:.4f}, "
            f"limit={self.config.motion_position_tolerance_rad:.4f}rad); "
            f"command remaining={self._last_command_remaining_rad:.4f}rad; "
            f"settle window={self._last_settle_elapsed_s or 0.0:.2f}/"
            f"{self.config.settle_dwell_s:.2f}s, position spread={spread}"
        )

    def begin_clean_release(self, *, operator_confirmed: bool) -> None:
        if self.state not in {ExecutorState.READY, ExecutorState.HOLDING}:
            raise RuntimeError("clean release requires a settled clearance executor")
        if not operator_confirmed:
            raise ValueError("operator confirmation is required for clean release")
        if self.current_pose_id != HANDOFF_POSE_ID:
            raise ValueError("arms must return to the validated source before release")
        now = self.clock.monotonic()
        self._phase_started_s = now
        self._transition(ExecutorState.RELEASING, "clean release approved", now)

    def observe_state(self):
        """Return one transport observation under the synchronized wrapper lock."""

        return self.transport.observe()

    def emergency_stop(self, reason: str) -> None:
        if self.state in {ExecutorState.STOPPED, ExecutorState.OBSERVING}:
            if self.state is ExecutorState.OBSERVING:
                self.transport.close()
                self._transition(ExecutorState.STOPPED, reason, self.clock.monotonic())
            return
        self._enter_fault(reason, self.clock.monotonic())

    def confirm_external_damping(self, reason: str) -> None:
        if self.state is ExecutorState.STOPPED:
            return
        if not reason.strip():
            raise ValueError("external damping reason must be non-empty")
        self.fault_reason = reason.strip()
        self.transport.close_after_external_takeover()
        self._transition(
            ExecutorState.STOPPED,
            f"external damping confirmed: {reason.strip()}",
            self.clock.monotonic(),
        )

    def _validate_fresh_state(self, sample, now: float) -> None:
        if not sample.is_mode5:
            raise ValueError("robot state is not mode_machine=5")
        age = sample.age_s(now)
        if age > self.config.state_freshness_timeout_s:
            raise ValueError(
                f"robot state age {age:.3f}s exceeds "
                f"{self.config.state_freshness_timeout_s:.3f}s"
            )

    def _reset_settle_window(self) -> None:
        self._settle_started_s = None
        self._settle_min_q = None
        self._settle_max_q = None
        self._last_settle_elapsed_s = 0.0
        self._last_settle_spread_rad = None

    def _send(self, now: float, *, emergency: bool = False) -> None:
        if self._command_q14 is None:
            raise RuntimeError("cannot command before measured-state seeding")
        torque = (
            np.zeros(DUAL_ARM_DOF, dtype=np.float64)
            if self.gravity_feedforward is None
            else self.gravity_feedforward.torque_for(self._command_q14)
        )
        self.transport.send_command(
            ArmCommand.create(
                self._command_q14,
                weight=self._weight,
                issued_monotonic_s=now,
                emergency_release=emergency,
                tau_ff14=torque,
            )
        )

    def _tick_release(self, now: float, *, emergency: bool) -> None:
        assert self._phase_started_s is not None
        elapsed = max(now - self._phase_started_s, 0.0)
        initial = self._fault_initial_weight if emergency else 1.0
        self._weight = initial * max(1.0 - elapsed / self.config.release_ramp_s, 0.0)
        self._send(now, emergency=emergency)
        if self._weight <= 0.0:
            self.transport.close()
            self._transition(
                ExecutorState.STOPPED,
                "emergency weight reached zero"
                if emergency
                else "clean release complete",
                now,
            )

    def _enter_fault(self, reason: str, now: float) -> None:
        if self.state in {ExecutorState.FAULT, ExecutorState.STOPPED}:
            return
        self.fault_reason = reason
        self._fault_initial_weight = self._weight
        self._phase_started_s = now
        self._transition(ExecutorState.FAULT, reason, now)
        self._send(now, emergency=True)

    def _transition(self, state: ExecutorState, reason: str, now: float) -> None:
        previous = self.state
        self.state = state
        self.events.append(
            ExecutorEvent(
                sequence=len(self.events),
                occurred_monotonic_s=now,
                previous_state=previous,
                state=state,
                reason=reason,
            )
        )
