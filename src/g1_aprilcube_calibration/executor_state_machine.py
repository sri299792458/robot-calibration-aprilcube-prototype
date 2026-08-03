"""Safety-first deterministic G1 pose execution state machine."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum

import numpy as np

from g1_aprilcube_calibration.clock import MonotonicClock
from g1_aprilcube_calibration.joint_map import (
    arm_joint_names,
    dual_arm_vector,
    opposite_arm,
    validate_arm_vector,
)
from g1_aprilcube_calibration.motion_profile import velocity_limited_step
from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID, PoseSet
from g1_aprilcube_calibration.transports.base import ArmCommand, ArmTransport

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ExecutorState(str, Enum):
    OBSERVING = "observing"
    ACQUIRING = "acquiring"
    HOLDING = "holding"
    MOVING = "moving"
    SETTLING = "settling"
    READY = "ready"
    CAPTURING = "capturing"
    RELEASING = "releasing"
    FAULT = "fault"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class ExecutorConfig:
    maximum_joint_velocity_rad_s: float = 0.2
    coarse_arrival_tolerance_rad: float = 0.05
    target_position_tolerance_rad: float = 0.05
    activation_position_tolerance_rad: float = 0.02
    held_arm_position_tolerance_rad: float = 0.02
    settled_position_spread_rad: float = 0.01
    settle_dwell_s: float = 0.5
    state_freshness_timeout_s: float = 0.1
    maximum_tick_gap_s: float = 0.05
    acquisition_ramp_s: float = 1.0
    release_ramp_s: float = 1.0
    motion_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.target_position_tolerance_rad > self.coarse_arrival_tolerance_rad:
            raise ValueError("target tolerance cannot exceed coarse arrival tolerance")


@dataclass(frozen=True, slots=True)
class TransitionApproval:
    from_pose_id: str
    to_pose_id: str
    pose_set_sha256: str
    validation_report_sha256: str
    passed: bool

    def __post_init__(self) -> None:
        if not self.from_pose_id or not self.to_pose_id:
            raise ValueError("transition pose IDs must be non-empty")
        if not _SHA256_PATTERN.fullmatch(self.pose_set_sha256):
            raise ValueError("pose_set_sha256 must be lowercase SHA-256")
        if not _SHA256_PATTERN.fullmatch(self.validation_report_sha256):
            raise ValueError("validation_report_sha256 must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class ExecutorEvent:
    sequence: int
    occurred_monotonic_s: float
    previous_state: ExecutorState
    state: ExecutorState
    reason: str


class PoseExecutor:
    def __init__(
        self,
        *,
        transport: ArmTransport,
        clock: MonotonicClock,
        pose_set: PoseSet,
        approved_validation_report_sha256: str,
        config: ExecutorConfig | None = None,
    ) -> None:
        if not _SHA256_PATTERN.fullmatch(approved_validation_report_sha256):
            raise ValueError(
                "approved validation report hash must be lowercase SHA-256"
            )
        self.transport = transport
        self.clock = clock
        self.pose_set = pose_set
        self.approved_validation_report_sha256 = approved_validation_report_sha256
        self.config = config or ExecutorConfig()
        self.state = ExecutorState.OBSERVING
        self.events: list[ExecutorEvent] = []
        self.fault_reason: str | None = None
        self.current_pose_id: str | None = None
        self._pending_pose_id: str | None = None
        self._goal_q14: np.ndarray | None = None
        self._command_q14: np.ndarray | None = None
        self._hold_q: np.ndarray | None = None
        self._calibration_goal_q: np.ndarray | None = None
        self._weight = 0.0
        self._phase_started_s: float | None = None
        self._motion_started_s: float | None = None
        self._settle_started_s: float | None = None
        self._settle_min_q: np.ndarray | None = None
        self._settle_max_q: np.ndarray | None = None
        self._last_tick_s: float | None = None
        self._fault_initial_weight = 0.0
        self._last_motion_phase: ExecutorState | None = None
        self._last_motion_elapsed_s: float | None = None
        self._last_motion_measured_q: np.ndarray | None = None
        self._last_motion_position_errors: np.ndarray | None = None
        self._last_command_remaining_rad: float | None = None
        self._last_settle_elapsed_s: float | None = None
        self._last_settle_spread_rad: float | None = None

    def acquire(
        self,
        *,
        operator_confirmed: bool,
    ) -> None:
        if self.state is not ExecutorState.OBSERVING:
            raise RuntimeError("control can only be acquired from observing")
        if not operator_confirmed:
            raise ValueError("operator confirmation is required before acquisition")
        now = self.clock.monotonic()
        sample = self.transport.observe()
        self._validate_fresh_state(sample, now)
        hold_arm = opposite_arm(self.pose_set.calibration_arm)
        hold_error = float(
            np.max(np.abs(sample.arm_q(hold_arm) - np.asarray(self.pose_set.hold_q)))
        )
        if hold_error > self.config.activation_position_tolerance_rad:
            raise ValueError(
                f"measured {hold_arm} arm differs from pose-set hold by "
                f"{hold_error:.4f}rad"
            )
        calibration_error = float(
            np.max(
                np.abs(
                    sample.arm_q(self.pose_set.calibration_arm)
                    - np.asarray(self.pose_set.handoff_q)
                )
            )
        )
        if calibration_error > self.config.activation_position_tolerance_rad:
            raise ValueError(
                f"measured {self.pose_set.calibration_arm} arm differs from "
                f"handoff pose by {calibration_error:.4f}rad"
            )
        self._hold_q = np.asarray(self.pose_set.hold_q)
        self._calibration_goal_q = np.asarray(self.pose_set.handoff_q)
        self._command_q14 = dual_arm_vector(sample.left_q, sample.right_q)
        self._goal_q14 = self._command_q14
        self.current_pose_id = HANDOFF_POSE_ID
        self._phase_started_s = now
        self._last_tick_s = now
        self._weight = 0.0
        self._send(now)
        self._transition(ExecutorState.ACQUIRING, "operator confirmed acquisition", now)

    def start_pose(
        self,
        pose_id: str,
        *,
        approval: TransitionApproval,
        operator_confirmed: bool,
    ) -> None:
        if self.state not in {ExecutorState.HOLDING, ExecutorState.READY}:
            raise RuntimeError("a move can only start while holding or ready")
        if not operator_confirmed:
            raise ValueError("operator confirmation is required for every move")
        if (
            self.current_pose_id is None
            or approval.from_pose_id != self.current_pose_id
        ):
            raise ValueError("transition approval does not match the current pose")
        if approval.to_pose_id != pose_id:
            raise ValueError("transition approval does not match the requested pose")
        if approval.pose_set_sha256 != self.pose_set.content_sha256:
            raise ValueError("transition approval pose-set hash is stale")
        if approval.validation_report_sha256 != self.approved_validation_report_sha256:
            raise ValueError("transition approval validation-report hash is stale")
        if not approval.passed:
            raise ValueError("transition validation did not pass")
        if pose_id == HANDOFF_POSE_ID:
            calibration_target = self.pose_set.handoff_q
        else:
            matching = [pose for pose in self.pose_set.poses if pose.id == pose_id]
            if len(matching) != 1:
                raise ValueError(f"pose ID is not present exactly once: {pose_id}")
            calibration_target = matching[0].measured_calibration_q
        if self._hold_q is None or self._command_q14 is None:
            raise RuntimeError("executor has no acquired arm state")
        calibration_goal = validate_arm_vector(
            calibration_target,
            side=self.pose_set.calibration_arm,
        )
        self._calibration_goal_q = calibration_goal
        self._goal_q14 = self._compose_command(calibration_goal)
        self._pending_pose_id = pose_id
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
        self._transition(ExecutorState.MOVING, f"approved move to {pose_id}", now)

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
        if duration < 0:
            self._enter_fault("monotonic clock moved backwards", now)
            return self.state
        if duration > self.config.maximum_tick_gap_s:
            self._enter_fault(
                f"control loop gap {duration:.3f}s exceeds "
                f"{self.config.maximum_tick_gap_s:.3f}s",
                now,
            )
            return self.state
        self._last_tick_s = now

        try:
            sample = self.transport.observe()
            self._validate_fresh_state(sample, now)
            self._validate_hold_arm(sample)
        except (TypeError, ValueError, RuntimeError) as error:
            self._enter_fault(str(error), now)
            return self.state

        if self.state is ExecutorState.ACQUIRING:
            assert self._phase_started_s is not None
            elapsed = now - self._phase_started_s
            self._weight = min(elapsed / self.config.acquisition_ramp_s, 1.0)
            self._send(now)
            if self._weight >= 1.0:
                self._transition(
                    ExecutorState.READY,
                    "acquisition ramp complete",
                    now,
                )
            return self.state

        if self.state is ExecutorState.RELEASING:
            self._tick_release(now, emergency=False)
            return self.state

        if self.state in {
            ExecutorState.HOLDING,
            ExecutorState.READY,
            ExecutorState.CAPTURING,
        }:
            self._send(now)
            return self.state

        assert self._command_q14 is not None
        assert self._goal_q14 is not None
        self._command_q14 = velocity_limited_step(
            self._command_q14,
            self._goal_q14,
            maximum_velocity_rad_s=self.config.maximum_joint_velocity_rad_s,
            duration_s=duration,
        )
        self._send(now)
        assert self._motion_started_s is not None
        assert self._calibration_goal_q is not None
        measured_q = sample.arm_q(self.pose_set.calibration_arm)
        position_errors = np.abs(measured_q - self._calibration_goal_q)
        position_error = float(np.max(position_errors))
        commanded_q = (
            self._command_q14[:7]
            if self.pose_set.calibration_arm == "left"
            else self._command_q14[7:]
        )
        self._last_motion_elapsed_s = now - self._motion_started_s
        self._last_motion_measured_q = measured_q.copy()
        self._last_motion_position_errors = position_errors.copy()
        self._last_command_remaining_rad = float(
            np.max(np.abs(commanded_q - self._calibration_goal_q))
        )
        if self.state is ExecutorState.MOVING:
            if position_error <= self.config.coarse_arrival_tolerance_rad:
                self._reset_settle_window()
                self._transition(ExecutorState.SETTLING, "coarse arrival reached", now)
        elif position_error > self.config.coarse_arrival_tolerance_rad:
            self._reset_settle_window()
            self._transition(ExecutorState.MOVING, "left coarse arrival region", now)
        elif position_error > self.config.target_position_tolerance_rad:
            self._reset_settle_window()
        elif self._settle_started_s is None:
            self._settle_started_s = now
            self._settle_min_q = measured_q.copy()
            self._settle_max_q = measured_q.copy()
            self._last_settle_elapsed_s = 0.0
            self._last_settle_spread_rad = 0.0
        else:
            assert self._settle_min_q is not None and self._settle_max_q is not None
            self._settle_min_q = np.minimum(self._settle_min_q, measured_q)
            self._settle_max_q = np.maximum(self._settle_max_q, measured_q)
            maximum_spread = float(np.max(self._settle_max_q - self._settle_min_q))
            self._last_settle_elapsed_s = now - self._settle_started_s
            self._last_settle_spread_rad = maximum_spread
            if maximum_spread > self.config.settled_position_spread_rad:
                self._settle_started_s = now
                self._settle_min_q = measured_q.copy()
                self._settle_max_q = measured_q.copy()
                self._last_settle_elapsed_s = 0.0
            elif now - self._settle_started_s >= self.config.settle_dwell_s:
                self.current_pose_id = self._pending_pose_id
                self._pending_pose_id = None
                self._transition(
                    ExecutorState.READY,
                    "continuous measured position-spread settle passed",
                    now,
                )
        if self.state in {ExecutorState.MOVING, ExecutorState.SETTLING}:
            self._last_motion_phase = self.state
            if self._last_motion_elapsed_s > self.config.motion_timeout_s:
                self._enter_fault(self.motion_diagnostic(prefix="motion timed out"), now)
        return self.state

    def motion_diagnostic(self, *, prefix: str = "motion status") -> str:
        """Describe the latest measured tracking/settling evidence."""

        if (
            self._pending_pose_id is None
            or self._last_motion_phase is None
            or self._last_motion_elapsed_s is None
            or self._last_motion_measured_q is None
            or self._last_motion_position_errors is None
            or self._last_command_remaining_rad is None
        ):
            return f"{prefix}: no active measured-motion diagnostic"
        worst_index = int(np.argmax(self._last_motion_position_errors))
        joint_name = arm_joint_names(self.pose_set.calibration_arm)[worst_index]
        error = float(self._last_motion_position_errors[worst_index])
        measured = float(self._last_motion_measured_q[worst_index])
        assert self._calibration_goal_q is not None
        target = float(self._calibration_goal_q[worst_index])
        settle_elapsed = self._last_settle_elapsed_s or 0.0
        spread = (
            "n/a"
            if self._last_settle_spread_rad is None
            else f"{self._last_settle_spread_rad:.4f}rad"
        )
        return (
            f"{prefix} after {self._last_motion_elapsed_s:.2f}s while "
            f"{self._last_motion_phase.value} for {self._pending_pose_id}: "
            f"{joint_name} has maximum position error {error:.4f}rad "
            f"(measured={measured:.4f}, target={target:.4f}, "
            f"target limit={self.config.target_position_tolerance_rad:.4f}); "
            f"command remaining={self._last_command_remaining_rad:.4f}rad; "
            f"settle window={settle_elapsed:.2f}/{self.config.settle_dwell_s:.2f}s, "
            f"position spread={spread} "
            f"(limit={self.config.settled_position_spread_rad:.4f}rad)"
        )

    def begin_capture(self) -> None:
        if self.state is not ExecutorState.READY:
            raise RuntimeError("capture can only begin from ready")
        self._transition(
            ExecutorState.CAPTURING,
            "stationary capture started",
            self.clock.monotonic(),
        )

    def finish_capture(self, *, outcome: str) -> None:
        if self.state is not ExecutorState.CAPTURING:
            raise RuntimeError("capture can only finish while capturing")
        if not outcome.strip():
            raise ValueError("capture outcome must be non-empty")
        self._transition(
            ExecutorState.HOLDING,
            f"capture finished: {outcome.strip()}",
            self.clock.monotonic(),
        )

    def begin_clean_release(
        self,
        *,
        operator_confirmed: bool,
    ) -> None:
        if self.state not in {ExecutorState.HOLDING, ExecutorState.READY}:
            raise RuntimeError(
                "clean release requires holding at the measured handoff pose"
            )
        if not operator_confirmed:
            raise ValueError("operator confirmation is required for clean release")
        if self.current_pose_id != HANDOFF_POSE_ID:
            raise ValueError("executor is not at the measured handoff pose")
        now = self.clock.monotonic()
        sample = self.transport.observe()
        self._validate_fresh_state(sample, now)
        assert self._goal_q14 is not None
        assert self._calibration_goal_q is not None
        position_error = float(
            np.max(
                np.abs(
                    sample.arm_q(self.pose_set.calibration_arm)
                    - self._calibration_goal_q
                )
            )
        )
        if position_error > self.config.target_position_tolerance_rad:
            raise ValueError("measured arm is outside the settled handoff tolerance")
        self._phase_started_s = now
        self._transition(ExecutorState.RELEASING, "clean release approved", now)

    def emergency_stop(self, reason: str) -> None:
        if self.state in {ExecutorState.STOPPED, ExecutorState.OBSERVING}:
            if self.state is ExecutorState.OBSERVING:
                self.transport.close()
                self._transition(ExecutorState.STOPPED, reason, self.clock.monotonic())
            return
        self._enter_fault(reason, self.clock.monotonic())

    def confirm_external_damping(self, reason: str) -> None:
        """Terminate command ownership after the G1 accepted whole-body damping."""

        if self.state is ExecutorState.STOPPED:
            return
        if not reason.strip():
            raise ValueError("external damping reason must be non-empty")
        now = self.clock.monotonic()
        self.fault_reason = reason.strip()
        self.transport.close_after_external_damping()
        self._transition(
            ExecutorState.STOPPED,
            f"external damping confirmed: {reason.strip()}",
            now,
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

    def _validate_hold_arm(self, sample) -> None:
        if self._hold_q is None:
            raise RuntimeError("executor has no measured hold-arm target")
        hold_arm = opposite_arm(self.pose_set.calibration_arm)
        error = float(np.max(np.abs(sample.arm_q(hold_arm) - self._hold_q)))
        if error > self.config.held_arm_position_tolerance_rad:
            raise ValueError(
                f"held {hold_arm} arm drifted by {error:.4f}rad; limit is "
                f"{self.config.held_arm_position_tolerance_rad:.4f}rad"
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
        self.transport.send_command(
            ArmCommand.create(
                self._command_q14,
                weight=self._weight,
                issued_monotonic_s=now,
                emergency_release=emergency,
            )
        )

    def _compose_command(self, calibration_q: np.ndarray) -> np.ndarray:
        if self._hold_q is None:
            raise RuntimeError("executor has no measured hold-arm state")
        if self.pose_set.calibration_arm == "left":
            return dual_arm_vector(calibration_q, self._hold_q)
        return dual_arm_vector(self._hold_q, calibration_q)

    def _tick_release(self, now: float, *, emergency: bool) -> None:
        assert self._phase_started_s is not None
        duration = self.config.release_ramp_s
        elapsed = max(now - self._phase_started_s, 0.0)
        initial = self._fault_initial_weight if emergency else 1.0
        self._weight = initial * max(1.0 - elapsed / duration, 0.0)
        self._send(now, emergency=emergency)
        if self._weight <= 0:
            self.transport.close()
            reason = (
                "emergency weight reached zero"
                if emergency
                else "clean release complete"
            )
            self._transition(ExecutorState.STOPPED, reason, now)

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


def validation_report_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
