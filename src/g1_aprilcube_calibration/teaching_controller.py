"""Continuously owned, manually guided G1 calibration-arm controller.

This controller deliberately has no pose-target or autonomous-motion API.  It
acquires ``rt/arm_sdk`` once at the measured Regular-mode rest position, follows
the measured calibration arm while the operator guides it, freezes that exact
measured position for hands-off capture, and changes GUIDE/HOLD through explicit
gain ramps at that frozen target. Neither transition commands an autonomous
trajectory. The harnessed workflow terminates through externally verified
whole-body Damp rather than blending back into locomotion control.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from g1_aprilcube_calibration.clock import MonotonicClock
from g1_aprilcube_calibration.joint_map import (
    dual_arm_vector,
    validate_arm_side,
    validate_arm_vector,
)
from g1_aprilcube_calibration.opposite_arm_hold import OppositeArmHold
from g1_aprilcube_calibration.transports.base import ArmCommand, ArmTransport


class TeachingState(str, Enum):
    OBSERVING = "observing"
    ACQUIRING = "acquiring"
    GUIDE = "guide"
    ENTERING_HOLD = "entering_hold"
    HOLDING = "holding"
    CAPTURING = "capturing"
    ENTERING_GUIDE = "entering_guide"
    FAULT = "fault"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class TeachingConfig:
    state_freshness_timeout_s: float = 0.1
    acquisition_ramp_s: float = 1.0
    gain_transition_ramp_s: float = 1.0
    guide_kp_scale: float = 0.5
    guide_kd_scale: float = 0.25
    activation_position_tolerance_rad: float = 0.02
    opposite_arm_hold_tolerance_rad: float = 0.02
    calibration_arm_hold_tolerance_rad: float = 0.05
    guide_joint_limit_margin_rad: float = 0.03

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("guide_kp_scale", "guide_kd_scale"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must not exceed the hold gain scale")


@dataclass(frozen=True, slots=True)
class TeachingEvent:
    sequence: int
    occurred_monotonic_s: float
    previous_state: TeachingState
    state: TeachingState
    reason: str


class TeachingArmController:
    """Joint-guiding controller that can freeze but can never command a move."""

    def __init__(
        self,
        *,
        transport: ArmTransport,
        clock: MonotonicClock,
        calibration_arm: str,
        activation_q14: Sequence[float] | np.ndarray,
        calibration_lower_q: Sequence[float] | np.ndarray,
        calibration_upper_q: Sequence[float] | np.ndarray,
        config: TeachingConfig | None = None,
    ) -> None:
        self.transport = transport
        self.clock = clock
        self.calibration_arm = validate_arm_side(calibration_arm)
        activation = np.asarray(activation_q14, dtype=np.float64).reshape(-1)
        if activation.shape != (14,) or not np.all(np.isfinite(activation)):
            raise ValueError("activation_q14 must contain 14 finite values")
        self.activation_q14 = activation.copy()
        self.activation_q14.setflags(write=False)
        self.config = config or TeachingConfig()
        self.calibration_lower_q = validate_arm_vector(
            calibration_lower_q, side=self.calibration_arm
        )
        self.calibration_upper_q = validate_arm_vector(
            calibration_upper_q, side=self.calibration_arm
        )
        if np.any(self.calibration_lower_q >= self.calibration_upper_q):
            raise ValueError("calibration-arm lower limits must be below upper limits")
        self._opposite_hold = OppositeArmHold.from_dual_arm_vector(
            calibration_arm=self.calibration_arm,
            q14=self.activation_q14,
        )

        self.state = TeachingState.OBSERVING
        self.events: list[TeachingEvent] = []
        self.fault_reason: str | None = None
        self._command_q14: np.ndarray | None = None
        self._held_calibration_q: np.ndarray | None = None
        self._weight = 0.0
        self._calibration_kp_scale = self.config.guide_kp_scale
        self._calibration_kd_scale = self.config.guide_kd_scale
        self._phase_started_s: float | None = None
        self._last_tick_s: float | None = None
        self._hold_command_count = 0
        self._minimum_joint_limit_clearance_rad = float("inf")

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def hold_command_count(self) -> int:
        return self._hold_command_count

    @property
    def calibration_kp_scale(self) -> float:
        return self._calibration_kp_scale

    @property
    def calibration_kd_scale(self) -> float:
        return self._calibration_kd_scale

    @property
    def held_calibration_q(self) -> tuple[float, ...] | None:
        if self._held_calibration_q is None:
            return None
        return tuple(float(value) for value in self._held_calibration_q)

    @property
    def near_joint_limit(self) -> bool:
        return (
            self._minimum_joint_limit_clearance_rad
            < self.config.guide_joint_limit_margin_rad
        )

    def acquire(self, *, operator_confirmed: bool) -> None:
        if self.state is not TeachingState.OBSERVING:
            raise RuntimeError("teaching control can only be acquired once")
        if not operator_confirmed:
            raise ValueError("operator confirmation is required before acquisition")
        now = self.clock.monotonic()
        sample = self.transport.observe()
        self._validate_fresh_state(sample, now)
        measured = dual_arm_vector(sample.left_q, sample.right_q)
        error = float(np.max(np.abs(measured - self.activation_q14)))
        if error > self.config.activation_position_tolerance_rad:
            raise ValueError(
                "measured arms differ from the stationary activation reference by "
                f"{error:.4f}rad; limit is "
                f"{self.config.activation_position_tolerance_rad:.4f}rad"
            )
        # The dynamic handoff reference is a median captured before runtime
        # validation, watchdog setup, and publisher creation. Once the latest
        # measured state passes that handoff gate, command and monitor the
        # opposite arm from this same zero-displacement ownership sample. Using
        # the older median as the monitor reference while commanding this newer
        # sample creates a permanent artificial drift offset.
        self._opposite_hold.seed_from_sample(sample)
        self._command_q14 = measured
        self._phase_started_s = now
        self._last_tick_s = now
        self._weight = 0.0
        self._send(now)
        self._transition(
            TeachingState.ACQUIRING,
            "operator confirmed measured-state acquisition",
            now,
        )

    def begin_hold(self, *, operator_confirmed: bool) -> None:
        """Freeze the current measured calibration-arm position without moving."""

        if self.state is not TeachingState.GUIDE:
            raise RuntimeError("a teaching hold can only begin from guide mode")
        if not operator_confirmed:
            raise ValueError("operator confirmation is required before freezing")
        self._freeze_current("operator requested zero-displacement measured-position hold")

    def protective_hold(self, reason: str) -> None:
        """Freeze GUIDE after a recoverable external monitoring failure."""

        if self.state is not TeachingState.GUIDE:
            raise RuntimeError("a protective hold can only begin from guide mode")
        if not reason.strip():
            raise ValueError("protective-hold reason must be non-empty")
        self._freeze_current(f"protective hold: {reason.strip()}")

    def _freeze_current(self, reason: str) -> None:
        now = self.clock.monotonic()
        sample = self.transport.observe()
        self._validate_fresh_state(sample, now)
        self._opposite_hold.validate(
            sample,
            tolerance_rad=self.config.opposite_arm_hold_tolerance_rad,
        )
        calibration_q = sample.arm_q(self.calibration_arm)
        self._validate_hard_limits(calibration_q)
        self._held_calibration_q = calibration_q.copy()
        self._command_q14 = self._compose_command(self._held_calibration_q)
        self._weight = 1.0
        self._phase_started_s = now
        self._hold_command_count = 0
        self._send(now)
        self._transition(
            TeachingState.ENTERING_HOLD,
            reason,
            now,
        )

    def resume_guide(self, *, operator_confirmed: bool) -> None:
        """Ramp to guide gains after support, without changing blend weight."""

        if self.state is not TeachingState.HOLDING:
            raise RuntimeError("guide mode can only resume from a stationary hold")
        if not operator_confirmed:
            raise ValueError("operator support confirmation is required")
        now = self.clock.monotonic()
        sample = self.transport.observe()
        self._validate_fresh_state(sample, now)
        self._opposite_hold.validate(
            sample,
            tolerance_rad=self.config.opposite_arm_hold_tolerance_rad,
        )
        calibration_q = sample.arm_q(self.calibration_arm)
        self._validate_hard_limits(calibration_q)
        self._update_limit_clearance(calibration_q)
        self._held_calibration_q = calibration_q.copy()
        self._command_q14 = self._compose_command(calibration_q)
        self._weight = 1.0
        self._phase_started_s = now
        self._send(now)
        self._transition(
            TeachingState.ENTERING_GUIDE,
            "operator confirmed support; reducing calibration-arm gains",
            now,
        )

    def begin_capture(self) -> None:
        if self.state is not TeachingState.HOLDING:
            raise RuntimeError("capture requires an active teaching hold")
        self._transition(
            TeachingState.CAPTURING,
            "hands-off stationary capture started",
            self.clock.monotonic(),
        )

    def finish_capture(self, *, outcome: str) -> None:
        if self.state is not TeachingState.CAPTURING:
            raise RuntimeError("capture can only finish while capturing")
        if not outcome.strip():
            raise ValueError("capture outcome must be non-empty")
        self._transition(
            TeachingState.HOLDING,
            f"capture finished: {outcome.strip()}",
            self.clock.monotonic(),
        )

    def tick(self) -> TeachingState:
        now = self.clock.monotonic()
        if self.state in {
            TeachingState.OBSERVING,
            TeachingState.STOPPED,
            TeachingState.FAULT,
        }:
            return self.state
        if self._last_tick_s is None:
            return self._enter_fault("control loop has no previous tick", now)
        duration = now - self._last_tick_s
        if duration < 0:
            return self._enter_fault("monotonic clock moved backwards", now)
        self._last_tick_s = now
        try:
            sample = self.transport.observe()
            self._validate_fresh_state(sample, now)
            self._opposite_hold.validate(
                sample,
                tolerance_rad=(
                    self.config.calibration_arm_hold_tolerance_rad
                    if self.state is TeachingState.ACQUIRING
                    else self.config.opposite_arm_hold_tolerance_rad
                ),
                transition=self.state is TeachingState.ACQUIRING,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            return self._enter_fault(str(error), now)

        if self.state is TeachingState.ACQUIRING:
            assert self._phase_started_s is not None
            elapsed = now - self._phase_started_s
            self._weight = min(elapsed / self.config.acquisition_ramp_s, 1.0)
            self._send(now)
            if self._weight >= 1.0:
                # Keep commanding the original acquisition position: its
                # position error supplies the torque that supports the arm in
                # the absence of gravity feedforward. Monitor subsequent drift
                # from the loaded, full-ownership equilibrium instead of from
                # the unloaded pre-ownership position.
                self._opposite_hold.rebase_monitor(sample)
                self._transition(
                    TeachingState.GUIDE,
                    "ownership ramp complete; measured-state guide active",
                    now,
                )
            return self.state

        if self.state is TeachingState.GUIDE:
            calibration_q = sample.arm_q(self.calibration_arm)
            try:
                self._validate_hard_limits(calibration_q)
            except ValueError as error:
                return self._enter_fault(str(error), now)
            self._update_limit_clearance(calibration_q)
            self._command_q14 = self._compose_command(calibration_q)
            self._send(now)
            return self.state

        if self.state is TeachingState.ENTERING_HOLD:
            assert self._phase_started_s is not None
            try:
                self._validate_held_calibration_arm(sample)
            except (RuntimeError, TypeError, ValueError) as error:
                return self._enter_fault(str(error), now)
            progress = min(
                max(now - self._phase_started_s, 0.0)
                / self.config.gain_transition_ramp_s,
                1.0,
            )
            self._calibration_kp_scale = self._interpolate_gain_scale(
                self.config.guide_kp_scale,
                1.0,
                progress,
            )
            self._calibration_kd_scale = self._interpolate_gain_scale(
                self.config.guide_kd_scale,
                1.0,
                progress,
            )
            self._send(now)
            if progress >= 1.0:
                self._hold_command_count = 1
                self._transition(
                    TeachingState.HOLDING,
                    "full hold gains established at the frozen measured position",
                    now,
                )
            return self.state

        if self.state is TeachingState.ENTERING_GUIDE:
            assert self._phase_started_s is not None
            try:
                self._validate_held_calibration_arm(sample)
            except (RuntimeError, TypeError, ValueError) as error:
                return self._enter_fault(str(error), now)
            progress = min(
                max(now - self._phase_started_s, 0.0)
                / self.config.gain_transition_ramp_s,
                1.0,
            )
            self._calibration_kp_scale = self._interpolate_gain_scale(
                1.0,
                self.config.guide_kp_scale,
                progress,
            )
            self._calibration_kd_scale = self._interpolate_gain_scale(
                1.0,
                self.config.guide_kd_scale,
                progress,
            )
            if progress >= 1.0:
                calibration_q = sample.arm_q(self.calibration_arm)
                self._validate_hard_limits(calibration_q)
                self._update_limit_clearance(calibration_q)
                self._command_q14 = self._compose_command(calibration_q)
                self._held_calibration_q = None
            self._send(now)
            if progress >= 1.0:
                self._transition(
                    TeachingState.GUIDE,
                    "guide gains established; measured-state following resumed",
                    now,
                )
            return self.state

        try:
            self._validate_held_calibration_arm(sample)
            self._send(now)
            self._hold_command_count += 1
        except (RuntimeError, TypeError, ValueError) as error:
            return self._enter_fault(str(error), now)
        return self.state

    def confirm_external_damping(self, reason: str) -> None:
        if self.state is TeachingState.STOPPED:
            return
        if not reason.strip():
            raise ValueError("external damping reason must be non-empty")
        self.fault_reason = reason.strip()
        self.transport.close_after_external_takeover()
        self._transition(
            TeachingState.STOPPED,
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

    def _validate_held_calibration_arm(self, sample) -> None:
        if self._held_calibration_q is None:
            raise RuntimeError("teaching hold has no frozen measured target")
        error = float(
            np.max(
                np.abs(
                    sample.arm_q(self.calibration_arm) - self._held_calibration_q
                )
            )
        )
        if error > self.config.calibration_arm_hold_tolerance_rad:
            raise ValueError(
                f"held {self.calibration_arm} arm drifted by {error:.4f}rad; "
                f"limit is {self.config.calibration_arm_hold_tolerance_rad:.4f}rad"
            )

    def _validate_hard_limits(self, calibration_q: np.ndarray) -> None:
        below = np.flatnonzero(calibration_q < self.calibration_lower_q)
        above = np.flatnonzero(calibration_q > self.calibration_upper_q)
        if not len(below) and not len(above):
            return
        index = int(below[0] if len(below) else above[0])
        raise ValueError(
            f"joint {index} at {calibration_q[index]:.4f}rad exceeds URDF limit "
            f"[{self.calibration_lower_q[index]:.4f}, "
            f"{self.calibration_upper_q[index]:.4f}]"
        )

    def _update_limit_clearance(self, calibration_q: np.ndarray) -> None:
        self._minimum_joint_limit_clearance_rad = float(
            np.min(
                np.minimum(
                    calibration_q - self.calibration_lower_q,
                    self.calibration_upper_q - calibration_q,
                )
            )
        )

    def _compose_command(self, calibration_q: np.ndarray) -> np.ndarray:
        return self._opposite_hold.compose(calibration_q)

    def _compose_gain_scale(self, calibration_scale: float) -> np.ndarray:
        calibration_scales = np.full(7, calibration_scale)
        opposite_scale = np.ones(7)
        if self.calibration_arm == "left":
            return dual_arm_vector(calibration_scales, opposite_scale)
        return dual_arm_vector(opposite_scale, calibration_scales)

    @staticmethod
    def _interpolate_gain_scale(start: float, end: float, progress: float) -> float:
        return start + (end - start) * progress

    def _send(self, now: float) -> None:
        if self._command_q14 is None:
            raise RuntimeError("cannot command before measured-state seeding")
        self.transport.send_command(
            ArmCommand.create(
                self._command_q14,
                weight=self._weight,
                issued_monotonic_s=now,
                kp_scale14=self._compose_gain_scale(self._calibration_kp_scale),
                kd_scale14=self._compose_gain_scale(self._calibration_kd_scale),
            )
        )

    def _enter_fault(self, reason: str, now: float) -> TeachingState:
        self.fault_reason = reason
        self._transition(TeachingState.FAULT, reason, now)
        return self.state

    def _transition(
        self,
        state: TeachingState,
        reason: str,
        now: float,
    ) -> None:
        if not reason.strip():
            raise ValueError("transition reason must be non-empty")
        previous = self.state
        self.state = state
        self.events.append(
            TeachingEvent(
                sequence=len(self.events) + 1,
                occurred_monotonic_s=now,
                previous_state=previous,
                state=state,
                reason=reason.strip(),
            )
        )
