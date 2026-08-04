from __future__ import annotations

import numpy as np
import pytest

from g1_aprilcube_calibration.clock import ManualClock
from g1_aprilcube_calibration.models import RobotStateSample
from g1_aprilcube_calibration.teaching_controller import (
    TeachingArmController,
    TeachingConfig,
    TeachingState,
)

UTC = "2026-08-04T12:00:00Z"


class FakeTransport:
    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        self.position = np.zeros(29)
        self.commands = []
        self.closed = False
        self.closed_after_damping = False

    def observe(self) -> RobotStateSample:
        return RobotStateSample(
            receipt_monotonic_s=self.clock.monotonic(),
            receipt_utc=UTC,
            mode_machine=5,
            position=self.position,
            velocity=np.zeros(29),
            estimated_torque=np.zeros(29),
        )

    def send_command(self, command) -> None:
        if self.closed:
            raise RuntimeError("transport closed")
        self.commands.append(command)

    def close(self) -> None:
        self.closed = True

    def close_after_external_damping(self) -> None:
        self.closed = True
        self.closed_after_damping = True


def subject() -> tuple[TeachingArmController, FakeTransport, ManualClock]:
    clock = ManualClock(1.0)
    transport = FakeTransport(clock)
    controller = TeachingArmController(
        transport=transport,
        clock=clock,
        calibration_arm="left",
        activation_q14=np.zeros(14),
        calibration_lower_q=np.full(7, -1.0),
        calibration_upper_q=np.full(7, 1.0),
        config=TeachingConfig(
            state_freshness_timeout_s=0.1,
            acquisition_ramp_s=0.04,
            gain_transition_ramp_s=0.04,
            guide_kp_scale=0.5,
            guide_kd_scale=0.25,
            activation_position_tolerance_rad=0.02,
            opposite_arm_hold_tolerance_rad=0.02,
            calibration_arm_hold_tolerance_rad=0.05,
            guide_joint_limit_margin_rad=0.1,
        ),
    )
    return controller, transport, clock


def acquire(controller, clock) -> None:
    controller.acquire(operator_confirmed=True)
    assert controller.state is TeachingState.ACQUIRING
    for _ in range(2):
        clock.advance(0.02)
        controller.tick()
    assert controller.state is TeachingState.GUIDE


def enter_hold(controller, clock) -> None:
    controller.begin_hold(operator_confirmed=True)
    assert controller.state is TeachingState.ENTERING_HOLD
    for _ in range(2):
        clock.advance(0.02)
        controller.tick()
    assert controller.state is TeachingState.HOLDING


def test_acquires_once_and_guide_follows_only_measured_calibration_arm() -> None:
    controller, transport, clock = subject()
    acquire(controller, clock)

    transport.position[15:22] = np.arange(7) / 20.0
    clock.advance(0.01)
    controller.tick()

    command = transport.commands[-1]
    np.testing.assert_allclose(command.q14[:7], np.arange(7) / 20.0)
    np.testing.assert_allclose(command.q14[7:], np.zeros(7))
    np.testing.assert_allclose(command.kp_scale14[:7], np.full(7, 0.5))
    np.testing.assert_allclose(command.kd_scale14[:7], np.full(7, 0.25))
    np.testing.assert_allclose(command.kp_scale14[7:], np.ones(7))
    np.testing.assert_allclose(command.kd_scale14[7:], np.ones(7))
    assert command.weight == 1.0
    assert not hasattr(controller, "start_pose")
    with pytest.raises(RuntimeError, match="only be acquired once"):
        controller.acquire(operator_confirmed=True)


def test_opposite_arm_monitor_latches_loaded_post_acquisition_equilibrium() -> None:
    controller, transport, clock = subject()
    # This is inside the validated dynamic-handoff tolerance, but close enough
    # that comparing future samples with the older zero median would false-fault.
    transport.position[22] = 0.0199
    controller.acquire(operator_confirmed=True)
    np.testing.assert_allclose(transport.commands[-1].q14[7], 0.0199)

    # Full-gain position impedance has no gravity feedforward. Its loaded
    # equilibrium can therefore differ from the commanded acquisition position
    # by more than the strict post-acquisition drift gate.
    transport.position[22] = 0.0420
    clock.advance(0.02)
    assert controller.tick() is TeachingState.ACQUIRING

    clock.advance(0.02)
    assert controller.tick() is TeachingState.GUIDE
    # The original command remains unchanged so its position error continues
    # supplying holding torque.
    np.testing.assert_allclose(transport.commands[-1].q14[7], 0.0199)

    transport.position[22] = 0.0621
    clock.advance(0.01)
    assert controller.tick() is TeachingState.FAULT
    assert "held right arm drifted" in controller.fault_reason


def test_opposite_arm_large_acquisition_motion_still_faults() -> None:
    controller, transport, clock = subject()
    controller.acquire(operator_confirmed=True)
    transport.position[22] = 0.0501
    clock.advance(0.01)

    assert controller.tick() is TeachingState.FAULT
    assert "moved during ownership transition" in controller.fault_reason
    assert "limit is 0.0500rad" in controller.fault_reason


def test_freeze_capture_and_resume_never_change_blend_weight() -> None:
    controller, transport, clock = subject()
    acquire(controller, clock)
    transport.position[15:22] = 0.3

    controller.begin_hold(operator_confirmed=True)
    assert controller.state is TeachingState.ENTERING_HOLD
    np.testing.assert_allclose(controller.held_calibration_q, np.full(7, 0.3))
    assert transport.commands[-1].weight == 1.0
    np.testing.assert_allclose(transport.commands[-1].kp_scale14[:7], 0.5)
    np.testing.assert_allclose(transport.commands[-1].kd_scale14[:7], 0.25)

    clock.advance(0.02)
    controller.tick()
    assert controller.state is TeachingState.ENTERING_HOLD
    np.testing.assert_allclose(transport.commands[-1].kp_scale14[:7], 0.75)
    np.testing.assert_allclose(transport.commands[-1].kd_scale14[:7], 0.625)

    clock.advance(0.02)
    controller.tick()
    assert controller.state is TeachingState.HOLDING
    np.testing.assert_allclose(transport.commands[-1].kp_scale14[:7], 1.0)
    np.testing.assert_allclose(transport.commands[-1].kd_scale14[:7], 1.0)

    transport.position[15:22] = 0.31
    clock.advance(0.01)
    controller.tick()
    np.testing.assert_allclose(transport.commands[-1].q14[:7], np.full(7, 0.3))
    controller.begin_capture()
    assert controller.state is TeachingState.CAPTURING
    controller.finish_capture(outcome="accepted")
    assert controller.state is TeachingState.HOLDING

    controller.resume_guide(operator_confirmed=True)
    assert controller.state is TeachingState.ENTERING_GUIDE
    np.testing.assert_allclose(transport.commands[-1].q14[:7], np.full(7, 0.31))
    assert transport.commands[-1].weight == 1.0
    np.testing.assert_allclose(transport.commands[-1].kp_scale14[:7], 1.0)
    np.testing.assert_allclose(transport.commands[-1].kd_scale14[:7], 1.0)

    clock.advance(0.02)
    controller.tick()
    assert controller.state is TeachingState.ENTERING_GUIDE
    np.testing.assert_allclose(transport.commands[-1].kp_scale14[:7], 0.75)
    np.testing.assert_allclose(transport.commands[-1].kd_scale14[:7], 0.625)

    clock.advance(0.02)
    controller.tick()
    assert controller.state is TeachingState.GUIDE
    np.testing.assert_allclose(transport.commands[-1].kp_scale14[:7], 0.5)
    np.testing.assert_allclose(transport.commands[-1].kd_scale14[:7], 0.25)


def test_guide_margin_is_warning_only_but_actual_urdf_violation_faults() -> None:
    controller, transport, clock = subject()
    acquire(controller, clock)
    transport.position[15] = 0.91
    clock.advance(0.01)

    assert controller.tick() is TeachingState.GUIDE
    assert controller.near_joint_limit
    assert transport.commands[-1].q14[0] == pytest.approx(0.91)
    assert transport.commands[-1].weight == 1.0

    command_count = len(transport.commands)
    transport.position[15] = 1.01
    clock.advance(0.01)
    assert controller.tick() is TeachingState.FAULT
    assert "exceeds URDF limit" in controller.fault_reason
    assert len(transport.commands) == command_count


def test_hold_drift_or_opposite_arm_drift_faults_and_stops_commands() -> None:
    controller, transport, clock = subject()
    acquire(controller, clock)
    enter_hold(controller, clock)
    command_count = len(transport.commands)
    transport.position[15] = 0.06
    clock.advance(0.01)

    assert controller.tick() is TeachingState.FAULT
    assert "held left arm drifted" in controller.fault_reason
    assert len(transport.commands) == command_count

    controller, transport, clock = subject()
    acquire(controller, clock)
    transport.position[22] = 0.03
    clock.advance(0.01)
    assert controller.tick() is TeachingState.FAULT
    assert "held right arm drifted" in controller.fault_reason


def test_external_damping_is_the_only_nonzero_weight_close() -> None:
    controller, transport, clock = subject()
    acquire(controller, clock)

    controller.confirm_external_damping("PC2 confirmed Damp")

    assert controller.state is TeachingState.STOPPED
    assert transport.closed_after_damping
    assert controller.fault_reason == "PC2 confirmed Damp"
    assert not hasattr(controller, "begin_clean_release")
