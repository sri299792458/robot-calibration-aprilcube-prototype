from dataclasses import replace

import numpy as np
import pytest

from g1_aprilcube_calibration.clock import ManualClock
from g1_aprilcube_calibration.executor_state_machine import (
    ExecutorConfig,
    ExecutorState,
    PoseExecutor,
    TransitionApproval,
)
from g1_aprilcube_calibration.pose_schema import (
    HANDOFF_POSE_ID,
    PoseAuditEvent,
    PoseRecord,
    PoseSet,
)
from g1_aprilcube_calibration.transports.fake import FakeArmTransport

UTC = "2026-08-02T12:00:00Z"
REPORT_HASH = "b" * 64


def pose(pose_id: str, calibration_q: np.ndarray) -> PoseRecord:
    full = np.zeros(29)
    full[15:22] = calibration_q
    return PoseRecord(
        id=pose_id,
        group="test",
        measured_calibration_q=tuple(calibration_q),
        measured_full_q=tuple(full),
        calibration_q_spread=(0.0,) * 7,
        recorded_at_utc=UTC,
        recorded_monotonic_s=1.0,
    )


def pose_set() -> PoseSet:
    result = PoseSet(
        robot_model="g1_29dof_rev_1_0",
        mode_machine=5,
        urdf_sha256="a" * 64,
        calibration_arm="left",
    )
    for item in (pose("pose_001", np.full(7, 0.08)),):
        result = result.with_pose(item, PoseAuditEvent("add", item.id, UTC))
    return result


def config() -> ExecutorConfig:
    return ExecutorConfig(
        maximum_joint_velocity_rad_s=0.2,
        coarse_arrival_tolerance_rad=0.02,
        target_position_tolerance_rad=0.005,
        activation_position_tolerance_rad=0.005,
        held_arm_position_tolerance_rad=0.005,
        settled_position_spread_rad=0.002,
        settle_dwell_s=0.06,
        state_freshness_timeout_s=0.1,
        maximum_tick_gap_s=0.05,
        acquisition_ramp_s=0.06,
        release_ramp_s=0.06,
        motion_timeout_s=2.0,
    )


def subject() -> tuple[ManualClock, FakeArmTransport, PoseExecutor]:
    clock = ManualClock(1.0)
    transport = FakeArmTransport(clock=clock, initial_full_q=np.zeros(29))
    executor = PoseExecutor(
        transport=transport,
        clock=clock,
        pose_set=pose_set(),
        handoff_q=(0.0,) * 7,
        hold_q=(0.0,) * 7,
        approved_validation_report_sha256=REPORT_HASH,
        config=config(),
    )
    return clock, transport, executor


def approval(executor: PoseExecutor, source: str, target: str, **kwargs):
    values = {
        "from_pose_id": source,
        "to_pose_id": target,
        "pose_set_sha256": executor.pose_set.content_sha256,
        "validation_report_sha256": REPORT_HASH,
        "passed": True,
    }
    values.update(kwargs)
    return TransitionApproval(**values)


def advance_until(
    transport: FakeArmTransport,
    executor: PoseExecutor,
    desired: ExecutorState,
    *,
    limit: int = 200,
) -> None:
    for _ in range(limit):
        if executor.state is desired:
            return
        transport.step(0.02)
        executor.tick()
    raise AssertionError(f"executor did not reach {desired}; state={executor.state}")


def test_measured_handoff_changes_only_blend_weight() -> None:
    initial = np.linspace(-0.2, 0.2, 29)
    clock = ManualClock(1.0)
    transport = FakeArmTransport(clock=clock, initial_full_q=initial)
    executor = PoseExecutor(
        transport=transport,
        clock=clock,
        pose_set=pose_set(),
        handoff_q=initial[15:22],
        hold_q=initial[22:29],
        approved_validation_report_sha256=REPORT_HASH,
        config=config(),
    )
    expected_q14 = tuple(initial[15:29])

    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)

    assert transport.commands[0].weight == 0.0
    assert transport.commands[-1].weight == 1.0
    assert all(command.q14 == expected_q14 for command in transport.commands)
    np.testing.assert_allclose(transport.position, initial)

    executor.begin_clean_release(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.STOPPED)
    assert transport.commands[-1].weight == 0.0
    assert all(command.q14 == expected_q14 for command in transport.commands)


def test_acquisition_move_capture_handoff_and_clean_release() -> None:
    _, transport, executor = subject()
    assert transport.commands == []
    executor.acquire(operator_confirmed=True)

    assert executor.state is ExecutorState.ACQUIRING
    assert transport.commands[0].weight == 0.0
    assert transport.commands[0].q14 == (0.0,) * 14
    advance_until(transport, executor, ExecutorState.READY)
    assert np.allclose(transport.position, 0.0)

    executor.start_pose(
        "pose_001",
        approval=approval(executor, HANDOFF_POSE_ID, "pose_001"),
        operator_confirmed=True,
    )
    first_motion_command = len(transport.commands)
    advance_until(transport, executor, ExecutorState.READY)

    commanded = np.asarray(
        [command.q14 for command in transport.commands[first_motion_command:]]
    )
    increments = np.abs(np.diff(commanded, axis=0))
    assert np.max(increments) <= 0.2 * 0.02 + 1e-12
    assert np.allclose(transport.position[15:22], 0.08, atol=0.005)
    assert np.allclose(transport.position[22:29], 0.0, atol=0.005)
    executor.begin_capture()
    with pytest.raises(RuntimeError, match="move"):
        executor.start_pose(
            HANDOFF_POSE_ID,
            approval=approval(executor, "pose_001", HANDOFF_POSE_ID),
            operator_confirmed=True,
        )
    executor.finish_capture(outcome="accepted")
    assert executor.state is ExecutorState.HOLDING
    executor.begin_capture()
    executor.finish_capture(outcome="retry accepted")

    executor.start_pose(
        HANDOFF_POSE_ID,
        approval=approval(executor, "pose_001", HANDOFF_POSE_ID),
        operator_confirmed=True,
    )
    advance_until(transport, executor, ExecutorState.READY)
    executor.begin_clean_release(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.STOPPED)

    assert transport.closed
    assert transport.commands[-1].weight == 0.0
    assert not transport.commands[-1].emergency_release
    assert [event.state for event in executor.events].count(ExecutorState.READY) == 3


def test_every_move_requires_exact_approved_pose_and_hash() -> None:
    _, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)

    with pytest.raises(ValueError, match="operator"):
        executor.start_pose(
            "pose_001",
            approval=approval(executor, HANDOFF_POSE_ID, "pose_001"),
            operator_confirmed=False,
        )
    with pytest.raises(ValueError, match="pose-set hash"):
        executor.start_pose(
            "pose_001",
            approval=approval(
                executor,
                HANDOFF_POSE_ID,
                "pose_001",
                pose_set_sha256="c" * 64,
            ),
            operator_confirmed=True,
        )
    with pytest.raises(ValueError, match="validation-report"):
        executor.start_pose(
            "pose_001",
            approval=approval(
                executor,
                HANDOFF_POSE_ID,
                "pose_001",
                validation_report_sha256="c" * 64,
            ),
            operator_confirmed=True,
        )
    with pytest.raises(ValueError, match="did not pass"):
        executor.start_pose(
            "pose_001",
            approval=approval(executor, HANDOFF_POSE_ID, "pose_001", passed=False),
            operator_confirmed=True,
        )
    assert executor.state is ExecutorState.READY


def test_drift_of_held_right_arm_faults_during_left_arm_control() -> None:
    clock, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)
    transport.position[22] = 0.006
    clock.advance(0.01)
    executor.tick()
    assert executor.state is ExecutorState.FAULT
    assert executor.fault_reason is not None
    assert "held right arm" in executor.fault_reason


def test_looser_target_arrival_does_not_loosen_held_arm_safety() -> None:
    clock, transport, executor = subject()
    executor.config = replace(
        executor.config,
        coarse_arrival_tolerance_rad=0.05,
        target_position_tolerance_rad=0.05,
    )
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)
    transport.position[22] = 0.006
    clock.advance(0.01)

    executor.tick()

    assert executor.state is ExecutorState.FAULT
    assert executor.fault_reason is not None
    assert "held right arm drifted by 0.0060rad; limit is 0.0050rad" in (
        executor.fault_reason
    )


@pytest.mark.parametrize("failure", ["wrong_mode", "stale_state", "loop_overrun"])
def test_active_safety_failure_emergency_ramps_to_terminal_zero(failure: str) -> None:
    _, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)

    if failure == "wrong_mode":
        transport.mode_machine = 4
        transport.step(0.02)
        executor.tick()
    elif failure == "stale_state":
        transport.freeze_state_receipt = True
        for _ in range(6):
            transport.step(0.02)
            executor.tick()
            if executor.state is ExecutorState.FAULT:
                break
    else:
        transport.step(0.06)
        executor.tick()

    assert executor.state is ExecutorState.FAULT
    assert executor.fault_reason
    advance_until(transport, executor, ExecutorState.STOPPED)
    assert transport.commands[-1].weight == 0.0
    assert transport.commands[-1].emergency_release


def test_acquire_fails_closed_without_mode5_or_confirmation() -> None:
    _, transport, executor = subject()
    with pytest.raises(ValueError, match="confirmation"):
        executor.acquire(operator_confirmed=False)
    assert transport.commands == []

    transport.mode_machine = 4
    with pytest.raises(ValueError, match="mode_machine=5"):
        executor.acquire(operator_confirmed=True)
    assert transport.commands == []


def test_acquisition_ignores_unreliable_raw_dq() -> None:
    _, transport, executor = subject()
    transport.velocity[15] = 10.0

    executor.acquire(operator_confirmed=True)

    assert executor.state is ExecutorState.ACQUIRING
    assert transport.commands[0].weight == 0.0


def test_stationary_right_arm_dq_noise_does_not_fault_active_hold() -> None:
    clock, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    transport.velocity[22] = 10.0
    clock.advance(0.01)

    executor.tick()

    assert executor.state is ExecutorState.ACQUIRING
    assert executor.fault_reason is None


def test_settling_requires_bounded_measured_position_spread_for_full_dwell() -> None:
    clock, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)
    executor.start_pose(
        "pose_001",
        approval=approval(executor, HANDOFF_POSE_ID, "pose_001"),
        operator_confirmed=True,
    )
    advance_until(transport, executor, ExecutorState.SETTLING)

    for index in range(10):
        offset = 0.0015 if index % 2 else -0.0015
        transport.position[15:22] = 0.08 + offset
        clock.advance(0.01)
        executor.tick()
    assert executor.state is ExecutorState.SETTLING

    transport.position[15:22] = 0.08
    for _ in range(8):
        clock.advance(0.01)
        executor.tick()
    assert executor.state is ExecutorState.READY


def test_motion_timeout_reports_exact_tracking_and_settling_evidence() -> None:
    _, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)
    transport.tracking_velocity_rad_s = 0.0
    executor.start_pose(
        "pose_001",
        approval=approval(executor, HANDOFF_POSE_ID, "pose_001"),
        operator_confirmed=True,
    )

    advance_until(transport, executor, ExecutorState.FAULT)

    assert executor.fault_reason is not None
    assert "motion timed out after" in executor.fault_reason
    assert "while moving for pose_001" in executor.fault_reason
    assert "left_shoulder_pitch_joint" in executor.fault_reason
    assert "maximum position error 0.0800rad" in executor.fault_reason
    assert "measured=0.0000, target=0.0800" in executor.fault_reason
    assert "command remaining=0.0000rad" in executor.fault_reason
    assert "position spread=n/a" in executor.fault_reason


def test_clean_release_refuses_non_handoff_pose() -> None:
    _, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)
    executor.start_pose(
        "pose_001",
        approval=approval(executor, HANDOFF_POSE_ID, "pose_001"),
        operator_confirmed=True,
    )
    advance_until(transport, executor, ExecutorState.READY)
    with pytest.raises(ValueError, match="not at"):
        executor.begin_clean_release(operator_confirmed=True)


def test_observing_emergency_stop_closes_without_publishing() -> None:
    _, transport, executor = subject()
    executor.emergency_stop("operator quit")
    assert executor.state is ExecutorState.STOPPED
    assert transport.closed
    assert transport.commands == []


def test_confirmed_external_damping_stops_without_weight_zero() -> None:
    _, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)
    assert transport.commands[-1].weight == 1.0

    executor.confirm_external_damping("PC2 accepted G1 Damp")

    assert executor.state is ExecutorState.STOPPED
    assert executor.fault_reason == "PC2 accepted G1 Damp"
    assert transport.closed
    assert transport.commands[-1].weight == 1.0
    assert executor.events[-1].reason.startswith("external damping confirmed")
