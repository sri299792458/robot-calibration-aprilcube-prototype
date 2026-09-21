from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import g1_aprilcube_calibration.dual_arm_clearance as clearance_module
from g1_aprilcube_calibration.clock import ManualClock
from g1_aprilcube_calibration.collision import CollisionConfig, FCLCollisionChecker
from g1_aprilcube_calibration.dual_arm_clearance import (
    DUAL_CLEARANCE_POSE_ID,
    RIGHT_CLEARANCE_POSE_ID,
    DualArmClearanceExecutor,
    DualArmClearancePlan,
    FingerSweepResult,
    live_dex3_collision_config,
    plan_dual_arm_shoulder_clearance,
    validate_dex3_finger_sweep_at_state,
)
from g1_aprilcube_calibration.executor_state_machine import (
    ExecutorConfig,
    ExecutorState,
)
from g1_aprilcube_calibration.joint_map import LEFT_ARM_INDICES, RIGHT_ARM_INDICES
from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID
from g1_aprilcube_calibration.pose_validator import PathValidationConfig
from g1_aprilcube_calibration.transports.fake import FakeArmTransport
from g1_aprilcube_calibration.transports.unitree_dex3 import (
    NVIDIA_MIDDLE_CLOSE_LEFT_Q_RAD,
    NVIDIA_MIDDLE_CLOSE_RIGHT_Q_RAD,
)
from g1_aprilcube_calibration.urdf_model import URDFModel

ROOT = Path(__file__).parents[1]

# A real stationary Ready observation retained here as a deterministic geometry
# regression. Runtime planning always uses a fresh live observation instead.
READY_Q29 = np.asarray(
    [
        -0.2690687478,
        -0.0112236263,
        -0.0044862852,
        0.5961049199,
        -0.3356498480,
        0.0015728838,
        -0.2664609849,
        -0.0010141317,
        -0.0011918787,
        0.5935909152,
        -0.3181147873,
        -0.0058997213,
        -0.0065620290,
        -0.0012682243,
        -0.0036700224,
        0.2904976308,
        0.1258934885,
        -0.0160109252,
        0.9753121734,
        0.2436992228,
        0.0562539548,
        -0.0057883807,
        0.2852725089,
        -0.1137654483,
        0.0060040969,
        0.9790632725,
        -0.1103986800,
        -0.0100307968,
        -0.0128470892,
    ],
    dtype=np.float64,
)


@pytest.fixture(scope="module")
def clearance_plan() -> DualArmClearancePlan:
    collision = CollisionConfig.from_yaml(
        ROOT / "config/collision_pairs_dex3_aruco.yaml"
    )
    body_model = URDFModel(
        ROOT / "unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"
    )
    dex3_model = URDFModel(
        ROOT / "unitree_ros/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf"
    )
    # A 0.05 rad sampling increment keeps this mesh-level regression compact.
    # Hardware uses the stricter PathValidationConfig default of 0.02 rad.
    return plan_dual_arm_shoulder_clearance(
        model=body_model,
        collision_checker=FCLCollisionChecker(body_model, collision),
        dex3_model=dex3_model,
        dex3_collision_checker=FCLCollisionChecker(
            dex3_model,
            live_dex3_collision_config(collision),
        ),
        reference_full_q=READY_Q29,
        initial_left_hand_q_rad=np.zeros(7),
        initial_right_hand_q_rad=np.zeros(7),
        target_left_hand_q_rad=NVIDIA_MIDDLE_CLOSE_LEFT_Q_RAD,
        target_right_hand_q_rad=NVIDIA_MIDDLE_CLOSE_RIGHT_Q_RAD,
        path_config=PathValidationConfig(maximum_joint_increment_rad=0.05),
    )


def test_live_clearance_search_starts_at_008_rad(
    clearance_plan: DualArmClearancePlan,
) -> None:
    plan = clearance_plan
    source = np.asarray(plan.source_q14)
    right = np.asarray(plan.right_clearance_q14)
    dual = np.asarray(plan.dual_clearance_q14)

    assert plan.shoulder_roll_offset_rad == pytest.approx(0.08)
    np.testing.assert_allclose(right[:7], source[:7])
    np.testing.assert_allclose(
        np.delete(right, 8), np.delete(source, 8), atol=0.0, rtol=0.0
    )
    assert right[8] == pytest.approx(source[8] - 0.08)
    np.testing.assert_allclose(dual[7:], right[7:])
    np.testing.assert_allclose(
        np.delete(dual, 1), np.delete(right, 1), atol=0.0, rtol=0.0
    )
    assert dual[1] == pytest.approx(source[1] + 0.08)
    assert plan.minimum_clearance_m >= 0.005
    assert plan.finger_sweep_minimum_clearance_m >= 0.005
    assert plan.finger_sweep_minimum_pair == (
        "right_hand_index_1_link",
        "right_hip_yaw_link",
    )
    assert plan.right_validation.passed
    assert plan.left_validation.passed


def test_live_clearance_search_reports_each_rejected_offset(monkeypatch) -> None:
    class Model:
        sha256 = "0" * 64

        def joint_limits(self, joint_names):
            return tuple(SimpleNamespace(lower=-2.0, upper=2.0) for _ in joint_names)

    class CollisionChecker:
        config = SimpleNamespace(content_sha256="1" * 64)

    attempts: list[tuple[int, int, float]] = []
    rejections: list[tuple[float, str]] = []

    monkeypatch.setattr(
        clearance_module,
        "_validate_finger_sweep",
        lambda **_kwargs: FingerSweepResult(
            passed=False,
            sample_count=2,
            minimum_clearance_m=0.004,
            minimum_pair=("left_hand_index_0_link", "left_hip_yaw_link"),
            failure="sample 0: test clearance failure",
        ),
    )

    with pytest.raises(ValueError, match="across 89 candidates"):
        plan_dual_arm_shoulder_clearance(
            model=Model(),
            collision_checker=CollisionChecker(),
            dex3_model=Model(),
            dex3_collision_checker=CollisionChecker(),
            reference_full_q=READY_Q29,
            initial_left_hand_q_rad=np.zeros(7),
            initial_right_hand_q_rad=np.zeros(7),
            target_left_hand_q_rad=NVIDIA_MIDDLE_CLOSE_LEFT_Q_RAD,
            target_right_hand_q_rad=NVIDIA_MIDDLE_CLOSE_RIGHT_Q_RAD,
            path_config=PathValidationConfig(),
            progress=lambda index, total, offset: attempts.append(
                (index, total, offset)
            ),
            rejection=lambda offset, reason: rejections.append((offset, reason)),
        )

    assert len(attempts) == len(rejections) == 89
    assert attempts[0] == pytest.approx((1, 89, 0.08))
    assert attempts[-1] == pytest.approx((89, 89, 1.84))
    assert rejections[0] == (
        pytest.approx(0.08),
        "finger sweep: sample 0: test clearance failure",
    )


def _executor_config() -> ExecutorConfig:
    return ExecutorConfig(
        maximum_joint_velocity_rad_s=0.2,
        motion_position_tolerance_rad=0.01,
        ownership_transition_position_tolerance_rad=0.01,
        activation_position_tolerance_rad=0.01,
        held_arm_position_tolerance_rad=0.01,
        settled_position_spread_rad=0.002,
        settle_dwell_s=0.04,
        state_freshness_timeout_s=0.1,
        nominal_tick_period_s=0.02,
        control_gap_fault_s=0.25,
        acquisition_ramp_s=0.04,
        release_ramp_s=0.04,
        motion_timeout_s=2.0,
    )


def _advance_until(
    transport: FakeArmTransport,
    executor: DualArmClearanceExecutor,
    desired: ExecutorState,
) -> None:
    for _ in range(200):
        if executor.state is desired:
            return
        transport.step(0.02)
        executor.tick()
    raise AssertionError(f"executor did not reach {desired}; state={executor.state}")


def test_executor_enforces_route_order_and_returns_to_live_source(
    clearance_plan: DualArmClearancePlan,
) -> None:
    clock = ManualClock(1.0)
    transport = FakeArmTransport(
        clock=clock,
        initial_full_q=READY_Q29,
        tracking_velocity_rad_s=1.0,
    )
    executor = DualArmClearanceExecutor(
        transport=transport,
        clock=clock,
        plan=clearance_plan,
        config=_executor_config(),
    )

    executor.acquire(operator_confirmed=True)
    _advance_until(transport, executor, ExecutorState.READY)
    with pytest.raises(ValueError, match="not validated"):
        executor.start_pose(DUAL_CLEARANCE_POSE_ID, operator_confirmed=True)

    for pose_id in (
        RIGHT_CLEARANCE_POSE_ID,
        DUAL_CLEARANCE_POSE_ID,
        RIGHT_CLEARANCE_POSE_ID,
        HANDOFF_POSE_ID,
    ):
        executor.start_pose(pose_id, operator_confirmed=True)
        _advance_until(transport, executor, ExecutorState.READY)

    np.testing.assert_allclose(
        transport.position[np.asarray(LEFT_ARM_INDICES)],
        READY_Q29[np.asarray(LEFT_ARM_INDICES)],
        atol=0.01,
    )
    np.testing.assert_allclose(
        transport.position[np.asarray(RIGHT_ARM_INDICES)],
        READY_Q29[np.asarray(RIGHT_ARM_INDICES)],
        atol=0.01,
    )
    executor.begin_clean_release(operator_confirmed=True)
    _advance_until(transport, executor, ExecutorState.STOPPED)
    assert transport.commands[-1].weight == 0.0
    assert transport.closed


def test_clearance_executor_resumes_unchanged_owned_command_after_long_pause(
    clearance_plan: DualArmClearancePlan,
) -> None:
    clock = ManualClock(1.0)
    transport = FakeArmTransport(
        clock=clock,
        initial_full_q=READY_Q29,
        tracking_velocity_rad_s=1.0,
    )
    executor = DualArmClearanceExecutor(
        transport=transport,
        clock=clock,
        plan=clearance_plan,
        config=_executor_config(),
    )
    executor.acquire(operator_confirmed=True)
    _advance_until(transport, executor, ExecutorState.READY)
    for pose_id in (RIGHT_CLEARANCE_POSE_ID, DUAL_CLEARANCE_POSE_ID):
        executor.start_pose(pose_id, operator_confirmed=True)
        _advance_until(transport, executor, ExecutorState.READY)

    clock.advance(1.0)
    executor.resume_owned_control()
    transport.step(0.01)
    executor.tick()

    assert executor.state is ExecutorState.READY
    assert executor.current_pose_id == DUAL_CLEARANCE_POSE_ID
    assert transport.commands[-1].weight == pytest.approx(1.0)


def test_clearance_scheduler_gap_uses_nominal_motion_step(
    clearance_plan: DualArmClearancePlan,
) -> None:
    clock = ManualClock(1.0)
    transport = FakeArmTransport(
        clock=clock,
        initial_full_q=READY_Q29,
        tracking_velocity_rad_s=1.0,
    )
    executor = DualArmClearanceExecutor(
        transport=transport,
        clock=clock,
        plan=clearance_plan,
        config=_executor_config(),
    )
    executor.acquire(operator_confirmed=True)
    _advance_until(transport, executor, ExecutorState.READY)
    initial_command = np.asarray(transport.commands[-1].q14)
    executor.start_pose(RIGHT_CLEARANCE_POSE_ID, operator_confirmed=True)

    transport.step(0.051)
    executor.tick()

    assert executor.state is ExecutorState.MOVING
    assert executor.fault_reason is None
    assert all("overrun" not in event.reason for event in executor.events)
    maximum_step = float(
        np.max(np.abs(np.asarray(transport.commands[-1].q14) - initial_command))
    )
    assert maximum_step == pytest.approx(0.004)


def test_executor_faults_if_an_arm_drifts_during_the_finger_hold(
    clearance_plan: DualArmClearancePlan,
) -> None:
    clock = ManualClock(1.0)
    transport = FakeArmTransport(clock=clock, initial_full_q=READY_Q29)
    executor = DualArmClearanceExecutor(
        transport=transport,
        clock=clock,
        plan=clearance_plan,
        config=_executor_config(),
    )
    executor.acquire(operator_confirmed=True)
    _advance_until(transport, executor, ExecutorState.READY)

    transport.position[LEFT_ARM_INDICES[0]] += 0.011
    clock.advance(0.02)
    executor.tick()

    assert executor.state is ExecutorState.FAULT
    assert "clearance-held arms drifted" in executor.fault_reason


def test_measured_out_of_limit_finger_start_may_only_recover_inward() -> None:
    class Model:
        def joint_limits(self, joint_names):
            return tuple(
                SimpleNamespace(lower=-1.74532925, upper=1.74532925)
                for _ in joint_names
            )

        def forward_kinematics(self, positions):
            return positions

    class CollisionChecker:
        def check(self, transforms):
            return SimpleNamespace(
                minimum_clearance_m=0.03,
                minimum_pair=None,
                colliding_pairs=(),
            )

    initial_left = np.zeros(7)
    initial_left[6] = -1.7790
    result = validate_dex3_finger_sweep_at_state(
        model=Model(),
        collision_checker=CollisionChecker(),
        body_q=READY_Q29,
        initial_left_hand_q_rad=initial_left,
        initial_right_hand_q_rad=np.zeros(7),
        target_left_hand_q_rad=NVIDIA_MIDDLE_CLOSE_LEFT_Q_RAD,
        target_right_hand_q_rad=NVIDIA_MIDDLE_CLOSE_RIGHT_Q_RAD,
        config=PathValidationConfig(maximum_joint_increment_rad=0.02),
    )

    assert result.passed
    assert result.recovered_start_limit_joints == ("left_hand_index_1_joint",)


def test_dex3_finger_target_must_remain_within_unitree_limits() -> None:
    class Model:
        def joint_limits(self, joint_names):
            return tuple(
                SimpleNamespace(lower=-1.74532925, upper=1.74532925)
                for _ in joint_names
            )

        def forward_kinematics(self, positions):
            return positions

    class CollisionChecker:
        def check(self, transforms):
            raise AssertionError("invalid target must fail before FCL")

    invalid_target = np.asarray(NVIDIA_MIDDLE_CLOSE_LEFT_Q_RAD).copy()
    invalid_target[6] = -1.80
    result = validate_dex3_finger_sweep_at_state(
        model=Model(),
        collision_checker=CollisionChecker(),
        body_q=READY_Q29,
        initial_left_hand_q_rad=np.zeros(7),
        initial_right_hand_q_rad=np.zeros(7),
        target_left_hand_q_rad=invalid_target,
        target_right_hand_q_rad=NVIDIA_MIDDLE_CLOSE_RIGHT_Q_RAD,
        config=PathValidationConfig(maximum_joint_increment_rad=0.02),
    )

    assert not result.passed
    assert result.failure == (
        "target: left_hand_index_1_joint=-1.8000rad outside hard limits"
    )
