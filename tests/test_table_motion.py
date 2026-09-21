from pathlib import Path

import numpy as np
import pytest

from g1_aprilcube_calibration.collision import CollisionConfig
from g1_aprilcube_calibration.hardware_cli import (
    _build_table_motion_preflight_worker,
    _run_isolated_control_work,
    _validate_final_table_motion_worker,
)
from g1_aprilcube_calibration.inverse_kinematics import IKConfig
from g1_aprilcube_calibration.joint_map import G1_29_JOINT_NAMES
from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID
from g1_aprilcube_calibration.pose_validator import PathValidationConfig
from g1_aprilcube_calibration.table_motion import (
    TABLE_ESCAPE_POSE_ID,
    TABLE_TARGET_POSE_ID,
    TablePlaneConfig,
    build_lifted_start_cube_target,
    build_table_motion_preflight,
    validate_table_plane_escape_path,
    validate_table_plane_path,
)
from g1_aprilcube_calibration.urdf_model import URDFModel

ROOT = Path(__file__).parents[1]
URDF = ROOT / "unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"
UTC = "2026-08-08T12:00:00Z"
PLANE_CONFIG = TablePlaneConfig(
    board_width_m=10.0,
    board_height_m=10.0,
    table_margin_m=10.0,
)


def _target(model: URDFModel) -> np.ndarray:
    full = np.zeros(29)
    full[15:22] = [0.25, 0.15, -0.2, 0.5, 0.15, -0.1, 0.1]
    return model.transform(
        "torso_link",
        "left_rubber_hand",
        dict(zip(G1_29_JOINT_NAMES, full, strict=True)),
    )


def _torso_T_board() -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = np.diag([1.0, -1.0, -1.0])
    transform[:3, 3] = [0.0, 0.0, -1.0]
    return transform


def test_table_motion_selects_hash_bound_round_trip_ik_target() -> None:
    model = URDFModel(URDF)
    preflight = build_table_motion_preflight(
        model=model,
        collision_config=CollisionConfig((), hardware_ready=True),
        desired_torso_T_escape=_target(model),
        desired_torso_T_hand=_target(model),
        torso_T_board=_torso_T_board(),
        reference_full_q=np.zeros(29),
        calibration_arm="left",
        plan_sha256="a" * 64,
        recorded_at_utc=UTC,
        recorded_monotonic_s=1.0,
        ik_config=IKConfig(restart_count=8),
        table_plane_config=PLANE_CONFIG,
    )

    escape_pose, target_pose = preflight.pose_set.poses
    assert escape_pose.id == TABLE_ESCAPE_POSE_ID
    assert escape_pose.source == "table_accuracy_escape_ik"
    assert target_pose.id == TABLE_TARGET_POSE_ID
    assert target_pose.source == "table_accuracy_target_ik"
    assert escape_pose.measured_calibration_q == (0.0,) * 7
    assert target_pose.measured_calibration_q == (0.0,) * 7
    assert (
        escape_pose.replay_calibration_q == preflight.escape_ik_solution.calibration_q
    )
    assert (
        target_pose.replay_calibration_q == preflight.target_ik_solution.calibration_q
    )
    assert target_pose.visual_quality["table_accuracy_plan_sha256"] == "a" * 64
    assert preflight.validation_report.passed
    assert {
        (edge.from_pose_id, edge.to_pose_id)
        for edge in preflight.validation_report.edges
    } == {
        (HANDOFF_POSE_ID, TABLE_ESCAPE_POSE_ID),
        (TABLE_ESCAPE_POSE_ID, TABLE_TARGET_POSE_ID),
        (TABLE_TARGET_POSE_ID, TABLE_ESCAPE_POSE_ID),
        (TABLE_ESCAPE_POSE_ID, HANDOFF_POSE_ID),
    }


def test_table_motion_preflight_round_trips_through_isolated_worker() -> None:
    class Driver:
        def check(self):
            return None

    model = URDFModel(URDF)
    preflight = _run_isolated_control_work(
        label="isolated table preflight test",
        worker=_build_table_motion_preflight_worker,
        worker_kwargs={
            "urdf_path": str(URDF),
            "collision_config": CollisionConfig((), hardware_ready=True),
            "desired_torso_T_escape": _target(model),
            "desired_torso_T_hand": _target(model),
            "torso_T_board": _torso_T_board(),
            "reference_full_q": np.zeros(29),
            "calibration_arm": "left",
            "plan_sha256": "a" * 64,
            "recorded_at_utc": UTC,
            "recorded_monotonic_s": 1.0,
            "ik_config": IKConfig(restart_count=4),
            "table_plane_config": PLANE_CONFIG,
        },
        driver=Driver(),
    )

    assert preflight.validation_report.passed
    assert preflight.pose_set.content_sha256

    report, escape_clearance, route_clearance = _run_isolated_control_work(
        label="isolated final table validation test",
        worker=_validate_final_table_motion_worker,
        worker_kwargs={
            "urdf_path": str(URDF),
            "collision_config": CollisionConfig((), hardware_ready=True),
            "pose_set": preflight.pose_set,
            "reference_full_q": np.zeros(29),
            "torso_T_board": _torso_T_board(),
            "calibration_arm": "left",
            "escape_target_q": np.asarray(
                preflight.escape_ik_solution.calibration_q
            ),
            "table_target_q": np.asarray(
                preflight.target_ik_solution.calibration_q
            ),
            "table_plane_config": PLANE_CONFIG,
        },
        driver=Driver(),
    )

    assert report.passed
    assert escape_clearance.passed
    assert route_clearance.passed


def test_table_motion_rejects_every_path_that_exceeds_policy() -> None:
    model = URDFModel(URDF)

    with pytest.raises(ValueError, match="no escape/target IK pair passed"):
        build_table_motion_preflight(
            model=model,
            collision_config=CollisionConfig((), hardware_ready=True),
            desired_torso_T_escape=_target(model),
            desired_torso_T_hand=_target(model),
            torso_T_board=_torso_T_board(),
            reference_full_q=np.zeros(29),
            calibration_arm="left",
            plan_sha256="a" * 64,
            recorded_at_utc=UTC,
            recorded_monotonic_s=1.0,
            ik_config=IKConfig(restart_count=4),
            path_config=PathValidationConfig(maximum_path_length_rad=0.001),
            table_plane_config=PLANE_CONFIG,
        )


def test_table_motion_identifies_an_unreachable_target_waypoint() -> None:
    model = URDFModel(URDF)
    unreachable = _target(model)
    unreachable[0, 3] += 10.0

    with pytest.raises(ValueError, match="table target waypoint IK failed"):
        build_table_motion_preflight(
            model=model,
            collision_config=CollisionConfig((), hardware_ready=True),
            desired_torso_T_escape=_target(model),
            desired_torso_T_hand=unreachable,
            torso_T_board=_torso_T_board(),
            reference_full_q=np.zeros(29),
            calibration_arm="left",
            plan_sha256="a" * 64,
            recorded_at_utc=UTC,
            recorded_monotonic_s=1.0,
            ik_config=IKConfig(restart_count=1, maximum_function_evaluations=50),
            table_plane_config=PLANE_CONFIG,
        )


def test_lifted_start_preserves_live_xy_orientation_at_target_height() -> None:
    current = np.eye(4)
    current[:3, :3] = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    current[:3, 3] = [0.02, 0.25, -0.04]
    target = np.eye(4)
    target[:3, 3] = [0.0, 0.27, -0.10]

    escape = build_lifted_start_cube_target(
        current_board_T_cube=current,
        desired_board_T_cube=target,
    )

    assert np.allclose(escape[:3, :3], current[:3, :3])
    assert np.allclose(escape[:2, 3], current[:2, 3])
    assert escape[2, 3] == pytest.approx(-0.10)

    invalid_target = target.copy()
    invalid_target[2, 3] = -0.03
    with pytest.raises(ValueError, match="does not lift"):
        build_lifted_start_cube_target(
            current_board_T_cube=current,
            desired_board_T_cube=invalid_target,
        )


def test_table_plane_validator_rejects_arm_geometry_below_plane() -> None:
    model = URDFModel(URDF)
    # With an identity board frame, much of the arm lies on positive board Z,
    # which is defined as inside/below the tabletop.
    result = validate_table_plane_path(
        model=model,
        collision_config=CollisionConfig((), hardware_ready=True),
        torso_T_board=np.eye(4),
        reference_full_q=np.zeros(29),
        calibration_arm="left",
        target_calibration_q=np.zeros(7),
        config=TablePlaneConfig(
            board_width_m=10.0,
            board_height_m=10.0,
            table_margin_m=10.0,
            minimum_clearance_m=0.01,
        ),
    )

    assert not result.passed
    assert result.minimum_clearance_m < 0


def test_supported_start_escape_must_finish_clear_of_the_plane() -> None:
    model = URDFModel(URDF)
    result = validate_table_plane_escape_path(
        model=model,
        collision_config=CollisionConfig((), hardware_ready=True),
        torso_T_board=np.eye(4),
        reference_full_q=np.zeros(29),
        calibration_arm="left",
        target_calibration_q=np.zeros(7),
        config=TablePlaneConfig(
            board_width_m=10.0,
            board_height_m=10.0,
            table_margin_m=10.0,
            minimum_clearance_m=0.01,
        ),
    )

    assert result.source_clearance_m is not None
    assert result.source_clearance_m < 0
    assert result.maximum_additional_penetration_m == pytest.approx(0.0)
    assert not result.passed


def test_already_clear_supported_start_satisfies_escape_policy() -> None:
    model = URDFModel(URDF)
    result = validate_table_plane_escape_path(
        model=model,
        collision_config=CollisionConfig((), hardware_ready=True),
        torso_T_board=_torso_T_board(),
        reference_full_q=np.zeros(29),
        calibration_arm="left",
        target_calibration_q=np.zeros(7),
        config=PLANE_CONFIG,
    )

    assert result.passed
    assert result.destination_clearance_m is not None
    assert result.destination_clearance_m >= result.required_destination_clearance_m


def test_table_plane_validator_includes_attached_cube_envelope() -> None:
    from g1_aprilcube_calibration.collision import AttachedBox

    model = URDFModel(URDF)
    without_cube = validate_table_plane_path(
        model=model,
        collision_config=CollisionConfig((), hardware_ready=True),
        torso_T_board=_torso_T_board(),
        reference_full_q=np.zeros(29),
        calibration_arm="left",
        target_calibration_q=np.zeros(7),
        config=PLANE_CONFIG,
    )
    with_cube = validate_table_plane_path(
        model=model,
        collision_config=CollisionConfig(
            (),
            attached_boxes=(
                AttachedBox(
                    name="cube",
                    parent_link="left_rubber_hand",
                    size_m=(0.06, 0.06, 0.06),
                    xyz_m=(0.0, 0.0, 0.0),
                    rpy_rad=(0.0, 0.0, 0.0),
                ),
            ),
            hardware_ready=True,
        ),
        torso_T_board=_torso_T_board(),
        reference_full_q=np.zeros(29),
        calibration_arm="left",
        target_calibration_q=np.zeros(7),
        config=PLANE_CONFIG,
    )

    assert without_cube.passed
    assert with_cube.passed
    assert with_cube.minimum_clearance_m <= without_cube.minimum_clearance_m
