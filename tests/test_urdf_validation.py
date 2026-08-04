from pathlib import Path

import numpy as np
import pytest

from g1_aprilcube_calibration.collision import (
    AttachedBox,
    CollisionConfig,
    CollisionPair,
    FCLCollisionChecker,
)
from g1_aprilcube_calibration.joint_map import G1_29_JOINT_NAMES
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
from g1_aprilcube_calibration.urdf_model import URDFModel

ROOT = Path(__file__).parents[1]
URDF = ROOT / "unitree_ros" / "robots" / "g1_description" / "g1_29dof_rev_1_0.urdf"
COLLISIONS = ROOT / "config" / "collision_pairs.yaml"
UTC = "2026-08-02T12:00:00Z"


def make_pose(pose_id: str, calibration_q: np.ndarray, urdf_sha: str) -> PoseRecord:
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


def make_pose_set(model: URDFModel, target: np.ndarray) -> PoseSet:
    result = PoseSet(
        robot_model="g1_29dof_rev_1_0",
        mode_machine=5,
        urdf_sha256=model.sha256,
        calibration_arm="left",
    )
    for item in (make_pose("target", target, model.sha256),):
        result = result.with_pose(item, PoseAuditEvent("add", item.id, UTC))
    return result


def hardware_ready_collision_config() -> CollisionConfig:
    source = CollisionConfig.from_yaml(COLLISIONS)
    return CollisionConfig(
        pairs=source.pairs,
        visual_fallback_links=source.visual_fallback_links,
        attached_boxes=source.attached_boxes,
        hardware_ready=True,
    )


def test_official_mode5_urdf_chain_limits_and_fk_regression() -> None:
    model = URDFModel(URDF)
    assert model.name == "g1_29dof_rev_1_0"
    assert model.root_link == "pelvis"
    assert (
        model.sha256
        == "c0ae739c640c3e2c00d1bdd8810b5d6e59601487bd1a3995859f9543269ee5c8"
    )
    assert [joint.name for joint in model.chain("torso_link", "left_rubber_hand")] == [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "left_hand_palm_joint",
    ]
    positions = dict.fromkeys(G1_29_JOINT_NAMES, 0.0)
    transform = model.transform("torso_link", "left_rubber_hand", positions)
    assert np.allclose(
        transform[:3, 3],
        [0.2452383588, 0.151653753121, 0.051230731362],
        atol=1e-10,
    )
    d435 = model.transform("torso_link", "d435_link", positions)
    assert np.allclose(d435[:3, 3], [0.0576235, 0.01753, 0.42987])
    left_limits = model.joint_limits(tuple(G1_29_JOINT_NAMES[15:22]))
    assert left_limits[0].lower == pytest.approx(-3.0892)
    assert left_limits[3].upper == pytest.approx(2.0944)


def test_selected_real_g1_collision_configuration_has_clearance_at_zero() -> None:
    model = URDFModel(URDF)
    checker = FCLCollisionChecker(model, CollisionConfig.from_yaml(COLLISIONS))
    transforms = model.forward_kinematics(dict.fromkeys(G1_29_JOINT_NAMES, 0.0))
    result = checker.check(transforms)
    assert result.colliding_pairs == ()
    assert result.minimum_clearance_m > 0.03
    assert result.minimum_pair == CollisionPair("left_elbow_link", "torso_link")


def test_modeled_left_palm_envelope_contains_cube_tape_and_margin() -> None:
    config = CollisionConfig.from_yaml(COLLISIONS)
    assert config.hardware_ready
    assert config.required_attached_boxes == ("aprilcube_envelope",)
    assert len(config.attached_boxes) == 1
    envelope = config.attached_boxes[0]
    assert envelope.parent_link == "left_rubber_hand"
    assert envelope.rpy_rad == (0.0, 0.0, 0.0)
    lower = np.asarray(envelope.xyz_m) - np.asarray(envelope.size_m) / 2.0
    upper = np.asarray(envelope.xyz_m) + np.asarray(envelope.size_m) / 2.0
    tape_lower = np.asarray([0.015, -0.013, -0.025])
    tape_upper = np.asarray([0.065, -0.0105, 0.025])
    assert np.all(lower <= tape_lower - 0.005 + 1e-12)
    assert np.all(upper >= tape_upper + 0.005 - 1e-12)
    assert any(
        "aprilcube_envelope" in (pair.first, pair.second) for pair in config.pairs
    )


def test_required_attachment_must_exist_and_be_collision_checked() -> None:
    attachment = AttachedBox(
        name="target",
        parent_link="left_rubber_hand",
        size_m=(0.06, 0.058, 0.06),
        xyz_m=(0.04, -0.034, 0.0),
        rpy_rad=(0.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="not used by any pair"):
        CollisionConfig(
            pairs=(CollisionPair("left_elbow_link", "torso_link"),),
            attached_boxes=(attachment,),
            required_attached_boxes=("target",),
            hardware_ready=True,
        )


def test_real_fcl_checker_detects_deliberate_overlap(tmp_path) -> None:
    urdf = tmp_path / "overlap.urdf"
    urdf.write_text(
        """<robot name="overlap">
        <link name="a"><collision><geometry><box size="1 1 1"/></geometry></collision></link>
        <link name="b"><collision><geometry><sphere radius="0.2"/></geometry></collision></link>
        <joint name="ab" type="fixed"><parent link="a"/><child link="b"/></joint>
        </robot>"""
    )
    model = URDFModel(urdf)
    config = CollisionConfig((CollisionPair("a", "b"),), hardware_ready=True)
    result = FCLCollisionChecker(model, config).check(model.forward_kinematics({}))
    assert result.minimum_clearance_m < 0.0
    assert result.colliding_pairs == (CollisionPair("a", "b"),)


def test_path_validator_samples_by_increment_and_emits_approval() -> None:
    model = URDFModel(URDF)
    checker = FCLCollisionChecker(model, hardware_ready_collision_config())
    pose_set = make_pose_set(model, np.array([0.1, 0, 0, 0, 0, 0, 0], dtype=float))
    validator = PosePathValidator(
        model=model,
        collision_checker=checker,
        config=PathValidationConfig(
            maximum_joint_increment_rad=0.02,
            joint_limit_margin_rad=0.03,
            minimum_collision_clearance_m=0.005,
            maximum_path_length_rad=5.0,
            assumed_joint_velocity_rad_s=0.2,
        ),
    )
    report = validator.validate(
        pose_set,
        directed_edges=((HANDOFF_POSE_ID, "target"),),
        reference_full_q=np.zeros(29),
    )
    edge = report.edge(HANDOFF_POSE_ID, "target")
    assert report.passed
    assert edge.sample_count == 6
    assert edge.estimated_duration_s == pytest.approx(0.5)
    approval = report.approval(HANDOFF_POSE_ID, "target")
    assert approval.passed
    assert approval.pose_set_sha256 == pose_set.content_sha256
    assert approval.validation_report_sha256 == report.content_sha256
    assert (
        ValidationReport.from_dict(report.to_dict()).content_sha256
        == report.content_sha256
    )


def test_path_validator_rejects_joint_limit_and_stale_urdf() -> None:
    model = URDFModel(URDF)
    checker = FCLCollisionChecker(model, hardware_ready_collision_config())
    target = np.zeros(7)
    target[0] = 2.66  # Inside the raw upper limit, outside the configured margin.
    pose_set = make_pose_set(model, target)
    validator = PosePathValidator(
        model=model,
        collision_checker=checker,
        config=PathValidationConfig(maximum_joint_increment_rad=0.5),
    )
    report = validator.validate(
        pose_set,
        directed_edges=((HANDOFF_POSE_ID, "target"),),
        reference_full_q=np.zeros(29),
    )
    assert not report.passed
    assert any("left_shoulder_pitch_joint" in item for item in report.edges[0].failures)

    stale = PoseSet(
        robot_model=pose_set.robot_model,
        mode_machine=5,
        urdf_sha256="d" * 64,
        calibration_arm=pose_set.calibration_arm,
    )
    with pytest.raises(ValueError, match="different URDF"):
        validator.validate(stale, directed_edges=(), reference_full_q=np.zeros(29))


def test_modeled_aprilcube_envelope_allows_hardware_approval() -> None:
    model = URDFModel(URDF)
    checker = FCLCollisionChecker(model, CollisionConfig.from_yaml(COLLISIONS))
    pose_set = make_pose_set(model, np.full(7, 0.01))
    report = PosePathValidator(model=model, collision_checker=checker).validate(
        pose_set,
        directed_edges=((HANDOFF_POSE_ID, "target"),),
        reference_full_q=np.zeros(29),
    )
    assert report.passed
    assert report.edges[0].failures == ()
