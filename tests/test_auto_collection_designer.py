import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from g1_aprilcube_calibration.authored_collection import (
    AuthoredCollectionPlan,
    AuthoredPoseTarget,
    exposed_target_normal_from_hardware,
    modeled_hand_T_target_from_hardware,
    validate_hardware_target_profile,
    validate_modeled_hand_target_binding,
)
from g1_aprilcube_calibration.auto_collection_designer import (
    AutoCollectionDesignConfig,
    _auto_design_worker_count,
    _CoverageSignature,
    _FeasibleCandidate,
    _sample_camera_T_target,
    _select_information_targets,
    _target_dimensions_m,
    _target_projects_inside,
    _validate_final_route,
    camera_info_from_hardware,
)
from g1_aprilcube_calibration.collision import CollisionConfig, FCLCollisionChecker
from g1_aprilcube_calibration.dex3_dorsal_mount import palm_T_marker_face
from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID
from g1_aprilcube_calibration.pose_validator import (
    PathValidationConfig,
    PosePathValidator,
)
from g1_aprilcube_calibration.urdf_model import URDFModel

ROOT = Path(__file__).parents[1]
URDF = ROOT / "unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"
DEX3_COLLISIONS = ROOT / "config/collision_pairs_dex3_aruco.yaml"


def test_arm_profiles_bind_distinct_ids_to_mirrored_dorsal_mounts() -> None:
    right_hardware = yaml.safe_load(
        (ROOT / "config/hardware_dex3_aruco.yaml").read_text()
    )
    left_hardware = yaml.safe_load(
        (ROOT / "config/hardware_dex3_left_aruco_id5.yaml").read_text()
    )
    right_target = json.loads(
        (ROOT / "config/dex3_dorsal_aruco_target.json").read_text()
    )
    left_target = json.loads(
        (ROOT / "config/dex3_left_dorsal_aruco_id5_target.json").read_text()
    )

    validate_hardware_target_profile(right_hardware, right_target)
    validate_hardware_target_profile(left_hardware, left_target)
    assert right_hardware["robot"]["calibration_arm"] == "right"
    assert left_hardware["robot"]["calibration_arm"] == "left"
    assert right_target["tag_ids"] == [4]
    assert left_target["tag_ids"] == [5]
    np.testing.assert_allclose(
        modeled_hand_T_target_from_hardware(right_hardware),
        palm_T_marker_face(side="right"),
    )
    np.testing.assert_allclose(
        modeled_hand_T_target_from_hardware(left_hardware),
        palm_T_marker_face(side="left"),
    )
    assert np.sign(
        modeled_hand_T_target_from_hardware(right_hardware)[1, 2]
    ) != np.sign(modeled_hand_T_target_from_hardware(left_hardware)[1, 2])
    assert right_hardware["camera"] == left_hardware["camera"]
    assert right_hardware["pose_recording"] == left_hardware["pose_recording"]
    assert right_hardware["safety"] == left_hardware["safety"]
    assert right_hardware["ros"] == left_hardware["ros"]
    right_control = dict(right_hardware["control"], calibration_arm="selected")
    left_control = dict(left_hardware["control"], calibration_arm="selected")
    assert right_control == left_control

    with pytest.raises(ValueError, match="hardware marker ID"):
        validate_hardware_target_profile(left_hardware, right_target)
    duplicate_hardware = {
        **left_hardware,
        "robot": {
            **left_hardware["robot"],
            "opposite_hand_calibration_target_id": 5,
        },
    }
    with pytest.raises(ValueError, match="must differ"):
        validate_hardware_target_profile(duplicate_hardware, left_target)


def test_authored_plan_is_bound_to_its_side_specific_mount_transform() -> None:
    right = palm_T_marker_face(side="right")
    left = palm_T_marker_face(side="left")
    target = AuthoredPoseTarget(
        id="candidate_0001",
        authored_calibration_q=(0.0,) * 7,
        desired_camera_T_cube=np.eye(4),
    )
    plan = AuthoredCollectionPlan(
        robot_model="g1_29dof_rev_1_0",
        mode_machine=5,
        urdf_sha256="0" * 64,
        calibration_arm="left",
        camera_profile_sha256="1" * 64,
        reference_full_q_sha256="2" * 64,
        targets=(target,),
        route_pose_ids=(HANDOFF_POSE_ID, target.id, HANDOFF_POSE_ID),
        capture_pose_ids=(target.id,),
        generation_config={"modeled_hand_T_target": left.tolist()},
    )

    validate_modeled_hand_target_binding(plan, modeled_hand_T_target=left)
    with pytest.raises(ValueError, match="differs from hardware"):
        validate_modeled_hand_target_binding(plan, modeled_hand_T_target=right)


def test_auto_design_worker_count_is_bounded_and_overrideable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("os.cpu_count", lambda: 32)

    assert _auto_design_worker_count(requested=None, task_count=1600) == 8
    assert _auto_design_worker_count(requested=2, task_count=1600) == 2
    assert _auto_design_worker_count(requested=8, task_count=3) == 3
    with pytest.raises(ValueError, match="within"):
        _auto_design_worker_count(requested=9, task_count=1600)


def test_camera_frustum_sample_keeps_complete_target_inside_real_profile() -> None:
    hardware = yaml.safe_load((ROOT / "config/hardware_dex3_aruco.yaml").read_text())
    target = yaml.safe_load((ROOT / "config/dex3_dorsal_aruco_target.json").read_text())
    camera_info = camera_info_from_hardware(hardware)
    exposed_target_normal = exposed_target_normal_from_hardware(hardware)
    target_dimensions_m = _target_dimensions_m(target)
    config = AutoCollectionDesignConfig(target_count=1, candidate_count=1)

    camera_T_target = _sample_camera_T_target(
        np.asarray([0.5, 0.5, 0.5, 0.2, 0.5, 0.5]),
        camera_info=camera_info,
        target_dimensions_m=target_dimensions_m,
        exposed_target_normal=exposed_target_normal,
        config=config,
    )

    assert camera_T_target is not None
    assert _target_projects_inside(
        camera_T_target,
        camera_info=camera_info,
        target_dimensions_m=target_dimensions_m,
        margin_px=config.image_margin_px,
        minimum_span_px=config.minimum_projected_target_span_px,
    )
    assert config.minimum_depth_m <= camera_T_target[2, 3] <= config.maximum_depth_m


@pytest.mark.parametrize("view_sample", np.linspace(0.01, 0.99, 8))
def test_camera_frustum_sample_stays_inside_front_view_cone(
    view_sample: float,
) -> None:
    hardware = yaml.safe_load((ROOT / "config/hardware_dex3_aruco.yaml").read_text())
    camera_info = camera_info_from_hardware(hardware)
    exposed_target_normal = exposed_target_normal_from_hardware(hardware)
    config = AutoCollectionDesignConfig(target_count=1, candidate_count=1)

    camera_T_target = _sample_camera_T_target(
        np.asarray([0.5, 0.5, 0.5, view_sample, view_sample, 0.5]),
        camera_info=camera_info,
        target_dimensions_m=np.asarray([0.04, 0.04, 0.001]),
        exposed_target_normal=exposed_target_normal,
        config=config,
    )

    assert camera_T_target is not None
    target_to_camera = camera_T_target[:3, :3].T @ -camera_T_target[:3, 3]
    target_to_camera /= np.linalg.norm(target_to_camera)
    minimum_cosine = np.cos(np.deg2rad(config.maximum_view_obliquity_deg))
    assert float(target_to_camera @ exposed_target_normal) >= minimum_cosine - 1e-12


def test_camera_frustum_rejects_far_oblique_marker_below_short_side_gate() -> None:
    hardware = yaml.safe_load((ROOT / "config/hardware_dex3_aruco.yaml").read_text())
    config = AutoCollectionDesignConfig(target_count=1, candidate_count=1)

    camera_T_target = _sample_camera_T_target(
        np.asarray([0.5, 0.5, 1.0, 0.5, 1.0, 0.5]),
        camera_info=camera_info_from_hardware(hardware),
        target_dimensions_m=np.asarray([0.04, 0.04, 0.001]),
        exposed_target_normal=exposed_target_normal_from_hardware(hardware),
        config=config,
    )

    assert camera_T_target is None


def test_design_config_requires_more_candidates_than_targets() -> None:
    with pytest.raises(ValueError, match="at least target_count"):
        AutoCollectionDesignConfig(target_count=20, candidate_count=19)


def test_information_selection_balances_every_reachable_image_cell() -> None:
    candidates = []
    identity = np.eye(4)
    for row in range(3):
        for col in range(3):
            for variant in range(2):
                index = len(candidates) + 1
                candidates.append(
                    _FeasibleCandidate(
                        target=AuthoredPoseTarget(
                            id=f"candidate_{index:04d}",
                            authored_calibration_q=(0.0,) * 7,
                            desired_camera_T_cube=identity,
                            ik_diagnostics={},
                        ),
                        information_matrix=np.eye(12) * (index + variant + 1),
                        coverage=_CoverageSignature(
                            image_cell=(col, row),
                            depth_bin=variant,
                            obliquity_bin=(row + variant) % 3,
                            azimuth_bin=(col + variant) % 6,
                            in_plane_rotation_bin=(row + col + variant) % 6,
                            normalized_centroid=((col + 0.5) / 3, (row + 0.5) / 3),
                            depth_m=0.3 + 0.1 * variant,
                            obliquity_deg=10.0 * row,
                            azimuth_deg=60.0 * col,
                            in_plane_rotation_deg=30.0 * (row + col),
                        ),
                    )
                )
    config = AutoCollectionDesignConfig(
        target_count=9,
        candidate_count=len(candidates),
    )

    selected, steps = _select_information_targets(
        candidates,
        count=9,
        config=config,
        role="selected",
    )

    assert {item.coverage.image_cell for item in selected} == {
        (col, row) for row in range(3) for col in range(3)
    }
    assert len(steps) == 9


def test_parallel_final_route_validation_matches_serial_report() -> None:
    model = URDFModel(URDF)
    collision = CollisionConfig.from_yaml(DEX3_COLLISIONS)
    path = PathValidationConfig()
    reference = np.zeros(29)
    target = AuthoredPoseTarget(
        id="target",
        authored_calibration_q=(0.01,) * 7,
        desired_camera_T_cube=np.eye(4),
        ik_diagnostics={},
    )
    plan = AuthoredCollectionPlan(
        robot_model=model.name,
        mode_machine=5,
        urdf_sha256=model.sha256,
        calibration_arm="right",
        camera_profile_sha256="0" * 64,
        reference_full_q_sha256="1" * 64,
        targets=(target,),
        route_pose_ids=(HANDOFF_POSE_ID, target.id, HANDOFF_POSE_ID),
        capture_pose_ids=(target.id,),
        generation_config={},
    )
    directed_edges = (
        (HANDOFF_POSE_ID, target.id),
        (target.id, HANDOFF_POSE_ID),
    )

    def validate(worker_count: int):
        return _validate_final_route(
            model=model,
            collision_config=collision,
            path_config=path,
            validator=PosePathValidator(
                model=model,
                collision_checker=FCLCollisionChecker(model, collision),
                config=path,
            ),
            plan=plan,
            directed_edges=directed_edges,
            reference=reference,
            worker_count=worker_count,
        )

    assert validate(2).to_dict() == validate(1).to_dict()


def test_dex3_plate_profile_uses_nominal_cad_transform_and_front_face() -> None:
    hardware = yaml.safe_load((ROOT / "config/hardware_dex3_aruco.yaml").read_text())
    target = yaml.safe_load((ROOT / "config/dex3_dorsal_aruco_target.json").read_text())

    np.testing.assert_allclose(
        modeled_hand_T_target_from_hardware(hardware), palm_T_marker_face()
    )
    np.testing.assert_allclose(
        exposed_target_normal_from_hardware(hardware),
        [0.0, 0.0, 1.0],
    )
    np.testing.assert_allclose(_target_dimensions_m(target), [0.04, 0.04, 0.001])
    assert hardware["robot"]["calibration_arm"] == "right"
    assert hardware["control"]["calibration_arm"] == "right"
