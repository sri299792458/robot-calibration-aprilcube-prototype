import importlib.util
import zipfile
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import trimesh
import yaml

from aprilcube import CorrespondenceDetector
from g1_aprilcube_calibration.collision import CollisionConfig
from g1_aprilcube_calibration.dex3_dorsal_mount import (
    ARUCO_DICTIONARY_ID,
    ARUCO_DICTIONARY_NAME,
    DEFAULT_DEX3_DORSAL_MOUNT_SPEC,
    hole_centers_plate_mm,
    marker_grid,
    mount_manifest,
    palm_T_marker_face,
    palm_T_plate_mm,
    validate_mount_spec,
)
from g1_aprilcube_calibration.transports.unitree_dex3 import (
    DEX3_MOTOR_JOINT_SUFFIXES,
    NVIDIA_MIDDLE_CLOSE_LEFT_Q_RAD,
    NVIDIA_MIDDLE_CLOSE_RIGHT_Q_RAD,
)
from g1_aprilcube_calibration.urdf_model import URDFModel

SCRIPT = (
    Path(__file__).resolve().parents[1] / "tools/generate_dex3_dorsal_aruco_mount.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "generate_dex3_dorsal_aruco_mount", SCRIPT
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
generator = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(generator)


def test_target_matches_marker_decoded_from_graspgenx_video() -> None:
    spec = DEFAULT_DEX3_DORSAL_MOUNT_SPEC
    assert ARUCO_DICTIONARY_NAME == "DICT_6X6_50"
    assert spec.marker_id == 4

    marker = (marker_grid(spec) * 255).astype(np.uint8)
    marker = cv2.resize(marker, (800, 800), interpolation=cv2.INTER_NEAREST)
    preview = cv2.copyMakeBorder(
        marker,
        100,
        100,
        100,
        100,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY_ID)
    _corners, ids, _rejected = cv2.aruco.ArucoDetector(dictionary).detectMarkers(
        preview
    )
    assert ids is not None
    assert ids.ravel().tolist() == [4]


def test_left_marker_id_five_decodes_in_the_same_dictionary() -> None:
    spec = replace(DEFAULT_DEX3_DORSAL_MOUNT_SPEC, marker_id=5)
    validate_mount_spec(spec)
    marker = (marker_grid(spec) * 255).astype(np.uint8)
    marker = cv2.resize(marker, (800, 800), interpolation=cv2.INTER_NEAREST)
    preview = cv2.copyMakeBorder(
        marker,
        100,
        100,
        100,
        100,
        cv2.BORDER_CONSTANT,
        value=255,
    )

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY_ID)
    _corners, ids, _rejected = cv2.aruco.ArucoDetector(dictionary).detectMarkers(
        preview
    )

    assert ids is not None
    assert ids.ravel().tolist() == [5]
    right = palm_T_marker_face(DEFAULT_DEX3_DORSAL_MOUNT_SPEC, side="right")
    left = palm_T_marker_face(spec, side="left")
    assert np.isclose(np.linalg.det(left[:3, :3]), 1.0)
    np.testing.assert_allclose(right[:3, 2], [0.0199430329, -0.999801118, 0.0])
    np.testing.assert_allclose(left[:3, 2], [0.0199430329, 0.999801118, 0.0])
    np.testing.assert_allclose(left[:3, 3], [right[0, 3], -right[1, 3], 0.0])


def test_arm_specific_detector_ignores_other_hand_and_rejects_duplicate_id() -> None:
    root = Path(__file__).resolve().parents[1]
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY_ID)
    marker_4 = cv2.aruco.generateImageMarker(dictionary, 4, 220)
    marker_5 = cv2.aruco.generateImageMarker(dictionary, 5, 220)
    image = np.full((400, 800), 255, dtype=np.uint8)
    image[90:310, 90:310] = marker_4
    image[90:310, 490:710] = marker_5

    right = CorrespondenceDetector(root / "config/dex3_dorsal_aruco_target.json")
    left = CorrespondenceDetector(
        root / "config/dex3_left_dorsal_aruco_id5_target.json"
    )
    right_result = right.detect(image)
    left_result = left.detect(image)

    assert right_result.tag_ids == (4,)
    assert right_result.ignored_tag_ids == (5,)
    assert left_result.tag_ids == (5,)
    assert left_result.ignored_tag_ids == (4,)

    duplicate = image.copy()
    duplicate[90:310, 490:710] = marker_4
    duplicate_result = right.detect(duplicate)
    assert duplicate_result.tag_ids == ()
    assert duplicate_result.duplicate_tag_ids == (4,)
    assert not duplicate_result.valid


def test_two_m3_heads_match_official_15_mm_pattern() -> None:
    spec = DEFAULT_DEX3_DORSAL_MOUNT_SPEC
    validate_mount_spec(spec)
    holes = hole_centers_plate_mm(spec)
    assert np.isclose(np.linalg.norm(holes[1] - holes[0]), 15.0)
    assert np.allclose(holes[:, 1], 55.0)
    head_radius = spec.countersink_major_diameter_mm / 2.0
    assert np.all(holes[:, 1] - head_radius >= spec.plate_size_mm)
    assert np.all(holes[:, 1] + head_radius <= spec.plate_length_mm)
    assert np.allclose(
        holes[:, 1] - head_radius - spec.plate_size_mm,
        spec.minimum_tab_ligament_mm,
    )
    assert np.allclose(
        spec.plate_length_mm - holes[:, 1] - head_radius,
        spec.minimum_tab_ligament_mm,
    )


def test_nominal_marker_frame_is_rigid_and_points_out_of_dorsal_palm() -> None:
    transform = palm_T_marker_face(side="right")
    rotation = transform[:3, :3]
    assert np.allclose(rotation.T @ rotation, np.eye(3))
    assert np.isclose(np.linalg.det(rotation), 1.0)
    assert np.allclose(
        rotation[:, 2],
        [0.0199430329, -0.999801118, 0.0],
    )
    assert np.allclose(transform[:3, 3], [0.03681565, -0.0274272, 0.0])


def test_runtime_hardware_profile_freezes_generated_nominal_transform() -> None:
    root = Path(__file__).resolve().parents[1]
    hardware = yaml.safe_load((root / "config/hardware_dex3_aruco.yaml").read_text())
    configured = np.asarray(
        hardware["robot"]["calibration_target_modeled_hand_T_target"]
    )

    np.testing.assert_allclose(configured, palm_T_marker_face(side="right"), atol=1e-12)

    left_hardware = yaml.safe_load(
        (root / "config/hardware_dex3_left_aruco_id5.yaml").read_text()
    )
    left_configured = np.asarray(
        left_hardware["robot"]["calibration_target_modeled_hand_T_target"]
    )
    np.testing.assert_allclose(
        left_configured, palm_T_marker_face(side="left"), atol=1e-12
    )


def test_pitched_plate_maps_all_three_support_centers_to_the_shell() -> None:
    spec = DEFAULT_DEX3_DORSAL_MOUNT_SPEC
    transform = palm_T_plate_mm(spec)
    hole_line = hole_centers_plate_mm(spec)[:, 1].mean()
    boss_center = transform @ np.asarray(
        [
            spec.plate_size_mm / 2.0,
            hole_line,
            spec.plate_thickness_mm + spec.screw_boss_height_mm,
            1.0,
        ]
    )
    pad_center = transform @ np.asarray(
        [
            spec.plate_size_mm / 2.0,
            spec.wrist_pad_from_proximal_edge_mm,
            spec.plate_thickness_mm + spec.wrist_pad_height_mm,
            1.0,
        ]
    )
    assert np.allclose(
        boss_center[:3],
        [
            spec.palm_hole_midpoint_x_mm,
            spec.palm_dorsal_surface_y_at_holes_mm,
            spec.palm_hole_midpoint_z_mm,
        ],
    )
    assert np.isclose(
        pad_center[1],
        spec.palm_dorsal_surface_y_at_wrist_pad_mm,
        atol=0.001,
    )
    assert np.isclose(spec.mounting_pitch_deg, 1.1427, atol=0.001)
    assert np.isclose(spec.wrist_pad_face_pitch_deg, -1.9935, atol=0.001)


def test_m3_x_8_fastener_stack_engages_without_bottoming() -> None:
    spec = DEFAULT_DEX3_DORSAL_MOUNT_SPEC
    assert np.isclose(spec.nominal_thread_engagement_mm, 2.5)
    assert np.isclose(spec.nominal_bottoming_clearance_mm, 0.5)


def test_manifest_records_official_hole_datum_and_fastener_stack() -> None:
    manifest = mount_manifest()
    assert manifest["spec"]["palm_hole_midpoint_x_mm"] == 66.7
    assert manifest["spec"]["hole_spacing_mm"] == 15.0
    assert manifest["spec"]["thread_depth_mm"] == 3.0
    assert manifest["fastener_stack"]["nominal_thread_engagement_mm"] == 2.5
    assert manifest["fastener_stack"]["nominal_bottoming_clearance_mm"] == 0.5
    assert "User_Manual.html" in manifest["sources"]["official_dimensioned_drawing"]


def test_generated_plate_and_coupon_are_closed_printable_volumes() -> None:
    spec = DEFAULT_DEX3_DORSAL_MOUNT_SPEC
    parts = generator.build_parts(spec)
    assert set(parts) == {"carrier_white_pla", "aruco_black_pla"}
    assert all(mesh.is_volume for mesh in parts.values())
    assert all(mesh.is_watertight for mesh in parts.values())
    assert np.allclose(
        parts["carrier_white_pla"].bounds[1],
        [50.0, 60.0, 6.6744],
        atol=0.001,
    )
    assert np.allclose(parts["aruco_black_pla"].bounds[1], [45.0, 45.0, 0.6])

    coupon = generator.make_fit_coupon(spec)
    assert coupon.is_volume
    assert coupon.is_watertight
    assert np.allclose(coupon.extents, [25.0, 8.0, 2.0])


def test_mounted_plate_clears_all_neutral_finger_links() -> None:
    spec = DEFAULT_DEX3_DORSAL_MOUNT_SPEC
    plate_transform = generator.palm_T_plate(spec)
    mounted = trimesh.util.concatenate(
        [
            generator.transformed(mesh, plate_transform)
            for mesh in generator.build_parts(spec).values()
        ]
    )
    hand_meshes = generator.load_neutral_hand_meshes()
    assert len(hand_meshes) == 8

    for finger_link in hand_meshes[1:]:
        collision = trimesh.collision.CollisionManager()
        collision.add_object("finger", finger_link)
        assert not collision.in_collision_single(mounted)


def test_segmented_collision_boxes_contain_middle_close_posture_meshes() -> None:
    root = Path(__file__).resolve().parents[1]
    collision = CollisionConfig.from_yaml(
        root / "config/collision_pairs_dex3_aruco.yaml"
    )
    boxes = {item.name: item for item in collision.attached_boxes}
    groups = {
        "palm": ("palm",),
        "thumb": ("thumb_0", "thumb_1", "thumb_2"),
        "middle": ("middle_0", "middle_1"),
        "index": ("index_0", "index_1"),
    }

    for short_side, side in (("l", "left"), ("r", "right")):
        model = URDFModel(
            root
            / "unitree_ros/robots/dexterous_hand_description/dex3_1"
            / f"dex3_1_{short_side}.urdf"
        )
        base = f"{side}_hand_palm_link"
        target = (
            NVIDIA_MIDDLE_CLOSE_LEFT_Q_RAD
            if side == "left"
            else NVIDIA_MIDDLE_CLOSE_RIGHT_Q_RAD
        )
        positions = {
            f"{side}_hand_{suffix}_joint": value
            for suffix, value in zip(
                DEX3_MOTOR_JOINT_SUFFIXES[side], target, strict=True
            )
        }
        for group, suffixes in groups.items():
            vertices = []
            for suffix in suffixes:
                link = base if suffix == "palm" else f"{side}_hand_{suffix}_link"
                link_transform = model.transform(base, link, positions)
                for geometry in model.link_geometries(link):
                    points = np.column_stack(
                        (geometry.mesh.vertices, np.ones(len(geometry.mesh.vertices)))
                    )
                    vertices.append(
                        (link_transform @ geometry.local_transform @ points.T).T[:, :3]
                    )
            points = np.vstack(vertices)
            box_name = f"{side}_dex3_{group}"
            if group == "palm":
                box_name = f"{side}_dex3_marker_palm"
            box = boxes[box_name]
            lower = np.asarray(box.xyz_m) - np.asarray(box.size_m) / 2.0
            upper = np.asarray(box.xyz_m) + np.asarray(box.size_m) / 2.0
            assert np.all(lower <= np.min(points, axis=0) - 0.00249)
            assert np.all(upper >= np.max(points, axis=0) + 0.00249)

    for side in ("left", "right"):
        plate_transform = generator.palm_T_plate(DEFAULT_DEX3_DORSAL_MOUNT_SPEC, side)
        mounted = [
            generator.transformed(mesh, plate_transform)
            for mesh in generator.build_parts(DEFAULT_DEX3_DORSAL_MOUNT_SPEC).values()
        ]
        plate_points_m = np.vstack([mesh.vertices for mesh in mounted]) / 1000.0
        palm = boxes[f"{side}_dex3_marker_palm"]
        lower = np.asarray(palm.xyz_m) - np.asarray(palm.size_m) / 2.0
        upper = np.asarray(palm.xyz_m) + np.asarray(palm.size_m) / 2.0
        assert np.all(lower <= np.min(plate_points_m, axis=0) - 0.00249)
        assert np.all(upper >= np.max(plate_points_m, axis=0) + 0.00249)


def test_multicolor_3mf_is_deterministic_and_assigns_both_filaments(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "dex3_mount.3mf"
    monkeypatch.setattr(generator, "MULTICOLOR_3MF", output)
    white = trimesh.creation.box(extents=(10.0, 10.0, 1.0))
    black = trimesh.creation.box(extents=(2.0, 2.0, 0.6))

    generator.write_multicolor_3mf(white, black)
    first = output.read_bytes()
    generator.write_multicolor_3mf(white, black)
    assert output.read_bytes() == first

    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        model = archive.read("3D/3dmodel.model").decode()
        settings = archive.read("Metadata/model_settings.config").decode()
    assert 'name="carrier_white_PLA"' in model
    assert 'name="aruco_black_PLA"' in model
    assert '<metadata key="extruder" value="2"/>' in settings
    assert '<metadata key="extruder" value="1"/>' in settings
