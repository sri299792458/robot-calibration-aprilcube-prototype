from pathlib import Path

import numpy as np

from g1_aprilcube_calibration.camera_initialization import (
    nominal_torso_T_color_optical,
)
from g1_aprilcube_calibration.torso_board_mount import (
    NOMINAL_FRONT_M6_SURFACE_POINTS,
    board_view_incidence_deg,
    image_border_margins_px,
    nominal_front_m6_points_in_pattern,
    nominal_torso_T_front_m6_pattern,
    project_planar_board_outer_corners,
    torso_T_camera_from_fixed_board,
    torso_T_charuco_outer_corner,
)
from g1_aprilcube_calibration.transforms import invert_transform
from g1_aprilcube_calibration.urdf_model import URDFModel

ROOT = Path(__file__).parents[1]
URDF = ROOT / "unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"
BOARD_WIDTH_M = 0.180
BOARD_HEIGHT_M = 0.270
CAMERA_MATRIX = np.asarray(
    [
        [905.4686279296875, 0.0, 652.4732055664062],
        [0.0, 905.5797729492188, 368.78778076171875],
        [0.0, 0.0, 1.0],
    ]
)


def test_nominal_front_pattern_frame_reconstructs_surface_points() -> None:
    torso_T_pattern = nominal_torso_T_front_m6_pattern()
    local_points = nominal_front_m6_points_in_pattern()

    assert np.allclose(torso_T_pattern[:3, :3], np.eye(3))
    assert np.allclose(
        torso_T_pattern[:3, 3], [0.06557272185, 0.0, 0.1679961682]
    )
    for point in NOMINAL_FRONT_M6_SURFACE_POINTS:
        reconstructed = (
            torso_T_pattern[:3, :3] @ np.asarray(local_points[point.name])
            + torso_T_pattern[:3, 3]
        )
        assert np.allclose(reconstructed, point.surface_xyz_m)


def test_nominal_front_pattern_preserves_published_dimensions() -> None:
    points = {
        point.name: np.asarray(point.surface_xyz_m)
        for point in NOMINAL_FRONT_M6_SURFACE_POINTS
    }

    assert np.isclose(points["upper_left"][1] - points["upper_right"][1], 0.0624)
    assert np.isclose(points["lower_left"][1] - points["lower_right"][1], 0.0645)
    assert np.isclose(points["upper_left"][2] - points["lower_left"][2], 0.1942)
    assert np.allclose(
        nominal_torso_T_front_m6_pattern()[:3, 3],
        np.mean(np.stack(tuple(points.values())), axis=0),
    )


def test_fixed_board_transform_recovers_camera_pose() -> None:
    torso_T_camera = nominal_torso_T_color_optical(URDFModel(URDF))
    torso_T_board = torso_T_charuco_outer_corner(
        board_center_x_m=0.250,
        board_center_y_m=0.0,
        board_center_z_m=0.180,
        board_width_m=BOARD_WIDTH_M,
        board_height_m=BOARD_HEIGHT_M,
        board_tilt_y_deg=18.0,
    )
    camera_T_board = invert_transform(torso_T_camera) @ torso_T_board
    recovered = torso_T_camera_from_fixed_board(
        torso_T_board=torso_T_board, camera_T_board=camera_T_board
    )
    assert np.allclose(recovered, torso_T_camera)
    assert np.allclose(torso_T_board[:3, 3], [0.2917173, 0.090, 0.3083925])


def test_proposed_board_is_visible_but_flush_board_is_not() -> None:
    torso_T_camera = nominal_torso_T_color_optical(URDFModel(URDF))
    proposed = torso_T_charuco_outer_corner(
        board_center_x_m=0.285,
        board_center_y_m=0.0,
        board_center_z_m=0.155,
        board_width_m=BOARD_WIDTH_M,
        board_height_m=BOARD_HEIGHT_M,
        board_tilt_y_deg=18.0,
    )
    proposed_pixels, _ = project_planar_board_outer_corners(
        torso_T_camera=torso_T_camera,
        torso_T_board=proposed,
        camera_matrix=CAMERA_MATRIX,
        board_width_m=BOARD_WIDTH_M,
        board_height_m=BOARD_HEIGHT_M,
    )
    proposed_margins = image_border_margins_px(
        proposed_pixels, image_width=1280, image_height=720
    )
    assert min(proposed_margins.values()) > 50.0
    assert board_view_incidence_deg(
        torso_T_camera=torso_T_camera,
        torso_T_board=proposed,
        board_width_m=BOARD_WIDTH_M,
        board_height_m=BOARD_HEIGHT_M,
    ) < 35.0

    flush = torso_T_charuco_outer_corner(
        board_center_x_m=0.080,
        board_center_y_m=0.0,
        board_center_z_m=0.150,
        board_width_m=BOARD_WIDTH_M,
        board_height_m=BOARD_HEIGHT_M,
    )
    flush_pixels, _ = project_planar_board_outer_corners(
        torso_T_camera=torso_T_camera,
        torso_T_board=flush,
        camera_matrix=CAMERA_MATRIX,
        board_width_m=BOARD_WIDTH_M,
        board_height_m=BOARD_HEIGHT_M,
    )
    flush_margins = image_border_margins_px(
        flush_pixels, image_width=1280, image_height=720
    )
    assert min(flush_margins.values()) < 0.0
