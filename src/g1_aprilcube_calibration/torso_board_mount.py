"""Geometry helpers for a temporary torso-mounted planar calibration target.

This module intentionally separates the exact transform algebra from nominal
CAD reconstruction.  Unitree's vector mounting drawing can be registered to
the URDF through the visible shoulder-roll axes, recovering the front M6
pattern in ``torso_link``.  The data below remains nominal CAD geometry rather
than a tolerance-controlled inspection result for a particular robot.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin

import numpy as np

from g1_aprilcube_calibration.transforms import (
    invert_transform,
    transform_points,
    validate_transform,
)


@dataclass(frozen=True, slots=True)
class NominalMountPoint:
    """A recovered M6 axis intersected with the public exterior torso STL."""

    name: str
    surface_xyz_m: tuple[float, float, float]
    outward_surface_normal: tuple[float, float, float]
    source: str = "unitree_vector_drawing_urdf_registration_plus_public_stl"


# Unitree's drawing gives 62.4 mm upper spacing, 64.5 mm lower spacing, and
# 194.2 mm vertical spacing.  Its vector shoulder-circle centers register to
# the shoulder-roll centers computed from the URDF with a 0.06% scale residual.
# That shared datum puts the row midpoint at torso z=0.167996 m.  Rays along
# torso -x then intersect torso_link_rev_1_0.STL at the exterior-shell points
# below.  These x coordinates locate shell contact, not the recessed insert
# face or the start/end of its thread.
NOMINAL_FRONT_M6_SURFACE_POINTS = (
    NominalMountPoint(
        "upper_left",
        (0.0617071476, 0.0312, 0.2650961682),
        (0.8934860523, 0.0848042913, 0.4410112318),
    ),
    NominalMountPoint(
        "upper_right",
        (0.0615783281, -0.0312, 0.2650961682),
        (0.8961632616, -0.0589538262, 0.4397906944),
    ),
    NominalMountPoint(
        "lower_left",
        (0.0695162713, 0.03225, 0.0708961682),
        (0.9916213467, 0.0955151656, -0.0869710178),
    ),
    NominalMountPoint(
        "lower_right",
        (0.0694891404, -0.03225, 0.0708961682),
        (0.9929173643, -0.0956241074, -0.0705062962),
    ),
)

# Pattern frame used for nominal CAD layout. Its axes are parallel to
# torso_link and its origin is the arithmetic mean of the four recovered shell
# intersections. The x coordinate is a convenient layout plane through four
# curved shell contacts; it is not the physical threaded-insert face.
NOMINAL_FRONT_M6_PATTERN_ORIGIN_M = (0.06557272185, 0.0, 0.1679961682)


def nominal_torso_T_front_m6_pattern() -> np.ndarray:
    """Return the nominal CAD-layout frame for the front M6 pattern."""
    result = np.eye(4)
    result[:3, 3] = NOMINAL_FRONT_M6_PATTERN_ORIGIN_M
    result.setflags(write=False)
    return result


def nominal_front_m6_points_in_pattern() -> dict[str, tuple[float, float, float]]:
    """Return nominal shell intersections expressed in the pattern frame."""
    origin = np.asarray(NOMINAL_FRONT_M6_PATTERN_ORIGIN_M, dtype=np.float64)
    return {
        point.name: tuple(np.asarray(point.surface_xyz_m) - origin)
        for point in NOMINAL_FRONT_M6_SURFACE_POINTS
    }


def torso_T_charuco_outer_corner(
    *,
    board_center_x_m: float,
    board_center_y_m: float,
    board_center_z_m: float,
    board_width_m: float,
    board_height_m: float,
    board_tilt_y_deg: float = 0.0,
) -> np.ndarray:
    """Return ``torso_link_T_board`` for an upright front-facing ChArUco board.

    OpenCV's board frame starts at the outer top-left corner as seen by the
    camera, with +x to image-right, +y image-down, and +z through the board away
    from the camera.  G1 ``torso_link`` uses +x forward, +y robot-left, +z up.
    The target is in front of the chest and its printed face looks back toward
    the head camera.  A positive torso-y tilt moves the board's upper edge
    forward (+torso-x) and points the printed face upward toward the camera.
    """
    values = np.asarray(
        [
            board_center_x_m,
            board_center_y_m,
            board_center_z_m,
            board_width_m,
            board_height_m,
            board_tilt_y_deg,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("board dimensions and pose must be finite")
    if board_width_m <= 0 or board_height_m <= 0:
        raise ValueError("board dimensions must be positive")

    upright_rotation = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    tilt_rad = radians(board_tilt_y_deg)
    torso_y_rotation = np.asarray(
        [
            [cos(tilt_rad), 0.0, sin(tilt_rad)],
            [0.0, 1.0, 0.0],
            [-sin(tilt_rad), 0.0, cos(tilt_rad)],
        ]
    )
    result = np.eye(4)
    result[:3, :3] = torso_y_rotation @ upright_rotation
    board_center = np.asarray(
        [board_center_x_m, board_center_y_m, board_center_z_m], dtype=np.float64
    )
    local_center = np.asarray(
        [board_width_m / 2.0, board_height_m / 2.0, 0.0], dtype=np.float64
    )
    result[:3, 3] = board_center - result[:3, :3] @ local_center
    return validate_transform(result)


def board_view_incidence_deg(
    *,
    torso_T_camera: np.ndarray,
    torso_T_board: np.ndarray,
    board_width_m: float,
    board_height_m: float,
) -> float:
    """Return board-normal/view-ray angle; zero degrees is straight-on."""
    if board_width_m <= 0 or board_height_m <= 0:
        raise ValueError("board dimensions must be positive")
    torso_T_camera = validate_transform(torso_T_camera)
    torso_T_board = validate_transform(torso_T_board)
    board_center = transform_points(
        torso_T_board,
        np.asarray([[board_width_m / 2.0, board_height_m / 2.0, 0.0]]),
    )[0]
    camera_to_board = board_center - torso_T_camera[:3, 3]
    distance = np.linalg.norm(camera_to_board)
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("camera and board centers must be distinct")
    cosine = float(torso_T_board[:3, 2] @ (camera_to_board / distance))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def planar_board_outer_corners(
    *, board_width_m: float, board_height_m: float
) -> np.ndarray:
    """Return the four outer board corners in OpenCV board coordinates."""
    if board_width_m <= 0 or board_height_m <= 0:
        raise ValueError("board dimensions must be positive")
    result = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [board_width_m, 0.0, 0.0],
            [board_width_m, board_height_m, 0.0],
            [0.0, board_height_m, 0.0],
        ],
        dtype=np.float64,
    )
    result.setflags(write=False)
    return result


def project_planar_board_outer_corners(
    *,
    torso_T_camera: np.ndarray,
    torso_T_board: np.ndarray,
    camera_matrix: np.ndarray,
    board_width_m: float,
    board_height_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project board corners and return ``(pixels, optical_depth_m)``."""
    torso_T_camera = validate_transform(torso_T_camera)
    torso_T_board = validate_transform(torso_T_board)
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("camera_matrix must have finite shape (3, 3)")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("camera focal lengths must be positive")

    object_points = planar_board_outer_corners(
        board_width_m=board_width_m, board_height_m=board_height_m
    )
    torso_points = transform_points(torso_T_board, object_points)
    camera_points = transform_points(invert_transform(torso_T_camera), torso_points)
    depths = camera_points[:, 2].copy()
    if np.any(depths <= 0):
        raise ValueError("one or more board corners lie behind the camera")
    homogeneous = (matrix @ camera_points.T).T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    pixels.setflags(write=False)
    depths.setflags(write=False)
    return pixels, depths


def image_border_margins_px(
    pixels: np.ndarray, *, image_width: int, image_height: int
) -> dict[str, float]:
    """Return positive distances from the target bounds to all image borders."""
    points = np.asarray(pixels, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("pixels must have finite shape (N, 2)")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    return {
        "left": float(np.min(points[:, 0])),
        "right": float((image_width - 1) - np.max(points[:, 0])),
        "top": float(np.min(points[:, 1])),
        "bottom": float((image_height - 1) - np.max(points[:, 1])),
    }


def torso_T_camera_from_fixed_board(
    *, torso_T_board: np.ndarray, camera_T_board: np.ndarray
) -> np.ndarray:
    """Recover the camera extrinsic from a known torso target and a PnP pose."""
    return validate_transform(
        validate_transform(torso_T_board) @ invert_transform(camera_T_board)
    )
