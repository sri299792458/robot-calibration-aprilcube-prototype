#!/usr/bin/env python3
"""Check a provisional torso-mounted ChArUco pose against the G1 color image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from g1_aprilcube_calibration.camera_initialization import nominal_torso_T_color_optical
from g1_aprilcube_calibration.torso_board_mount import (
    NOMINAL_FRONT_M6_SURFACE_POINTS,
    board_view_incidence_deg,
    image_border_margins_px,
    nominal_front_m6_points_in_pattern,
    nominal_torso_T_front_m6_pattern,
    project_planar_board_outer_corners,
    torso_T_charuco_outer_corner,
)
from g1_aprilcube_calibration.urdf_model import URDFModel

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hardware", type=Path, default=ROOT / "config" / "hardware.yaml"
    )
    parser.add_argument("--board-width-m", type=float, default=0.180)
    parser.add_argument("--board-height-m", type=float, default=0.270)
    parser.add_argument("--board-center-x-m", type=float, default=0.277)
    parser.add_argument("--board-center-y-m", type=float, default=0.0)
    parser.add_argument("--board-center-z-m", type=float, default=0.161)
    parser.add_argument("--board-tilt-y-deg", type=float, default=18.0)
    parser.add_argument("--required-margin-px", type=float, default=50.0)
    return parser.parse_args()


def matrix_list(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def main() -> int:
    args = parse_args()
    hardware_path = args.hardware.resolve()
    hardware = yaml.safe_load(hardware_path.read_text())
    urdf_path = ROOT / hardware["robot"]["urdf"]
    profile = hardware["camera"]["color_profile"]
    projection = np.asarray(profile["p"], dtype=np.float64).reshape(3, 4)
    camera_matrix = projection[:, :3]

    torso_T_camera = nominal_torso_T_color_optical(URDFModel(urdf_path))
    torso_T_board = torso_T_charuco_outer_corner(
        board_center_x_m=args.board_center_x_m,
        board_center_y_m=args.board_center_y_m,
        board_center_z_m=args.board_center_z_m,
        board_width_m=args.board_width_m,
        board_height_m=args.board_height_m,
        board_tilt_y_deg=args.board_tilt_y_deg,
    )
    pixels, depths = project_planar_board_outer_corners(
        torso_T_camera=torso_T_camera,
        torso_T_board=torso_T_board,
        camera_matrix=camera_matrix,
        board_width_m=args.board_width_m,
        board_height_m=args.board_height_m,
    )
    margins = image_border_margins_px(
        pixels,
        image_width=int(profile["width"]),
        image_height=int(profile["height"]),
    )
    result = {
        "status": "nominal_fov_check_only_not_print_release",
        "fov_assumption": (
            "Projection uses the URDF's nominal torso-to-camera pose. The fixed-board "
            "calibration equation itself does not require or preserve that assumption."
        ),
        "hardware": str(hardware_path),
        "urdf": str(urdf_path),
        "board": {
            "width_m": args.board_width_m,
            "height_m": args.board_height_m,
            "center_x_m": args.board_center_x_m,
            "center_y_m": args.board_center_y_m,
            "center_z_m": args.board_center_z_m,
            "tilt_y_deg": args.board_tilt_y_deg,
            "torso_T_board_outer_corner": matrix_list(torso_T_board),
        },
        "nominal_torso_T_color_optical": matrix_list(torso_T_camera),
        "board_view_incidence_deg": board_view_incidence_deg(
            torso_T_camera=torso_T_camera,
            torso_T_board=torso_T_board,
            board_width_m=args.board_width_m,
            board_height_m=args.board_height_m,
        ),
        "projected_outer_corners_px": pixels.tolist(),
        "optical_corner_depth_m": depths.tolist(),
        "image_border_margins_px": margins,
        "required_margin_px": args.required_margin_px,
        "passes_required_margin": min(margins.values()) >= args.required_margin_px,
        "nominal_front_m6_surface_points": [
            {"name": point.name, "surface_xyz_m": list(point.surface_xyz_m)}
            for point in NOMINAL_FRONT_M6_SURFACE_POINTS
        ],
        "nominal_front_m6_pattern": {
            "status": "nominal_cad_reconstruction_not_manufacturing_metrology",
            "registration_evidence": (
                "Unitree's vector drawing was scaled by its published 194.2 mm "
                "front-row spacing and registered vertically to the shoulder-roll "
                "axes computed from the URDF. Its independently predicted shoulder "
                "spacing agrees with the URDF by 0.06%."
            ),
            "torso_T_pattern": matrix_list(nominal_torso_T_front_m6_pattern()),
            "axis_direction_in_torso": [1.0, 0.0, 0.0],
            "surface_points_in_pattern_m": {
                name: list(point)
                for name, point in nominal_front_m6_points_in_pattern().items()
            },
            "fixture_datum_strategy": (
                "Model the four local root seats and board around the identity-"
                "positioned torso STL; the small root shoulders locate at the "
                "recovered exterior-surface intersections while the M6 screws "
                "supply clamp force."
            ),
        },
        "warning": (
            "Do not release a fixture from this FOV result alone. The physical head "
            "pitch must be locked and verified in live rectified color. The recovered "
            "pattern and shell contacts are nominal CAD values; insert-face depth, "
            "thread depth, and physical manufacturing tolerances are not published."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passes_required_margin"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
