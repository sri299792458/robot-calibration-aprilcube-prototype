#!/usr/bin/env python3
"""Reconstruct the G1 front M6 pattern from Unitree's drawing, URDF, and STL."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from g1_aprilcube_calibration.urdf_model import URDFModel

ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = (
    ROOT / "unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"
)

# Circle centers transcribed from the vector front view on page 6 of Unitree's
# G1 User Manual V1.4. PDF coordinates use +x right and +y down. These values
# are retained so the registration arithmetic is reviewable without raster
# measurement.
DRAWING_SHOULDER_LEFT = np.asarray([115.0156, 575.88615])
DRAWING_SHOULDER_RIGHT = np.asarray([236.3963, 575.87245])
DRAWING_UPPER_LEFT = np.asarray([162.39645, 567.2168])
DRAWING_UPPER_RIGHT = np.asarray([189.3359, 567.1641])
DRAWING_LOWER_LEFT = np.asarray([161.86135, 650.99415])
DRAWING_LOWER_RIGHT = np.asarray([189.7656, 650.99415])

PUBLISHED_UPPER_SPACING_M = 0.0624
PUBLISHED_LOWER_SPACING_M = 0.0645
PUBLISHED_VERTICAL_SPACING_M = 0.1942


def outer_x_intersection(
    mesh_vertices: np.ndarray, point_yz: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return the outer x intersection and outward local triangle normal."""
    triangles = mesh_vertices[:, :, 1:3]
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    v0 = b - a
    v1 = c - a
    v2 = point_yz - a
    denominator = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    usable = np.abs(denominator) > 1e-14
    u = np.full(len(triangles), np.nan)
    v = np.full(len(triangles), np.nan)
    u[usable] = (
        v2[usable, 0] * v1[usable, 1]
        - v1[usable, 0] * v2[usable, 1]
    ) / denominator[usable]
    v[usable] = (
        v0[usable, 0] * v2[usable, 1]
        - v2[usable, 0] * v0[usable, 1]
    ) / denominator[usable]
    inside = usable & (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1.0 + 1e-10)
    if not np.any(inside):
        raise ValueError(f"axis at yz={point_yz.tolist()} does not cross torso mesh")
    triangle_x = mesh_vertices[:, :, 0]
    x = (
        triangle_x[:, 0]
        + u * (triangle_x[:, 1] - triangle_x[:, 0])
        + v * (triangle_x[:, 2] - triangle_x[:, 0])
    )
    candidates = np.flatnonzero(inside)
    triangle_index = int(candidates[np.argmax(x[candidates])])
    triangle = mesh_vertices[triangle_index]
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    normal /= np.linalg.norm(normal)
    if normal[0] < 0.0:
        normal = -normal
    return float(x[triangle_index]), normal


def main() -> int:
    model = URDFModel(URDF_PATH)
    left_shoulder = model.transform(
        "torso_link",
        "left_shoulder_roll_link",
        {"left_shoulder_pitch_joint": 0.0, "left_shoulder_roll_joint": 0.0},
    )[:3, 3]
    right_shoulder = model.transform(
        "torso_link",
        "right_shoulder_roll_link",
        {"right_shoulder_pitch_joint": 0.0, "right_shoulder_roll_joint": 0.0},
    )[:3, 3]

    drawing_shoulder_center = (
        DRAWING_SHOULDER_LEFT + DRAWING_SHOULDER_RIGHT
    ) / 2.0
    drawing_upper_center = (DRAWING_UPPER_LEFT + DRAWING_UPPER_RIGHT) / 2.0
    drawing_lower_center = (DRAWING_LOWER_LEFT + DRAWING_LOWER_RIGHT) / 2.0
    drawing_vertical_span = drawing_lower_center[1] - drawing_upper_center[1]
    vertical_scale = PUBLISHED_VERTICAL_SPACING_M / drawing_vertical_span
    shoulder_scale = np.linalg.norm(left_shoulder - right_shoulder) / np.linalg.norm(
        DRAWING_SHOULDER_LEFT - DRAWING_SHOULDER_RIGHT
    )
    shoulder_z = float((left_shoulder[2] + right_shoulder[2]) / 2.0)
    upper_z = shoulder_z + (
        drawing_shoulder_center[1] - drawing_upper_center[1]
    ) * vertical_scale
    lower_z = upper_z - PUBLISHED_VERTICAL_SPACING_M

    yz_points = {
        "upper_left": np.asarray([PUBLISHED_UPPER_SPACING_M / 2.0, upper_z]),
        "upper_right": np.asarray([-PUBLISHED_UPPER_SPACING_M / 2.0, upper_z]),
        "lower_left": np.asarray([PUBLISHED_LOWER_SPACING_M / 2.0, lower_z]),
        "lower_right": np.asarray([-PUBLISHED_LOWER_SPACING_M / 2.0, lower_z]),
    }
    torso_geometry = model.link_geometries("torso_link")[0]
    torso_mesh = torso_geometry.mesh.copy()
    torso_mesh.apply_transform(torso_geometry.local_transform)
    vertices = torso_mesh.vertices[torso_mesh.faces]
    intersections = {
        name: outer_x_intersection(vertices, yz) for name, yz in yz_points.items()
    }
    points = {
        name: [intersection[0], float(yz_points[name][0]), float(yz_points[name][1])]
        for name, intersection in intersections.items()
    }
    normals = {
        name: intersection[1].tolist()
        for name, intersection in intersections.items()
    }
    pattern_origin = np.mean(np.asarray(tuple(points.values())), axis=0)
    result = {
        "status": "nominal_cad_reconstruction_not_manufacturing_metrology",
        "urdf": str(URDF_PATH),
        "shoulder_roll_centers_in_torso_m": {
            "left": left_shoulder.tolist(),
            "right": right_shoulder.tolist(),
        },
        "scale_cross_check": {
            "published_vertical_scale_m_per_vector_unit": float(vertical_scale),
            "urdf_shoulder_scale_m_per_vector_unit": float(shoulder_scale),
            "relative_difference": float(shoulder_scale / vertical_scale - 1.0),
        },
        "row_z_in_torso_m": {"upper": upper_z, "lower": lower_z},
        "axis_direction_in_torso": [1.0, 0.0, 0.0],
        "outer_shell_intersections_in_torso_m": points,
        "outer_shell_normals_in_torso": normals,
        "nominal_pattern_origin_in_torso_m": pattern_origin.tolist(),
        "limitations": (
            "The reconstruction supplies nominal screw-axis lines and outer-shell "
            "contacts. It does not supply insert recess, thread depth, or physical "
            "manufacturing tolerances."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
