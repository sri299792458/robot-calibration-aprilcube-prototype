#!/usr/bin/env python3
"""Render both middle-close Dex3 hands, dorsal plates, and collision boxes."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import generate_dex3_dorsal_aruco_mount as mount_generator

from g1_aprilcube_calibration.collision import CollisionConfig
from g1_aprilcube_calibration.dex3_dorsal_mount import (
    DEFAULT_DEX3_DORSAL_MOUNT_SPEC,
)
from g1_aprilcube_calibration.transports.unitree_dex3 import (
    DEX3_MOTOR_JOINT_SUFFIXES,
    NVIDIA_MIDDLE_CLOSE_LEFT_Q_RAD,
    NVIDIA_MIDDLE_CLOSE_RIGHT_Q_RAD,
)
from g1_aprilcube_calibration.urdf_model import URDFModel

OUTPUT = ROOT / "renders/dual_dex3_segmented_collision_envelopes.png"
BACKGROUND = "#0c1016"
BOX_COLORS = {
    "marker_palm": "#2dd4d7",
    "thumb": "#eb5bcf",
    "middle": "#ff9d3b",
    "index": "#ffe34d",
}


def _hand_meshes(side: str) -> list[trimesh.Trimesh]:
    short = "l" if side == "left" else "r"
    model = URDFModel(
        ROOT
        / "unitree_ros/robots/dexterous_hand_description/dex3_1"
        / f"dex3_1_{short}.urdf"
    )
    base = f"{side}_hand_palm_link"
    target = (
        NVIDIA_MIDDLE_CLOSE_LEFT_Q_RAD
        if side == "left"
        else NVIDIA_MIDDLE_CLOSE_RIGHT_Q_RAD
    )
    positions = {
        f"{side}_hand_{suffix}_joint": value
        for suffix, value in zip(DEX3_MOTOR_JOINT_SUFFIXES[side], target, strict=True)
    }
    meshes: list[trimesh.Trimesh] = []
    for link in model.links:
        if not link.startswith(f"{side}_hand_"):
            continue
        try:
            base_T_link = model.transform(base, link, positions)
        except ValueError:
            continue
        for geometry in model.link_geometries(link):
            mesh = geometry.mesh.copy()
            mesh.apply_transform(base_T_link @ geometry.local_transform)
            meshes.append(mesh)
    return meshes


def _mounted_plate(side: str, marker_id: int) -> dict[str, trimesh.Trimesh]:
    spec = replace(DEFAULT_DEX3_DORSAL_MOUNT_SPEC, marker_id=marker_id)
    palm_T_plate = mount_generator.palm_T_plate(spec, side)
    result = {}
    for name, mesh in mount_generator.build_parts(spec).items():
        mounted = mount_generator.transformed(mesh, palm_T_plate)
        mounted.apply_scale(0.001)
        result[name] = mounted
    return result


def _draw_mesh(ax, mesh: trimesh.Trimesh, color: str, alpha: float) -> None:
    triangles = mesh.vertices[mesh.faces[::2]]
    ax.add_collection3d(
        Poly3DCollection(
            triangles,
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
        )
    )


def _box_vertices(center: np.ndarray, size: np.ndarray) -> np.ndarray:
    half = size / 2.0
    return np.asarray(
        [
            center + np.asarray([x, y, z]) * half
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ]
    )


def _draw_box(ax, center: np.ndarray, size: np.ndarray, color: str) -> None:
    vertices = _box_vertices(center, size)
    for first, second in (
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ):
        points = vertices[[first, second]]
        ax.plot(*points.T, color=color, linewidth=2.2)


def _render_side(ax, side: str, marker_id: int, boxes: dict) -> None:
    hand_meshes = _hand_meshes(side)
    plate_meshes = _mounted_plate(side, marker_id)
    for mesh in hand_meshes:
        _draw_mesh(ax, mesh, "#56606d", 0.52)
    for name, mesh in plate_meshes.items():
        color = "#f3f0e9" if "white" in name else "#08090b"
        _draw_mesh(ax, mesh, color, 1.0)

    prefixes = ("marker_palm", "thumb", "middle", "index")
    for suffix in prefixes:
        box = boxes[f"{side}_dex3_{suffix}"]
        center = np.asarray(box.xyz_m)
        size = np.asarray(box.size_m)
        color = BOX_COLORS[suffix]
        _draw_box(ax, center, size, color)
        ax.text(
            *(center + np.asarray([0.0, 0.0, size[2] / 2.0 + 0.006])),
            suffix.replace("marker_", "") + " box",
            color=color,
            fontsize=9,
            ha="center",
        )

    points = np.vstack(
        [mesh.vertices for mesh in [*hand_meshes, *plate_meshes.values()]]
        + [
            _box_vertices(
                np.asarray(boxes[f"{side}_dex3_{suffix}"].xyz_m),
                np.asarray(boxes[f"{side}_dex3_{suffix}"].size_m),
            )
            for suffix in prefixes
        ]
    )
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    center = (lower + upper) / 2.0
    radius = max(upper - lower) * 0.58
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_proj_type("ortho")
    ax.view_init(elev=20, azim=-62, roll=88)
    ax.set_axis_off()
    ax.set_facecolor(BACKGROUND)
    ax.set_title(
        f"{side.title()} Dex3 · DICT_6X6_50 ID {marker_id}",
        color="#f4f7fa",
        fontsize=15,
        weight="bold",
        pad=14,
    )


def main() -> None:
    collision = CollisionConfig.from_yaml(
        ROOT / "config/collision_pairs_dex3_aruco.yaml"
    )
    boxes = {box.name: box for box in collision.attached_boxes}
    fig = plt.figure(figsize=(16, 9), facecolor=BACKGROUND)
    left = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
    right = fig.add_subplot(1, 2, 2, projection="3d", computed_zorder=False)
    _render_side(left, "left", 5, boxes)
    _render_side(right, "right", 4, boxes)
    fig.suptitle(
        "Dual Dex3 dorsal-marker collision envelopes · NVIDIA middle-close posture",
        color="#f4f7fa",
        fontsize=20,
        weight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.035,
        "Solid geometry: official Dex3 meshes + mounted plate  ·  "
        "Wire boxes: 2.5 mm padded segmented FCL attachments",
        color="#aeb8c5",
        fontsize=11,
        ha="center",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
