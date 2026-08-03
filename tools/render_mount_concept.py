#!/usr/bin/env python3
"""Render the provisional printed AprilCube clamshell concept.

This is a communication model, not printable CAD. The official Unitree dummy-hand
mesh is shown at its actual URDF scale; the cradle clearance and contact surface
remain provisional until the physical hand is measured.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]
HAND_STL = (
    ROOT
    / "unitree_ros/robots/g1_description/meshes/right_rubber_hand.STL"
)
OUTPUT = ROOT / "renders/dummy_hand_aprilcube_mount_concept.png"

COLORS = {
    "background": "#11151c",
    "grid": "#3b4350",
    "text": "#eef2f7",
    "muted": "#aeb7c4",
    "hand": "#b7bec8",
    "primary": "#f39c3d",
    "cap": "#4e9de0",
    "target": "#f3f0e8",
    "tag": "#17191d",
    "fastener": "#d8dde4",
    "guide": "#727d8d",
}


def box(extents: tuple[float, float, float], center: tuple[float, float, float]):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def cylinder_y(
    radius: float,
    length: float,
    center: tuple[float, float, float],
    sections: int = 24,
):
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    mesh.apply_transform(
        trimesh.geometry.align_vectors(np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]))
    )
    mesh.apply_translation(center)
    return mesh


def elliptical_cylinder_x(
    length: float,
    radius_y: float,
    radius_z: float,
    center: tuple[float, float, float],
    sections: int = 96,
):
    mesh = trimesh.creation.cylinder(radius=1.0, height=length, sections=sections)
    mesh.apply_transform(
        trimesh.geometry.align_vectors(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    )
    mesh.apply_scale((1.0, radius_y, radius_z))
    mesh.apply_translation(center)
    return mesh


def clamshell_halves():
    """Create a provisional oval sleeve split into camera-side and cap halves."""
    center = (32.0, -2.7, 8.5)
    outer = elliptical_cylinder_x(54.0, 23.0, 52.0, center)
    # Extend the inner solid past both ends so the boolean leaves an open sleeve.
    inner = elliptical_cylinder_x(60.0, 18.8, 47.8, center)
    sleeve = trimesh.boolean.difference([outer, inner], engine="manifold")

    # Split on the hand's thin y axis. The negative-y half carries the target.
    negative_half = box((70, 100, 140), (32, -52.7, 8.5))
    positive_half = box((70, 100, 140), (32, 47.3, 8.5))
    primary = trimesh.boolean.intersection([sleeve, negative_half], engine="manifold")
    cap = trimesh.boolean.intersection([sleeve, positive_half], engine="manifold")
    return primary, cap


def translated(mesh: trimesh.Trimesh, offset: tuple[float, float, float]):
    copy = mesh.copy()
    copy.apply_translation(offset)
    return copy


def target_cube(center=(32.0, -77.0, 8.0), size=62.5, tag_size=50.0):
    cx, cy, cz = center
    half = size / 2.0
    skin = 0.7
    cube = box((size, size, size), center)
    tags = [
        box((tag_size, skin, tag_size), (cx, cy - half - skin / 2, cz)),
        box((skin, tag_size, tag_size), (cx - half - skin / 2, cy, cz)),
        box((skin, tag_size, tag_size), (cx + half + skin / 2, cy, cz)),
        box((tag_size, tag_size, skin), (cx, cy, cz - half - skin / 2)),
        box((tag_size, tag_size, skin), (cx, cy, cz + half + skin / 2)),
    ]
    return cube, tags


def build_components():
    hand = trimesh.load_mesh(HAND_STL, process=False)
    hand.apply_scale(1000.0)  # official mesh is in meters; render in millimeters

    # Provisional oval saddle around x=5..59 mm of the rigid palm.
    primary_shell, cap_shell = clamshell_halves()
    primary = [
        primary_shell,
        box((22, 24, 24), (32, -35, 8)),  # integrated short target stalk
        box((34, 5, 34), (32, -46, 8)),   # keyed target seating flange
    ]
    cap = [cap_shell]

    # Four side-flange pairs: two near the wrist, two toward the fingers.
    screw_positions = [(18, -47), (47, -47), (18, 64), (47, 64)]
    for x, z in screw_positions:
        primary.append(box((13, 9, 10), (x, -7.2, z)))
        cap.append(box((13, 9, 10), (x, 1.8, z)))

    m4_screws = [cylinder_y(2.0, 23.0, (x, -2.7, z)) for x, z in screw_positions]
    m3_screws = [cylinder_y(1.5, 18.0, (x, -46.0, 8.0)) for x in (25.0, 39.0)]
    cube, tags = target_cube()
    return hand, primary, cap, m4_screws, m3_screws, cube, tags


def draw_mesh(ax, mesh, color, alpha=1.0, zorder=1):
    triangles = mesh.vertices[mesh.faces]
    collection = Poly3DCollection(
        triangles,
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_collection3d(collection)


def style_axis(ax, title):
    ax.set_xlim(-18, 152)
    ax.set_ylim(-170, 105)
    ax.set_zlim(-70, 95)
    ax.set_box_aspect((170, 275, 165))
    ax.view_init(elev=23, azim=-53)
    ax.set_title(title, color=COLORS["text"], pad=12, fontsize=14, weight="bold")
    ax.set_xlabel("x  wrist → fingers (mm)", color=COLORS["muted"], labelpad=8)
    ax.set_ylabel("y  target offset (mm)", color=COLORS["muted"], labelpad=8)
    ax.set_zlabel("z (mm)", color=COLORS["muted"], labelpad=8)
    ax.tick_params(colors=COLORS["muted"], labelsize=8)
    ax.set_facecolor(COLORS["background"])
    ax.grid(True, alpha=0.28)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(COLORS["background"])
        axis.pane.set_edgecolor(COLORS["grid"])
        axis._axinfo["grid"]["color"] = COLORS["grid"]


def draw_assembly(ax, exploded=False):
    hand, primary, cap, m4, m3, cube, tags = build_components()
    draw_mesh(ax, hand, COLORS["hand"], alpha=0.82)

    if exploded:
        primary_offset = (0, -18, 0)
        cap_offset = (0, 46, 0)
        cube_offset = (0, -60, 0)
        m4_offset = (0, 72, 0)
        m3_offset = (0, -39, 0)
    else:
        primary_offset = cap_offset = cube_offset = m4_offset = m3_offset = (0, 0, 0)

    for part in primary:
        draw_mesh(ax, translated(part, primary_offset), COLORS["primary"])
    for part in cap:
        draw_mesh(ax, translated(part, cap_offset), COLORS["cap"])
    for screw in m4:
        draw_mesh(ax, translated(screw, m4_offset), COLORS["fastener"])
    for screw in m3:
        draw_mesh(ax, translated(screw, m3_offset), COLORS["fastener"])
    draw_mesh(ax, translated(cube, cube_offset), COLORS["target"])
    for tag in tags:
        draw_mesh(ax, translated(tag, cube_offset), COLORS["tag"])

    if exploded:
        # Assembly direction guides along the y axis.
        for x, z in ((18, -47), (47, -47), (18, 64), (47, 64)):
            ax.plot([x, x], [-32, 78], [z, z], color=COLORS["guide"], linestyle="--", linewidth=1)
        ax.plot([32, 32], [-152, -26], [8, 8], color=COLORS["guide"], linestyle="--", linewidth=1)
        ax.text(75, -142, 69, "AprilCube\n62.5 mm", color=COLORS["text"], fontsize=9)
        ax.text(72, -35, 56, "printed primary cradle\n+ integrated stalk", color=COLORS["primary"], fontsize=9)
        ax.text(70, 68, 46, "printed clamp cap", color=COLORS["cap"], fontsize=9)
        ax.text(70, 90, -48, "4 × M4×20", color=COLORS["fastener"], fontsize=9)
    else:
        ax.text(73, -107, 68, "multi-face target", color=COLORS["text"], fontsize=9)
        ax.text(70, -20, -49, "clamp wraps rigid palm", color=COLORS["text"], fontsize=9)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(15, 8.5), facecolor=COLORS["background"])
    assembled = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
    exploded = fig.add_subplot(1, 2, 2, projection="3d", computed_zorder=False)
    draw_assembly(assembled, exploded=False)
    draw_assembly(exploded, exploded=True)
    style_axis(assembled, "Assembled concept")
    style_axis(exploded, "Exploded concept")

    legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["hand"], markersize=10, label="official rigid dummy-hand mesh"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["primary"], markersize=10, label="printed primary cradle + stalk"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["cap"], markersize=10, label="printed clamp cap"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["target"], markersize=10, label="printed AprilCube"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["fastener"], markersize=8, label="M4/M3 purchased fasteners"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=5,
        frameon=False,
        labelcolor=COLORS["text"],
        bbox_to_anchor=(0.5, 0.015),
        fontsize=9,
    )
    fig.text(
        0.5,
        0.982,
        "Fallback printed G1 dummy-hand AprilCube fixture",
        ha="center",
        va="top",
        color=COLORS["text"],
        fontsize=18,
        weight="bold",
    )
    fig.text(
        0.5,
        0.948,
        "Only needed if direct thin transfer tape fails the mount-stability test.",
        ha="center",
        va="top",
        color=COLORS["muted"],
        fontsize=10,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.865, bottom=0.09, wspace=0.02)
    fig.savefig(OUTPUT, dpi=170, facecolor=fig.get_facecolor())
    print(OUTPUT)


if __name__ == "__main__":
    main()
