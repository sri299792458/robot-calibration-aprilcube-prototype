#!/usr/bin/env python3
"""Render the existing 40 mm AprilCube on the left rubber-hand palm.

This is a nominal visual placement, not a physical measurement.  Only the hand,
cube, and conservative collision AABB are drawn.  The mounting tape is omitted
from the picture but covered by the AABB allowance.
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
HAND_STL = ROOT / "unitree_ros/robots/g1_description/meshes/left_rubber_hand.STL"
CUBE_3MF = ROOT / "aprilcube/models/dex3_safe_cube/cube.3mf"
OUTPUT = ROOT / "renders/left_palm_aprilcube_envelope.png"
GEOMETRY_OUTPUT = ROOT / "renders/left_palm_aprilcube_envelope.json"

CUBE_CENTER_MM = np.array([40.0, -33.0, 0.0])
CUBE_SIZE_MM = np.array([40.0, 40.0, 40.0])
ENVELOPE_CENTER_MM = np.array([40.0, -34.0, 0.0])
ENVELOPE_SIZE_MM = np.array([60.0, 58.0, 60.0])

# Cube-local +Z (tag 4) faces into the palm (+hand Y).  The outward face is
# therefore -Z (tag 5), while +X (tag 0) is visible from the chosen view.  This
# matches the two faces visible in the user's mounting photograph.
HAND_R_CUBE = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ]
)

COLORS = {
    "background": "#0b0e13",
    "hand": "#343a43",
    "hand_edge": "#555e6a",
    "cube_white": "#f1eee6",
    "cube_black": "#111318",
    "envelope": "#30d6dc",
    "text": "#f3f6fa",
    "muted": "#aeb8c5",
}


def shaded_colors(
    triangles: np.ndarray,
    color: str,
    *,
    ambient: float,
) -> np.ndarray:
    """Create simple directional lighting for a triangle collection."""
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(cross, axis=1)
    normals = np.divide(
        cross,
        lengths[:, None],
        out=np.zeros_like(cross),
        where=lengths[:, None] > 1e-12,
    )
    light = np.array([0.25, -0.72, 0.65])
    light /= np.linalg.norm(light)
    intensity = ambient + (1.0 - ambient) * np.clip(normals @ light, 0.0, 1.0)
    rgb = np.asarray(mcolors.to_rgb(color))
    rgba = np.empty((len(triangles), 4))
    rgba[:, :3] = np.clip(rgb[None, :] * intensity[:, None], 0.0, 1.0)
    rgba[:, 3] = 1.0
    return rgba


def draw_triangles(
    ax,
    triangles: np.ndarray,
    facecolors: np.ndarray,
    *,
    edgecolor: str = "none",
    linewidth: float = 0.0,
    zorder: int = 1,
) -> None:
    ax.add_collection3d(
        Poly3DCollection(
            triangles,
            facecolors=facecolors,
            edgecolors=edgecolor,
            linewidths=linewidth,
            antialiased=True,
            zorder=zorder,
        )
    )


def load_hand_triangles() -> np.ndarray:
    hand = trimesh.load_mesh(HAND_STL, process=False)
    hand.apply_scale(1000.0)  # Unitree mesh coordinates are metres.
    return hand.vertices[hand.faces]


def load_cube_triangles() -> tuple[np.ndarray, np.ndarray]:
    """Read the released 3MF directly and preserve its triangle paint."""
    namespace = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
    with ZipFile(CUBE_3MF) as archive:
        root = ElementTree.fromstring(archive.read("3D/Objects/object_1.model"))

    vertices = np.array(
        [
            [float(vertex.attrib[axis]) for axis in ("x", "y", "z")]
            for vertex in root.findall(".//m:vertex", namespace)
        ]
    )
    triangle_nodes = root.findall(".//m:triangle", namespace)
    faces = np.array(
        [
            [int(triangle.attrib[index]) for index in ("v1", "v2", "v3")]
            for triangle in triangle_nodes
        ],
        dtype=int,
    )
    painted_white = np.array(
        [triangle.attrib.get("paint_color") == "8" for triangle in triangle_nodes]
    )

    vertices = vertices @ HAND_R_CUBE.T + CUBE_CENTER_MM
    return vertices[faces], painted_white


def cube_facecolors(triangles: np.ndarray, painted_white: np.ndarray) -> np.ndarray:
    white = shaded_colors(triangles, COLORS["cube_white"], ambient=0.72)
    black = shaded_colors(triangles, COLORS["cube_black"], ambient=0.62)
    return np.where(painted_white[:, None], white, black)


def aabb_corners(center: np.ndarray, size: np.ndarray) -> np.ndarray:
    half = size / 2.0
    return np.array(
        [
            center + np.array([sx * half[0], sy * half[1], sz * half[2]])
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
    )


def draw_envelope(ax) -> None:
    corners = aabb_corners(ENVELOPE_CENTER_MM, ENVELOPE_SIZE_MM)
    for i, corner_a in enumerate(corners):
        for corner_b in corners[i + 1 :]:
            if np.count_nonzero(np.abs(corner_a - corner_b) > 1e-9) != 1:
                continue
            ax.plot(
                [corner_a[0], corner_b[0]],
                [corner_a[1], corner_b[1]],
                [corner_a[2], corner_b[2]],
                color=COLORS["envelope"],
                linewidth=1.8,
                linestyle=(0, (3, 2)),
                alpha=0.95,
                zorder=20,
            )


def write_geometry_record() -> None:
    envelope_min = ENVELOPE_CENTER_MM - ENVELOPE_SIZE_MM / 2.0
    envelope_max = ENVELOPE_CENTER_MM + ENVELOPE_SIZE_MM / 2.0
    record = {
        "frame": "left_rubber_hand",
        "units": "mm",
        "status": "nominal_visual_placement_not_physical_measurement",
        "cube": {
            "source": str(CUBE_3MF.relative_to(ROOT)),
            "center": CUBE_CENTER_MM.tolist(),
            "size": CUBE_SIZE_MM.tolist(),
            "hand_R_cube": HAND_R_CUBE.tolist(),
            "hidden_face": "+Z (tag 4)",
            "outward_face": "-Z (tag 5)",
        },
        "conservative_aabb": {
            "center": ENVELOPE_CENTER_MM.tolist(),
            "size": ENVELOPE_SIZE_MM.tolist(),
            "min": envelope_min.tolist(),
            "max": envelope_max.tolist(),
            "collision_allowance": (
                "Padded beyond the 40 mm cube to include the mounting tape footprint "
                "numerically; tape is intentionally not rendered."
            ),
        },
    }
    GEOMETRY_OUTPUT.write_text(json.dumps(record, indent=2) + "\n")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    hand_triangles = load_hand_triangles()
    cube_triangles, painted_white = load_cube_triangles()

    fig = plt.figure(figsize=(13.5, 8.5), facecolor=COLORS["background"])
    ax = fig.add_axes(
        (0.015, 0.095, 0.75, 0.80), projection="3d", computed_zorder=False
    )
    ax.set_facecolor(COLORS["background"])
    ax.set_proj_type("ortho")

    draw_triangles(
        ax,
        hand_triangles,
        shaded_colors(hand_triangles, COLORS["hand"], ambient=0.34),
        edgecolor=COLORS["hand_edge"],
        linewidth=0.025,
        zorder=1,
    )
    draw_triangles(
        ax,
        cube_triangles,
        cube_facecolors(cube_triangles, painted_white),
        zorder=8,
    )
    draw_envelope(ax)

    ax.set_xlim(-4, 140)
    ax.set_ylim(-68, 28)
    ax.set_zlim(-52, 72)
    ax.set_box_aspect((144, 96, 124))
    ax.view_init(elev=24, azim=-55)
    ax.set_axis_off()

    fig.suptitle(
        "40 mm AprilCube on the left rubber-hand palm",
        x=0.5,
        y=0.96,
        color=COLORS["text"],
        fontsize=19,
        weight="bold",
    )
    fig.text(
        0.5,
        0.916,
        "Nominal visual placement in left_rubber_hand — no physical measurement required",
        ha="center",
        color=COLORS["muted"],
        fontsize=10.5,
    )

    fig.text(
        0.765,
        0.72,
        "NOMINAL GEOMETRY",
        color=COLORS["envelope"],
        fontsize=10,
        weight="bold",
    )
    fig.text(
        0.765,
        0.665,
        "Cube\n40 × 40 × 40 mm",
        color=COLORS["text"],
        fontsize=12,
        linespacing=1.35,
    )
    fig.text(
        0.765,
        0.555,
        "Cube center\n[40, −33, 0] mm",
        color=COLORS["text"],
        fontsize=12,
        linespacing=1.35,
    )
    fig.text(
        0.765,
        0.445,
        "Conservative AABB\n60 × 58 × 60 mm\ncenter [40, −34, 0] mm",
        color=COLORS["envelope"],
        fontsize=12,
        weight="bold",
        linespacing=1.35,
    )
    fig.text(
        0.765,
        0.30,
        "Dashed box includes numerical\nallowance for the mounting tape.\nTape is not rendered.",
        color=COLORS["muted"],
        fontsize=9.5,
        linespacing=1.45,
    )

    legend = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=COLORS["hand"],
            markersize=10,
            label="official left rubber-hand STL",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=COLORS["cube_white"],
            markersize=10,
            label="released rounded AprilCube 3MF",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["envelope"],
            linestyle=(0, (3, 2)),
            linewidth=2,
            label="conservative collision AABB",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        frameon=False,
        labelcolor=COLORS["text"],
        bbox_to_anchor=(0.5, 0.035),
        fontsize=9.5,
    )
    fig.text(
        0.5,
        0.012,
        "Visual estimate only; no physical measurement is implied.",
        ha="center",
        color=COLORS["muted"],
        fontsize=8.5,
    )
    fig.savefig(OUTPUT, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    write_geometry_record()
    print(OUTPUT)
    print(GEOMETRY_OUTPUT)


if __name__ == "__main__":
    main()
