#!/usr/bin/env python3
"""Generate the direct two-hole Dex3 dorsal ArUco carrier.

The carrier is deliberately one structural print.  Two M3 countersunk screw
heads occupy a short finger-side tab outside the marker and its quiet zone,
preserving the optical target without a cover plate or adhesive.  A two-hole
coupon is exported alongside the full carrier because Unitree's public palm
STL omits the physical dorsal holes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g1_aprilcube_calibration.dex3_dorsal_mount import (
    ARUCO_DICTIONARY_NAME,
    DEFAULT_DEX3_DORSAL_MOUNT_SPEC,
    Dex3DorsalMountSpec,
    hole_centers_plate_mm,
    marker_grid,
    mount_manifest,
    palm_T_plate_mm,
    validate_mount_spec,
)

CAD_DIR = ROOT / "cad/dex3_dorsal_aruco_mount"
RENDER = ROOT / "renders/dex3_dorsal_aruco_mount.png"
MULTICOLOR_3MF = CAD_DIR / "dex3_dorsal_aruco_mount_multicolor_h2d.3mf"
PALM_URDF_DIRECTORY = ROOT / "unitree_ros/robots/dexterous_hand_description/dex3_1"

COLORS = {
    "background": "#0c1016",
    "hand": "#313842",
    "hand_edge": "#4c5664",
    "white": "#f1eee7",
    "black": "#0c0e12",
    "fastener": "#161a20",
    "accent": "#2dd4d7",
    "text": "#f4f7fa",
    "muted": "#aeb8c5",
}


def transformed(mesh: trimesh.Trimesh, transform: np.ndarray) -> trimesh.Trimesh:
    result = mesh.copy()
    result.apply_transform(transform)
    return result


def translated(
    mesh: trimesh.Trimesh, xyz: tuple[float, float, float]
) -> trimesh.Trimesh:
    result = mesh.copy()
    result.apply_translation(np.asarray(xyz, dtype=np.float64))
    return result


def boolean_union(parts: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    result = trimesh.boolean.union(parts, engine="manifold")
    if not isinstance(result, trimesh.Trimesh):
        raise TypeError("manifold union did not return one mesh")
    return result


def subtract(mesh: trimesh.Trimesh, cutters: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    result = trimesh.boolean.difference([mesh, *cutters], engine="manifold")
    if not isinstance(result, trimesh.Trimesh):
        raise TypeError("manifold difference did not return one mesh")
    return result


def box_at(
    extents: tuple[float, float, float], center: tuple[float, float, float]
) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=np.asarray(extents, dtype=np.float64))
    mesh.apply_translation(np.asarray(center, dtype=np.float64))
    return mesh


def cylinder_at(
    radius: float,
    height: float,
    center: tuple[float, float, float],
    *,
    sections: int = 64,
) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    mesh.apply_translation(np.asarray(center, dtype=np.float64))
    return mesh


def sloped_cylinder_at(
    radius: float,
    base_z: float,
    center_xy: tuple[float, float],
    center_height: float,
    dz_dy: float,
    *,
    sections: int = 96,
) -> trimesh.Trimesh:
    """Return a cylinder whose top plane has the requested local y slope."""
    angles = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    x = center_xy[0] + radius * np.cos(angles)
    y = center_xy[1] + radius * np.sin(angles)
    bottom = np.column_stack((x, y, np.full(sections, base_z)))
    top_z = base_z + center_height + dz_dy * (y - center_xy[1])
    top = np.column_stack((x, y, top_z))
    vertices = np.vstack(
        (
            bottom,
            top,
            np.asarray([[center_xy[0], center_xy[1], base_z]]),
            np.asarray([[center_xy[0], center_xy[1], base_z + center_height]]),
        )
    )
    bottom_center = 2 * sections
    top_center = bottom_center + 1
    faces: list[tuple[int, int, int]] = []
    for index in range(sections):
        following = (index + 1) % sections
        faces.extend(
            (
                (index, following, sections + following),
                (index, sections + following, sections + index),
                (bottom_center, following, index),
                (top_center, sections + index, sections + following),
            )
        )
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=True)
    mesh.fix_normals()
    return mesh


def rounded_plate(spec: Dex3DorsalMountSpec) -> trimesh.Trimesh:
    """Return a rounded optical square with a compact central mounting tab."""
    width = spec.plate_size_mm
    radius = spec.plate_corner_radius_mm
    depth = spec.plate_thickness_mm
    parts = [
        box_at(
            (width - 2.0 * radius, width, depth),
            (width / 2.0, width / 2.0, depth / 2.0),
        ),
        box_at(
            (width, width - 2.0 * radius, depth),
            (width / 2.0, width / 2.0, depth / 2.0),
        ),
    ]
    for x in (radius, width - radius):
        for y in (radius, width - radius):
            parts.append(cylinder_at(radius, depth, (x, y, depth / 2.0)))

    tab_width = spec.mounting_tab_width_mm
    tab_overlap = 0.5
    tab_length = spec.mounting_tab_depth_mm + tab_overlap
    tab_center_x = width / 2.0
    tab_center_y = width - tab_overlap + tab_length / 2.0
    parts.extend(
        (
            box_at(
                (tab_width - 2.0 * radius, tab_length, depth),
                (tab_center_x, tab_center_y, depth / 2.0),
            ),
            box_at(
                (tab_width, tab_length - 2.0 * radius, depth),
                (tab_center_x, tab_center_y, depth / 2.0),
            ),
        )
    )
    for x in (
        tab_center_x - tab_width / 2.0 + radius,
        tab_center_x + tab_width / 2.0 - radius,
    ):
        for y in (
            tab_center_y - tab_length / 2.0 + radius,
            tab_center_y + tab_length / 2.0 - radius,
        ):
            parts.append(cylinder_at(radius, depth, (x, y, depth / 2.0)))
    return boolean_union(parts)


def black_pattern_solid(
    spec: Dex3DorsalMountSpec,
    *,
    depth_mm: float | None = None,
    z_min_mm: float = 0.0,
) -> trimesh.Trimesh:
    """Return one printable solid for all black ArUco cells.

    Adjacent row runs overlap by 0.01 mm away from the outside marker edge.
    This is far below FFF resolution and prevents edge-touching cells from
    becoming non-manifold after STL's float32 conversion.
    """
    grid = marker_grid(spec)
    cell = spec.marker_cell_size_mm
    quiet = spec.quiet_zone_mm
    depth = spec.black_inlay_depth_mm if depth_mm is None else depth_mm
    overlap = 0.01
    parts: list[trimesh.Trimesh] = []
    for row in range(grid.shape[0]):
        start: int | None = None
        for column in range(grid.shape[1] + 1):
            black = column < grid.shape[1] and grid[row, column] == 0
            if black and start is None:
                start = column
            if not black and start is not None:
                x_min = quiet + start * cell
                x_max = quiet + column * cell
                y_min = quiet + row * cell
                y_max = quiet + (row + 1) * cell
                if start > 0:
                    x_min -= overlap
                if column < grid.shape[1]:
                    x_max += overlap
                if row > 0:
                    y_min -= overlap
                if row + 1 < grid.shape[0]:
                    y_max += overlap
                parts.append(
                    box_at(
                        (x_max - x_min, y_max - y_min, depth),
                        (
                            (x_min + x_max) / 2.0,
                            (y_min + y_max) / 2.0,
                            z_min_mm + depth / 2.0,
                        ),
                    )
                )
                start = None
    return boolean_union(parts)


def countersunk_hole_cutter(
    spec: Dex3DorsalMountSpec, center_xy: np.ndarray, total_height: float
) -> trimesh.Trimesh:
    """Return a through cutter with a 90-degree flat-head countersink."""
    clear_r = spec.hole_clearance_diameter_mm / 2.0
    head_r = spec.countersink_major_diameter_mm / 2.0
    start = -0.25
    end = total_height + 0.5
    profile = np.asarray(
        [
            [0.0, start],
            [head_r, start],
            [head_r, 0.0],
            [clear_r, spec.countersink_depth_mm],
            [clear_r, end],
            [0.0, end],
        ],
        dtype=np.float64,
    )
    cutter = trimesh.creation.revolve(profile, sections=64)
    cutter.apply_translation((center_xy[0], center_xy[1], 0.0))
    return cutter


def build_parts(
    spec: Dex3DorsalMountSpec,
) -> dict[str, trimesh.Trimesh]:
    validate_mount_spec(spec)
    holes = hole_centers_plate_mm(spec)
    structure = rounded_plate(spec)

    pads = [
        cylinder_at(
            spec.screw_boss_diameter_mm / 2.0,
            spec.screw_boss_height_mm,
            (
                hole[0],
                hole[1],
                spec.plate_thickness_mm + spec.screw_boss_height_mm / 2.0,
            ),
        )
        for hole in holes
    ]
    pads.append(
        sloped_cylinder_at(
            spec.wrist_pad_diameter_mm / 2.0,
            spec.plate_thickness_mm,
            (
                spec.plate_size_mm / 2.0,
                spec.wrist_pad_from_proximal_edge_mm,
            ),
            spec.wrist_pad_height_mm,
            spec.wrist_pad_face_slope_dz_dy,
        )
    )
    structure = boolean_union([structure, *pads])

    black = black_pattern_solid(spec)
    max_height = spec.plate_thickness_mm + max(
        spec.screw_boss_height_mm, spec.wrist_pad_height_mm
    )
    hole_cutters = [
        countersunk_hole_cutter(spec, hole[:2], max_height) for hole in holes
    ]
    # Extending the pattern cutter by 0.02 mm avoids coincident boolean faces;
    # the exported black inlay retains the exact requested depth.
    pattern_cutter = black_pattern_solid(
        spec,
        depth_mm=spec.black_inlay_depth_mm + 0.02,
        z_min_mm=-0.01,
    )
    white = subtract(structure, [pattern_cutter, *hole_cutters])
    black = subtract(black, hole_cutters)
    return {"carrier_white_pla": white, "aruco_black_pla": black}


def make_fit_coupon(spec: Dex3DorsalMountSpec) -> trimesh.Trimesh:
    """Return a small print-first coupon for the inferred two-hole pattern."""
    length = spec.hole_spacing_mm + 10.0
    width = 8.0
    height = 2.0
    radius = 2.0
    parts = [
        box_at(
            (length - 2.0 * radius, width, height),
            (length / 2.0, width / 2.0, height / 2.0),
        ),
        box_at(
            (length, width - 2.0 * radius, height),
            (length / 2.0, width / 2.0, height / 2.0),
        ),
    ]
    for x in (radius, length - radius):
        for y in (radius, width - radius):
            parts.append(cylinder_at(radius, height, (x, y, height / 2.0), sections=40))
    coupon = boolean_union(parts)
    hole_x = (length - spec.hole_spacing_mm) / 2.0
    cutters = [
        cylinder_at(
            spec.hole_clearance_diameter_mm / 2.0,
            height + 1.0,
            (x, width / 2.0, height / 2.0),
            sections=48,
        )
        for x in (hole_x, hole_x + spec.hole_spacing_mm)
    ]
    return subtract(coupon, cutters)


def _three_mf_mesh_xml(
    mesh: trimesh.Trimesh,
    *,
    object_id: int,
    name: str,
    material_index: int,
) -> str:
    lines = [
        (
            f'    <object id="{object_id}" name="{name}" type="model" '
            f'pid="1" pindex="{material_index}">'
        ),
        "      <mesh>",
        "        <vertices>",
    ]
    lines.extend(
        f'          <vertex x="{x:.6f}" y="{y:.6f}" z="{z:.6f}"/>'
        for x, y, z in mesh.vertices
    )
    lines.extend(("        </vertices>", "        <triangles>"))
    lines.extend(
        f'          <triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in mesh.faces
    )
    lines.extend(("        </triangles>", "      </mesh>", "    </object>"))
    return "\n".join(lines)


def write_multicolor_3mf(
    white: trimesh.Trimesh,
    black: trimesh.Trimesh,
    output: Path | None = None,
) -> None:
    output = MULTICOLOR_3MF if output is None else output
    model = "\n".join(
        (
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<model unit="millimeter" xml:lang="en-US"',
            ' xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"',
            ' xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">',
            '  <metadata name="Title">Dex3 dorsal ArUco mount</metadata>',
            "  <resources>",
            '    <m:basematerials id="1">',
            '      <m:base name="Black PLA" displaycolor="#000000FF"/>',
            '      <m:base name="White PLA" displaycolor="#FFFFFFFF"/>',
            "    </m:basematerials>",
            _three_mf_mesh_xml(
                white,
                object_id=2,
                name="carrier_white_PLA",
                material_index=1,
            ),
            _three_mf_mesh_xml(
                black,
                object_id=3,
                name="aruco_black_PLA",
                material_index=0,
            ),
            '    <object id="4" name="dex3_dorsal_aruco_mount" type="model">',
            "      <components>",
            '        <component objectid="2"/>',
            '        <component objectid="3"/>',
            "      </components>",
            "    </object>",
            "  </resources>",
            "  <build>",
            '    <item objectid="4"/>',
            "  </build>",
            "</model>",
        )
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
        '  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
        "</Types>\n"
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        '  <Relationship Target="/3D/3dmodel.model" Id="rel-1" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
        "</Relationships>\n"
    )
    model_settings = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<config>\n"
        '  <object id="4">\n'
        '    <metadata key="name" value="Dex3 dorsal ArUco mount"/>\n'
        '    <part id="2" subtype="normal_part">\n'
        '      <metadata key="name" value="carrier_white_PLA"/>\n'
        '      <metadata key="extruder" value="2"/>\n'
        "    </part>\n"
        '    <part id="3" subtype="normal_part">\n'
        '      <metadata key="name" value="aruco_black_PLA"/>\n'
        '      <metadata key="extruder" value="1"/>\n'
        "    </part>\n"
        "  </object>\n"
        "</config>\n"
    )
    project_settings = json.dumps(
        {
            "version": "02.00.00.00",
            "printer_technology": "FFF",
            "filament_colour": ["#000000", "#FFFFFF"],
            "filament_type": ["PLA", "PLA"],
            "nozzle_diameter": ["0.4", "0.4"],
        },
        indent=2,
    )
    entries = {
        "[Content_Types].xml": content_types,
        "_rels/.rels": relationships,
        "3D/3dmodel.model": model,
        "Metadata/model_settings.config": model_settings,
        "Metadata/project_settings.config": project_settings,
    }
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, payload in entries.items():
            info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload)


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def xyz_rpy_transform(origin: ElementTree.Element | None) -> np.ndarray:
    transform = np.eye(4)
    if origin is None:
        return transform
    xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
    transform[:3, :3] = rpy_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def load_neutral_hand_meshes(side: str = "right") -> list[trimesh.Trimesh]:
    """Load official side-specific Dex3 meshes at the zero configuration."""
    if side not in {"left", "right"}:
        raise ValueError("Dex3 render side must be left or right")
    short = "l" if side == "left" else "r"
    palm_urdf = PALM_URDF_DIRECTORY / f"dex3_1_{short}.urdf"
    root = ElementTree.parse(palm_urdf).getroot()
    visual_mesh: dict[str, tuple[Path, np.ndarray]] = {}
    for link in root.findall("link"):
        visual = link.find("visual")
        if visual is None:
            continue
        mesh_node = visual.find("geometry/mesh")
        if mesh_node is None:
            continue
        visual_mesh[link.attrib["name"]] = (
            palm_urdf.parent / mesh_node.attrib["filename"],
            xyz_rpy_transform(visual.find("origin")),
        )

    children: dict[str, list[tuple[str, np.ndarray]]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.attrib["link"]
        child_name = child.attrib["link"]
        children.setdefault(parent_name, []).append(
            (child_name, xyz_rpy_transform(joint.find("origin")))
        )
    # Render in the actual palm-link frame.  The URDF also contains virtual
    # retargeting base links above the palm which are irrelevant here.
    base = f"{side}_hand_palm_link"
    link_T: dict[str, np.ndarray] = {base: np.eye(4)}
    stack = [base]
    while stack:
        parent = stack.pop()
        for child, parent_T_child in children.get(parent, []):
            link_T[child] = link_T[parent] @ parent_T_child
            stack.append(child)

    meshes: list[trimesh.Trimesh] = []
    for link_name, (mesh_path, link_T_visual) in visual_mesh.items():
        mesh = trimesh.load_mesh(mesh_path, process=False)
        mesh.apply_transform(link_T[link_name] @ link_T_visual)
        mesh.apply_scale(1000.0)
        meshes.append(mesh)
    return meshes


def palm_T_plate(spec: Dex3DorsalMountSpec, side: str = "right") -> np.ndarray:
    """Map printable plate coordinates to the palm-link frame in millimetres."""
    return palm_T_plate_mm(spec, side=side)


def draw_mesh(
    ax,
    mesh: trimesh.Trimesh,
    color: str,
    *,
    alpha: float = 1.0,
    stride: int = 1,
    edgecolor: str = "none",
    linewidth: float = 0.0,
) -> None:
    triangles = mesh.vertices[mesh.faces[::stride]]
    ax.add_collection3d(
        Poly3DCollection(
            triangles,
            facecolor=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
        )
    )


def style_3d(ax) -> None:
    ax.set_facecolor(COLORS["background"])
    ax.set_axis_off()
    ax.set_proj_type("ortho")


def render(
    parts: dict[str, trimesh.Trimesh],
    spec: Dex3DorsalMountSpec,
    output: Path | None = None,
    *,
    side: str = "right",
) -> None:
    output = RENDER if output is None else output
    hand_meshes = load_neutral_hand_meshes(side)
    plate_transform = palm_T_plate(spec, side)
    mounted = {name: transformed(mesh, plate_transform) for name, mesh in parts.items()}

    fig = plt.figure(figsize=(17, 9.5), facecolor=COLORS["background"])
    assembled_ax = fig.add_subplot(1, 3, 1, projection="3d", computed_zorder=False)
    exploded_ax = fig.add_subplot(1, 3, 2, projection="3d", computed_zorder=False)
    face_ax = fig.add_subplot(1, 3, 3)

    for ax in (assembled_ax, exploded_ax):
        for hand in hand_meshes:
            draw_mesh(
                ax,
                hand,
                COLORS["hand"],
                alpha=0.94,
                stride=2,
                edgecolor=COLORS["hand_edge"],
                linewidth=0.015,
            )
        style_3d(ax)
        ax.set_xlim(-8, 165)
        ax.set_ylim(-70, 70)
        ax.set_zlim(-70, 70)
        ax.set_box_aspect((173, 100, 140))

    for name, mesh in mounted.items():
        draw_mesh(
            assembled_ax,
            mesh,
            COLORS["white"] if "white" in name else COLORS["black"],
        )
    for hole in hole_centers_plate_mm(spec):
        head = cylinder_at(
            spec.countersink_major_diameter_mm / 2.0,
            0.16,
            (hole[0], hole[1], -0.03),
            sections=48,
        )
        draw_mesh(
            assembled_ax,
            transformed(head, plate_transform),
            COLORS["fastener"],
        )
    assembled_ax.view_init(elev=1, azim=-91, roll=90)
    assembled_ax.set_title(
        "One rigid plate on the two dorsal holes",
        color=COLORS["text"],
        fontsize=12,
        weight="bold",
        pad=8,
    )

    exploded_transform = plate_transform.copy()
    dorsal_sign = -1.0 if side == "right" else 1.0
    exploded_transform[1, 3] += dorsal_sign * 24.0
    for name, mesh in parts.items():
        draw_mesh(
            exploded_ax,
            transformed(mesh, exploded_transform),
            COLORS["white"] if "white" in name else COLORS["black"],
        )
    for hole in hole_centers_plate_mm(spec):
        palm_hole = plate_transform @ np.asarray([hole[0], hole[1], 0.0, 1.0])
        screw = trimesh.creation.cylinder(radius=1.0, height=30.0, sections=32)
        screw.apply_transform(
            trimesh.geometry.align_vectors(
                np.asarray([0.0, 0.0, 1.0]), np.asarray([0.0, 1.0, 0.0])
            )
        )
        screw.apply_translation(
            (palm_hole[0], palm_hole[1] + dorsal_sign * 12.0, palm_hole[2])
        )
        draw_mesh(exploded_ax, screw, COLORS["fastener"])
    exploded_ax.view_init(elev=18, azim=-62, roll=84)
    exploded_ax.set_title(
        "Exploded: 2× M3 only; three shell datum pads",
        color=COLORS["text"],
        fontsize=12,
        weight="bold",
        pad=8,
    )

    plate_size = spec.plate_size_mm
    grid = marker_grid(spec) * 255
    pixels_per_cell = 40
    marker = cv2.resize(
        grid,
        (spec.total_marker_cells * pixels_per_cell,) * 2,
        interpolation=cv2.INTER_NEAREST,
    )
    quiet_px = round(spec.quiet_zone_mm / spec.marker_cell_size_mm * pixels_per_cell)
    face = cv2.copyMakeBorder(
        marker,
        quiet_px,
        quiet_px,
        quiet_px,
        quiet_px,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    face_ax.imshow(
        face,
        cmap="gray",
        vmin=0,
        vmax=255,
        extent=(0.0, plate_size, 0.0, plate_size),
        origin="lower",
    )
    face_ax.add_patch(
        Rectangle(
            (0.0, plate_size),
            plate_size,
            spec.mounting_tab_depth_mm,
            facecolor="white",
            edgecolor="#d7d7d7",
            linewidth=1.0,
        )
    )
    for hole in hole_centers_plate_mm(spec):
        face_ax.add_patch(
            Circle(
                (hole[0], hole[1]),
                spec.countersink_major_diameter_mm / 2.0,
                fill=False,
                edgecolor=COLORS["accent"],
                linewidth=2.0,
            )
        )
    left, right = hole_centers_plate_mm(spec)[:, 0]
    y_arrow = spec.plate_length_mm + 4.0
    face_ax.annotate(
        "",
        xy=(left, y_arrow),
        xytext=(right, y_arrow),
        arrowprops={"arrowstyle": "<->", "color": COLORS["accent"], "lw": 1.8},
    )
    face_ax.text(
        (left + right) / 2.0,
        y_arrow + 1.2,
        f"{spec.hole_spacing_mm:g} mm",
        color=COLORS["accent"],
        ha="center",
        va="bottom",
        fontsize=10,
        weight="bold",
    )
    face_ax.text(
        plate_size / 2.0,
        -5.2,
        "40 mm active marker · 5 mm quiet zone\nM3 heads stay on the separate finger-side tab",
        color=COLORS["muted"],
        ha="center",
        va="top",
        fontsize=9.5,
        linespacing=1.4,
    )
    face_ax.set_xlim(-3, plate_size + 3)
    face_ax.set_ylim(-10, spec.plate_length_mm + 9)
    face_ax.set_aspect("equal")
    face_ax.axis("off")
    face_ax.set_title(
        f"{ARUCO_DICTIONARY_NAME}, ID {spec.marker_id}",
        color=COLORS["text"],
        fontsize=12,
        weight="bold",
        pad=8,
    )

    legend = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=COLORS["hand"],
            markersize=10,
            label="official Unitree Dex3 meshes",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=COLORS["white"],
            markersize=10,
            label="white PLA structural carrier",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=COLORS["black"],
            markersize=10,
            label="black PLA inlay / black M3 heads",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        frameon=False,
        labelcolor=COLORS["text"],
        bbox_to_anchor=(0.5, 0.015),
        fontsize=9.5,
    )
    fig.suptitle(
        "Dex3 dorsal ArUco mount — GraspGen-X geometry, reduced to one plate",
        color=COLORS["text"],
        fontsize=18,
        weight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.938,
        "No adhesive · no secondary carrier · no moving-finger attachment",
        color=COLORS["muted"],
        fontsize=10.5,
        ha="center",
    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.89, bottom=0.09, wspace=0.03)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, facecolor=fig.get_facecolor())
    plt.close(fig)


def export(
    parts: dict[str, trimesh.Trimesh],
    spec: Dex3DorsalMountSpec,
    output_directory: Path | None = None,
    *,
    side: str = "right",
) -> None:
    output_directory = CAD_DIR if output_directory is None else output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, mesh in parts.items():
        mesh.export(output_directory / f"{name}.stl")
    coupon_name = f"m3_{spec.hole_spacing_mm:g}mm_fit_coupon.stl"
    make_fit_coupon(spec).export(output_directory / coupon_name)
    write_multicolor_3mf(
        parts["carrier_white_pla"],
        parts["aruco_black_pla"],
        output_directory / "dex3_dorsal_aruco_mount_multicolor_h2d.3mf",
    )
    (output_directory / "design_manifest.json").write_text(
        json.dumps(mount_manifest(spec, side=side), indent=2) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--marker-id",
        type=int,
        default=DEFAULT_DEX3_DORSAL_MOUNT_SPEC.marker_id,
        help="DICT_6X6_50 marker ID; right defaults to 4 and left uses 5",
    )
    parser.add_argument(
        "--side",
        choices=("left", "right"),
        default="right",
        help="physical Dex3 hand side used for the palm-frame transform/render",
    )
    parser.add_argument(
        "--hole-spacing-mm",
        type=float,
        default=DEFAULT_DEX3_DORSAL_MOUNT_SPEC.hole_spacing_mm,
        help="measured dorsal-hole center spacing",
    )
    parser.add_argument(
        "--hole-midpoint-x-mm",
        type=float,
        default=DEFAULT_DEX3_DORSAL_MOUNT_SPEC.palm_hole_midpoint_x_mm,
        help="nominal palm-link x coordinate used for render/transform only",
    )
    parser.add_argument(
        "--hole-midpoint-z-mm",
        type=float,
        default=DEFAULT_DEX3_DORSAL_MOUNT_SPEC.palm_hole_midpoint_z_mm,
        help="nominal palm-link z coordinate used for render/transform only",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="CAD output directory; defaults to an ID-specific repository path",
    )
    parser.add_argument(
        "--render-output",
        type=Path,
        help="render path; defaults to an ID-specific repository path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = replace(
        DEFAULT_DEX3_DORSAL_MOUNT_SPEC,
        marker_id=args.marker_id,
        hole_spacing_mm=args.hole_spacing_mm,
        palm_hole_midpoint_x_mm=args.hole_midpoint_x_mm,
        palm_hole_midpoint_z_mm=args.hole_midpoint_z_mm,
    )
    validate_mount_spec(spec)
    output_directory = args.output_directory or (
        CAD_DIR
        if spec.marker_id == DEFAULT_DEX3_DORSAL_MOUNT_SPEC.marker_id
        else ROOT / f"cad/dex3_dorsal_aruco_mount_id{spec.marker_id}"
    )
    render_output = args.render_output or (
        RENDER
        if spec.marker_id == DEFAULT_DEX3_DORSAL_MOUNT_SPEC.marker_id
        else ROOT / f"renders/dex3_dorsal_aruco_mount_id{spec.marker_id}.png"
    )
    parts = build_parts(spec)
    export(parts, spec, output_directory, side=args.side)
    render(parts, spec, render_output, side=args.side)
    for path in sorted(output_directory.iterdir()):
        print(path)
    print(render_output)


if __name__ == "__main__":
    main()
