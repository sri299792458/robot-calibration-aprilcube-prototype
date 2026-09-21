#!/usr/bin/env python3
"""Overlay G1Pilot's primitive collision geometry on Unitree's G1 URDF.

The output keeps every link, joint, limit, inertial, visual, and sensor frame
from Unitree's official revision-1.0 model.  For links where the pinned
G1Pilot URDF supplies primitive collision geometry, the corresponding
``<collision>`` elements are copied verbatim.  Mesh collision elements are
left on the Unitree model so the generated file has no runtime dependency on
G1Pilot's package:// mesh resolver.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNITREE_URDF = ROOT / "unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"
DEFAULT_G1PILOT_URDF = ROOT.parent / "g1pilot/description_files/urdf/g1_29dof.urdf"
DEFAULT_OUTPUT = ROOT / "config/urdf/g1_29dof_rev_1_0_g1pilot_collision.urdf"
PRIMITIVE_TAGS = {"box", "cylinder", "sphere"}
UNITREE_URDF_SHA256 = "c0ae739c640c3e2c00d1bdd8810b5d6e59601487bd1a3995859f9543269ee5c8"
G1PILOT_COMMIT = "72acc803edefe583c24f53e76a21d8d4ed10ed14"
G1PILOT_URDF_SHA256 = "59a3f308bc0b9ef14c3aa4009009ced2c931b1eb703a5468485978a7f0dbde7d"
REQUIRED_ARM_TORSO_LINKS = {
    "torso_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
}


def _links(root: ET.Element) -> dict[str, ET.Element]:
    return {element.attrib["name"]: element for element in root.findall("link")}


def _has_only_primitive_collisions(link: ET.Element) -> bool:
    collisions = link.findall("collision")
    if not collisions:
        return False
    for collision in collisions:
        geometry = collision.find("geometry")
        if (
            geometry is None
            or len(geometry) != 1
            or geometry[0].tag not in PRIMITIVE_TAGS
        ):
            return False
    return True


def build_overlay(
    unitree_urdf: Path,
    g1pilot_urdf: Path,
    output: Path,
) -> tuple[ET.ElementTree, tuple[str, ...]]:
    unitree_sha256 = hashlib.sha256(unitree_urdf.read_bytes()).hexdigest()
    source_sha256 = hashlib.sha256(g1pilot_urdf.read_bytes()).hexdigest()
    if unitree_sha256 != UNITREE_URDF_SHA256:
        raise ValueError(
            f"Unitree URDF hash {unitree_sha256} does not match the pinned source"
        )
    if source_sha256 != G1PILOT_URDF_SHA256:
        raise ValueError(
            f"G1Pilot URDF hash {source_sha256} does not match commit " + G1PILOT_COMMIT
        )
    unitree_tree = ET.parse(unitree_urdf)
    unitree_root = unitree_tree.getroot()
    g1pilot_root = ET.parse(g1pilot_urdf).getroot()
    unitree_links = _links(unitree_root)
    g1pilot_links = _links(g1pilot_root)

    copied_names = tuple(
        sorted(
            name
            for name in unitree_links.keys() & g1pilot_links.keys()
            if _has_only_primitive_collisions(g1pilot_links[name])
        )
    )
    missing = sorted(REQUIRED_ARM_TORSO_LINKS - set(copied_names))
    if missing:
        raise ValueError(
            "G1Pilot URDF is missing required primitive collision links: "
            + ", ".join(missing)
        )

    for name in copied_names:
        destination = unitree_links[name]
        existing = destination.findall("collision")
        insertion_index = (
            min(list(destination).index(element) for element in existing)
            if existing
            else len(destination)
        )
        for element in existing:
            destination.remove(element)
        for offset, collision in enumerate(g1pilot_links[name].findall("collision")):
            destination.insert(insertion_index + offset, copy.deepcopy(collision))

    # The generated file lives outside Unitree's ignored checkout. Preserve
    # every remaining official mesh reference by rebasing only its path.
    for mesh in unitree_root.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        if filename.startswith("package://"):
            raise ValueError(
                "Unitree source unexpectedly contains an unresolved package mesh: "
                + filename
            )
        source_mesh = (unitree_urdf.parent / filename).resolve()
        mesh.attrib["filename"] = Path(
            os.path.relpath(source_mesh, output.parent.resolve())
        ).as_posix()

    unitree_root.insert(
        0,
        ET.Comment(
            " Generated mechanically by tools/generate_g1pilot_collision_urdf.py; "
            f"Unitree source sha256={unitree_sha256}; "
            f"G1Pilot collision source sha256={source_sha256}. "
        ),
    )
    ET.indent(unitree_tree, space="  ")
    return unitree_tree, copied_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unitree-urdf", type=Path, default=DEFAULT_UNITREE_URDF)
    parser.add_argument("--g1pilot-urdf", type=Path, default=DEFAULT_G1PILOT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tree, copied_names = build_overlay(
        args.unitree_urdf,
        args.g1pilot_urdf,
        args.output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    print(
        f"wrote {args.output} with exact G1Pilot primitive collisions for "
        f"{len(copied_names)} links"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
