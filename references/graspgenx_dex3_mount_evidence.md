# GraspGen-X Dex3 dorsal-mount evidence record

This record preserves the reproducible measurements used for the printable
mount without redistributing the downloaded 29 MB presentation video or the
user's full-resolution physical photograph.

## Sources

- GraspGen-X project: <https://graspgenx.github.io/>
- Original project video:
  <https://raw.githubusercontent.com/graspgenx/graspgenx.github.io/main/static/videos/graspgen-x.mp4>
- Linked CVPR 2026 presentation: <https://www.youtube.com/watch?v=a2sv9EVQJXE>
- Paper and source archive: <https://arxiv.org/abs/2606.00998>
- Official code: <https://github.com/NVlabs/GraspGenX>
- Official Dex3 URDF/meshes used locally:
  `unitree_ros/robots/dexterous_hand_description/dex3_1/dex3_1_r.urdf`
- Official dimensioned Dex3-1 user manual:
  <https://marketing.unitree.com/article/zh/Dex3-1/User_Manual.html>

## Public-CAD audit

The detailed manufacturing CAD necessarily exists inside Unitree, but no
public release containing the two dorsal holes was found as of 2026-08-11:

- Unitree's official `unitree_ros` Dex3 directory contains only the left/right
  URDFs and per-link STL meshes. It contains no STEP, Parasolid, or native CAD.
- The current and historical `right_hand_palm_link.STL` revisions from
  `unitree_ros` and `xr_teleoperate` were inspected. The current mesh is copied
  byte-for-byte into the local GraspGen-X, G1Pilot, cuRobo, and simulation
  assets; none exposes the two dorsal threaded holes.
- Unitree's official `unitree_cad` repository contains A1, AlienGo, and Laikago
  STL files only—no G1 or Dex3.
- Unitree's current Hugging Face `unitree_model` archive contains G1 23-DoF and
  29-DoF USD packages, but no Dex3 directory or detailed hand CAD.
- A global GitHub code search found no public Dex3 STEP/STP file. GraspGen-X
  also ships the same simplified Unitree palm STL and does not publish its
  physical marker fixture.

Public source locations checked:

- <https://github.com/unitreerobotics/unitree_ros/tree/master/robots/dexterous_hand_description/dex3_1>
- <https://github.com/unitreerobotics/unitree_cad>
- <https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main/G1>
- <https://github.com/unitreerobotics/xr_teleoperate>

The manufacturing solid model was not released, but the official user manual
provides the missing dimensioned mounting-hole view. The palm STL remains the
source for the surrounding shell envelope, surface coordinate, and palm-link
frame.

The selected standalone `unitree_ros` URDF is the latest official Dex3-1 model
in the project. Its `right_hand_palm_link.STL` SHA-256 is
`86c0b231cc44477d64a6493e5a427ba16617a00738112dd187c652675b086fb9`,
byte-identical to the palm mesh used by the GraspGen-X cuRobo G1 model. All
seven right-hand joint origins and axes also match the cuRobo and
`xr_teleoperate` variants exactly; their URDF differences are joint limits,
materials, inertial orientation, and XR-only auxiliary links, none of which
changes this fixture geometry.

## Marker reconstruction

OpenCV's predefined-dictionary detector was run over the G1 sequence in the
linked presentation. The dorsal marker repeatedly decoded as ID 4 in the
6-by-6 family. For example, in the locally extracted 20-second G1 crop,
`DICT_6X6_50` returned these four corners:

```text
[[619, 937], [604, 950], [572, 937], [589, 925]]
```

The same wrist target decoded as ID 4 in the 22- and 24-second views. The
larger `DICT_6X6_100`, `250`, and `1000` dictionaries share their first marker
codes; `DICT_6X6_50` is the smallest compatible OpenCV family and matches the
project's observed ID range. A 40 mm active square plus a 5 mm white quiet zone
reconstructs the approximately 50 mm visible plate relative to the known
88 mm Dex3 hand width.

## Hole-pattern reconstruction

The supplied physical photograph was 696 x 928 pixels. Circle detection around
the two dorsal features returned representative outer-recess estimates:

```text
left  center ~= [225.5, 537.5] px, radius ~= 17.4 px
right center ~= [304.5, 546.5] px, radius ~= 15.2 px
center distance ~= 79.5 px
```

Registering the palm outline against the official 86.7 x 87.8 mm palm-envelope
dimensions gives approximately 5.2–5.4 pixels/mm in the local dorsal plane.
The resulting photograph estimates were:

```text
hole center spacing ~= 15.0–15.3 mm
inside opening ~= 3 mm
head recess ~= 5.5–6 mm
```

Direct caliper measurements on the real hand gave approximately 15.5 mm
center-to-center and an accurately measured 12.75 mm gap between the nearest
inner edges. The user directly inserted an M3 screw and confirmed the thread.

The official manual resolves the remaining ambiguity. Its mounting-hole drawing
calls out `2 x M3` to 3 mm depth, 15.00 mm center spacing, and locates the pair
centerline 66.70 mm from the wrist-side datum. The 15.5 mm caliper result was an
approximate check; the CAD default follows the explicit 15.00 mm dimension.

In the URDF, the palm mesh begins at the wrist datum at `x=0`, the finger-base
joints are at `x=77.7 mm`, and the mesh extends toward the fingers to
`x=86.7 mm`. The drawing's 66.70 mm wrist-to-hole dimension therefore maps
directly to palm `x=66.70 mm`. The holes straddle the drawing and URDF centerline
at palm `z=+/-7.50 mm`, so their midpoint is `[x,z]=[66.70,0] mm`.

## Palm surface and nominal pose

Ray intersections against the official `right_hand_palm_link.STL` show that the
dorsal shell under the plate is shallowly convex. Ray intersections at the
documented holes, palm `x=66.70 mm`, `z=+/-7.50 mm`, give `y=-21.330 mm` at
both positions. The first design incorrectly treated that support plane as
parallel to the palm-link `x-z` plane. A least-squares tangent fit over the M3
boss footprints gives `dy/dx=0.019947`, or 1.143 degrees. The first physical
print confirmed the consequence: its former 2.0 mm wrist pad remained clear
of the shell and the plate flexed.

The corrected design moves the third datum to plate `y=20 mm`, approximately
palm `x=31.69 mm`, where the official shell is `y=-20.958 mm`. Its center
height is 2.57 mm and its contact face is tilted by 1.994 degrees relative to
the plate to follow the local shell tangent. The nominal marker-center
translation is now approximately `[0.036816,-0.027427,0] m`; its orientation
also includes the 1.143-degree mounting pitch.

The plate's marker center is 30.0 mm wristward of the screw line. Its transform
is tied to the official hole drawing and the local surface tangent recovered
from the official palm mesh rather than an image-registration estimate.
