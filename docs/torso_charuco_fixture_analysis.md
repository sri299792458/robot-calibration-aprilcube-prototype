# Torso calibration fixture: geometry and evidence

The current mechanical design is the
[V7 native Fusion mount](../artifacts/g1_mount_v7/README.md), using two
crossbars and two flat side plates with the accepted printed carrier.
Current physical observations and unresolved screw lengths are in
[FIT_STATUS.md](../artifacts/g1_mount_v7/FIT_STATUS.md).

## Calibration relationship

For a stationary torso target:

```text
torso_T_camera = torso_T_board @ inverse(camera_T_board)
```

The ChArUco observation supplies `camera_T_board`. The CAD transform supplies
a nominal `torso_T_board`, which must be checked against physical datums.
Reprojection error and repeated images alone do not establish absolute
camera-to-torso accuracy. The intended use is tabletop manipulation within
a few millimetres; the README's repeatability and flatness limits are initial
acceptance targets, not achieved results.

## Preserved interfaces

The camera and front accessory pattern belong to `torso_link`, above the
waist articulation. The design uses the four intended front M6 installation
holes. The nominal spacing is 62.4 mm across the upper row, 64.5 mm across
the lower row, and 194.2 mm between rows.

| Location | Nominal outer shell point in torso coordinates, mm |
| --- | --- |
| Upper left | `[61.7071476, 31.2, 265.0961682]` |
| Upper right | `[61.5783281, -31.2, 265.0961682]` |
| Lower left | `[69.5162713, 32.25, 70.8961682]` |
| Lower right | `[69.4891404, -32.25, 70.8961682]` |

The source drawing was registered to the shoulder-roll axes from the URDF;
the parallel screw axes were intersected with the public exterior shell
mesh. These points are shell intersections, not insert faces or thread-start
positions. The nominal helpers remain in
`src/g1_aprilcube_calibration/torso_board_mount.py`; the input reconstruction
is documented by `tools/recover_g1_front_m6_pattern.py`.

The 210 x 300 mm carrier contains a 180 x 270 mm ChArUco pattern with 30 mm
squares, 22 mm markers and `DICT_5X5_50`. Its four existing M4 coordinates
are frozen. The nominal board center is `[245, 0, 127.5]` mm, at +21 degrees
about torso y. The carrier STL's print coordinates have a +15 mm offset on
both board-plane axes; V7 removes that offset before applying the optical
transform. See `artifacts/g1_mount_v7/design_inputs.json` for the full matrix.

## Verification limits

Recorded geometry checks show four aligned carrier contacts, zero positive
structural volume interference, clear screw/tool approaches, and 49,051
unobstructed active-target rays from the supplied measured camera pose.
These checks do not establish actual shell stiffness, screw engagement,
print stiffness, creep, board flatness or mounted-pose accuracy.

The first physical coupon checks found that M6 x 25 reportedly works at the
top but does not engage through the lower coupons. Upper washer/engagement
details and lower usable thread depth remain unmeasured. All current root
seats are 12 mm at the screw axis; no proposed deeper recess has been
implemented. Hardware remains governed by the physical fit record.

## Sources

- [Unitree G1 user manual](https://marketing.unitree.com/article/en/G1/User_Manual.html)
- [Official Unitree G1 URDF/model](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description)
- [Frozen input provenance](../artifacts/g1_mount_v7/reference_inputs/PROVENANCE.md)
- [Current CAD verification and acceptance targets](../artifacts/g1_mount_v7/README.md)
