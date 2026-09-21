# Physical fit status — recorded 2026-09-13

These observations were reported by the user during the design conversation;
they are not instrumented measurements or a completed acceptance test.

| Item | Latest evidence | Remaining check |
| --- | --- | --- |
| Original V6 frame | Small M6 boss broke during support removal/handling in PLA. | Replaced by the V7 broad crossbar architecture. |
| Upper R2 coupons | User reports M6 x 25 worked with the top coupons. | Washer use/thickness, turns from first engagement to clamping, seating and usable thread depth were not recorded. Keep 25 mm as the upper length under test. |
| Lower R2 coupons | M6 x 25 does not engage through the coupons; it threads into the robot with the coupon removed. | Measure screw reach, seating/alignment, usable thread depth and bottom clearance; lower screw length is unresolved. |
| Bare-screw comparison | About six exposed threads at the lower shell and twelve at the upper shell when first engaging. | Approximate thread counts, not caliper measurements. They suggest different recess depths but do not establish final bolt lengths. |
| Full V7 assembly | CAD/mesh checks passed. | Physical assembly, stiffness, creep, board flatness, reinstall repeatability and calibration accuracy have not been established. |

The old four-M6-x-20 hardware assumption is superseded. Upper M6 x 35 and
lower M6 x 40 were discussed as rough trial estimates; neither is a verified
specification. The later report that 25 mm works for the upper coupons takes
precedence over the rough upper estimate. Do not order or install 35/40 mm
as a confirmed bill of materials based on this handoff.

The screw head and washer must clamp the printed part before the tip or
thread runout contacts an internal stop. Threading a bare screw fully in
does not establish the usable threaded depth with the mount installed.

All four current coupons and crossbars retain the same nominal **12 mm
shell-to-washer-face stack at the screw axis**. A deeper lower washer/head
recess was discussed but has not been modeled, exported, structurally
validated or printed. No 6 mm or 8 mm seat is a released variant. A future
coupon must match the final structural washer-seat position.

The frame still requires **eight M5 x 30 screws**, eight approximately 1 mm
flat washers and eight standard M5 hex nuts. An M5 x 20 alternative was
discussed and not implemented. The accepted carrier still uses four M4 x 30
screws, eight M4 washers and four M4 nyloc nuts.

Next physical check: record the washer dimensions and the upper engagement
turns; measure the lower head-to-shell gap at first engagement and usable
threaded depth. Resolve the lower hardware or a matching structural recess
before printing the full mount. The initial accuracy/repeatability targets
in the design README remain targets, not achieved specifications.
