# G1 torso calibration mount — native Fusion redesign

Status: mechanically motivated CAD prototype for stationary calibration. The four structural parts have passed the recorded geometry and clearance checks. R2 coupons have been printed and tried: M6 x 25 reportedly works at the upper locations but does not engage through the lower coupons. Lower screw length, full assembly fit, stiffness, repeatability and calibration accuracy remain unresolved. See [physical fit status](FIT_STATUS.md).

## Design decision

Use two torso crossbars and two flat side plates. The accepted 210 × 300 mm carrier, its four existing M4 holes, the 180 × 270 mm ChArUco pattern and its nominal placement are unchanged.

The failed design had a small offset connection to an M6 boss, which broke during support removal/handling in PLA. This redesign puts each 28 × 28 mm shell seat directly into a continuous crossbar. The side plates have approximately 18 mm rails, a broad rear spine, rounded windows, and fully backed 28 × 12 mm carrier tabs.

The four-part architecture is a deliberate print-orientation choice. Side plates lie on their exterior faces; crossbars lie on their forward faces. The principal long members then lie in their print layers. An integral two-frame alternative reduces screw count but makes the lateral torso attachments harder to orient favorably. The four-part layout is the current recommendation for calibration, not a claim that it is globally optimal.

Eight additional M5 bolts attach the four side-plate/crossbar junctions, two bolts per junction with 20 mm spacing. These are fixed assembly joints. There are no adjustment slots or extra carrier holes. Splitting parts to improve print orientation follows the general approach described in [Prusa's design-for-printing guidance](https://help.prusa3d.com/article/modeling-with-3d-printing-in-mind_164135).

## Files

- `cad/G1_calibration_mount_native.f3d`: native Fusion archive with named sketches, extrusions, lofts, construction planes, imported reference meshes and hidden fit coupons.
- `cad/G1_calibration_mount_assembly.step`: four structural solids in their assembled positions. Reference meshes and hidden coupons are excluded.
- `print_parts/`: four centered, correctly oriented structural STLs, one copy each.
- `fit_coupons/`: four compact R2 interface coupons, centered on the bed. Their full contact patches, M6 bores and washer bearing planes are retained from the crossbars. The unused front block and access wells are removed. See `fit_coupons/README.md` for identification marks and coupon settings.
- `renders/fusion_assembly.png`: assembly shown in Fusion, with the existing board and robot as references.
- `renders/board_connections.png`: centered rear view showing the four board mounting pads.
- `renders/m6_root_section.png`: section through the upper-left M6 connection.
- `renders/print_orientations.png`: broad faces to place on the bed.
- `design_inputs.json`: dimensions, frozen interfaces, camera/board placement and nominal shell samples.
- `mesh_and_clearance_qa.json`, `fusion_assembly_qa.json`, `fusion_interference_qa.json`, `board_contact_qa.json`: verification records.

The original accepted board is included in the Fusion file as reference geometry. It does not need reprinting. Mesh references are not converted into a new optical pattern.

The supplied carrier STLs use print-bed coordinates, translated +15 mm along both board-plane axes. The final Fusion assembly reverses this offset before applying the optical-board transform. Earlier assembly previews omitted that conversion and showed the carrier shifted relative to its supports; the final archive and included previews are corrected.

## Hardware

| Use | Quantity | Hardware |
| --- | ---: | --- |
| Upper torso mounting | 2 | M6 × 25 socket-head screws under physical test; engagement and washer stack unmeasured |
| Lower torso mounting | 2 | M6 socket-head screws; length unresolved. M6 × 25 does not engage through the R2 coupons |
| Torso washers | 4 | M6 flat washers, approximately 1.6 mm thick, maximum 13 mm OD |
| Existing board attachment | 4 | M4 × 30 socket-head screws |
| Board washers and nuts | 8 + 4 | M4 flat washers and M4 nyloc nuts |
| New frame joints | 8 | M5 × 30 socket-head screws |
| New frame washers and nuts | 8 + 8 | M5 flat washers, approximately 1 mm thick, and standard M5 hex nuts |

The M5 nut slots are 4.4 mm thick and 8.3 mm across the retaining flats. They are for ordinary approximately 4 mm thick M5 hex nuts, not taller nyloc nuts. Load the nuts through the forward-facing slots and insert each bolt from the exterior of the side plate.

The M5 stack leaves approximately 3.8 mm of bolt protrusion beyond the modeled inner nut face. Check the actual nut/washer dimensions before assembly. All tightening is through washers and printed material; use a short hand tool and stop before compressing or deforming the print. Witness-mark the joints and inspect for loosening.

The earlier M6 × 20 assumption is superseded by the physical results. No deeper lower washer recess or M5 × 20 conversion has been made. See [FIT_STATUS.md](FIT_STATUS.md) before selecting hardware.

### M6 reach is not the same as engagement

At each M6 center the modeled shell-to-washer-face distance is 12 mm. A 20 mm screw with a 1.6 mm washer therefore projects about **6.4 mm past the modeled outer shell**. Actual threaded engagement is reduced by any recess before the first usable thread. The handoff does not measure that recess or the blind-hole clearance.

Do not assume 6.4 mm of thread engagement, use screw bottoming as a datum, or substitute a longer screw without measuring the usable depth. A screw that merely begins to clamp is not proof of adequate engagement. Record both thread-start depth and available bottom clearance with each fit coupon before finalizing the M6 length.

## Print and assembly

1. Print the four compact R2 fit coupons first, using 0.20 mm layers, three walls, four top/bottom layers and 10% gyroid infill in PLA. Each retains the seating face, M6 bore and washer bearing plane. The deep tool well is removed. Fit the matching coupon with its arrow pointing up and the normal silver shell installed. Dot counts identify 1 upper-left, 2 upper-right, 3 lower-left, 4 lower-right, using the robot's own left/right.
2. Check full seating without rocking, adequate measured screw engagement and clearance before bottoming. The shell must not shift or spring against internal foam under light clamping. Compact coupons test individual seats and screw reach; the full crossbar hole spacing, simultaneous seating and deep tool-access corridors still need checking with the actual crossbars.
3. After the coupons pass, print the four structural STLs in their provided orientations. Centering in the slicer is fine; preserve rotation and 100% scale. One side plate per bed. Both crossbars can share a bed if the slicer confirms spacing.
4. Start with PLA or PLA+, 0.20 mm layers, six walls, six top/bottom layers and 35% gyroid infill. These are starting settings, not a tested printer profile. There are no detached bosses or large unsupported carrier tabs. Inspect the short horizontal holes and the M6 counterbore ceilings in the slicer; small bridges remain. Do not enable support that will be trapped in a well or nut slot.
5. Assemble the crossbars and side plates loosely with the eight M5 bolts. Fit the existing board against the four carrier tabs and install its four M4 bolts, washers and nyloc nuts. Keep all joints loose until every interface seats naturally; do not pull misaligned parts together with screw torque.
6. Mount on the stationary robot, then snug the joints progressively without bowing the board or crushing the PLA. Record the tightening sequence and witness marks for repeatable reassembly.

The side plates measure approximately 238.1 × 271.4 × 34.5 mm in print orientation. Upper and lower crossbars measure 230 × 40 × 53.1 mm and 230 × 40 × 37.6 mm. All fit within the handoff's conservative 300 × 320 mm bed area.

PLA remains the baseline. A change to PETG changes stiffness and printing behavior and does not replace the fit and flex checks. The fixture is for stationary calibration; remove it before robot walking, waist motion or arm trajectories.

## Recorded CAD checks

- Four structural STLs and four coupons: watertight, consistently oriented, one connected solid each.
- Native Fusion timeline: no feature errors or warnings at export.
- Four native structural solids: zero volume interference; coincident mating faces excluded.
- Full active target: 49,051 rays at 1 mm spacing from the supplied fitted camera pose, zero obstruction.
- Four M6 corridors: a sampled 14.4 mm diameter, 90 mm long straight approach is clear.
- Four M4 locations: optical-side and rear nut approach centerlines are clear.
- Existing carrier to support pads: all four screw centerlines align; 72 sampled points around each hole match the carrier's bearing face within 0.02 mm mesh tolerance. This checks nominal CAD contact, not printed fit.
- Eight M5 centerlines: plate and crossbar holes are aligned and clear.
- Nominal shell approximation: 96 contact samples per root; maximum absolute sampled difference from the public shell is approximately 0.113 mm. This is only CAD approximation error, not physical robot accuracy or a complete surface-tolerance certification.

R2 coupon fit observations are recorded in FIT_STATUS.md. No FEA, complete V7 structural assembly trial, creep test, measured stiffness test or calibration capture has been performed. The full-density structural CAD volume corresponds to about 1.01 kg of PLA at 1.25 g/cm³, excluding the accepted carrier and metal hardware. This is an upper bound for solid prints, not a slicer mass estimate. The corresponding structure-only gravity moment about the M6 pattern is approximately 0.78 N·m; it is not a load rating.

## Calibration acceptance for tabletop work within a few millimetres

The intended camera extrinsic remains:

`torso_T_camera = torso_T_board × inverse(camera_T_board)`

The nominal board center is `[245, 0, 127.5]` mm in `torso_link`, pitched +21° about torso y. The active outer top-left corner defines the OpenCV board origin. The complete transform is in `design_inputs.json` in millimetres; convert translation to metres when using the calibration code.

Use these as initial acceptance targets, not achieved specifications:

- Active pattern dimensions remain 180 × 270 mm; measure both axes and diagonals. Check mounted face bow, initially targeting less than 0.5 mm.
- Live rectified color detects all 27 markers and 40 ChArUco corners without fixture obstruction.
- With the camera fixed, collect stationary bursts across at least five full fixture removal/reinstall cycles. Aim for board-pose variation below 0.5 mm and 0.1°. If it exceeds those targets, investigate seating, shell compliance, bolt slip and board flex before accepting the calibration.
- A 0.1° rotation error produces about 0.87 mm lateral error at 0.5 m, before translation and other errors. Evaluate the budget at the actual working distance.
- Reinstall repeatability does not establish absolute accuracy. Independently check the installed board's relationship to a torso datum and validate the result on a separately measured tabletop target or task points. Neither CAD coordinates nor low image reprojection error alone prove millimetre accuracy.

The interface remains nominal until the physical observations are recorded. Keep those observations and the installed transform separate from the source CAD values.

## Editing

The four structural parts are native solids with editable feature history. The supplied Python builders make the fixed interface coordinates explicit. Changing an extrusion or sketch can invalidate the assembled mating faces, screw stack or optical clearance; regenerate and rerun the checks after geometry changes. The single plate-thickness user parameter is not a complete assembly-level constraint system.

The packaging scripts can generate a review ZIP with copies of the V7 scripts in `source/`; generated ZIPs are kept outside Git. Their paths resolve from the repository checkout; they are not a one-click installer. Frozen inputs are in `reference_inputs/`; geometry construction runs inside Fusion. See [the repository script guide](../../scripts/G1_MOUNT_README.md) for dependencies and execution context. Scripts included under `source/` in a generated ZIP are provenance copies; run them from the repository `scripts/` folder. The native F3D file is self-contained and can be opened without these scripts or Python dependencies. For slicing, use `print_parts/` and `fit_coupons/`; the raw STLs under `cad/` retain their modeling coordinates.
