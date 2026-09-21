# Compact G1 fit coupons — R2

Use these revised coupons for the first shell-fit and M6 screw-reach check. All four together are estimated at **19.8 g and 43 minutes**, versus **61.6 g and 100 minutes** for the original coupons with the same settings. That is approximately **68% less filament**. These are Bambu Studio estimates, not a completed print measurement.

The full 28 × 28 mm curved contact patch, 6.8 mm M6 bore, and original washer bearing position are retained. The old 34 mm outer block and deep access well are removed. Upper coupons are now about 22.5–22.8 mm high; lower coupons about 15.1 mm. The four structural mount parts are unchanged.

## Physical fit status

These coupons have now been tried. The user reports M6 × 25 works at the upper locations but does not engage at the lower locations with coupons installed. Lower screw length is unresolved, upper engagement and washer details are unmeasured, and all four CAD seats remain 12 mm at the screw axis. The M6 × 20 example below is nominal geometry, not a hardware recommendation. See [FIT_STATUS.md](FIT_STATUS.md).

## Print

Open `G1_R2_fit_coupons.3mf` in Bambu Studio for all four coupons on one plate, or load the four `FIT_ONLY_*_PRINT.stl` files in this folder. Keep the supplied orientation and 100% scale. The marked flat face goes on the bed; the curved face prints upward.

The supplied profile uses an H2D with a 0.4 mm standard nozzle, Bambu PLA Basic, 0.20 mm layers, **three walls, 10% gyroid infill and four top/bottom layers**, without supports or a brim. Use the actual filament profile for your material before printing; its time estimate can differ. The engraved marks span at most 4 mm on the underside and create small bridges. The main M6 bore is vertical in print orientation.

## Identify and fit

The triangle on the flat washer side points toward the robot's head when installed. Dots on that same side identify the location:

| Dots | Robot location |
| ---: | --- |
| 1 | Upper left |
| 2 | Upper right |
| 3 | Lower left |
| 4 | Lower right |

Left/right refer to the robot's own sides. The markings are engraved on the flat washer side.

Place the curved face against the matching location on the installed silver shell, arrow upward. Use the intended M6 screw and washer, and tighten only gently for the fit check. Check for rocking and verify usable thread engagement and clearance before the screw bottoms out. These coupons retain the nominal 12 mm center stack: M6 × 20 with a 1.6 mm washer projects approximately 6.4 mm beyond the modeled outer shell, before subtracting any unmeasured insert recess.

These are individual fit gauges. They do not validate the assembled crossbar's hole spacing, simultaneous seating, deep tool access, stiffness or calibration accuracy.

## Verification

All four STLs are watertight single solids and sit at Z = 0 in their print files. The CAD revision only trims away material in front of the contact patch and outside its footprint, then adds shallow markings away from the washer. Across 1,592 contact samples per coupon, the revised meshes agree with fresh exports of the original crossbars within 0.038 mm (0.05 mm comparison tolerance); all 96 sampled washer-seat points per coupon agree, and every screw centerline is clear. Native Fusion has no feature errors or warnings. Detailed records are in `../compact_coupon_geometry_qa.json` and `../compact_coupon_slicer_comparison.json`.
