# `manual_run_001` calibration analysis

## Dataset integrity

- Finalized immutable session: `manual_run_001`
- Dataset SHA-256: `8369ba65246418f0f0844a6adfde173b19fb9ff542ca95497dc5253cf518cf11`
- Accepted captures / calibration samples: 39 / 39
- Lossless PNG and complete state-window pairs: 273 (seven per capture)
- Image, state, camera-profile, pairing, and correspondence hashes verified by
  the dataset builder
- The original ignored raw session remains untouched. Its historical version-2
  pose artifact is not a supported input to the cleaned version-3 pose API; the
  finalized `dataset.json` is schema-independent and remains the solver input.

## Native `mikeferguson/robot_calibration` full-data fits

Both fits use the pinned ROS 2 revision
`db991b040d1dc28af09d8865fc72f09720e12b73`, its native
`Chain3dToCamera2d` residual, all 29 measured joint positions, and all matched
AprilCube 3D corner / image-pixel observations. These are fits to all 39
samples, not held-out deployment validation.

| Model | Joint-offset candidate | Coordinate RMS | Radial RMS | Radial median | Radial p90 | Radial p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Camera + target extrinsics | none | 5.8016 px | 8.2047 px | 6.2127 px | 12.7357 px | 16.4655 px |
| Camera + target extrinsics | left shoulder roll, 5° regularized prior | 3.4472 px | 4.8751 px | 4.2278 px | 7.0473 px | 7.7663 px |

The regularized native fit estimates a left-shoulder-roll correction of
`+0.07585 rad` (`+4.3459°`). This is evidence of a repeatable model discrepancy,
not proof that the encoder zero alone is wrong: target mounting, URDF geometry,
camera mounting, compliance, and correlated parameters can project into the
same scalar correction.

## Pose-level held-out comparison

A deterministic pooled five-fold comparison was also run with complete poses
held out. Raw-corner radial RMS was:

| Free joint-offset candidates | Pooled held-out radial RMS |
| --- | ---: |
| none (12 extrinsic parameters) | 10.875 px |
| shoulder roll | 6.032 px |
| shoulder yaw | 10.936 px |
| elbow | 10.886 px |
| shoulder roll + shoulder yaw | 5.717 px |

The shoulder-roll estimate across folds was `+4.256° ± 0.205°` (one standard
deviation), independently agreeing with the native full-data result. Adding
shoulder yaw improves the pooled score further, but the present dataset is not
sufficient evidence to deploy a second correlated joint parameter.

## Decision

Do not deploy either fitted camera transform yet. The 39 samples are healthy
and strongly identify a shoulder-roll-correlated error, but the next run should
be a fresh version-3 manual session with deliberately broader target depth,
image coverage, and joint excitation. Select model complexity using held-out
poses, inspect per-capture residuals, and require a physically repeatable result
before changing the robot model or camera extrinsics.

Generated native artifacts are under the ignored path
`runs/manual_run_001/robot_calibration/`; the reproducible exporter, upstream
pins, and account-local installer are tracked in this repository.
