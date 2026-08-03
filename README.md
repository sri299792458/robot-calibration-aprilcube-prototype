# G1 RealSense–AprilCube calibration prototype

This workspace is building a lightweight calibration path for the head-mounted
RealSense on a mode-5 Unitree G1. The camera observes the rounded AprilCube
rigidly taped to the palm of the left dummy hand. The arm side is a pose-set and
runtime argument (`calibration_arm: left` here), not a separate hard-coded
workflow. The optimizer jointly estimates `torso_T_color_camera` and
`left_rubber_hand_T_aprilcube` from raw image corners and measured joint states.

The codebase now covers the full workflow: read-only teaching, offline path
validation, staged G1 commissioning, immutable rectified capture, deterministic
dataset rebuilding, a twelve-parameter extrinsic solve, held-out residuals, and
joint-compensated repeated-anchor stability. It uses only tags decoded in the
current frame—no optical flow, prediction, recovery, or temporal filtering.

![Preview on the mounted physical cube](renders/aprilcube_quality_preview.png)

This saved example uses approximate bench intrinsics only to exercise the PnP
overlay. It is not a calibration measurement.

## Setup

The checked-out `aprilcube/` repository is an editable local dependency:

```bash
uv sync --python /usr/bin/python3.12 --group dev
```

Use system Python 3.12 for this project. ROS Jazzy's `rclpy` and `cv_bridge` on
the laptop are built for Python 3.12 and NumPy 1.x; a Conda Python 3.13 or NumPy
2.x environment is not ABI-compatible with those installed ROS extensions.

## Preview a saved image

```bash
uv run g1-calib preview \
  --image /path/to/image.png \
  --fx 900 --fy 900 \
  --output renders/quality_preview.png \
  --report-json renders/quality_preview.json \
  --no-window
```

`cx` and `cy` default to the image center for a bench preview. Real collection
will consume the exact rectified `CameraInfo`; approximate focal lengths are not
acceptable for calibration data.

## Preview a local camera

```bash
uv run g1-calib preview --camera 0 --fx 900 --fy 900
```

Keys in the live window:

- `S`: save a visual coverage signature; yellow requires pressing `S` twice.
- `U`: undo the latest visual signature.
- `Q` or Escape: quit.

The standalone OpenCV preview deliberately says **VISUAL QUALITY ONLY** because
it has no robot-state input. Use `teach-poses` for real pose recording; that
command is also read-only, but combines the rectified ROS stream with complete
receipt-stamped `rt/lowstate` samples.

## Pre-hardware verification

```bash
uv run pytest -q
uv run ruff check src tests
uv run g1-calib inspect-artifacts
uv run g1-calib synthetic-check \
  --output-directory runs/synthetic_001 \
  --pose-count 40 --pixel-noise 0.3
```

`inspect-artifacts` returns status 0 only when the required attached geometry is
present and used by collision pairs. The current prototype wraps the known
40 mm cube and tape footprint in a hand-axis-aligned `60 x 58 x 60 mm` box
centered approximately at `[40, -34, 0] mm` in `left_rubber_hand`, matching the
supplied installation photo. Motion reports are content-bound to this config.

## Optional ROS and Unitree runtime

Hardware imports are lazy; the offline environment never opens DDS. In the ROS
Jazzy terminal used with the robot:

```bash
./tools/install_hardware_dependencies.sh
```

The bootstrap verifies exact commits from `config/upstream_pins.yaml`. The
installer builds Unitree's pinned Python CycloneDDS binding against the
CycloneDDS headers and libraries already provided by ROS Jazzy, then verifies a
35-slot HG message and the real CRC implementation. Run all hardware commands
through `tools/g1_calib_hardware.sh`; it supplies the required ROS and dynamic-
library environment without changing the offline workflow. The
camera input must be a rectified color `sensor_msgs/Image` paired with its
matching `CameraInfo`. The adapter rejects non-zero distortion coefficients,
frame-ID mismatches, changed resolution/intrinsics, and raw topics. RealSense
intrinsics are used directly from that matching `CameraInfo`; they are not
estimated by this calibration.

## Read-only pose teaching

First inspect one complete mode-5 state without creating a publisher, then
initialize the content-hashed pose set:

```bash
./tools/g1_calib_hardware.sh inspect-hardware \
  --network-interface enp3s0 --state-json work/initial_state.json

uv run g1-calib init-pose-set \
  --state-json work/initial_state.json \
  --output work/poses.yaml \
  --calibration-arm left
```

After physically checking the fixed head-pitch witness mark, run the live
teacher (replace the topic names and serial with observed values):

```bash
./tools/g1_calib_hardware.sh teach-poses \
  --network-interface enp3s0 \
  --pose-set work/poses.yaml \
  --image-topic /camera/color/image_rect \
  --camera-info-topic /camera/color/camera_info \
  --camera-name g1_head_color --camera-serial REPLACE_ME \
  --head-witness-ack
```

Keys are `S` to save, `A` to mark the next pose as an anchor, `U` to undo, and
`Q`/Escape to quit. Yellow needs a supplied `--yellow-override-reason` and a
second `S`; red can never be saved. Saving waits for state samples after the
chosen image and rechecks image/state bracketing, mode 5, both-arm stationarity,
the frozen right-arm hold, and the witness acknowledgement. It records measured
left joints 15–21, never commanded values. This command constructs no DDS
publisher.

## Validate and commission motion

Review the padded `left_rubber_hand` AprilCube envelope in
`config/collision_pairs.yaml`, then list the exact directed routes in an edges
file based on `config/edges.example.yaml`:

```bash
uv run g1-calib validate-poses \
  --pose-set work/poses.yaml \
  --reference-state-json work/initial_state.json \
  --edges-yaml work/edges.yaml \
  --output work/validation_report.json
```

Commission in increasing-risk stages. The CLI prints the exact acknowledgement
phrase required by each command; it must be passed verbatim. Do not skip a
stage, and use only the actual robot network interface:

```bash
./tools/g1_calib_hardware.sh commission-weight-zero --network-interface enp3s0 \
  --confirm 'I UNDERSTAND THIS WRITES RT/ARM_SDK'

./tools/g1_calib_hardware.sh commission-hold --network-interface enp3s0 \
  --pose-set work/poses.yaml \
  --confirm 'I CONFIRM THE G1 WORKSPACE IS CLEAR'

./tools/g1_calib_hardware.sh commission-pose --network-interface enp3s0 \
  --pose-set work/poses.yaml \
  --validation-report work/validation_report.json \
  --home-pose home --target-pose pose_001 \
  --confirm 'I CONFIRM THE G1 WORKSPACE IS CLEAR'
```

Every stage has exclusive local command ownership, measured-state seeding,
mode/freshness checks, fixed gains matching the official G1 arm example,
velocity limiting, continuous measured settling, CRC, and a terminal blend
weight of zero. Only arm slots 15–28 and blend slot 29 are written; no IK is
used. A fault or operator interruption takes the emergency zero-weight path.

## Collect, solve, and qualify the mount

Copy `config/session_plan.example.yaml`, replace it with the validated pose
order, and revisit `home` at least three times across the run:

```bash
./tools/g1_calib_hardware.sh collect-session \
  --network-interface enp3s0 \
  --pose-set work/poses.yaml \
  --validation-report work/validation_report.json \
  --plan-yaml work/session_plan.yaml \
  --session-directory sessions/run_001 --session-id run_001 \
  --image-topic /camera/color/image_rect \
  --camera-info-topic /camera/color/camera_info \
  --camera-name g1_head_color --camera-serial REPLACE_ME \
  --confirm 'I CONFIRM THE G1 WORKSPACE IS CLEAR'

uv run g1-calib build-dataset \
  --session sessions/run_001 --output sessions/run_001/dataset.json

uv run g1-calib solve \
  --dataset sessions/run_001/dataset.json \
  --output-directory runs/run_001

uv run g1-calib anchor-stability \
  --dataset sessions/run_001/dataset.json \
  --result-json runs/run_001/result.json \
  --pose-id home --output runs/run_001/anchor_stability.json
```

Collection requires a passed report bound to the exact pose-set, URDF, and
collision-config hashes. It freezes all three plus the camera/target inputs into
the immutable session. Each move is confirmed interactively unless
`--auto-confirm-transitions` is explicitly supplied. The 250 Hz command tick
runs on a dedicated synchronized thread, so detection and lossless disk writes
cannot starve arm holding. A run returns to the measured home pose, publishes
terminal weight zero, then makes the raw session read-only.

Do not accept the calibration from RMS alone. Inspect `report.md`, held-out and
per-capture residuals, tag grouping, calibration-joint correlations, and the
repeated anchor report. A failed anchor report means the taped cube/camera mount or data
must be investigated and the run repeated; it is not a reason to add joint
offset parameters.
