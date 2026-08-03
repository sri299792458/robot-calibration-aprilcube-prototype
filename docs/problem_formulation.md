# G1 RealSense Extrinsic Calibration: Problem Formulation

## Objective

Estimate the six-degree-of-freedom transform between the Unitree G1 and its
head-mounted RealSense accurately enough for manipulation. The first target is
the RGB/color optical frame because AprilTag corners are observed in the color
image. The RealSense driver's factory transforms can subsequently connect the
color optical frame to the depth and IMU frames.

The robot-side reference frame must be selected explicitly. `torso_link` is the
best primary calibration frame when the camera mount is rigid to the torso. A
transform to `pelvis` or another control frame can then be composed from the
robot model and measured joint state. This avoids folding waist and leg motion
into a nominally fixed camera extrinsic.

Frame notation in this document is `A_T_B`: a transform that maps coordinates
from frame B into frame A.

## What the HERO paper establishes

The paper's calibration and control results expose three separate error sources:

1. **Camera extrinsic error.** The D435i mount includes an adjustable head pitch,
   so the manufacturer transform does not reliably describe the physical robot.
2. **Upper-body FK error.** On their G1, analytical end-effector FK had mean
   errors of 1.76 cm and 5.87 degrees over 60 workspace samples. Their residual
   model reduced these to 0.27 cm and 2.30 degrees.
3. **Base-motion error.** Whole-body reaching moves the floating base even when
   the feet are intended to remain planted. Their analytical leg-FK odometry had
   1.10 cm translation error, which a learned residual model reduced to 0.33 cm.

HERO used a 13-camera OptiTrack system for two different jobs:

- measuring base and end-effector poses to train/evaluate residual FK and base
  odometry models; and
- registering a hand-held ArUco board to the robot base for eye-to-hand camera
  calibration. They collected 60-70 diverse board poses and reported 2.5 mm
  reprojection error.

This means that "replace MOCAP with AprilTags" is not one operation. We need a
camera extrinsic data source, an end-effector ground-truth data source, and—if
we reproduce HERO's base correction—a base-motion data source.

## Relevant G1 model facts

The official `unitreerobotics/unitree_ros` repository contains the supported G1
URDF variants under `robots/g1_description/`. Our robot reports
`mode_machine = 5`, selecting the revision-1.0 29-DOF family. Use
`g1_29dof_rev_1_0.urdf` as the base model. Its fixed
`left_hand_palm_joint` terminates at the installed rigid dummy-hand carrier,
`left_rubber_hand`. The current pose set moves left-arm indices 15–21 and holds
right-arm indices 22–28 at their measured activation pose.

Most 29-DOF variants model the D435 as a fixed child of `torso_link` with the
nominal transform:

```text
xyz = [0.0576235, 0.01753, 0.42987] m
rpy = [0, 0.8307767239493009, 0] rad
```

That is an initial estimate, not a calibration result. On our robot the camera
pitch is manually adjustable and has no measured joint state. Therefore the
physical camera-to-torso transform is constant only while the adjustment remains
untouched. The standard URDF's fixed `d435_joint` is the correct model structure,
but its six numerical transform parameters must be calibrated for the chosen
physical setting.

An unsensed manual pitch cannot be corrected in software at runtime. For this
prototype, the camera will be mechanically fixed at one chosen operating angle
and witness-marked before calibration. The witness mark is a quick validity check;
the mechanical fixation is what preserves the transform. If the mark moves or the
mount is loosened, the saved extrinsic is invalid and must be recalibrated.

The URDF only defines `d435_link`; the RealSense ROS driver defines color and
depth optical frames. Calibration output must state which exact optical frame it
targets, otherwise a correct numerical transform can still be applied with the
wrong axis convention.

## Measurement model with an AprilCube on the hand

Define:

- B: chosen robot reference frame, initially `torso_link`;
- C: RealSense color optical frame;
- E: end-effector or rigid hand-mount frame;
- O: AprilCube object frame;
- `q_i`: synchronized robot joint state at capture i.

The AprilCube detector supplies `C_T_O_i`. Analytical FK supplies
`B_T_E(q_i)`. A consistent sample must satisfy:

```text
B_T_C * C_T_O_i = B_T_E(q_i; kinematic_parameters) * E_T_O
```

The unknowns are at least `B_T_C` and the target mount `E_T_O`. If the physical
head moves, `B_T_C` becomes a kinematic chain containing the head joint and a
fixed head-to-camera transform.

For calibration, it is better to optimize the raw detected image corners than
already-solved PnP poses. For AprilCube corner `p_O_j` observed at pixel
`u_i_j`, minimize a robust reprojection residual:

```text
u_i_j - project(K, C_T_B * FK_B_E(q_i; theta) * E_T_O * p_O_j)
```

Here `theta` can include selected joint-zero offsets and frame corrections. This
is the same optimization structure already supported by `robot_calibration` for
a camera observing a checkerboard on an arm. AprilCube will require a new feature
finder/adapter that emits its known 3D corners and detected 2D pixels.

## The observability trap

A stationary tag board of unknown pose does **not** calibrate the camera to the
robot. It only gives the board pose relative to the camera. A missing transform
cannot be recovered by collecting more images of the same unknown relationship.

Likewise, solving camera extrinsics from a hand-mounted tag while treating
nominal G1 FK as truth will absorb FK bias into the camera transform. The result
may have low reprojection error on calibration poses while being wrong elsewhere.
That is exactly the failure warned about in HERO.

We need at least one of these anchors:

1. a target rigidly registered to `torso_link` by a measured/CAD calibration
   fixture and visible to the camera;
2. a world target field registered to the robot by a physical foot/base fixture;
3. joint optimization of camera extrinsics, target mount, and a deliberately
   limited set of kinematic corrections over diverse arm configurations; or
4. an independent external camera observing both robot-mounted and world tags.

Option 3 is the fastest useful prototype with our current repositories. Option 1
is the cleanest way to break the camera/FK coupling if we can fabricate a rigid,
repeatable torso-mounted fixture.

## Proposed staged approach

### Stage 0: Identify and freeze conventions

- Record exact RealSense model and serial number, ROS topics, firmware, image
  resolution, and active camera profile. The G1 is `mode_machine = 5`.
- Mechanically fix the manually adjustable, unsensed camera pitch at the intended
  operating angle and add a witness mark that can be checked before each run.
- Choose `torso_link` as the primary reference unless the physical mount dictates
  another rigid link.
- Use only the rectified color image with its matching `CameraInfo`. Projection
  uses the rectified 3 x 3 block of `P`, not raw-image `K`.

### Stage 1: Validate the visual target

- Start with the already printed default AprilCube: a 40 mm cube with six 30 mm
  OpenCV ArUco `4x4_100` markers. Reprint a larger/different target only if the
  physical visibility and corner-noise test fails.
- Use the released rounded-target geometry: a 40 mm envelope with 30 mm tags.
  The exact artifact/config hashes are frozen into every session.
- Disable temporal filtering during calibration capture. Retain raw tag IDs and
  subpixel corner coordinates, and reject blurred or high-reprojection-error
  frames.
- Run a repeatability experiment with a stationary camera and target before using
  the target as metrology equipment.

AprilCube improves visibility and PnP conditioning by combining corners on
multiple faces, but its technical report does not provide a calibrated physical
accuracy benchmark. Its accuracy depends on print geometry, contrast, focus,
exposure, intrinsics, and mechanical mounting, so this validation is mandatory.

### Stage 2: Static joint camera/kinematic calibration

- Attach the AprilCube directly to the left rigid dummy palm with the purchased
  mounting tape, in the approximate placement shown by the supplied robot photo.
  A padded hand-frame AABB covers the cube and tape for collision planning; the
  exact six-DoF hand-to-cube transform remains a free calibration parameter.
- Capture 60-80 quasi-static samples with synchronized image, `CameraInfo`,
  joint states, and raw corner detections.
- Record and replay the complete measured seven-joint left-arm vector. Wrist
  joints may vary when that improves target orientation and image coverage;
  their measured positions are included in FK. Continuously hold and monitor the
  right arm. Keep the waist/legs fixed and avoid a set dominated by
  fronto-parallel views.
- Initially solve only `torso_link` to color-optical-frame extrinsics and the
  hand-to-AprilCube mount.
- Add at most one strongly regularized, varied shoulder/elbow zero offset only
  after residual and parameter-observability analysis. A fixed wrist offset is
  indistinguishable from the unknown hand-to-cube transform in this dataset and
  must not be estimated.
- Optimize a single fixed six-DoF `torso_link` to camera transform. Do not model
  the manual pitch as a joint because no runtime measurement is available.

The existing `robot_calibration` optimizer is suitable for this batch problem,
but its checkerboard-only 2D finder must be adapted to AprilCube correspondences.

### Stage 3: Break the FK/extrinsic coupling

Build a torso-registered AprilTag fixture or another independently measured
reference. Use it to validate—and preferably initialize—`torso_link` to camera
extrinsics without arm FK. Compare this independent result to Stage 2. A large
difference indicates that the joint solution has hidden G1 FK error inside the
camera transform.

### Stage 4: Collect residual FK labels without MOCAP

After the camera extrinsic is anchored, the hand-mounted AprilCube directly gives
end-effector pose relative to the torso while it remains visible:

```text
B_T_E_measured = B_T_C * C_T_O * inverse(E_T_O)
```

Log this alongside joint state to characterize analytical FK residuals and, if
needed, train a HERO-style residual model. Capture both arms separately and
include payload/load cases because elastic deflection is state- and load-dependent.

### Stage 5: Base-motion labels

Place several large AprilTag/AprilCube targets at fixed, surveyed poses around the
workspace. Observing that mapped field with the head camera estimates camera—and
therefore torso—motion relative to the room while the robot bends or squats. This
can label the base/torso motion that HERO measured with OptiTrack. Multiple targets
are needed because the paper reports loss of visual feedback under whole-body
motion and a narrow useful field of view.

## Validation and acceptance criteria

Do not accept a solution based only on its training reprojection error.

- Split samples by robot pose into calibration and held-out sets.
- Report median, 90th percentile, and worst-case corner reprojection error.
- Report pose-loop translation/orientation error on held-out poses.
- Repeat the complete capture on a different day and after physically disturbing
  and reseating the target mount.
- Compare left-arm and right-arm estimates of the same camera extrinsic.
- Check depth/color consistency by placing the calibrated cube at known distances.
- Target sub-centimeter held-out translation consistency for manipulation; the
  paper's grasp-close threshold was 1.5 cm, leaving little budget for extrinsic,
  depth, target, FK, and controller errors combined.

## Immediate implementation boundary

The prototype now implements the offline-first recorder, measured-pose replay,
immutable capture, exact URDF FK, raw-corner optimization, pose-group holdout,
residual reporting, and repeated-anchor check described above. The hardware
boundary remains staged: subscriber-only teaching first, then weight-zero,
measured hold, one validated round trip, and finally the approved capture plan.
