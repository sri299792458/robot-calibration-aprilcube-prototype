# Running Notes

## 2026-08-03 — Left-palm AprilCube configuration completed

### Decisions

- The existing 40 mm AprilCube is attached directly to the rigid left rubber-
  hand palm with the purchased double-sided mounting tape, approximately as
  shown in the reference photograph. There is no printed carrier, skirt,
  clamp, or fastener assembly and no physical-measurement gate.
- The cube-to-hand transform remains a free calibration variable. Its nominal
  placement is used only for initialization, visualization, and conservative
  collision checking; the visible tag-face orientation need not be exact.
- The collision model uses a 60 x 58 x 60 mm axis-aligned box centered at
  `[40, -34, 0]` mm in `left_rubber_hand`. This contains the 40 mm cube and
  numerical allowance for the tape footprint. The tape is not separately
  rendered.

### Implemented

- Made the calibration carrier an explicit `calibration_arm` setting throughout
  pose recording, schema validation, readiness gating, replay, collision
  validation, dataset construction, solving, residual reporting, synthetic
  checks, and both CLIs. The selected configuration moves left joints 15--21
  while continuously monitoring and holding right joints 22--28.
- Added continuous velocity and position-drift interlocks for both the moving
  calibration arm and the opposite held arm. Collision preflight now requires
  the AprilCube attachment to be present in an enabled collision pair before a
  command publisher can be constructed.
- Freeze `collision_pairs.yaml` alongside each immutable raw session and bind it
  to the transition-validation report by content hash.
- Use the rectified projection matrix `CameraInfo.P[:3, :3]` for detection,
  PnP, optimization, anchor checks, and synthetic generation. The RealSense
  intrinsics are still used; this selects the intrinsics belonging to the
  rectified color stream instead of the raw/distorted matrix.
- Keep repeated captures from leaking between train and holdout sets by
  splitting and bootstrapping whole pose groups. Added a linear warm start for
  the robust solve to avoid poor local minima.
- Added a reproducible left-palm geometry render and JSON record using the
  official Unitree hand STL and released AprilCube 3MF. Only the hand, cube,
  and conservative collision box are drawn.

### Verification

- Root prototype test suite: 102 passed.
- Ruff lint and format checks pass across source, tests, and the render tool;
  source and wheel builds also pass.
- Artifact inspection reports no blockers and binds the ready collision model
  to SHA-256 `bb8a9352b00dfa5f155f29a819887f4ebde8a3edc75eaa897ea63a2efb17c97f`.
- The real FCL model reports no collision at the configured zero pose; its
  minimum checked clearance is about 36.5 mm.
- A noisy 20-pose left-arm synthetic run recovered the 12-parameter solution
  at full Jacobian rank, with 0.279 px holdout radial RMS, 0.110 mm camera
  translation error, and 0.0123 degree camera rotation error.

## 2026-08-02 — Operator teaching, commissioning, and collection completed

### Added

- Added the read-only `teach-poses` application. It combines subscriber-only
  Unitree `LowState` with rectified ROS image/`CameraInfo`, displays the same
  red/yellow/green detector overlay, freezes a selected image, waits for the
  required post-image state bracket, and atomically records the measured pose.
  It creates no Unitree command publisher.
- Pose recording now verifies that the left arm is stationary and remains near
  the pose-set's frozen left hold, in addition to all existing right-arm gates.
- Added explicit weight-zero, measured-pose hold, and single-pose round-trip
  commissioning commands. The increasing-risk stages require exact typed
  acknowledgements before the publisher is constructed and share an exclusive
  command-owner lock.
- Added the production `collect-session` command: strict configuration/hash
  preflight, exact-home acquisition, explicit passed edges, optional per-move
  terminal confirmation, rectified stationary bursts, return home, terminal
  weight zero, and immutable session finalization.
- Added a synchronized 250 Hz executor driver. Detection, preview rendering,
  ROS spinning, PNG encoding, and manifest fsync cannot starve the command tick.
- Session plans may intentionally revisit an anchor pose. Added a post-solve,
  joint-compensated repeated-anchor report that reconstructs a hand-to-target
  transform for each anchor capture and fails configurable pairwise translation
  or rotation stability thresholds.
- Added example directed-edge and repeated-anchor session-plan YAML files and a
  complete command-by-command hardware runbook in the README.

### Verification

- Root prototype test suite: 94 passed.
- Ruff lint and format checks pass across 64 files.
- Source and wheel builds pass. The wheel contains all offline, ROS boundary,
  Unitree transport, operator, solver, report, and bundled JSON Schema modules.
- Hardware was deliberately not contacted. `config/collision_pairs.yaml` still
  blocks motion approval until the installed cube envelope pose is measured.

### Pinned SDK installation check

- A direct editable install correctly exposed that Unitree's pinned
  `cyclonedds==0.10.2` build cannot discover ROS Jazzy's CycloneDDS prefix by
  itself. Added a local prefix shim using the installed ROS headers, binaries,
  and libraries, matching the mechanism in the pinned G1Pilot setup.
- Added an idempotent hardware dependency installer and a hardware command
  wrapper that exports the ROS/CycloneDDS runtime paths before Python starts.
- Verified the real pinned SDK creates the expected 35-slot HG `LowCmd`, leaves
  blend slot 29 at zero by default, loads the correct HG LowState/LowCmd types,
  and computes a CRC successfully. No DDS domain/channel was initialized during
  this message/CRC check.

## 2026-08-02 — Live burst construction and offline workflow CLI implemented

### Added

- Made the complete state buffer thread-safe for concurrent DDS callback writes
  and operator-loop reads. The Unitree observer can now forward every valid
  receipt-stamped state into that buffer; malformed samples are never forwarded.
- Added a live burst source that consumes only new rectified frames, re-runs the
  stateless detector, applies red/yellow/green visual gates, reconstructs a
  centered stationary state window, verifies image/state pairing, and emits the
  exact raw `CaptureFrameInput` consumed by the immutable session store.
- Added CLI commands for offline artifact inspection, subscriber-only hardware
  inspection, pose-set initialization/summary/undo, explicit directed transition
  validation, raw-session verification, dataset construction, real dataset
  solving, and known-truth synthetic recovery.
- The solve command initializes the camera from the official URDF optical frame
  and the target from one rectified PnP frame, then still fits all twelve
  parameters to all training corners and exports held-out residual diagnostics.

### Verification

- Root prototype test suite: 88 passed after this slice.
- A CLI synthetic run with 20 poses and 0.2 px injected noise recovered both
  transforms with rank 12, about 0.30 px holdout radial RMS, 0.10 mm camera
  translation error, and 0.011 degree camera rotation error, and wrote all five
  expected report artifacts.
- `inspect-artifacts` correctly returns non-zero and names the intentionally
  unmeasured AprilCube collision-envelope pose as the hardware blocker.

## 2026-08-02 — Full fake capture/replay lifecycle implemented

### Added

- Acquisition can now be bound to a named, validated home pose. Before the first
  publish it verifies measured left-arm hold and right-arm home errors against
  the configured settled tolerance. A known-home acquisition becomes capture-
  ready after its blend ramp; an unbound commissioning acquisition retains the
  previous anonymous hold behavior.
- Added a capture interlock adapter around the immutable session store. Raw
  writes occur only while the executor is in `CAPTURING`, and every success,
  rejection, or write failure returns the executor to a non-capture hold state.
- Added a one-shot scheduled frame source and a complete approved-session
  orchestrator. It acquires at home, captures the requested pose sequence,
  requires confirmation and a hash-bound passed edge for every move, returns
  home, sends the terminal blend-weight-zero release, then finalizes the store.
- Any orchestration error initiates the executor's emergency release path. A
  failed run is not finalized and therefore remains inspectable/recoverable.

### Verification

- Root prototype test suite: 84 passed.
- The fake end-to-end run captures home and two arm poses, returns home, and
  closes on a non-emergency zero-weight command. Tests also prove that a measured
  home mismatch publishes nothing and that refused move confirmation triggers a
  terminal emergency zero-weight command without finalizing the session.

## 2026-08-02 — Rectified ROS camera boundary implemented

### Added

- Added strict ROS `Image` conversion for `bgr8`, `rgb8`, and `mono8`, including
  row-stride handling, truncation checks, and owned immutable BGR arrays.
- Added ROS `CameraInfo` conversion into the frozen camera profile. Non-zero
  distortion is rejected, so an accidentally selected raw RealSense topic
  cannot silently enter a solver that assumes rectified pixels.
- Preserved both ROS header time and local monotonic receipt time. The latter is
  the clock used to pair images with the independently received Unitree state.
- Added a bounded, thread-safe rectified frame buffer and optional subscriptions
  owned by a caller-provided ROS 2 node. ROS packages are imported only when the
  subscriber is constructed.
- Added a complete named `JointState` adapter for integration tests or bridges;
  it refuses missing measured velocities. Native `rt/lowstate` remains the
  production source because it also provides `mode_machine` directly.

### Verification

- Root prototype test suite: 81 passed.
- Tests cover zero-distortion enforcement, ROS time conversion, padded rows,
  RGB/BGR and mono conversion, immutable bounded buffering, monotonic receipts,
  authoritative named-joint ordering, and missing velocity rejection.

## 2026-08-02 — Hardware integration references pinned

- Cloned the current official `unitreerobotics/xr_teleoperate` reference at
  revision `64ed45b4177e6297936940866df623b72621643a`.
- Cloned the exact `lnotspotl/unitree_sdk2_python` fork and revision used by the
  pinned G1Pilot checkout:
  `7c661d27f4ae064ffd0dd633fd9d5b518ef0b508`.
- Recorded both pins in `config/upstream_pins.yaml` and added an idempotent
  bootstrap script. Both remain independent, ignored Git worktrees.
- Confirmed from the official controller that the G1 uses `rt/lowstate`,
  `rt/arm_sdk`, 35-slot HG messages, 29 physical joints, the dual-arm indices
  15–28, blend weight in slot 29, measured `mode_machine`, and a CRC on every
  command. The prototype adapter deliberately does not inherit the official
  example's eager publishing or zero-target startup behavior.

## 2026-08-02 — Raiden added

- Cloned `TRI-ML/raiden` into `raiden/`.
  - Branch: `main`
  - Revision: `2353b1040c8ffb67158fc7059a8ed61b4c4e672e`
- Materialized the repository's four Git LFS stereo-model weight files.
- Initialized the `third_party/i2rt` submodule at revision
  `7b6d5016f05ca63f9ef0185b7143e63f2c7a5708`.
- Kept `raiden/` as an independent Git worktree, consistent with the other
  upstream repositories.

## 2026-08-02 — G1 pose data boundary implemented

### Added

- Encoded the complete authoritative mode-5 joint order and explicit left-arm
  indices 15–21 and right-arm indices 22–28.
- Added immutable, receipt-stamped 29-DoF robot-state records. Partial arrays,
  non-finite values, non-UTC timestamps, and invalid sequences fail closed.
- Added versioned pose-set and pose-record models. Every stored pose includes the
  measured full state, derived right-arm state, per-joint recording spread,
  visual-quality evidence, head witness acknowledgement, and preview path.
- Added a strict bundled JSON Schema and a canonical SHA-256 over every pose-set
  document. Loading detects schema drift or manual content corruption.
- Added an append/undo audit trail and validates that replaying the audit actions
  exactly reproduces the active pose order.
- Added atomic YAML persistence using a same-directory temporary file, file and
  directory `fsync`, `os.replace`, and a previous-version `.bak` file.

### Verification

- Root prototype test suite: 25 passed.
- Ruff lint and format checks pass.
- Tests cover exact joint ordering, named-state reordering, immutability,
  malformed state, pose/full-state disagreement, duplicate IDs, append/backup,
  undo, schema rejection, audit inconsistency, and content-hash tampering.

## 2026-08-02 — Read-only pose recording core implemented

### Added

- Added a strictly monotonic, bounded complete-state buffer and a centered
  stationary-window selector with bracketing samples at both edges.
- Added fail-closed recording readiness checks for mode 5, state freshness,
  sample count, continuous duration, state gaps, measured right-arm velocity,
  and measured right-arm position spread.
- Added local receipt-time camera/state pairing. An image must be bracketed by
  state observations, satisfy a maximum bracket span, and have a sufficiently
  close nearest measured state. The original camera header timestamp is retained
  separately for provenance but is not compared to unsynchronized LowState time.
- Added a read-only `PoseRecorder`. Green visual quality is accepted, yellow
  requires a non-empty recorded override reason, and red cannot be saved. The
  head-pitch witness mark must be acknowledged.
- Successful recording stores median full/right joint positions, per-right-joint
  peak-to-peak spread, visual metrics, state-readiness metrics, timestamp-pairing
  evidence, and the preview reference through the atomic pose store.
- Added the initial manual-recording thresholds to `config/hardware.yaml`.

### Verification

- Root prototype test suite: 40 passed.
- Tests exercise wrong mode, motion, drift, stale state, state gaps,
  non-monotonic timestamps, unbracketed/distant images, red/yellow/head gates,
  median pose extraction, evidence persistence, and prevention of writes after a
  failed assessment.

## 2026-08-02 — Deterministic safety executor implemented

### Added

- Added a narrow arm-transport protocol whose only mutation is a validated
  fourteen-joint command plus blend weight. Commands require finite exact-length
  arrays, a monotonic timestamp, and a weight in `[0, 1]`.
- Added a manual clock and deterministic fake mode-5 G1 transport. It simulates
  measured tracking, state loss, mode changes, command history, and shutdown
  without any DDS or hardware dependency.
- Added the explicit observing/acquiring/holding/moving/settling/ready/capturing/
  releasing/fault/stopped state machine.
- Acquisition seeds both arm targets from fresh measured state and publishes
  weight zero before ramping. The measured left arm is then held for the entire
  executor lifetime.
- Moves can only reference a pose in the immutable pose set. Every move requires
  operator confirmation plus an exact passed directed-edge approval whose pose-
  set and validation-report hashes match the artifacts loaded by the executor.
- Added per-tick joint-velocity limiting, measured coarse arrival, continuous
  position/velocity dwell, motion timeout, capture/motion interlocks, and
  control-loop overrun detection.
- Wrong mode, stale state, explicit emergency stop, or loop overrun enters a
  logged fault and ramps the blend weight to a terminal emergency weight-zero
  command. Clean release is only possible at an approved, measured-settled home
  pose and also ends with a terminal weight-zero command.
- Added an advisory process lock to prevent two copies of this calibration
  executor from owning the laptop command path.

### Verification

- Root prototype test suite: 53 passed.
- The fake end-to-end executor test performs acquisition, a bounded move,
  measured settling, capture, return home, and clean release. Additional tests
  cover incorrect/stale approvals, missing operator confirmation, capture/move
  interlocks, wrong robot mode, frozen state, loop overrun, emergency release,
  no-publish observation exit, motion-profile bounds, and exclusive ownership.

## 2026-08-02 — Immutable raw sessions and dataset builder implemented

### Added

- Added immutable rectified-camera metadata with exact stream dimensions,
  optical frame, camera name/serial, full `K/R/P`, zero-distortion enforcement,
  and a canonical camera-profile hash.
- Added canonical current-frame AprilCube correspondence serialization and
  hashing, including decoded corners, generated 3D geometry, and explicit
  duplicate/ignored/rejected metadata.
- Added a strict version-1 session manifest schema and content hash. A session
  freezes the exact pose set, hardware YAML, target JSON, and transition
  validation report and verifies the pose-set content hash before creation.
- Added crash-safe raw capture. Lossless PNGs and complete state-window JSON are
  written with exclusive atomic creation before the manifest advances. A
  failure between those operations is visible as an orphan and is never
  silently overwritten.
- Every frame rechecks reproducible image/state bracketing, stationary measured
  state, unchanged camera profile, current-frame detector validity, and visual
  quality. Accepted bursts deterministically select the real frame nearest the
  median corner vector among the most common visible-tag signature.
- Finalization marks all session files read-only. Accepted, rejected, retry,
  skipped, and aborted outcomes remain explicit in the manifest.
- Added a deterministic offline dataset builder. It verifies every raw frame,
  including non-selected burst frames; re-runs the stateless detector; compares
  correspondence hashes; reconstructs receipt-time pairing; converts target
  millimetres to metres exactly once; and emits one content-hashed sample per
  accepted pose.
- Added a stable content-derived train/holdout split and dataset load/validation
  contract.

### Verification

- Root prototype test suite: 63 passed.
- A synthetic-camera integration test writes five accepted captures, recreates
  the session writer to simulate interruption, appends five more, finalizes,
  and rebuilds the same ten-sample dataset twice with an identical content hash.
- Additional tests cover camera/profile changes, red frames, finalize/append
  exclusion, invalid metadata with no raw writes, source/manifest tampering,
  orphan files, selected and non-selected raw corruption, metre conversion, and
  deterministic holdout membership.

## 2026-08-02 — Exact URDF FK and offline transition validation implemented

### Added

- Added a strict lightweight URDF parser for the exact official
  `g1_29dof_rev_1_0.urdf`. It resolves the tree, fixed/revolute/prismatic joint
  transforms, finite joint limits, relative meshes, primitive collision shapes,
  full-tree FK, and arbitrary ancestor-to-child chains.
- Added a numerical regression for the eight-joint
  `torso_link -> right_rubber_hand` chain, the nominal D435 transform, all seven
  right-arm limit records, and the exact official URDF SHA-256.
- Added `python-fcl` and selected-pair Trimesh/FCL clearance checking. The
  official collision meshes are used except for the rigid dummy hand, whose
  official visual mesh is an explicit fallback because that URDF link has no
  collision element.
- Added configurable attached-box collision objects so the printed AprilCube
  envelope can be placed on `right_rubber_hand` once its installed pose is
  measured.
- Added directed transition validation. Every edge checks both endpoints and
  interpolation samples spaced by maximum joint increment, all 29 joint limits
  with margin, selected arm/body/leg collision distances, path length, and
  estimated duration.
- Added a content-hashed validation report tied to the pose-set, official URDF,
  collision configuration, reference full-body state, validator version, and
  every directed edge. It produces the exact `TransitionApproval` consumed by
  the executor; stale or failed hashes remain unusable.
- Session creation now parses the frozen validation report and refuses a failed
  report or one belonging to another pose set/URDF.

### Deliberate pre-hardware block

`config/collision_pairs.yaml` is intentionally `hardware_ready: false`. The
installed cube-to-hand pose cannot be obtained from software or the photograph.
After the cube is taped in its final position, measure a conservative cube
envelope pose on `right_rubber_hand`, add its body collision pairs, inspect the
result, and only then set the flag true. Until that is done, every production
path report fails and cannot authorize the executor.

### Verification

- Root prototype test suite: 69 passed.
- Real FCL reports over 30 mm minimum selected-pair clearance at the nominal G1
  zero-arm configuration and reports negative signed distance for a deliberately
  overlapping box/sphere URDF.
- Tests cover FK/limits, interpolation count, passed approval generation,
  report round-trip hashing, joint-limit margin rejection, stale URDF rejection,
  and the unmeasured-AprilCube hardware block.

## 2026-08-02 — Twelve-parameter calibration and residual reporting implemented

### Added

- Added one explicit transform convention: parent-from-child homogeneous
  matrices and six-vectors ordered as translation XYZ in metres followed by a
  rotation vector in radians.
- Added the extrinsics-only nonlinear solver for the intended twelve unknowns:
  `torso_T_color_camera` and `right_rubber_hand_T_aprilcube`. Each residual uses
  the sample's complete measured mode-5 state, exact URDF FK, rectified
  `CameraInfo`, target points in metres, and raw ordered image corners.
- Added configurable robust SciPy least squares, positive-depth penalties,
  optimization status, numerical Jacobian singular values/rank/condition,
  approximate parameter standard deviations, and fail-closed degeneracy
  rejection.
- Added nominal `d435_link -> color optical` initialization and a single-frame
  rectified PnP initializer for the unknown hand-to-target transform. PnP is only
  an initial value; the batch solve still minimizes every raw corner.
- Added stable content-derived training/holdout orchestration and pose-level
  bootstrap resampling. Dataset/solver URDF hashes must match.
- Added per-corner signed U/V and radial residuals, depth, capture/pose/frame,
  tag/corner identity, per-capture RMS, per-tag RMS, and correlations between
  pose-mean residual structure and all seven measured right-arm joints.
- Added versioned, schema-validated, content-hashed result export containing
  `result.json`, `calibrated_extrinsics.yaml`, `corner_residuals.csv`,
  `provenance.json`, and a concise Markdown report.
- The report explicitly requires examining train/holdout residual structure and
  joint correlations before proposing any joint offset. Correlation is a reason
  to investigate physical/model error, not permission to add parameters.

### Verification

- Root prototype test suite: 73 passed.
- A noiseless 40-pose synthetic G1 dataset recovers both six-DoF transforms to
  numerical precision from perturbed initial values with Jacobian rank 12.
- Repeating one pose is rejected at rank 6/12.
- A noisy 30-pose synthetic run performs deterministic holdout evaluation and
  three successful bootstrap solves, recovers each translation within 2 mm and
  each rotation within 0.3 degrees, stays below 0.8 px holdout radial RMS, emits
  all run artifacts, and detects manual result tampering.

## 2026-08-01 — Project setup

### Goal

Create a lightweight prototype workspace for quickly experimenting with robot
calibration and AprilCube-based perception.

### Added

- Cloned `mikeferguson/robot_calibration` into `robot_calibration/`.
  - Branch: `ros2`
  - Revision: `db991b040d1dc28af09d8865fc72f09720e12b73`
- Cloned `sri299792458/aprilcube` into `aprilcube/`.
  - Branch: `main`
  - Revision: `fc18d50c8bbaadc9646dfd0aa5fcd2404a9868c5`
- Downloaded arXiv paper `2602.16705` into `papers/2602.16705.pdf`.
  - Title: *HERO: Learning Humanoid End-Effector Control for Visual Whole-Body
    Open-Vocabulary Object Grasping*
  - SHA-256: `4cf2da3f6f1697edc747465beff918082f7ac6a6ae4ef1089a0767489f5b8d99`

### Workspace conventions

- Keep the upstream repositories as independent Git worktrees so experiments can
  be committed to the appropriate codebase without mixing histories.
- Track workspace-level research material and decisions in this parent repository.
- Favor small, reversible experiments and document commands, observations, and
  decisions here as the prototype evolves.

### Current state

- Initial assets are present and verified.
- No source changes have been made to either upstream repository.

## 2026-08-01 — Camera calibration problem formulation

### Added

- Cloned the official `unitreerobotics/unitree_ros` repository into
  `unitree_ros/`.
  - Branch: `master`
  - Revision: `f3772ce54c56ef2d34c6aee8100bc768896c7d19`
- Reviewed HERO sections III-B, V-C, and appendix B.1-B.2, focusing on G1
  analytical-FK error, base odometry, and MOCAP-assisted camera calibration.
- Inspected the official G1 URDF variants and nominal D435/D455 camera chains.
- Inspected `robot_calibration`'s 2D camera/3D chain reprojection optimizer and
  AprilCube's raw tag-corner geometry/detections.
- Wrote `docs/problem_formulation.md` with the measurement equations,
  observability constraints, staged AprilTag approach, and validation criteria.

### Key decision

Use `torso_link` as the provisional calibration reference because it is rigidly
connected to the standard G1 camera mount. Treat the URDF camera transform only
as an optimizer initial value. Confirm this after identifying the physical G1
variant and camera/head hardware.

### Main risk

A hand-mounted AprilCube plus nominal FK can produce a low reprojection error by
absorbing G1 kinematic error into the camera extrinsic. The prototype will jointly
estimate a limited set of parameters and use an independently registered target
fixture for validation when possible.

### Information needed from the physical robot

- Hand configuration.
- RealSense model and serial number.
- ROS 2 image, `CameraInfo`, and joint-state topic names.

## 2026-08-01 — Physical G1 variant confirmed

### Confirmed

- `mode_machine = 5`, selecting the 29-DOF revision-1.0 G1 family.
- Camera pitch is manually adjustable and unsensed.
- Base URDF: `unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf`.

### Calibration consequence

Treat the camera as a fixed six-DoF child of `torso_link` only after physically
locking it at the intended operating angle. Any manual pitch change invalidates
the complete extrinsic—not only its pitch component—and requires a fresh
calibration.

### Hardware decision

Physically fix the head pitch at one operating angle and apply a witness mark.
Check the mark before each calibration/capture run. If it shifts or the mount is
loosened, discard the saved extrinsic and recalibrate.

## 2026-08-01 — Calibration target carrier

### Confirmed

- Both Dex3 and dummy rubber hands are available.
- Use the rubber-hand kinematic configuration for the first calibration fixture.
- The physical dummy hand is rigid, despite the URDF link name
  `right_rubber_hand`; the earlier concern about rubber deformation does not
  apply.
- The mode-5 URDF attaches `right_rubber_hand` to `right_wrist_yaw_link` with a
  fixed `right_hand_palm_joint` at `[0.0415, -0.003, 0]` meters.

### Mount recommendation

Prefer a keyed two-piece clamshell registered directly against the rigid dummy
hand. Use hard locating pads, an end stop, an asymmetric anti-rotation feature,
and two screw clamps without compliant liners. A replacement wrist flange is the
fallback if clamshell removal/reinstallation is insufficiently repeatable. Do not
use the Dex3 grasp, tape, or hook-and-loop material as the metrology interface.

### Next dependency

Obtain photographs and caliper measurements of two candidate clamp sections on
the physical dummy hand. Use the official STL for the nominal cradle surface and
print a small fit coupon before the complete mount. See
`docs/target_mount_design.md`.

## 2026-08-01 — Fixture fabrication constraint

### Confirmed

- Custom parts can only be made by FDM 3D printing; machining is unavailable.
- Standard metric screws, washers, and nuts can be purchased.

### Design response

Use three printed custom parts: a primary conformal cradle with integrated target
stalk, a clamp cap, and the AprilCube. Join them with the M4/M3 fastener set below,
captured by printed counterbores and nut traps. Validate hand-fit clearance with a
short printed cradle slice before printing the full fixture.

### Fastener selection

Revised the preliminary two-bolt idea to a fully specified six-bolt assembly:

- four M4 x 20 mm ISO 4762 socket-head screws, four M4 DIN 985 nyloc nuts, and
  four M4 flat washers for two clamping flanges per side; and
- two M3 x 16 mm ISO 4762 socket-head screws, two M3 DIN 985 nyloc nuts, and two
  M3 flat washers for the keyed AprilCube-to-stalk connection.

All nuts sit in printed captive pockets; no heat-set inserts or machining are
required. Print a tolerance coupon before the fixture.

## 2026-08-01 — Mount concept render

### Added

- Created a project-local `uv` environment with locked dependencies for Trimesh,
  Manifold, NumPy, SciPy, Matplotlib, and Pillow.
- Added `tools/render_mount_concept.py`.
- Rendered assembled and exploded views to
  `renders/dummy_hand_aprilcube_mount_concept.png` using the official Unitree
  dummy-hand STL at URDF scale.

### Interpretation

The render communicates the intended assembly: an orange printed cradle and
integrated short target stalk, a blue printed clamp cap, four M4 side-flange
fasteners, and a two-M3 keyed AprilCube attachment. The current oval cradle and
target offset are provisional and must not be sliced as production CAD.

## 2026-08-01 — Physical dummy-hand fit check

### Observation

The provided physical photo shows that the dummy palm is rigid, broad, and flat,
and that the existing AprilCube has a stable full-face seating area near the
wrist. This removes the need for a printed fixture in the first experiment.

### Revised MVP

Mount the cube directly to the palm using thin, high-tack double-sided film tape.
Avoid Velcro and foam tape because their compliant layers permit pose changes
under gravity and wrist rotation. The covered bottom tag is not needed.

The hand-to-cube transform is optimized as a free parameter, so its exact value
and reinstall repeatability are unnecessary. Rigidity during one dataset is the
requirement. Run a wrist-orientation return test against stationary detector noise
before capture. Retain the printed clamshell only as a fallback if tape moves.

## 2026-08-01 — Walmart tape verification

The inexpensive Scotch permanent office tape is advertised for paper and crafts
and has no relevant mounting-load specification, so it is not an acceptable
robot target-mount recommendation. The approximately $3.56 Scotch Indoor
Mounting Tape 110H is load-rated and easily supports a 40 g cube, but 3M lists
its carrier as flexible polyethylene foam. It is therefore suitable for fall
prevention but not preferred as the calibration interface because its elastic
pose change is unspecified.

The cheapest Walmart candidate found with a thin, non-foam carrier and
manufacturer technical data is T-Rex Double-Sided Super Glue Tape, 0.5 in x
7.5 yd, approximately $5.43 at the time of checking. Its technical data lists a
0.19 mm polyester-film carrier, acrylic adhesive, plastic as an intended surface,
and representative peel adhesion to steel of 180 oz/in of width. Use three
adjacent 40 mm strips to cover 38.1 x 40 mm of the cube base. Store price and
availability are location-dependent.

A 40 g target produces 0.392 N at rest and 1.96 N under a conservative 5 g total
load. Across the 1,524 mm^2 taped area, the corresponding average loads are only
0.257 kPa and 1.29 kPa. These figures establish a comfortable bulk-strength
margin, but do not establish adhesion to the unknown dummy-hand polymer or bound
peel under the cube's lever arm. The wrist-orientation return test remains
mandatory; passing means no pose change beyond stationary detector noise.

## 2026-08-02 — Three-repository review and implementation plan

### Reviewed revisions

- `robot_calibration` at `db991b040d1dc28af09d8865fc72f09720e12b73`.
- `aprilcube` at `fc18d50c8bbaadc9646dfd0aa5fcd2404a9868c5`.
- `raiden` at `2353b1040c8ffb67158fc7059a8ed61b4c4e672e`,
  including the initialized `i2rt` submodule.
- Used `unitree_ros` at `f3772ce54c56ef2d34c6aee8100bc768896c7d19`
  only to validate the mode-5 G1 frame and joint chain.

### Main technical conclusions

- `robot_calibration` already has the correct raw-corner batch residual:
  `Chain3dToCamera2d`. Its free-frame mechanism can estimate both the
  `d435_joint` correction and the unknown `right_rubber_hand -> AprilCube`
  transform.
- Its 2D camera model ignores distortion, uses plain squared loss, and does not
  export final per-corner residuals or observability metrics. Use rectified color
  pixels for the MVP and add residual/Jacobian reporting before considering
  kinematic offsets.
- The built-in capture path is not sufficiently synchronized for this use case;
  create a small G1-specific recorder that pairs each stationary image with the
  nearest complete `JointState` by timestamp.
- AprilCube's default printed target is a 40 mm cube with six 30 mm OpenCV ArUco
  `4x4_100` markers. Its 3D geometry is in millimetres and its detector corners
  are ordered TL/TR/BR/BL. The ROS bridge must convert geometry to metres once.
- AprilCube's tracking stack can use temporal filtering, prediction, rejected
  quad recovery, and optical flow. Calibration observations must instead come
  from a new stateless current-frame correspondence API.
- Raiden's solver is tied to YAM kinematics and a fixed ChArUco board, so it is
  not the G1 solution. Its RealSense intrinsics/bag code, clock cautions, Rerun
  pattern, and versioned JSON schema are useful references. Keep it out of the
  first ROS runtime dependency graph.
- A hand-mounted target without an independent reference yields a
  kinematics-conditioned camera extrinsic. Held-out residuals establish internal
  consistency but cannot prove that nominal G1 FK bias was not absorbed into the
  camera transform.

### Physical protocol corrections

The earlier tape recommendation and wrist-orientation test are superseded:

- use the Scotch foam mounting tape already purchased; do not buy replacement
  tape before testing the installed target;
- bulk holding strength is ample for the approximately 40 g cube;
- foam matters only if the cube-to-hand transform changes, so qualify it with
  stationary measurements and repeated anchor arm poses;
- keep wrist roll, pitch, and yaw fixed for the full dataset; and
- use slow shoulder pitch/roll/yaw and elbow motion to create pose diversity.

### Verification completed

- AprilCube generator/web tests: 14 passed.
- Default-cube synthetic detector: 48/48 viewpoints detected and passed with
  temporal filtering disabled.
- Python syntax compilation passed for AprilCube and Raiden sources.
- `robot_calibration_msgs` builds under ROS 2 Jazzy when CMake is directed to
  `/usr/bin/python3`. The main package configure is currently blocked by missing
  system dependencies: `camera_calibration_parsers`, Ceres, gflags, and protobuf
  development packages. This is an environment dependency gap, not a source
  failure.

### Plan

Added `docs/detailed_implementation_plan.md`, covering the repository audit,
data contract, software boundaries, pose design, synthetic recovery tests,
physical qualification, extrinsics-only solve, residual diagnostics,
observability gates, optional joint-offset criteria, independent validation,
deployment provenance, and risk register.

The next implementation milestone is a public stateless AprilCube
correspondence API, followed by a synthetic 12-parameter recovery test before
connecting to the physical G1.

## 2026-08-02 — G1Pilot added

- Cloned `sri299792458/g1pilot` into `g1pilot/`.
  - Branch: `dev`
  - Revision: `6b5af59b109e2ee687920fdf66ded6182725e945`
- The checkout has no configured Git submodules or Git LFS objects.
- Kept `g1pilot/` as an independent Git worktree, consistent with the other
  upstream repositories.

## 2026-08-02 — SPARK data-collection pose workflow reviewed

- Cloned `RPM-lab-UMN/spark-data-collection` into
  `spark-data-collection/`.
  - Branch: `main`
  - Revision: `be284c2f8138f383d260526f68613c7a28d364d4`
- The checkout has no configured Git submodules or Git LFS objects and is kept
  as an independent Git worktree.
- Its calibration workflow is a useful behavioral template rather than a
  reusable robot driver:
  - `record_calibration_poses.py` enables UR freedrive and records measured
    joint positions and TCP pose when the operator presses `r`;
  - `calibrate_rig.py` later replays each saved joint vector with UR `moveJ`,
    waits for stabilization, and captures camera observations; and
  - the UR implementation depends on RTDE APIs that the G1 does not provide.
- The equivalent G1 workflow is feasible by recording motor positions from
  `rt/lowstate` while the robot is in a verified manual-teaching/compliant mode,
  then replaying discrete joint vectors through one exclusive `rt/arm_sdk`
  owner. The official Unitree control layer identified below should own DDS;
  G1Pilot remains useful for offline limit/collision checks, not as the motor
  transport.
- For this calibration, save and replay the complete measured seven-joint
  right-arm vector. Replay only one operator-confirmed pose at a time,
  interpolate slowly, verify measured settling, and then trigger capture.
- Unitree's current official SDK example confirms that 29-DoF G1 arm positions
  are commanded over `rt/arm_sdk` while states arrive over `rt/lowstate`. The
  exact availability and behavior of the Explore app's manual Teaching mode is
  firmware-dependent and must be verified on the lab G1 before relying on it.

## 2026-08-02 — Full joint-space pose recording selected

- The earlier fixed-wrist collection rule is superseded. Wrist roll, pitch, and
  yaw may vary naturally between manually recorded configurations because their
  measured positions are saved and included in G1 FK for every capture.
- Replaying a recorded joint vector does not use IK. Each endpoint is reachable
  by construction, so there is no IK convergence or branch-selection failure.
- This does not make arbitrary transitions safe. A straight interpolation in
  joint space can collide even when both endpoints are safe. At minimum, sample
  and check each interpolated path against joint limits and the relevant
  arm-versus-body collision geometry, then execute it slowly under operator
  confirmation. Full MuJoCo simulation is optional rather than a prerequisite.
- Always pair an image with the measured joint state at image time rather than
  the recorded target. This captures finite tracking error and settling.
- Varying wrist orientation changes the gravity load on the taped cube. Qualify
  mechanical rigidity using repeated returns to the exact same complete
  seven-joint anchor vector; replace the mount only if measured drift exceeds
  stationary detector scatter.

## 2026-08-02 — Existing G1 trajectory implementations audited

The earlier conclusion that the G1 lacked a reusable `moveJ`-like layer was too
strong. The public ecosystem contains three distinct levels of implementation:

- Unitree's official `unitree_ros2` G1 arm example contains a private `MoveTo`
  helper. It publishes `/arm_sdk`, subscribes `/lowstate`, interpolates at
  50 Hz, applies a 0.5 rad/s per-joint clamp, and ramps out the arm blend
  weight. It is a hard-coded demonstration, not a callable service/action; it
  does not wait for arbitrary goals to settle from measured feedback.
- Unitree's official `xr_teleoperate` repository provides the reusable
  `G1_29_ArmController`. In motion mode it owns `rt/arm_sdk`, reads
  `rt/lowstate`, streams at 250 Hz, clips requested motion using measured arm
  positions, and accepts arbitrary 14-joint dual-arm targets through
  `ctrl_dual_arm`. This is the preferred transport/control base for the
  calibration prototype.
- Unitree's official `unitree_lerobot` repository already replays recorded G1
  datasets by sending each recorded arm frame through that controller at the
  dataset frequency. This proves official record-and-replay support, but its
  dense LeRobot episode loop is not the desired discrete
  move-settle-capture protocol.

No official Unitree repository currently exposes arbitrary G1 arm goals through
ROS 2 `control_msgs/action/FollowJointTrajectory`. The standard ROS 2
`joint_trajectory_controller` supplies interpolation, goal tolerances, feedback,
and a blocking action result, but it requires a G1 `ros2_control` hardware
interface that Unitree does not publish.

Community alternatives exist but are not the prototype baseline:

- MyBotShop's G1 integration documents 7-DoF left, right, and dual-arm
  `FollowJointTrajectory` actions, but it is a vendor integration rather than
  an official/open Unitree driver.
- `fiveages-sim/unitree-ros2-control` and `Adyansh04/grove-g1` bridge G1 to
  `ros2_control`; both are very new. The former contains questionable arm-index
  handling in its current `g1_arm_sdk` path. The latter is thoughtfully tested
  in simulation and uses only `/arm_sdk`, but was created in July 2026, has no
  repository license, and explicitly leaves real-hardware validation pending.
- A recent whole-body community workspace exposes the standard trajectory
  controller through raw `/lowcmd`; that would replace the onboard balance
  controller and is inappropriate for this standing calibration workflow.

### Revised implementation choice

Vendor `G1_29_ArmController` (or a minimal pinned copy of it) and add only the
calibration-specific wrapper that Unitree does not provide:

1. preserve the left-arm target at its measured value while moving the right
   arm to a saved seven-joint vector;
2. use conservative configurable velocity/timeout limits;
3. reject non-finite and out-of-URDF-limit targets and prevalidated unsafe
   transitions;
4. wait on measured right-arm position and velocity tolerances from
   `rt/lowstate`, then require a dwell interval; and
5. trigger image/joint capture only after settling.

This reproduces the useful blocking behavior of UR `moveJ` without reimplementing
Unitree DDS ownership or introducing a full ROS 2 control stack for the MVP.

## 2026-08-02 — Lightweight calibration package selected over G1Pilot runtime

Build the recorder/replayer as a small project-local package rather than adding
the G1Pilot manipulation stack to the calibration runtime.

The checked-out G1Pilot `dev` arm path is an OpenSoT Cartesian controller: its
public goals are right/left `PoseStamped` hand poses, it solves IK continuously,
and optional collision avoidance pulls in XBot, OpenSoT, FCL, and Python binding
builds. It does not expose a discrete measured joint-vector replay interface.
Using it as the executor would therefore add an unnecessary IK layer and a large
dependency surface to a joint-space record/replay problem.

Keep G1Pilot as a development-time validation tool only:

- reuse its mode-5 collision geometry and selected arm/body collision pairs;
- use lightweight kinematic path sampling for transition checks, with MuJoCo
  available only when dynamic simulation is useful;
- do not import G1Pilot modules into the hardware recorder/replayer; and
- never allow G1Pilot and the calibration executor to publish `rt/arm_sdk` at
  the same time.

The project-local package should have four narrow boundaries: an official
Unitree-controller adapter; versioned pose/session storage; a blocking
move-and-settle executor; and capture orchestration. Detection, calibration,
and collision-validation logic remain separate processes/packages. This keeps
the robot-facing code small enough to review and test while preserving a future
path to replace only the adapter with a standard ROS trajectory action.

## 2026-08-02 — Full implementation plan revised

Reworked `docs/detailed_implementation_plan.md` into the executable plan for the
lightweight package. It now defines the package layout, command-line surface,
transport protocol, motion/capture state machine, safety invariants, pose/session
and result schemas, immutable raw-versus-derived data split, transition
validation contract, hardware commissioning sequence, verification matrix,
risk register, and milestones M0 through M12.

One additional controller-safety finding changes how the official code is
reused. `xr_teleoperate` is Apache-2.0, but its `G1_29_ArmController` must not be
instantiated unchanged for calibration: motion mode starts its publisher thread
with a zero dual-arm target and weight one, and it has no explicit joined-thread
shutdown. Implement a small attributed G1-only derivative that waits for fresh
mode-5 state, seeds both arms from measured q, starts at weight zero, ramps under
explicit acquisition, checks state freshness, and provides clean/emergency
release.

Updated `config/hardware.yaml` with the initial commissioning configuration:
250 Hz commands, 0.2 rad/s joint speed, 0.02 rad position tolerance, 0.03 rad/s
velocity tolerance, 0.75 s dwell, and 0.1 s LowState timeout. These are starting
values requiring supervised hardware validation, not final performance claims.
The configuration and problem formulation now consistently record and replay
all seven measured right-arm joints rather than fixing the wrist.

## 2026-08-02 — Full versus lock-waist revision-1.0 URDF comparison

Compared the official `g1_29dof_rev_1_0.urdf` and
`g1_29dof_lock_waist_rev_1_0.urdf` directly and as parsed XML. They have the
same 40 links, 39 joints, link/joint name sets, origins, axes, limits, inertials,
meshes, collision geometry, right-arm chain, dummy-hand transform, and D435
transform. The only changes are:

- robot name;
- `waist_roll_joint`: revolute -> fixed;
- `waist_pitch_joint`: revolute -> fixed; and
- removal of the non-standard `dont_collapse="true"` attribute from the two
  fixed hand-palm joints.

Despite its name, the lock-waist file leaves `waist_yaw_joint` revolute and has
27 movable joints instead of 29. With `torso_link` as the calibration root, the
camera-to-right-hand FK is identical in both files. Retain the full 29-DoF file
for the project because it matches `LowState`, preserves measured waist
roll/pitch for pelvis/world TF and whole-body collision checks, and retains the
hand-frame preservation hint for converters that honor it.

## 2026-08-02 — AprilCube mounting-face convention

The face taped to the dummy hand does not need to have a particular ID or a
pre-measured orientation. The calibration model estimates the complete rigid
six-DoF transform `right_rubber_hand_T_aprilcube` together with
`torso_T_color_camera`, so choosing another cube face or rotating the whole
cube on the palm changes the first unknown transform rather than invalidating
the method. The attachment must remain rigid and unchanged throughout a
dataset; reattaching the cube creates a new hand-to-cube transform and requires
a new calibration run.

For `models/dex3_safe_cube/config.json`, the cube origin is at its center and the
40 mm cube faces are assigned as follows: tag 0 is `+X`, tag 1 is `-X`, tag 2
is `+Y`, tag 3 is `-Y`, tag 4 is `+Z`, and tag 5 is `-Z`. These are cube-local
directions, not robot, hand, or camera directions. The corresponding face
planes are at `X/Y/Z = +/-20 mm`, and each 30 mm tag's detected corners are
mapped in the generated TL/TR/BR/BL order. The ID-to-face assignment and the
printed rotation of each tag must therefore stay consistent with this exact
config file even though the whole cube may be mounted in any orientation.

Running the AprilCube detector on the supplied mounting photo decoded tag 5
(`-Z`) on top and tag 0 (`+X`) on the front-facing side. Therefore the hidden
face against the hand is tag 4 (`+Z`). This is a good usable mounting and does
not need to be changed. A hidden tag simply supplies no observations. Keep the
other faces unobstructed and collect many views in which two adjacent faces are
visible when practical; this improves corner geometry and rejection of bad
detections, but seeing all six tags is neither required nor possible in one
image.

## 2026-08-02 — Physical AprilCube provenance confirmed from demo workspace

Checked `/home/srinivas/Desktop/demo/third_party/aprilcube` after the user
identified it as the print workspace. Its session log records that the first
sharp `models/1x1x1_30_cube` candidate was superseded and that the physical
print completed on 2026-07-14 from `models/dex3_safe_cube/cube.3mf`. The actual
target is the compact 40 x 40 x 40 mm dual-color PLA release with a 3 mm tangent
edge/corner radius, six 30 mm `4x4_100` markers, and IDs 0 through 5.

Both Desktop checkouts are at AprilCube commit
`fc18d50c8bbaadc9646dfd0aa5fcd2404a9868c5`. The demo and calibration copies
of the rounded config and 3MF are byte-identical. The sharp and rounded configs
produce identical per-ID 3D tag-corner maps; rounding affects only the outer
perimeter, not the planar marker coordinates. Re-running detection on the
mounting photo with `models/dex3_safe_cube/config.json` again decoded IDs 5 and
0 on faces `-Z` and `+X`, confirming that hidden ID 4 / `+Z` and all prior
mounting conclusions remain correct. Use the rounded config in all calibration
session manifests for exact physical-artifact provenance.

## 2026-08-02 — Live laptop preview and pose acceptance workflow

Pose teaching and replay capture will both show the rectified RealSense color
stream on the laptop with current-frame AprilCube IDs, corner overlays, and
quality metrics. The calibration detector must be stateless: no optical flow,
prediction, rejected-quad recovery, or temporal filter may make a missing tag
appear valid for capture.

The operator display separates three judgments. Capture readiness checks fresh
camera/state data and the measured settle window. Visual quality checks tag
size, boundary margin, duplicate IDs, visible faces, and multi-face PnP only as
a diagnostic. Dataset value checks whether the view adds image-region, depth,
orientation, or FK diversity relative to saved poses. Red candidates cannot be
saved, yellow candidates require an explicit reason/confirmation, and green
candidates are recommended. Green calibration quality is not a motion-safety
approval; every recorded joint configuration and directed replay transition
still requires the offline collision/path report and operator clearance.

During manual teaching, Space stores the stationary median of measured joints
plus the preview and quality metadata. During replay, the same gates are rerun
after measured settling, a seven-frame stationary burst is retained, and only
one best/median-corner calibration observation is emitted per pose. Aim for
60–80 accepted diverse configurations rather than 60–80 consecutive frames;
record additional candidates as needed because weak, redundant, or unsafe poses
will be rejected. Initial tunable thresholds are now recorded in
`config/capture_quality.yaml`.

## 2026-08-02 — First runnable implementation: stateless visual preview

Implemented the first hardware-independent vertical slice. The AprilCube fork
now exposes `CorrespondenceDetector`, immutable per-tag 2D/3D observations,
explicit duplicate/ignored/rejected metadata, and a stateless PnP diagnostic.
It uses only markers decoded in the current image and deliberately has no
optical flow, temporal prediction, rejected-quad recovery, or prior-pose input.
This work is committed independently in the AprilCube checkout as
`80ed7c72ed00aef6dc70f77d8169a199e9a612cd`.

The root project is now an installable `uv`/Hatch package with a `g1-calib`
entry point. `g1-calib preview` accepts an image, video, or local camera, loads
the exact `models/dex3_safe_cube/config.json`, applies the configured visual
hard/preferred gates, tracks in-memory coverage novelty, and renders tag IDs,
corner order, cube axes, metrics, reasons, and a red/yellow/green side panel.
The panel explicitly says `VISUAL QUALITY ONLY` and reports robot
readiness/collision status as not connected so it cannot be mistaken for
authorization to save or replay a G1 joint configuration.

Verified the headless preview on the supplied physical mounting photograph with
temporary bench intrinsics (`fx=fy=1000 px`, centered principal point). It
decoded tags 0 and 5 on faces `+X` and `-Z`, found a 101.89 px minimum tag side,
221.56 px minimum image margin, and 1.11 px multi-face PnP reprojection error,
and graded the visual view green. The PnP values validate the preview path only;
they are not calibration measurements because the photograph is not a
rectified RealSense frame with its exact `CameraInfo`.

Verification completed:

- root package: 8/8 pytest tests passed;
- AprilCube correspondence plus existing generator/web suite: 18/18 passed;
- Ruff lint and format checks passed for all new root and AprilCube files;
- both packages compiled successfully; and
- the installed `g1-calib` and `g1-calib preview` help surfaces execute.

The next slice remains hardware-facing: discover and verify the rectified
RealSense image/`CameraInfo` topics and add receipt-stamped mode-5 `LowState`
readiness/settling to the same report before pose storage is enabled.

## 2026-08-02 — Python environment aligned with ROS Jazzy ABI

The first `uv sync` selected the active Conda Python 3.13 and NumPy 2.x. A
read-only environment probe then demonstrated that this is incompatible with
the installed ROS Jazzy extensions: `rclpy` is built for Python 3.12 and
`cv_bridge` is built against NumPy 1.x. Continuing with that environment would
make the standalone preview work but break the later ROS camera/state adapter.

Constrained the project to Python `>=3.12,<3.13`, NumPy `>=1.26,<2`, and a
compatible SciPy range, then recreated `.venv` explicitly from
`/usr/bin/python3.12`. The resolved environment uses Python 3.12.3, NumPy
1.26.4, OpenCV contrib 4.11.0, and SciPy 1.14.1. After sourcing
`/opt/ros/jazzy/setup.bash`, the same `uv` environment successfully imports
`rclpy`, `cv_bridge`, and `sensor_msgs`; all eight root tests and the physical
photo preview still pass.

The laptop currently has no Unitree ROS message package or
`unitree_sdk2py` checkout installed in this project environment, and no
RealSense device was returned by `rs-enumerate-devices`. Those are expected
hardware-integration prerequisites for the next slice, not reasons to weaken
the tested Python/ROS boundary.
