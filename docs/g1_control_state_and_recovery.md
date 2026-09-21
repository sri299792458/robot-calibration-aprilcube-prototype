# G1 control state and debug-lowcmd recovery

This is the operator runbook for the control behavior physically observed on
this G1. Keep the distinction between the active **motion service**, the
locomotion **FSM ID**, and LowState `mode_machine`; they are different state.

## State vocabulary

| State | Meaning on this robot |
|---|---|
| Motion service `{"form": "0", "name": "ai"}` | The normal Unitree high-level controller is active. The wireless remote and G1 locomotion client can request FSM transitions. |
| No active motion service | Unitree `ReleaseMode()` has made direct debug `rt/lowcmd` ownership available. This is not an FSM such as Damp or Seated. |
| FSM 0 | Zero-torque state observed when the `ai` service was restored after a failed debug session. |
| FSM 1 | Damp. |
| FSM 3 | Seated. |
| FSM 4 | Regular/Ready on this robot. |
| LowState `mode_machine=5` | The 29-DoF G1 machine/joint layout. It is not locomotion FSM 5. |

Selecting the `ai` motion service does **not** select Damp, Seated, Ready, or
Walking. It only makes the high-level controller available. An explicit FSM
request and a read-back are still required.

## Why the remote can become unresponsive

The seated tabletop controller follows Unitree's debug architecture:

1. verify the robot starts in seated FSM 3;
2. call MotionSwitcher `ReleaseMode()` to deactivate `ai`;
3. publish a complete measured-state 29-joint command on `rt/lowcmd`; and
4. before closing that publisher, restore `ai` and explicitly verify the
   requested terminal FSM.

While `ai` is released, the wireless remote has no active high-level controller
to receive its FSM requests. The G1 locomotion client's read-only `get_fsm_id`
may also time out. This does not by itself indicate a failed Ethernet link or a
dead remote.

If the lowcmd process exits without restoring `ai`, both of these can be true at
once:

- no laptop lowcmd writer remains; and
- no normal Unitree motion service is active.

The result looks like a zero-torque/debug state and the remote cannot select
another FSM until `ai` is restored.

## Hardware observation: 2026-08-10

The zero-displacement seated commissioning run established the following:

- PC2 verified initial seated FSM 3;
- the dynamic handoff passed;
- the laptop released the active service;
- the first complete measured-state `rt/lowcmd` packet was accepted; and
- no changing joint target was part of the test.

The run then failed because PC2 stopped receiving the laptop heartbeat. Its
fallback attempted to restore the motion service, but MotionSwitcher read-only
API 1001 (`CheckMode`) timed out three times. PC2 therefore could not proceed to
a verified Damp request. The lowcmd process exited with `ai` still released,
and the wireless remote could not change modes.

Restoring `ai` from the laptop with Unitree's official SDK succeeded and
reported:

```text
motion service restored: {'form': '0', 'name': 'ai'}
```

The robot then entered the observed zero-torque FSM 0. It did not automatically
enter Damp or return to Seated. The wireless remote became usable again.

This run proved zero-displacement debug takeover, but at that point it did
**not** commission normal seated restoration, heartbeat-loss zero-torque
recovery, or changing-target arm motion. The later lifecycle checks recorded
below closed the first two gaps.

A second zero-displacement run after the first software changes narrowed the
failure further:

- PC2's new read-only preflight succeeded while `ai` was active and reported
  `motion_switcher=ai` together with seated FSM 3;
- the heartbeat guard covered `ReleaseMode()` and completion of the first
  measured-state lowcmd write;
- the first packet again passed and no changing target was commanded;
- the heartbeat nevertheless expired during the subsequent hold; and
- once `ai` had been released, PC2 MotionSwitcher API 1001 timed out again on
  every fallback attempt.

This established an important asymmetry on the installed stack: PC2's ROS
MotionSwitcher path could answer `CheckMode` before service release but could
not be relied upon to answer after release. The laptop's raw Unitree SDK did
successfully verify the released state during takeover and later restored
`ai`. The watchdog therefore stopped using the PC2 ROS MotionSwitcher path for
debug recovery. It creates a persistent client from the pinned raw SDK runtime
before takeover and reuses that same client after release. This replacement was
installed and tested offline before the later physical lifecycle checks
commissioned its post-release behavior.

A third zero-displacement run used that private raw-SDK runtime. It established
four additional facts:

- PC2 received one ping and expired at 0.500258 seconds, while the laptop later
  reported a maximum send gap of 0.922032 seconds;
- the gap occurred before takeover during blocking laptop-side MotionSwitcher
  client and lowcmd-publisher initialization, not during the 250 Hz hold loop;
- PC2's raw `SelectMode("ai")` request did take effect—the robot returned to its
  observed zero-torque state and the remote became responsive—even though the
  RPC reply was reported as status 7002; and
- the subsequent automatic Damp RPC did not complete.

Accordingly, every non-commanding local DDS endpoint is now initialized before
the short PC2 heartbeat lease is armed. Debug-lowcmd failure recovery no longer
chains an automatic Damp request after AI restoration. It restores `ai` and
requires the locomotion service to report FSM 0. The standing `rt/arm_sdk`
workflow retains its independently commissioned terminal Damp fallback.

A fourth zero-displacement run then verified the heartbeat correction:

- the laptop sent 34 pings with a maximum gap of 0.104117 seconds, safely below
  the 0.5-second PC2 lease;
- no heartbeat timeout occurred;
- PC2 restored `ai` and repeatedly read FSM 0; and
- a direct Sit request did not change FSM 0 to FSM 3.

This supports the operator's transition model: clean seated restoration must
use `FSM 0 -> Damp/FSM 1 -> Sit/FSM 3`, verifying every state. This chain is
used only after the arm has returned to the measured, table-supported start.
Fault or heartbeat-loss recovery stops at verified FSM 0 and does not continue
through Damp or Sit. If clean restoration reaches FSM 1 but cannot reach FSM 3,
the installed PC2 locomotion client supports an explicit ZeroTorque request so
cleanup can return to verified FSM 0.

The final two zero-displacement commissioning runs passed both terminal
lifecycle paths:

- normal completion reported `PC2 verified AI FSM 0 -> 1 -> seated FSM 3
  before the lowcmd publisher closed`; and
- deliberate heartbeat loss reported `PC2 verified AI service in zero-torque
  FSM 0 before the lowcmd publisher closed`.

Both runs had already passed dynamic handoff, service release, the first
complete measured-state lowcmd packet, and the zero-displacement hold. These
results physically commission debug ownership, normal seated restoration, and
independent heartbeat-loss recovery. They did not claim changing-target motion;
at that point the first fully preflighted tabletop route was still the next
changing-target seated test. The reviewed repository configuration now sets
`control.seated_debug_lowcmd_commissioned: true`.

## Changing-target result and gravity-feedforward follow-up

A later complete seated table route physically verified changing left-arm
targets, image capture at the elevated target, reversal to the supported start,
and clean AI/FSM 0 -> Damp/FSM 1 -> Sit/FSM 3 restoration. Direct visual
scoring showed that the requested 100 mm lift achieved about 71.1 mm. The task
error was 29.0 mm total, including 28.9 mm vertical error; the seven-frame
measurement itself was repeatable to 0.524 mm and 0.183 degrees. At the table
target, the largest joint error was 0.0495 rad at left shoulder pitch. The run
used Unitree's pinned position gains but wrote zero to every arm `tau` field.

The pinned Unitree XR implementation does not use position PD alone. Its G1
IK code evaluates Pinocchio `rnea(q_command, 0, 0)` and its arm controller writes
the returned values to each motor's `tau` feedforward field. The local seated
controller now implements that same term using the exact configured 29-DoF
rubber-hand URDF, while freezing legs and waist at the measured takeover pose.
At the recorded table target, the local model predicts left-arm gravity torques
of approximately `[-3.121, -0.115, +0.870, -1.444, +0.335, -0.026, +0.176]`
Nm. The observed PD term at the settled pose was approximately
`[-3.959, -0.034, +0.619, -1.666, +0.782, +0.089, +0.176]` Nm, so the model
explains most—but not necessarily all—of the loaded shoulder offset.

The gravity-feedforward zero-displacement run passed physically on 2026-08-11.
It kept every position target fixed while ramping the model torque from zero to
full. Maximum measured drift across both arms was `0.008317 rad` (`0.4765`
degrees), only 16.6% of the `0.05 rad` commissioning bound. PC2 then verified
AI/FSM 0 -> Damp/FSM 1 -> Sit/FSM 3 before the lowcmd publisher closed. This
commissions the torque ramp and its clean restoration path—not the resulting
tabletop task accuracy. `control.seated_gravity_feedforward_commissioned` is
therefore `true`; the next full table run must measure whether the direct
board-relative vertical and total landing errors improve.

The first gravity-compensated accuracy run exposed a second, separate effect:
the torso/head-camera pose changed during complete lowcmd takeover. A route
computed from the seated locomotion state is therefore stale before its first
changing target. The table executor now performs no IK/FCL solve before
ownership. It acquires and settles at an unchanged measured command, captures a
fresh board/cube burst, rebuilds and validates all four route edges from that
loaded state, and atomically installs the matching pose-set/report hashes. A
second loaded burst rejects movement during planning. The achieved burst is
used only for one-shot scoring; it never corrects the commanded endpoint.

## Recovery when the remote cannot change modes

Prerequisites: the load-bearing harness is engaged, the workspace is clear, and
both arms are physically supported. Restoring a controller may re-enable
torque. Do not rerun the failed lowcmd program while the service state is
unknown.

Run this on the laptop; Codex must not run it on the robot operator's behalf:

```bash
cd /path/to/robot-calibration-aprilcube-prototype && \
CYCLONEDDS_HOME="$PWD/deps/cyclonedds_python_prefix" \
LD_LIBRARY_PATH="$PWD/deps/cyclonedds_python_prefix/lib:${LD_LIBRARY_PATH:-}" \
.venv/bin/python - <<'PY'
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

ChannelFactoryInitialize(0, "enp134s0")
client = MotionSwitcherClient()
client.SetTimeout(5.0)
client.Init()

status, mode = client.CheckMode()
if status != 0:
    raise SystemExit(f"CheckMode failed before recovery: status={status}")

name = str(mode.get("name", "")).strip()
if not name:
    status, _ = client.SelectMode("ai")
    if status != 0:
        raise SystemExit(f"SelectMode('ai') failed: status={status}")
    time.sleep(1.0)

status, mode = client.CheckMode()
if status != 0 or not str(mode.get("name", "")).strip():
    raise SystemExit(f"AI service was not restored: status={status}, mode={mode}")

print(f"motion service restored: {mode}")
PY
```

After it reports `name: ai`, confirm that the wireless remote responds. On this
robot service restoration has initialized zero-torque FSM 0. Keep the robot and
arms physically supported; zero torque is limp, whereas Damp adds velocity
resistance. Selecting Damp with the remote is an operator choice after control
has been recovered, not a required second step in automated debug recovery.

Do not repeatedly run `SelectMode("ai")` after it has succeeded. Do not treat
the printed motion-service record as an FSM verification.

## Protections added after the observation

The debug transport now maintains its takeover heartbeat continuously from the
blocking MotionSwitcher release through completion of the first measured-state
lowcmd write. Previously, the temporary guard ended after `ReleaseMode()` and
left the first write outside that protected interval.

The PC2 watchdog also performs its own read-only raw-SDK MotionSwitcher
`CheckMode` preflight while `ai` is still active. If it cannot answer,
commissioning stops before service release and before any lowcmd packet. The
same persistent client is retained for post-release restoration. No further
tabletop execution was allowed until normal restoration and heartbeat-loss
recovery were physically commissioned. The final two zero-displacement runs
verified both paths on hardware, including the corrected heartbeat lifetime.

The replacement runtime is installed under
`/home/unitree/.local/share/g1-aprilcube-watchdog`. PC2 initially lacked the
optional Ubuntu `python3.8-venv` package; the operator installed that standard
package explicitly using temporary hotspot access. The watchdog dependencies
themselves are still downloaded and checksum-verified on the laptop, copied to
PC2 over Ethernet, and installed offline into an immutable versioned venv. They
do not modify system Python, the user-wide site-packages directory, Unitree's
ROS workspace, shell startup files, or boot services.

### Installed private runtime

The runtime installed on 2026-08-10 is:

- immutable runtime ID `98da77c45b9d9931`;
- runtime path
  `/home/unitree/.local/share/g1-aprilcube-watchdog/runtimes/98da77c45b9d9931`;
- active selector
  `/home/unitree/.local/share/g1-aprilcube-watchdog/current`;
- Python 3.8.10 with user-site packages disabled;
- CycloneDDS Python 0.10.2 linked to PC2's existing
  `libddsc.so.0.10.2`; and
- `unitree_sdk2py` 1.0.1 built from pinned revision
  `7c661d27f4ae064ffd0dd633fd9d5b518ef0b508`.

Its manifest is at
`/home/unitree/.local/share/g1-aprilcube-watchdog/current/manifest.json` and
records every source checksum, offline dependency version, and the validation
scope. Installation validation imported the raw SDK bindings only; it did not
create a DDS participant, contact the robot, or send a robot command.

Reinstall or reproduce it from the laptop with:

```bash
cd /path/to/robot-calibration-aprilcube-prototype
./tools/install_pc2_watchdog_runtime.sh
```

The installer downloads and verifies inputs on the laptop, stages them over
Ethernet, builds the ARM64 bindings on PC2, and atomically selects the completed
runtime. Re-running it with the same inputs selects the same runtime ID.
