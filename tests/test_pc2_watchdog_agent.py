from __future__ import annotations

import os
import select
import subprocess
import sys
import time
from pathlib import Path

from g1_aprilcube_calibration import pc2_watchdog_agent


def _fake_loco_client(tmp_path: Path) -> tuple[Path, Path]:
    executable = tmp_path / "fake_g1_loco_client"
    damp_log = tmp_path / "damp.log"
    executable.write_text(
        """#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path

if "--get_fsm_id" in sys.argv:
    # Match the installed Unitree client: several log lines can arrive in one
    # OS-level read while the process remains alive.  A selector wrapped around
    # TextIOWrapper.readline() loses sight of lines buffered after the first.
    fsm_id = Path(os.environ["FAKE_FSM_STATE"]).read_text()
    os.write(
        sys.stdout.fileno(),
        (
            "Processing command: [get_fsm_id]\\n"
            f"current fsm_id: {fsm_id}\\n"
            "Done processing command: get_fsm_id\\n"
        ).encode(),
    )
elif "--damp" in sys.argv:
    Path(os.environ["FAKE_DAMP_LOG"]).write_text("damped\\n")
    if os.environ.get("FAKE_DAMP_CHANGES_FSM", "1") == "1":
        Path(os.environ["FAKE_FSM_STATE"]).write_text("1")
    print("Damp command sent", flush=True)
else:
    raise SystemExit(2)
time.sleep(30)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable, damp_log


def _start_agent(
    tmp_path: Path,
    *,
    token: str,
    heartbeat_timeout_s: float,
    damp_changes_fsm: bool = True,
) -> tuple[subprocess.Popen[str], Path]:
    executable, damp_log = _fake_loco_client(tmp_path)
    fsm_state = tmp_path / "fsm.state"
    fsm_state.write_text("500", encoding="utf-8")
    environment = os.environ.copy()
    environment["FAKE_DAMP_LOG"] = str(damp_log)
    environment["FAKE_FSM_STATE"] = str(fsm_state)
    environment["FAKE_DAMP_CHANGES_FSM"] = "1" if damp_changes_fsm else "0"
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(Path(pc2_watchdog_agent.__file__).resolve()),
            "--token",
            token,
            "--heartbeat-timeout-s",
            str(heartbeat_timeout_s),
            "--client-timeout-s",
            "1",
            "--damp-executable",
            str(executable),
            "--lock-file",
            str(tmp_path / "watchdog.lock"),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
    )
    return process, damp_log


def _read_until(process: subprocess.Popen[str], marker: str, timeout_s: float) -> str:
    assert process.stdout is not None
    deadline = time.monotonic() + timeout_s
    lines: list[str] = []
    while time.monotonic() < deadline:
        readable, _, _ = select.select([process.stdout], [], [], 0.1)
        if readable:
            line = process.stdout.readline()
            if line:
                lines.append(line.rstrip())
                if line.startswith(marker):
                    return line.rstrip()
        if process.poll() is not None:
            break
    raise AssertionError(f"did not receive {marker}: {lines}")


def test_agent_disarms_without_damping(tmp_path: Path) -> None:
    token = "test-token"
    process, damp_log = _start_agent(
        tmp_path,
        token=token,
        heartbeat_timeout_s=1.0,
    )
    try:
        _read_until(process, pc2_watchdog_agent.READY_MARKER, 2.0)
        assert process.stdin is not None
        process.stdin.write(f"PING {token}\nDISARM {token}\n")
        process.stdin.flush()
        _read_until(process, pc2_watchdog_agent.DISARMED_MARKER, 1.0)
        assert process.wait(timeout=1.0) == 0
        assert not damp_log.exists()
    finally:
        if process.poll() is None:
            process.kill()


def test_agent_heartbeat_timeout_invokes_supported_damping(tmp_path: Path) -> None:
    process, damp_log = _start_agent(
        tmp_path,
        token="timeout-token",
        heartbeat_timeout_s=0.2,
    )
    try:
        _read_until(process, pc2_watchdog_agent.READY_MARKER, 2.0)
        result = _read_until(process, pc2_watchdog_agent.DAMPED_MARKER, 2.0)
        assert "Damp command sent" in result
        assert "fsm_id=1" in result
        assert process.wait(timeout=1.0) == 0
        assert damp_log.read_text(encoding="utf-8") == "damped\n"
    finally:
        if process.poll() is None:
            process.kill()


def test_agent_explicit_request_invokes_supported_damping(tmp_path: Path) -> None:
    token = "damp-token"
    process, damp_log = _start_agent(
        tmp_path,
        token=token,
        heartbeat_timeout_s=1.0,
    )
    try:
        _read_until(process, pc2_watchdog_agent.READY_MARKER, 2.0)
        assert process.stdin is not None
        process.stdin.write(f"DAMP {token} operator_interrupt\n")
        process.stdin.flush()
        _read_until(process, pc2_watchdog_agent.DAMPED_MARKER, 2.0)
        assert process.wait(timeout=1.0) == 0
        assert damp_log.exists()
    finally:
        if process.poll() is None:
            process.kill()


def test_agent_rejects_unconfirmed_damping_transition(tmp_path: Path) -> None:
    process, _ = _start_agent(
        tmp_path,
        token="failed-damp-token",
        heartbeat_timeout_s=1.0,
        damp_changes_fsm=False,
    )
    try:
        _read_until(process, pc2_watchdog_agent.READY_MARKER, 2.0)
        assert process.stdin is not None
        process.stdin.write("DAMP failed-damp-token test_failure\n")
        process.stdin.flush()
        result = _read_until(process, pc2_watchdog_agent.ERROR_MARKER, 5.0)
        assert "did not reach locomotion FSM 1" in result
        assert process.wait(timeout=1.0) == 4
    finally:
        if process.poll() is None:
            process.kill()
