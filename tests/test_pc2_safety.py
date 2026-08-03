from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from g1_aprilcube_calibration.pc2_safety import (
    PC2DampingWatchdog,
    PC2SafetyConfig,
)


def _fake_ssh(tmp_path: Path) -> Path:
    script = tmp_path / "fake_ssh.py"
    script.write_text(
        """import sys

print("WATCHDOG_READY current fsm_id: 500", flush=True)
for line in sys.stdin:
    command = line.split(maxsplit=1)[0]
    if command == "DISARM":
        print("WATCHDOG_DISARMED", flush=True)
        raise SystemExit(0)
    if command == "DAMP":
        sys.stdout.write(
            "WATCHDOG_DAMPING explicit_request\\n"
            "WATCHDOG_DAMPED fsm_id=1 | Damp command sent\\n"
        )
        sys.stdout.flush()
        raise SystemExit(0)
""",
        encoding="utf-8",
    )
    return script


def _subject(tmp_path: Path) -> PC2DampingWatchdog:
    identity = tmp_path / "identity"
    identity.write_text("fake", encoding="utf-8")
    fake_ssh = _fake_ssh(tmp_path)

    def run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def popen(_args, **kwargs):
        return subprocess.Popen([sys.executable, "-u", str(fake_ssh)], **kwargs)

    config = PC2SafetyConfig(
        host="unitree@192.168.123.164",
        ssh_identity=identity,
        heartbeat_interval_s=0.05,
        heartbeat_timeout_s=0.2,
        connect_timeout_s=0.5,
        client_timeout_s=0.5,
    )
    return PC2DampingWatchdog(config, run=run, popen=popen)


def test_laptop_watchdog_can_disarm_cleanly(tmp_path: Path) -> None:
    watchdog = _subject(tmp_path)
    watchdog.start()
    assert watchdog.armed
    watchdog.pulse()
    watchdog.disarm()
    assert not watchdog.armed
    assert watchdog.terminal_action == "disarmed"


def test_laptop_watchdog_can_request_and_confirm_damping(tmp_path: Path) -> None:
    watchdog = _subject(tmp_path)
    watchdog.start()
    watchdog.damp("operator pressed ctrl-c")
    assert not watchdog.armed
    assert watchdog.terminal_action == "damped"
