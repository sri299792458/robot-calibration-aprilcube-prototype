"""Ephemeral PC2-side heartbeat watchdog for the supported G1 Damp RPC.

This module intentionally uses only the Python standard library.  The laptop
copies it to a unique path under ``/tmp`` and runs it inside PC2's sourced ROS 2
environment for the duration of one motion-producing calibration command.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path

READY_MARKER = "WATCHDOG_READY"
DAMPING_MARKER = "WATCHDOG_DAMPING"
DAMPED_MARKER = "WATCHDOG_DAMPED"
DISARMED_MARKER = "WATCHDOG_DISARMED"
ERROR_MARKER = "WATCHDOG_ERROR"
_FSM_ID_PATTERN = re.compile(r"current fsm_id:\s*(-?\d+)")


def _run_unitree_client(
    executable: Path,
    option: str,
    success_marker: str,
    *,
    timeout_s: float,
) -> str:
    """Run Unitree's client until its one-shot action reports success."""

    process = subprocess.Popen(
        [str(executable), f"--{option}"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout_s
    output = bytearray()
    success_bytes = success_marker.encode()
    try:
        while success_bytes not in output and time.monotonic() < deadline:
            remaining = max(deadline - time.monotonic(), 0.0)
            events = selector.select(min(remaining, 0.1))
            for key, _ in events:
                chunk = os.read(key.fileobj.fileno(), 4096)
                if chunk:
                    output.extend(chunk)
            if process.poll() is not None:
                remainder = process.stdout.read()
                if remainder:
                    output.extend(remainder)
                break
    finally:
        selector.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        remainder = process.stdout.read()
        if remainder:
            output.extend(remainder)
    lines = bytes(output).decode("utf-8", errors="replace").splitlines()
    detail = " | ".join(lines[-8:]) or "no output"
    if success_bytes not in output:
        raise RuntimeError(
            f"Unitree client --{option} did not report success: {detail}"
        )
    return detail


def _get_fsm_id(executable: Path, *, timeout_s: float) -> tuple[int, str]:
    """Read the Unitree locomotion FSM ID from the installed client."""

    detail = _run_unitree_client(
        executable,
        "get_fsm_id",
        "current fsm_id:",
        timeout_s=timeout_s,
    )
    match = _FSM_ID_PATTERN.search(detail)
    if match is None:
        raise RuntimeError(f"Unitree client returned no parseable FSM ID: {detail}")
    return int(match.group(1)), detail


def _emit(message: str) -> None:
    try:
        print(message, flush=True)
    except BrokenPipeError:
        # The safety action must still finish if the laptop link disappeared.
        pass


def run_watchdog(args: argparse.Namespace) -> int:
    if not 0.2 <= args.heartbeat_timeout_s <= 10.0:
        raise ValueError("heartbeat timeout must be within [0.2, 10.0] seconds")
    if not args.token or any(character.isspace() for character in args.token):
        raise ValueError("watchdog token must be non-empty and contain no whitespace")
    if not args.damp_executable.is_file() or not os.access(
        args.damp_executable, os.X_OK
    ):
        raise ValueError(
            f"G1 locomotion client is not executable: {args.damp_executable}"
        )

    lock_stream = args.lock_file.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_stream.close()
        raise RuntimeError("another PC2 damping watchdog is already armed") from error

    signal_reason: list[str] = []

    def request_damping(signum, _frame) -> None:
        signal_reason.append(f"signal_{signum}")

    signal.signal(signal.SIGHUP, request_damping)
    signal.signal(signal.SIGTERM, request_damping)
    signal.signal(signal.SIGINT, request_damping)

    def damp(reason: str) -> int:
        _emit(f"{DAMPING_MARKER} {reason}")
        errors: list[str] = []
        for _attempt in range(args.damp_attempts):
            try:
                detail = _run_unitree_client(
                    args.damp_executable,
                    "damp",
                    "Damp command sent",
                    timeout_s=args.client_timeout_s,
                )
                fsm_id, fsm_detail = _get_fsm_id(
                    args.damp_executable,
                    timeout_s=args.client_timeout_s,
                )
                if fsm_id != 1:
                    raise RuntimeError(
                        "Unitree Damp request did not reach locomotion FSM 1: "
                        f"{fsm_detail}"
                    )
                _emit(f"{DAMPED_MARKER} fsm_id=1 | {detail}")
                return 0
            except RuntimeError as error:
                errors.append(str(error))
                time.sleep(0.1)
        _emit(f"{ERROR_MARKER} damping failed: {' || '.join(errors)}")
        return 4

    try:
        initial_fsm_id, readiness = _get_fsm_id(
            args.damp_executable,
            timeout_s=args.client_timeout_s,
        )
        _emit(f"{READY_MARKER} fsm_id={initial_fsm_id} | {readiness}")
        deadline = time.monotonic() + args.heartbeat_timeout_s
        input_buffer = b""
        while True:
            if signal_reason:
                return damp(signal_reason[-1])
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return damp("heartbeat_timeout")
            readable, _, _ = select_with_signals(remaining)
            if signal_reason:
                return damp(signal_reason[-1])
            if not readable:
                continue
            chunk = os.read(sys.stdin.fileno(), 4096)
            if not chunk:
                return damp("watchdog_channel_eof")
            input_buffer += chunk
            lines = input_buffer.split(b"\n")
            input_buffer = lines.pop()
            for raw_line in lines:
                line = raw_line.decode("utf-8", errors="replace")
                parts = line.strip().split(maxsplit=2)
                if len(parts) < 2 or parts[1] != args.token:
                    continue
                command = parts[0]
                if command == "PING":
                    deadline = time.monotonic() + args.heartbeat_timeout_s
                elif command == "DISARM":
                    _emit(DISARMED_MARKER)
                    return 0
                elif command == "DAMP":
                    reason = parts[2] if len(parts) == 3 else "explicit_request"
                    return damp(reason.replace("\n", " "))
    finally:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_UN)
        finally:
            lock_stream.close()
        if args.self_delete:
            try:
                Path(__file__).unlink()
            except OSError:
                pass


def select_with_signals(timeout_s: float) -> tuple[list, list, list]:
    """Keep the event loop testable while allowing signals to interrupt select."""

    import select

    return select.select([sys.stdin], [], [], min(timeout_s, 0.1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--heartbeat-timeout-s", type=float, required=True)
    parser.add_argument("--client-timeout-s", type=float, default=3.0)
    parser.add_argument("--damp-attempts", type=int, default=3)
    parser.add_argument("--damp-executable", type=Path, required=True)
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=Path("/tmp/g1-aprilcube-damping-watchdog.lock"),
    )
    parser.add_argument("--self-delete", action="store_true")
    return parser


def main() -> int:
    try:
        return run_watchdog(build_parser().parse_args())
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        _emit(f"{ERROR_MARKER} {error}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
