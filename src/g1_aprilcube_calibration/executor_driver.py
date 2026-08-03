"""Thread-safe wrapper and fixed-rate driver for hardware capture workloads."""

from __future__ import annotations

import threading
import time
from typing import Any

from g1_aprilcube_calibration.executor_state_machine import PoseExecutor


class SynchronizedPoseExecutor:
    """Serialize operator actions with the dedicated control tick thread."""

    def __init__(self, executor: PoseExecutor) -> None:
        self._executor = executor
        self._lock = threading.RLock()

    @property
    def pose_set(self):
        return self._executor.pose_set

    @property
    def approved_validation_report_sha256(self) -> str:
        return self._executor.approved_validation_report_sha256

    @property
    def state(self):
        with self._lock:
            return self._executor.state

    @property
    def current_pose_id(self) -> str | None:
        with self._lock:
            return self._executor.current_pose_id

    @property
    def fault_reason(self) -> str | None:
        with self._lock:
            return self._executor.fault_reason

    @property
    def events(self):
        with self._lock:
            return tuple(self._executor.events)

    def acquire(self, **kwargs) -> None:
        self._call("acquire", **kwargs)

    def start_pose(self, *args, **kwargs) -> None:
        self._call("start_pose", *args, **kwargs)

    def begin_capture(self) -> None:
        self._call("begin_capture")

    def finish_capture(self, **kwargs) -> None:
        self._call("finish_capture", **kwargs)

    def begin_clean_release(self, **kwargs) -> None:
        self._call("begin_clean_release", **kwargs)

    def emergency_stop(self, reason: str) -> None:
        self._call("emergency_stop", reason)

    def tick(self):
        return self._call("tick")

    def _call(self, name: str, *args, **kwargs) -> Any:
        with self._lock:
            return getattr(self._executor, name)(*args, **kwargs)


class ExecutorControlDriver:
    """Run executor ticks independently of image detection and disk writes."""

    def __init__(
        self,
        executor: SynchronizedPoseExecutor,
        *,
        rate_hz: float = 250.0,
    ) -> None:
        if rate_hz <= 0:
            raise ValueError("control rate must be positive")
        self.executor = executor
        self.period_s = 1.0 / rate_hz
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("control driver is already started")
        self._thread = threading.Thread(
            target=self._run,
            name="g1-calibration-control",
            daemon=True,
        )
        self._thread.start()

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"control driver failed: {self.error}") from self.error

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError("control driver did not stop")

    def _run(self) -> None:
        deadline = time.monotonic()
        while not self._stop.is_set():
            try:
                self.executor.tick()
            except (RuntimeError, TypeError, ValueError) as error:
                self.error = error
                self._stop.set()
                return
            deadline += self.period_s
            remaining = deadline - time.monotonic()
            if remaining > 0:
                self._stop.wait(remaining)
            else:
                deadline = time.monotonic()
