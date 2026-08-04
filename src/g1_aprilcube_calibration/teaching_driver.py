"""Thread-safe fixed-rate driver for the manual teaching controller."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from g1_aprilcube_calibration.teaching_controller import (
    TeachingArmController,
    TeachingState,
)


class SynchronizedTeachingController:
    """Serialize operator actions with the 250 Hz teaching-control thread."""

    def __init__(self, controller: TeachingArmController) -> None:
        self._controller = controller
        self._lock = threading.RLock()

    @property
    def state(self) -> TeachingState:
        with self._lock:
            return self._controller.state

    @property
    def fault_reason(self) -> str | None:
        with self._lock:
            return self._controller.fault_reason

    @property
    def events(self):
        with self._lock:
            return tuple(self._controller.events)

    @property
    def hold_command_count(self) -> int:
        with self._lock:
            return self._controller.hold_command_count

    @property
    def calibration_kp_scale(self) -> float:
        with self._lock:
            return self._controller.calibration_kp_scale

    @property
    def calibration_kd_scale(self) -> float:
        with self._lock:
            return self._controller.calibration_kd_scale

    @property
    def held_calibration_q(self) -> tuple[float, ...] | None:
        with self._lock:
            return self._controller.held_calibration_q

    @property
    def near_joint_limit(self) -> bool:
        with self._lock:
            return self._controller.near_joint_limit

    def acquire(self, **kwargs) -> None:
        self._call("acquire", **kwargs)

    def begin_hold(self, **kwargs) -> None:
        self._call("begin_hold", **kwargs)

    def protective_hold(self, reason: str) -> None:
        self._call("protective_hold", reason)

    def resume_guide(self, **kwargs) -> None:
        self._call("resume_guide", **kwargs)

    def begin_capture(self) -> None:
        self._call("begin_capture")

    def finish_capture(self, **kwargs) -> None:
        self._call("finish_capture", **kwargs)

    def confirm_external_damping(self, reason: str) -> None:
        self._call("confirm_external_damping", reason)

    def tick(self) -> TeachingState:
        return self._call("tick")

    def _call(self, name: str, *args, **kwargs) -> Any:
        with self._lock:
            return getattr(self._controller, name)(*args, **kwargs)


class TeachingControlDriver:
    """Run guide/hold commands independently of image and disk work."""

    def __init__(
        self,
        controller: SynchronizedTeachingController,
        *,
        rate_hz: float = 250.0,
        safety_heartbeat: Callable[[], None] | None = None,
    ) -> None:
        if rate_hz <= 0:
            raise ValueError("control rate must be positive")
        self.controller = controller
        self.period_s = 1.0 / rate_hz
        self.safety_heartbeat = safety_heartbeat
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("teaching control driver is already started")
        self._thread = threading.Thread(
            target=self._run,
            name="g1-calibration-teaching-control",
            daemon=True,
        )
        self._thread.start()

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError(
                f"teaching control driver failed: {self.error}"
            ) from self.error

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError("teaching control driver did not stop")

    def _run(self) -> None:
        deadline = time.monotonic()
        while not self._stop.is_set():
            try:
                state = self.controller.tick()
                if state is TeachingState.FAULT:
                    raise RuntimeError(
                        "teaching controller entered fault: "
                        f"{self.controller.fault_reason or 'unknown reason'}; "
                        "PC2 heartbeat intentionally stopped"
                    )
                if (
                    self.safety_heartbeat is not None
                    and state is not TeachingState.STOPPED
                ):
                    self.safety_heartbeat()
            # The thread boundary must retain any transport/backend exception so
            # every failure stops the safety heartbeat and reaches the main CLI.
            except Exception as error:  # noqa: BLE001
                self.error = error
                self._stop.set()
                return
            deadline += self.period_s
            remaining = deadline - time.monotonic()
            if remaining > 0:
                self._stop.wait(remaining)
            else:
                deadline = time.monotonic()
