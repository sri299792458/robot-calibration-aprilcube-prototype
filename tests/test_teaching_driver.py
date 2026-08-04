from __future__ import annotations

import threading

import pytest

from g1_aprilcube_calibration.teaching_controller import TeachingState
from g1_aprilcube_calibration.teaching_driver import (
    SynchronizedTeachingController,
    TeachingControlDriver,
)


class Controller:
    def __init__(self) -> None:
        self.state = TeachingState.OBSERVING
        self.fault_reason = None
        self.events = []
        self.hold_command_count = 0
        self.held_calibration_q = None
        self.tick_thread_ids = []

    def tick(self):
        self.tick_thread_ids.append(threading.get_ident())
        return self.state

    def acquire(self, **_kwargs):
        self.state = TeachingState.ACQUIRING


class FailingController(Controller):
    def tick(self):
        raise OSError("transport write failed")


def test_teaching_driver_serializes_actions_and_heartbeats_off_main_thread() -> None:
    raw = Controller()
    synchronized = SynchronizedTeachingController(raw)
    heartbeat_threads = []
    driver = TeachingControlDriver(
        synchronized,
        rate_hz=500,
        safety_heartbeat=lambda: heartbeat_threads.append(threading.get_ident()),
    )
    driver.start()
    synchronized.acquire(operator_confirmed=True)
    for _ in range(10_000):
        if heartbeat_threads:
            break
    driver.close()
    driver.check()

    assert raw.tick_thread_ids
    assert heartbeat_threads
    assert set(raw.tick_thread_ids) != {threading.get_ident()}
    assert set(heartbeat_threads) != {threading.get_ident()}


def test_teaching_driver_preserves_fault_reason() -> None:
    raw = Controller()
    raw.state = TeachingState.FAULT
    raw.fault_reason = "held calibration arm drifted"
    driver = TeachingControlDriver(
        SynchronizedTeachingController(raw),
        rate_hz=500,
    )

    driver.start()
    driver.close()

    with pytest.raises(RuntimeError, match="held calibration arm drifted"):
        driver.check()


def test_teaching_driver_propagates_non_runtime_transport_errors() -> None:
    driver = TeachingControlDriver(
        SynchronizedTeachingController(FailingController()),
        rate_hz=500,
    )

    driver.start()
    driver.close()

    with pytest.raises(RuntimeError, match="transport write failed"):
        driver.check()


def test_stopped_teaching_controller_does_not_send_safety_heartbeat() -> None:
    raw = Controller()
    raw.state = TeachingState.STOPPED
    heartbeats = []
    driver = TeachingControlDriver(
        SynchronizedTeachingController(raw),
        rate_hz=500,
        safety_heartbeat=lambda: heartbeats.append(True),
    )

    driver.start()
    for _ in range(10_000):
        if raw.tick_thread_ids:
            break
    driver.close()
    driver.check()

    assert raw.tick_thread_ids
    assert heartbeats == []
