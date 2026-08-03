from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pytest

from g1_aprilcube_calibration.clock import ManualClock
from g1_aprilcube_calibration.transports.base import ArmCommand
from g1_aprilcube_calibration.transports.unitree_arm_sdk import (
    ARM_WEIGHT_SLOT,
    UnitreeArmSDKTransport,
    UnitreeLowStateObserver,
    UnitreeSDKBindings,
    UnitreeTransportConfig,
)


@dataclass
class Motor:
    mode: int = 0
    q: float = 0.0
    dq: float = 0.0
    tau: float = 0.0
    kp: float = 0.0
    kd: float = 0.0


@dataclass
class LowCommand:
    mode_pr: int = 99
    mode_machine: int = 0
    motor_cmd: list[Motor] = field(default_factory=lambda: [Motor() for _ in range(35)])
    crc: int = 0


@dataclass
class LowState:
    mode_machine: int
    tick: int
    motor_state: list[Motor]


class Subscriber:
    instances: ClassVar[list[Subscriber]] = []

    def __init__(self, topic, message_type):
        self.topic = topic
        self.message_type = message_type
        self.callback = None
        self.closed = False
        self.__class__.instances.append(self)

    def Init(self, callback, queue_length):
        self.callback = callback
        self.queue_length = queue_length

    def Close(self):
        self.closed = True

    def emit(self, message):
        self.callback(message)


class Publisher:
    instances: ClassVar[list[Publisher]] = []

    def __init__(self, topic, message_type):
        self.topic = topic
        self.message_type = message_type
        self.messages = []
        self.closed = False
        self.__class__.instances.append(self)

    def Init(self):
        self.initialized = True

    def Write(self, message):
        self.messages.append(deepcopy(message))
        return True

    def Close(self):
        self.closed = True


@pytest.fixture
def sdk():
    Subscriber.instances.clear()
    Publisher.instances.clear()
    initialized = []
    bindings = UnitreeSDKBindings(
        initialize=lambda domain, interface: initialized.append((domain, interface)),
        publisher_type=Publisher,
        subscriber_type=Subscriber,
        low_command_type=LowCommand,
        low_state_type=LowState,
        make_low_command=LowCommand,
        calculate_crc=lambda message: 0x1234,
    )
    return bindings, initialized


def state(*, mode=5, tick=17):
    return LowState(
        mode_machine=mode,
        tick=tick,
        motor_state=[Motor(q=index / 10, dq=-index / 100) for index in range(35)],
    )


def config():
    return UnitreeTransportConfig(network_interface="enp3s0", domain_id=4)


def test_read_only_observer_never_constructs_a_publisher(sdk):
    bindings, initialized = sdk
    clock = ManualClock(2.0)
    observer = UnitreeLowStateObserver(
        config(),
        bindings=bindings,
        clock=clock,
        utc_now=lambda: "2026-08-02T12:00:00Z",
    )

    assert initialized == [(4, "enp3s0")]
    assert Publisher.instances == []
    with pytest.raises(RuntimeError, match="no valid"):
        observer.observe()
    Subscriber.instances[0].emit(state())
    sample = observer.observe()
    assert sample.source_sequence == 17
    assert sample.mode_machine == 5
    np.testing.assert_allclose(sample.position, np.arange(29) / 10)
    np.testing.assert_allclose(sample.velocity, -np.arange(29) / 100)
    observer.close()
    assert Subscriber.instances[0].closed


def test_observer_forwards_only_valid_samples(sdk):
    bindings, _ = sdk
    received = []
    observer = UnitreeLowStateObserver(
        config(), bindings=bindings, on_sample=received.append
    )
    Subscriber.instances[0].emit(state(tick=2))
    bad = state(tick=3)
    bad.motor_state = bad.motor_state[:2]
    Subscriber.instances[0].emit(bad)
    assert [sample.source_sequence for sample in received] == [2]
    observer.close()


def test_invalid_lowstate_is_rejected_without_replacing_last_good_state(sdk):
    bindings, _ = sdk
    observer = UnitreeLowStateObserver(config(), bindings=bindings)
    Subscriber.instances[0].emit(state(tick=3))
    bad = state(tick=4)
    bad.motor_state[4].q = float("nan")
    Subscriber.instances[0].emit(bad)
    assert observer.observe().source_sequence == 3
    assert "NaN" in observer.last_error
    observer.close()


def test_transport_maps_only_arms_and_weight_and_requires_zero_close(sdk):
    bindings, _ = sdk
    clock = ManualClock(5.0)
    transport = UnitreeArmSDKTransport(
        config(),
        bindings=bindings,
        clock=clock,
        utc_now=lambda: "2026-08-02T12:00:00Z",
    )
    assert Publisher.instances[0].messages == []
    Subscriber.instances[0].emit(state())
    command = ArmCommand.create(np.arange(14) / 20, weight=0.6, issued_monotonic_s=5)
    transport.send_command(command)

    sent = Publisher.instances[0].messages[-1]
    assert sent.mode_pr == 0
    assert sent.mode_machine == 5
    assert sent.crc == 0x1234
    assert sent.motor_cmd[ARM_WEIGHT_SLOT].q == pytest.approx(0.6)
    for offset, index in enumerate(range(15, 29)):
        motor = sent.motor_cmd[index]
        assert motor.mode == 1
        assert motor.q == pytest.approx(command.q14[offset])
        assert motor.dq == 0
        assert motor.tau == 0
        expected = (40.0, 1.5) if index in {19, 20, 21, 26, 27, 28} else (80, 3)
        assert (motor.kp, motor.kd) == expected
    for index in list(range(15)) + list(range(30, 35)):
        assert sent.motor_cmd[index] == Motor()

    with pytest.raises(RuntimeError, match="terminal weight-zero"):
        transport.close()
    transport.send_command(
        ArmCommand.create(command.q14, weight=0.0, issued_monotonic_s=5)
    )
    transport.close()
    assert Publisher.instances[0].closed
    assert Subscriber.instances[0].closed


def test_transport_attaches_publisher_to_existing_read_only_observer(sdk):
    bindings, initialized = sdk
    observer = UnitreeLowStateObserver(config(), bindings=bindings)

    assert len(Subscriber.instances) == 1
    assert Publisher.instances == []
    transport = UnitreeArmSDKTransport(config(), observer=observer)

    assert initialized == [(4, "enp3s0")]
    assert len(Subscriber.instances) == 1
    assert len(Publisher.instances) == 1
    Subscriber.instances[0].emit(state())
    assert transport.observe().source_sequence == 17
    transport.close()
    assert Subscriber.instances[0].closed


def test_transport_rejects_mismatched_existing_observer(sdk):
    bindings, _ = sdk
    observer = UnitreeLowStateObserver(config(), bindings=bindings)
    mismatched = UnitreeTransportConfig(network_interface="other")

    with pytest.raises(ValueError, match="configuration does not match"):
        UnitreeArmSDKTransport(mismatched, observer=observer)

    assert Publisher.instances == []
    observer.close()


def test_transport_refuses_wrong_mode_before_publish(sdk):
    bindings, _ = sdk
    transport = UnitreeArmSDKTransport(config(), bindings=bindings)
    Subscriber.instances[0].emit(state(mode=4))
    with pytest.raises(RuntimeError, match="mode_machine=4"):
        transport.send_command(
            ArmCommand.create(np.zeros(14), weight=0, issued_monotonic_s=0)
        )
    assert Publisher.instances[0].messages == []
    transport.close()


def test_transport_can_close_nonzero_only_after_external_damping(sdk):
    bindings, _ = sdk
    transport = UnitreeArmSDKTransport(config(), bindings=bindings)
    Subscriber.instances[0].emit(state())
    transport.send_command(
        ArmCommand.create(np.zeros(14), weight=1.0, issued_monotonic_s=0)
    )
    transport.close_after_external_damping()
    assert Publisher.instances[0].closed
    assert Subscriber.instances[0].closed
