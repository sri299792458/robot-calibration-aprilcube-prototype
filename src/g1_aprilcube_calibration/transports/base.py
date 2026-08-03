"""Narrow command boundary owned exclusively by the calibration executor."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from g1_aprilcube_calibration.joint_map import DUAL_ARM_DOF
from g1_aprilcube_calibration.models import RobotStateSample


@dataclass(frozen=True, slots=True)
class ArmCommand:
    q14: tuple[float, ...]
    weight: float
    issued_monotonic_s: float
    emergency_release: bool = False

    def __post_init__(self) -> None:
        array = np.asarray(self.q14, dtype=np.float64).reshape(-1)
        if array.shape != (DUAL_ARM_DOF,):
            raise ValueError(f"arm command must contain {DUAL_ARM_DOF} joints")
        if not np.all(np.isfinite(array)):
            raise ValueError("arm command contains NaN or infinity")
        if not np.isfinite(self.weight) or not 0 <= self.weight <= 1:
            raise ValueError("arm command weight must be within [0, 1]")
        if not np.isfinite(self.issued_monotonic_s) or self.issued_monotonic_s < 0:
            raise ValueError("arm command timestamp must be finite and non-negative")
        object.__setattr__(self, "q14", tuple(float(value) for value in array))

    @classmethod
    def create(
        cls,
        q14: Sequence[float] | np.ndarray,
        *,
        weight: float,
        issued_monotonic_s: float,
        emergency_release: bool = False,
    ) -> ArmCommand:
        return cls(tuple(q14), weight, issued_monotonic_s, emergency_release)


class ArmTransport(Protocol):
    """The only object permitted to touch the arm command channel."""

    def observe(self) -> RobotStateSample: ...

    def send_command(self, command: ArmCommand) -> None: ...

    def close(self) -> None: ...
