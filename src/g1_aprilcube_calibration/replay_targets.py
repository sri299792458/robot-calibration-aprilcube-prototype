"""Derive replay-only targets that respect a joint-limit safety margin."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from g1_aprilcube_calibration.pose_schema import PoseSet
from g1_aprilcube_calibration.urdf_model import URDFModel


@dataclass(frozen=True, slots=True)
class ReplayTargetAdjustment:
    pose_id: str
    joint_name: str
    limit_side: str
    measured_target_rad: float
    replay_target_rad: float
    hard_limit_rad: float
    margin_rad: float

    @property
    def delta_rad(self) -> float:
        return self.replay_target_rad - self.measured_target_rad

    def to_dict(self) -> dict[str, str | float]:
        return {
            "pose_id": self.pose_id,
            "joint_name": self.joint_name,
            "limit_side": self.limit_side,
            "measured_target_rad": self.measured_target_rad,
            "replay_target_rad": self.replay_target_rad,
            "delta_rad": self.delta_rad,
            "hard_limit_rad": self.hard_limit_rad,
            "margin_rad": self.margin_rad,
        }


def back_off_replay_targets(
    pose_set: PoseSet,
    model: URDFModel,
    *,
    joint_limit_margin_rad: float,
) -> tuple[PoseSet, tuple[ReplayTargetAdjustment, ...]]:
    """Clamp every replay joint target into its margin-adjusted URDF range.

    The manually measured joint vectors are immutable calibration evidence.  A
    separate replay vector is added only to poses that require a changed target.
    """

    if not np.isfinite(joint_limit_margin_rad) or joint_limit_margin_rad <= 0:
        raise ValueError("joint_limit_margin_rad must be finite and positive")
    if pose_set.urdf_sha256 != model.sha256:
        raise ValueError("pose set was authored against a different URDF hash")

    limits = model.joint_limits(pose_set.joint_order)
    lower = np.asarray([item.lower for item in limits], dtype=np.float64)
    upper = np.asarray([item.upper for item in limits], dtype=np.float64)
    safe_lower = lower + joint_limit_margin_rad
    safe_upper = upper - joint_limit_margin_rad
    invalid = np.flatnonzero(safe_lower > safe_upper)
    if invalid.size:
        name = pose_set.joint_order[int(invalid[0])]
        raise ValueError(f"joint limit margin leaves no valid range for {name}")

    adjusted_poses = []
    adjustments: list[ReplayTargetAdjustment] = []
    for pose in pose_set.poses:
        measured = np.asarray(pose.measured_calibration_q, dtype=np.float64)
        replay = np.clip(measured, safe_lower, safe_upper)
        changed = np.flatnonzero(replay != measured)
        if not changed.size:
            adjusted_poses.append(replace(pose, replay_calibration_q=None))
            continue
        for index in changed:
            index = int(index)
            side = "lower" if measured[index] < safe_lower[index] else "upper"
            hard_limit = lower[index] if side == "lower" else upper[index]
            adjustments.append(
                ReplayTargetAdjustment(
                    pose_id=pose.id,
                    joint_name=pose_set.joint_order[index],
                    limit_side=side,
                    measured_target_rad=float(measured[index]),
                    replay_target_rad=float(replay[index]),
                    hard_limit_rad=float(hard_limit),
                    margin_rad=float(joint_limit_margin_rad),
                )
            )
        adjusted_poses.append(
            replace(pose, replay_calibration_q=tuple(float(item) for item in replay))
        )

    return replace(pose_set, poses=tuple(adjusted_poses)), tuple(adjustments)
