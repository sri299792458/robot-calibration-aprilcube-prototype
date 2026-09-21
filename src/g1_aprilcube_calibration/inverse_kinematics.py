"""Bounded deterministic inverse kinematics for one G1 seven-joint arm."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from g1_aprilcube_calibration.joint_map import (
    G1_29_JOINT_NAMES,
    arm_hand_link,
    arm_indices,
    arm_joint_names,
    validate_arm_side,
    validate_full_joint_vector,
)
from g1_aprilcube_calibration.transforms import validate_transform
from g1_aprilcube_calibration.urdf_model import URDFModel


@dataclass(frozen=True, slots=True)
class IKConfig:
    joint_limit_margin_rad: float = 0.03
    maximum_translation_error_m: float = 0.0005
    maximum_rotation_error_deg: float = 0.25
    restart_count: int = 24
    maximum_function_evaluations: int = 800
    translation_scale_m: float = 0.001
    rotation_scale_rad: float = np.deg2rad(0.5)
    seed_regularization: float = 1e-3

    def __post_init__(self) -> None:
        for name in (
            "joint_limit_margin_rad",
            "maximum_translation_error_m",
            "maximum_rotation_error_deg",
            "translation_scale_m",
            "rotation_scale_rad",
            "seed_regularization",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.restart_count < 1:
            raise ValueError("restart_count must be positive")
        if self.maximum_function_evaluations < 1:
            raise ValueError("maximum_function_evaluations must be positive")


@dataclass(frozen=True, slots=True)
class IKSolution:
    calibration_q: tuple[float, ...]
    torso_T_hand: np.ndarray
    translation_error_m: float
    rotation_error_deg: float
    seed_distance_rad: float
    function_evaluations: int
    optimizer_cost: float

    def __post_init__(self) -> None:
        q = np.asarray(self.calibration_q, dtype=np.float64).reshape(-1)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("IK solution must contain seven finite joints")
        object.__setattr__(self, "calibration_q", tuple(float(item) for item in q))
        object.__setattr__(self, "torso_T_hand", validate_transform(self.torso_T_hand))
        for name in (
            "translation_error_m",
            "rotation_error_deg",
            "seed_distance_rad",
            "optimizer_cost",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.function_evaluations < 1:
            raise ValueError("IK function-evaluation count must be positive")

    def to_dict(self) -> dict:
        return {
            "calibration_q": list(self.calibration_q),
            "torso_T_hand": self.torso_T_hand.tolist(),
            "translation_error_m": self.translation_error_m,
            "translation_error_mm": self.translation_error_m * 1000.0,
            "rotation_error_deg": self.rotation_error_deg,
            "seed_distance_rad": self.seed_distance_rad,
            "function_evaluations": self.function_evaluations,
            "optimizer_cost": self.optimizer_cost,
        }


def solve_arm_ik_candidates(
    model: URDFModel,
    *,
    desired_torso_T_hand: np.ndarray,
    reference_full_q: np.ndarray,
    calibration_arm: str,
    config: IKConfig | None = None,
) -> tuple[IKSolution, ...]:
    """Return deterministic, joint-bounded IK candidates nearest the live seed.

    Collision and path clearance are intentionally not optimized here.  The
    caller must pass every candidate through the project's discrete FCL path
    validator and may use only a candidate whose outward and return edges pass.
    """

    config = config or IKConfig()
    calibration_arm = validate_arm_side(calibration_arm)
    desired = validate_transform(desired_torso_T_hand)
    reference = validate_full_joint_vector(reference_full_q)
    indices = np.asarray(arm_indices(calibration_arm))
    seed = reference[indices]
    names = arm_joint_names(calibration_arm)
    limits = model.joint_limits(names)
    lower = np.asarray(
        [item.lower + config.joint_limit_margin_rad for item in limits],
        dtype=np.float64,
    )
    upper = np.asarray(
        [item.upper - config.joint_limit_margin_rad for item in limits],
        dtype=np.float64,
    )
    if np.any(lower >= upper):
        raise ValueError("IK joint-limit margin removes the feasible range")
    clipped_seed = np.clip(seed, lower, upper)
    starts = [clipped_seed]
    if config.restart_count > 1:
        unit_samples = qmc.Halton(d=7, scramble=False).random(config.restart_count - 1)
        starts.extend(qmc.scale(unit_samples, lower, upper))

    def forward(q: np.ndarray) -> np.ndarray:
        full = reference.copy()
        full[indices] = q
        positions = dict(zip(G1_29_JOINT_NAMES, full, strict=True))
        return model.transform("torso_link", arm_hand_link(calibration_arm), positions)

    def primary_error(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        translation = transform[:3, 3] - desired[:3, 3]
        rotation = Rotation.from_matrix(
            (desired[:3, :3].T @ transform[:3, :3]).copy()
        ).as_rotvec()
        return translation, rotation

    def residual(q: np.ndarray) -> np.ndarray:
        translation, rotation = primary_error(forward(q))
        return np.concatenate(
            (
                translation / config.translation_scale_m,
                rotation / config.rotation_scale_rad,
                config.seed_regularization * (q - clipped_seed),
            )
        )

    candidates: list[IKSolution] = []
    best_failure: tuple[float, float] | None = None
    for start in starts:
        optimized = least_squares(
            residual,
            np.asarray(start, dtype=np.float64),
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=config.maximum_function_evaluations,
        )
        transform = forward(optimized.x)
        translation, rotation = primary_error(transform)
        translation_error = float(np.linalg.norm(translation))
        rotation_error_deg = float(np.degrees(np.linalg.norm(rotation)))
        failure_key = (translation_error, rotation_error_deg)
        if best_failure is None or failure_key < best_failure:
            best_failure = failure_key
        if (
            translation_error > config.maximum_translation_error_m
            or rotation_error_deg > config.maximum_rotation_error_deg
        ):
            continue
        if any(
            np.max(np.abs(np.asarray(item.calibration_q) - optimized.x)) < 1e-4
            for item in candidates
        ):
            continue
        candidates.append(
            IKSolution(
                calibration_q=tuple(optimized.x),
                torso_T_hand=transform,
                translation_error_m=translation_error,
                rotation_error_deg=rotation_error_deg,
                seed_distance_rad=float(np.linalg.norm(optimized.x - seed)),
                function_evaluations=int(optimized.nfev),
                optimizer_cost=float(optimized.cost),
            )
        )
    if not candidates:
        translation_mm = (
            float("nan") if best_failure is None else 1000.0 * best_failure[0]
        )
        rotation_deg = float("nan") if best_failure is None else best_failure[1]
        raise ValueError(
            "no bounded IK solution met the Cartesian tolerances; best error was "
            f"{translation_mm:.3f}mm and {rotation_deg:.3f}deg"
        )
    candidates.sort(
        key=lambda item: (
            item.seed_distance_rad,
            item.translation_error_m,
            item.rotation_error_deg,
        )
    )
    return tuple(candidates)
