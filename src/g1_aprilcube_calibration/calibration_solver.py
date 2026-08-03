"""Twelve-parameter camera/target extrinsic least-squares calibration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.dataset_builder import CalibrationSample
from g1_aprilcube_calibration.joint_map import (
    G1_29_JOINT_NAMES,
    arm_hand_link,
    arm_joint_names,
    validate_arm_side,
)
from g1_aprilcube_calibration.transforms import (
    invert_transform,
    pose_vector_to_transform,
    transform_points,
    transform_to_pose_vector,
    validate_transform,
)
from g1_aprilcube_calibration.urdf_model import URDFModel

PARAMETER_NAMES = (
    "torso_T_camera_tx_m",
    "torso_T_camera_ty_m",
    "torso_T_camera_tz_m",
    "torso_T_camera_rx_rad",
    "torso_T_camera_ry_rad",
    "torso_T_camera_rz_rad",
    "hand_T_target_tx_m",
    "hand_T_target_ty_m",
    "hand_T_target_tz_m",
    "hand_T_target_rx_rad",
    "hand_T_target_ry_rad",
    "hand_T_target_rz_rad",
)


class DegenerateCalibrationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SolverConfig:
    robust_loss: str = "huber"
    robust_scale_px: float = 2.0
    maximum_evaluations: int = 500
    minimum_depth_m: float = 0.05
    depth_penalty_px_per_m: float = 1000.0
    jacobian_relative_rank_threshold: float = 1e-7
    maximum_condition_number: float = 1e8

    def __post_init__(self) -> None:
        if self.robust_loss not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
            raise ValueError("unsupported scipy least-squares loss")
        for name in (
            "robust_scale_px",
            "maximum_evaluations",
            "minimum_depth_m",
            "depth_penalty_px_per_m",
            "jacobian_relative_rank_threshold",
            "maximum_condition_number",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class ObservabilityReport:
    rank: int
    parameter_count: int
    singular_values: tuple[float, ...]
    condition_number: float
    observable: bool
    parameter_stddev: tuple[float, ...]

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "parameter_count": self.parameter_count,
            "singular_values": list(self.singular_values),
            "condition_number": self.condition_number,
            "observable": self.observable,
            "parameter_stddev": list(self.parameter_stddev),
        }


@dataclass(frozen=True, slots=True)
class SolveResult:
    torso_T_camera: np.ndarray
    hand_T_target: np.ndarray
    parameters: tuple[float, ...]
    parameter_names: tuple[str, ...]
    success: bool
    message: str
    evaluations: int
    cost: float
    optimization_rms_px: float
    observability: ObservabilityReport

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "torso_T_camera", validate_transform(self.torso_T_camera)
        )
        object.__setattr__(
            self, "hand_T_target", validate_transform(self.hand_T_target)
        )

    def to_dict(self) -> dict:
        return {
            "torso_T_camera": self.torso_T_camera.tolist(),
            "hand_T_target": self.hand_T_target.tolist(),
            "parameters": dict(zip(self.parameter_names, self.parameters, strict=True)),
            "success": self.success,
            "message": self.message,
            "evaluations": self.evaluations,
            "cost": self.cost,
            "optimization_rms_px": self.optimization_rms_px,
            "observability": self.observability.to_dict(),
        }


class ExtrinsicsSolver:
    """Fit the camera and configured carrier-hand target transforms jointly."""

    def __init__(
        self,
        model: URDFModel,
        *,
        torso_link: str = "torso_link",
        calibration_arm: str = "left",
        config: SolverConfig | None = None,
    ) -> None:
        self.model = model
        self.torso_link = torso_link
        self.calibration_arm = validate_arm_side(calibration_arm)
        self.hand_link = arm_hand_link(self.calibration_arm)
        self.config = config or SolverConfig()
        chain_names = [joint.name for joint in model.chain(torso_link, self.hand_link)]
        expected = list(arm_joint_names(self.calibration_arm))
        movable = [
            name
            for name in chain_names
            if name in model.joints and model.joints[name].joint_type != "fixed"
        ]
        if movable != expected:
            raise ValueError(
                "solver URDF hand chain does not match the configured mode-5 "
                f"{self.calibration_arm} arm"
            )

    def solve(
        self,
        samples: Sequence[CalibrationSample],
        *,
        initial_torso_T_camera: np.ndarray,
        initial_hand_T_target: np.ndarray,
        require_observable: bool = True,
    ) -> SolveResult:
        sample_tuple = tuple(samples)
        if len(sample_tuple) < 3:
            raise ValueError("at least three calibration poses are required")
        initial = np.concatenate(
            (
                transform_to_pose_vector(initial_torso_T_camera),
                transform_to_pose_vector(initial_hand_T_target),
            )
        )
        start = initial
        warm_evaluations = 0
        if self.config.robust_loss != "linear":
            warm = least_squares(
                lambda parameters: self.optimization_residuals(
                    parameters, sample_tuple
                ),
                initial,
                method="trf",
                loss="linear",
                max_nfev=self.config.maximum_evaluations,
                x_scale="jac",
            )
            if warm.success and np.all(np.isfinite(warm.x)):
                start = np.asarray(warm.x, dtype=np.float64)
            warm_evaluations = int(warm.nfev)
        raw = least_squares(
            lambda parameters: self.optimization_residuals(parameters, sample_tuple),
            start,
            method="trf",
            loss=self.config.robust_loss,
            f_scale=self.config.robust_scale_px,
            max_nfev=self.config.maximum_evaluations,
            x_scale="jac",
        )
        parameters = np.asarray(raw.x, dtype=np.float64)
        pixel_residuals = self.pixel_residuals(parameters, sample_tuple)
        observability = self._observability(raw.jac, pixel_residuals)
        result = SolveResult(
            torso_T_camera=pose_vector_to_transform(parameters[:6]),
            hand_T_target=pose_vector_to_transform(parameters[6:]),
            parameters=tuple(float(value) for value in parameters),
            parameter_names=PARAMETER_NAMES,
            success=bool(raw.success),
            message=str(raw.message),
            evaluations=warm_evaluations + int(raw.nfev),
            cost=float(raw.cost),
            optimization_rms_px=float(np.sqrt(np.mean(np.square(pixel_residuals)))),
            observability=observability,
        )
        if not result.success:
            raise RuntimeError(f"extrinsic optimization failed: {result.message}")
        if require_observable and not observability.observable:
            raise DegenerateCalibrationError(
                f"12-parameter solve is degenerate: rank={observability.rank}/12, "
                f"condition={observability.condition_number:.3g}"
            )
        return result

    def optimization_residuals(
        self,
        parameters: np.ndarray,
        samples: Sequence[CalibrationSample],
    ) -> np.ndarray:
        pixels, depths = self._project(parameters, samples)
        observed = np.vstack(
            [np.asarray(sample.image_points_px, dtype=np.float64) for sample in samples]
        )
        pixel_residuals = (pixels - observed).reshape(-1)
        depth_penalties = np.maximum(self.config.minimum_depth_m - depths, 0.0)
        return np.concatenate(
            (pixel_residuals, depth_penalties * self.config.depth_penalty_px_per_m)
        )

    def pixel_residuals(
        self,
        parameters: np.ndarray,
        samples: Sequence[CalibrationSample],
    ) -> np.ndarray:
        pixels, _ = self._project(parameters, samples)
        observed = np.vstack(
            [np.asarray(sample.image_points_px, dtype=np.float64) for sample in samples]
        )
        return (pixels - observed).reshape(-1)

    def project_sample(
        self, sample: CalibrationSample, result: SolveResult
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._project(np.asarray(result.parameters), (sample,))

    def _project(
        self,
        parameters: np.ndarray,
        samples: Sequence[CalibrationSample],
    ) -> tuple[np.ndarray, np.ndarray]:
        vector = np.asarray(parameters, dtype=np.float64).reshape(-1)
        if vector.shape != (12,) or not np.all(np.isfinite(vector)):
            raise ValueError("solver parameter vector must contain 12 finite values")
        torso_T_camera = pose_vector_to_transform(vector[:6])
        camera_T_torso = invert_transform(torso_T_camera)
        hand_T_target = pose_vector_to_transform(vector[6:])
        all_pixels: list[np.ndarray] = []
        all_depths: list[np.ndarray] = []
        for sample in samples:
            state = sample.measured_state
            if state.get("mode_machine") != 5:
                raise ValueError("calibration sample is not from mode_machine=5")
            position = np.asarray(state["position"], dtype=np.float64)
            if position.shape != (29,) or not np.all(np.isfinite(position)):
                raise ValueError(
                    "calibration sample contains an invalid measured state"
                )
            positions = dict(zip(G1_29_JOINT_NAMES, position, strict=True))
            torso_T_hand = self.model.transform(
                self.torso_link, self.hand_link, positions
            )
            target_points = np.asarray(sample.object_points_m, dtype=np.float64)
            torso_points = transform_points(torso_T_hand @ hand_T_target, target_points)
            camera_points = transform_points(camera_T_torso, torso_points)
            depths = camera_points[:, 2]
            safe_depths = np.where(
                np.abs(depths) < 1e-6,
                np.where(depths < 0, -1e-6, 1e-6),
                depths,
            )
            info = RectifiedCameraInfo.from_dict(sample.camera_info)
            matrix = info.rectified_camera_matrix
            pixels = np.column_stack(
                (
                    matrix[0, 0] * camera_points[:, 0] / safe_depths + matrix[0, 2],
                    matrix[1, 1] * camera_points[:, 1] / safe_depths + matrix[1, 2],
                )
            )
            all_pixels.append(pixels)
            all_depths.append(depths)
        return np.vstack(all_pixels), np.concatenate(all_depths)

    def _observability(
        self, jacobian: np.ndarray, pixel_residuals: np.ndarray
    ) -> ObservabilityReport:
        matrix = np.asarray(jacobian, dtype=np.float64)
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        maximum = float(singular_values[0]) if singular_values.size else 0.0
        threshold = maximum * self.config.jacobian_relative_rank_threshold
        rank = int(np.count_nonzero(singular_values > threshold))
        minimum = float(singular_values[-1]) if singular_values.size else 0.0
        condition = float("inf") if minimum <= 0 else maximum / minimum
        observable = rank == 12 and condition <= self.config.maximum_condition_number
        degrees_of_freedom = max(len(pixel_residuals) - 12, 1)
        variance = float(np.sum(np.square(pixel_residuals)) / degrees_of_freedom)
        covariance = variance * np.linalg.pinv(matrix.T @ matrix)
        standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        return ObservabilityReport(
            rank=rank,
            parameter_count=12,
            singular_values=tuple(float(value) for value in singular_values),
            condition_number=condition,
            observable=observable,
            parameter_stddev=tuple(float(value) for value in standard_deviation),
        )
