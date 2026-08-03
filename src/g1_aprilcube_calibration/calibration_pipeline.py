"""Train/holdout solve orchestration and pose-level bootstrap stability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from g1_aprilcube_calibration.calibration_solver import (
    DegenerateCalibrationError,
    ExtrinsicsSolver,
    SolveResult,
)
from g1_aprilcube_calibration.dataset_builder import (
    CalibrationDataset,
    CalibrationSample,
    deterministic_holdout_split,
)
from g1_aprilcube_calibration.residual_report import (
    CalibrationRunExporter,
    ResidualReport,
    build_residual_report,
)


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    holdout_fraction: float = 0.2
    bootstrap_trials: int = 50
    bootstrap_seed: int = 17

    def __post_init__(self) -> None:
        if not 0 < self.holdout_fraction < 1:
            raise ValueError("holdout fraction must lie strictly between zero and one")
        if self.bootstrap_trials < 0:
            raise ValueError("bootstrap trials cannot be negative")
        if self.bootstrap_seed < 0:
            raise ValueError("bootstrap seed cannot be negative")


@dataclass(frozen=True, slots=True)
class BootstrapReport:
    requested_trials: int
    successful_trials: int
    degenerate_trials: int
    parameter_stddev: tuple[float | None, ...]

    def to_dict(self) -> dict:
        return {
            "requested_trials": self.requested_trials,
            "successful_trials": self.successful_trials,
            "degenerate_trials": self.degenerate_trials,
            "parameter_stddev": list(self.parameter_stddev),
        }


@dataclass(frozen=True, slots=True)
class PipelineResult:
    solution: SolveResult
    residuals: ResidualReport
    bootstrap: BootstrapReport
    training_samples: tuple[CalibrationSample, ...]
    holdout_samples: tuple[CalibrationSample, ...]


class CalibrationPipeline:
    def __init__(
        self,
        solver: ExtrinsicsSolver,
        *,
        config: PipelineConfig | None = None,
    ) -> None:
        self.solver = solver
        self.config = config or PipelineConfig()

    def run(
        self,
        dataset: CalibrationDataset,
        *,
        initial_torso_T_camera: np.ndarray,
        initial_hand_T_target: np.ndarray,
        output_directory: str | Path | None = None,
        provenance: dict | None = None,
    ) -> PipelineResult:
        if dataset.urdf_sha256 != self.solver.model.sha256:
            raise ValueError("dataset and solver URDF hashes do not match")
        if dataset.calibration_arm != self.solver.calibration_arm:
            raise ValueError("dataset and solver calibration arms do not match")
        training, holdout = deterministic_holdout_split(
            dataset, holdout_fraction=self.config.holdout_fraction
        )
        solution = self.solver.solve(
            training,
            initial_torso_T_camera=initial_torso_T_camera,
            initial_hand_T_target=initial_hand_T_target,
        )
        residuals = build_residual_report(
            self.solver,
            solution,
            training_samples=training,
            holdout_samples=holdout,
        )
        bootstrap = self._bootstrap(training, solution)
        result = PipelineResult(solution, residuals, bootstrap, training, holdout)
        if output_directory is not None:
            CalibrationRunExporter().export(
                output_directory,
                dataset=dataset,
                result=solution,
                residuals=residuals,
                training_capture_ids=[sample.capture_id for sample in training],
                holdout_capture_ids=[sample.capture_id for sample in holdout],
                provenance={} if provenance is None else provenance,
                bootstrap=bootstrap.to_dict(),
            )
        return result

    def _bootstrap(
        self,
        training: tuple[CalibrationSample, ...],
        solution: SolveResult,
    ) -> BootstrapReport:
        if self.config.bootstrap_trials == 0:
            return BootstrapReport(0, 0, 0, (0.0,) * 12)
        rng = np.random.default_rng(self.config.bootstrap_seed)
        parameters: list[np.ndarray] = []
        degenerate = 0
        by_pose: dict[str, list[CalibrationSample]] = {}
        for sample in training:
            by_pose.setdefault(sample.pose_id, []).append(sample)
        pose_ids = tuple(sorted(by_pose))
        for _ in range(self.config.bootstrap_trials):
            indices = rng.integers(0, len(pose_ids), size=len(pose_ids))
            resampled = tuple(
                sample for index in indices for sample in by_pose[pose_ids[int(index)]]
            )
            try:
                trial = self.solver.solve(
                    resampled,
                    initial_torso_T_camera=solution.torso_T_camera,
                    initial_hand_T_target=solution.hand_T_target,
                )
            except (DegenerateCalibrationError, RuntimeError):
                degenerate += 1
                continue
            parameters.append(np.asarray(trial.parameters))
        if parameters:
            standard_deviation = np.std(np.vstack(parameters), axis=0, ddof=0)
        else:
            standard_deviation = np.full(12, np.nan)
        return BootstrapReport(
            requested_trials=self.config.bootstrap_trials,
            successful_trials=len(parameters),
            degenerate_trials=degenerate,
            parameter_stddev=tuple(
                None if not np.isfinite(value) else float(value)
                for value in standard_deviation
            ),
        )
