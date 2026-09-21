#!/usr/bin/env python3
"""Measure native Ferguson calibration generalization versus training pose count."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from g1_aprilcube_calibration.authored_collection import (
    modeled_hand_T_target_from_hardware,
)
from g1_aprilcube_calibration.calibration_evaluation import (
    NativeCalibrationProjection,
)
from g1_aprilcube_calibration.dataset_builder import CalibrationDataset
from g1_aprilcube_calibration.robot_calibration_bridge import (
    solve_robot_calibration_dataset,
)
from g1_aprilcube_calibration.urdf_model import URDFModel


def _folds(pose_ids: tuple[str, ...], count: int, seed: int) -> tuple[set[str], ...]:
    ranked = sorted(
        pose_ids,
        key=lambda pose_id: hashlib.sha256(f"fold:{seed}:{pose_id}".encode()).digest(),
    )
    return tuple(set(ranked[index::count]) for index in range(count))


def _ranked_training(
    pose_ids: list[str], seed: int, fold: int, replicate: int
) -> list[str]:
    return sorted(
        pose_ids,
        key=lambda pose_id: hashlib.sha256(
            f"train:{seed}:{fold}:{replicate}:{pose_id}".encode()
        ).digest(),
    )


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "mean": float(np.mean(array)),
        "stddev": float(np.std(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--hardware-config", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot-calibration-directory", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--free-joint-offset", action="append", default=[])
    parser.add_argument("--target-mode", choices=("fixed", "optimize"), required=True)
    parser.add_argument("--training-size", type=int, action="append", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--joint-offset-prior-sigma-deg", type=float, default=5.0)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"analysis output already exists: {args.output}")
    dataset = CalibrationDataset.from_json(args.dataset)
    model = URDFModel(args.urdf)
    hardware = yaml.safe_load(args.hardware_config.read_text(encoding="utf-8"))
    initial_target = modeled_hand_T_target_from_hardware(hardware)
    projection = NativeCalibrationProjection(
        model,
        calibration_arm=dataset.calibration_arm,
        fixed_hand_T_target=(initial_target if args.target_mode == "fixed" else None),
    )
    pose_ids = tuple(sorted({sample.pose_id for sample in dataset.samples}))
    folds = _folds(pose_ids, args.folds, args.seed)
    results = []
    with tempfile.TemporaryDirectory(prefix="g1_learning_curve_") as temporary:
        root = Path(temporary)
        for requested_size in sorted(set(args.training_size)):
            trials = []
            for fold, holdout_ids in enumerate(folds):
                pool = sorted(set(pose_ids) - holdout_ids)
                size = min(requested_size, len(pool))
                replicate_count = 1 if size == len(pool) else args.replicates
                for replicate in range(replicate_count):
                    selected_ids = set(
                        _ranked_training(pool, args.seed, fold, replicate)[:size]
                    )
                    training = tuple(
                        sample
                        for sample in dataset.samples
                        if sample.pose_id in selected_ids
                    )
                    holdout = tuple(
                        sample
                        for sample in dataset.samples
                        if sample.pose_id in holdout_ids
                    )
                    solution = solve_robot_calibration_dataset(
                        replace(dataset, samples=training),
                        model,
                        root / f"n{size:03d}_f{fold:02d}_r{replicate:02d}",
                        initial_hand_T_target=initial_target,
                        initial_target_source="configured_nominal_palm_T_marker",
                        optimize_hand_target=args.target_mode == "optimize",
                        free_joint_offsets=tuple(args.free_joint_offset),
                        joint_offset_prior_sigma_deg=args.joint_offset_prior_sigma_deg,
                        robot_calibration_directory=args.robot_calibration_directory,
                        runner_path=args.runner,
                        timeout_s=args.timeout_s,
                    )
                    residual = projection.pixel_residuals(
                        solution.offsets, holdout
                    ).reshape(-1, 2)
                    trials.append(
                        {
                            "fold": fold,
                            "replicate": replicate,
                            "actual_training_pose_count": size,
                            "holdout_radial_rms_px": float(
                                np.sqrt(np.mean(np.sum(np.square(residual), axis=1)))
                            ),
                            "observable": solution.observability.observable,
                            "rank": solution.observability.rank,
                            "parameter_count": solution.observability.parameter_count,
                            "condition_number": solution.observability.condition_number,
                        }
                    )
            values = [trial["holdout_radial_rms_px"] for trial in trials]
            item = {
                "requested_training_pose_count": requested_size,
                "actual_training_pose_count_range": [
                    min(trial["actual_training_pose_count"] for trial in trials),
                    max(trial["actual_training_pose_count"] for trial in trials),
                ],
                "trial_count": len(trials),
                "all_trials_observable": all(trial["observable"] for trial in trials),
                "holdout_radial_rms_px": _summary(values),
                "trials": trials,
            }
            results.append(item)
            print(
                f"n={requested_size}: holdout={item['holdout_radial_rms_px']['mean']:.3f}"
                f" +/- {item['holdout_radial_rms_px']['stddev']:.3f}px, "
                f"observable={item['all_trials_observable']}",
                flush=True,
            )
    document = {
        "schema_version": 1,
        "dataset_path": str(args.dataset),
        "dataset_sha256": dataset.content_sha256,
        "calibration_arm": dataset.calibration_arm,
        "target_mode": args.target_mode,
        "free_joint_offsets": args.free_joint_offset,
        "fold_count": args.folds,
        "replicates": args.replicates,
        "seed": args.seed,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
