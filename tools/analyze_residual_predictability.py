#!/usr/bin/env python3
"""Test whether held-out native residuals are predictable from robot state.

This is a diagnostic, not a calibration backend.  Every geometric parameter is
fit by Ferguson/Ceres; ridge regressions only test whether systematic structure
remains in held-out pixel residuals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from g1_aprilcube_calibration.urdf_model import URDFModel


def _features(sample, arm_indices: tuple[int, ...], mode: str) -> np.ndarray:
    q = np.asarray(sample.measured_state["position"], dtype=np.float64)[
        list(arm_indices)
    ]
    torque = np.asarray(sample.measured_state["estimated_torque"], dtype=np.float64)[
        list(arm_indices)
    ]
    if mode == "constant":
        return np.empty(0)
    if mode == "joints":
        return q
    if mode == "joints_torque":
        return np.concatenate((q, torque))
    if mode == "quadratic_joints_torque":
        return np.concatenate((q, np.square(q), torque))
    raise ValueError(f"unsupported feature mode: {mode}")


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, ...]:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (x - mean) / scale
    design = np.column_stack((np.ones(len(x)), standardized))
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return mean, scale, coefficients


def _ridge_predict(x: np.ndarray, model: tuple[np.ndarray, ...]) -> np.ndarray:
    mean, scale, coefficients = model
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    return design @ coefficients


def _inner_groups(pose_ids: list[str], fold: int) -> np.ndarray:
    return np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(f"{fold}:{pose_id}".encode()).digest()[:4], "big"
            )
            % 3
            for pose_id in pose_ids
        ]
    )


def _select_alpha(
    x: np.ndarray, y: np.ndarray, pose_ids: list[str], fold: int
) -> float:
    if x.shape[1] == 0:
        return 0.0
    candidates = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
    groups = _inner_groups(pose_ids, fold)
    scores = []
    for alpha in candidates:
        squared = 0.0
        count = 0
        for group in range(3):
            train = groups != group
            valid = groups == group
            if np.count_nonzero(train) <= x.shape[1] or not np.any(valid):
                continue
            prediction = _ridge_predict(x[valid], _ridge_fit(x[train], y[train], alpha))
            squared += float(np.sum(np.square(prediction - y[valid])))
            count += int(np.size(prediction))
        scores.append((squared / count if count else np.inf, alpha))
    return min(scores)[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--hardware-config", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"analysis output already exists: {args.output}")
    dataset = CalibrationDataset.from_json(args.dataset)
    sweep = json.loads(args.sweep.read_text(encoding="utf-8"))
    model = URDFModel(args.urdf)
    hardware = yaml.safe_load(args.hardware_config.read_text(encoding="utf-8"))
    target = modeled_hand_T_target_from_hardware(hardware)
    selected = next(
        item
        for item in sweep["models"]
        if item["all_folds_observable"]
        and item["successful_folds"] == sweep["fold_count"]
    )
    projection = NativeCalibrationProjection(
        model,
        calibration_arm=dataset.calibration_arm,
        fixed_hand_T_target=(target if selected["target_mode"] == "fixed" else None),
    )
    arm_start = 15 if dataset.calibration_arm == "left" else 22
    arm_indices = tuple(range(arm_start, arm_start + 7))
    modes = ("constant", "joints", "joints_torque", "quadratic_joints_torque")
    accumulators = {mode: {"squared": 0.0, "count": 0, "alphas": []} for mode in modes}
    baseline_squared = 0.0
    baseline_count = 0
    folds = []
    for fold_result in selected["folds"]:
        holdout_ids = {entry["pose_id"] for entry in fold_result["holdout_per_capture"]}
        train_samples = [
            sample for sample in dataset.samples if sample.pose_id not in holdout_ids
        ]
        holdout_samples = [
            sample for sample in dataset.samples if sample.pose_id in holdout_ids
        ]
        offsets = fold_result["offsets"]

        train_y = np.vstack(
            [
                projection.pixel_residuals(offsets, (sample,)).reshape(1, -1)
                for sample in train_samples
            ]
        )
        holdout_y = np.vstack(
            [
                projection.pixel_residuals(offsets, (sample,)).reshape(1, -1)
                for sample in holdout_samples
            ]
        )
        baseline_squared += float(np.sum(np.square(holdout_y)))
        baseline_count += int(np.size(holdout_y) // 2)
        fold_summary = {"fold": fold_result["fold"], "models": {}}
        for mode in modes:
            train_x = np.vstack(
                [_features(sample, arm_indices, mode) for sample in train_samples]
            )
            holdout_x = np.vstack(
                [_features(sample, arm_indices, mode) for sample in holdout_samples]
            )
            alpha = _select_alpha(
                train_x,
                train_y,
                [sample.pose_id for sample in train_samples],
                fold_result["fold"],
            )
            prediction = _ridge_predict(holdout_x, _ridge_fit(train_x, train_y, alpha))
            corrected = holdout_y - prediction
            squared = float(np.sum(np.square(corrected)))
            count = int(np.size(corrected) // 2)
            accumulators[mode]["squared"] += squared
            accumulators[mode]["count"] += count
            accumulators[mode]["alphas"].append(alpha)
            fold_summary["models"][mode] = {
                "alpha": alpha,
                "corrected_radial_rms_px": float(np.sqrt(squared / count)),
            }
        folds.append(fold_summary)
    baseline = float(np.sqrt(baseline_squared / baseline_count))
    results = []
    for mode in modes:
        item = accumulators[mode]
        rms = float(np.sqrt(item["squared"] / item["count"]))
        results.append(
            {
                "feature_model": mode,
                "corrected_radial_rms_px": rms,
                "relative_squared_error_reduction": 1.0 - rms**2 / baseline**2,
                "selected_alphas": item["alphas"],
            }
        )
    document = {
        "schema_version": 1,
        "dataset_path": str(args.dataset),
        "dataset_sha256": dataset.content_sha256,
        "calibration_arm": dataset.calibration_arm,
        "native_model": {
            "target_mode": selected["target_mode"],
            "joint_subset": selected["joint_subset"],
        },
        "baseline_native_cv_radial_rms_px": baseline,
        "interpretation": (
            "Ridge models are diagnostics fit inside each native outer fold; "
            "they do not alter or replace Ferguson/Ceres calibration."
        ),
        "predictability_models": results,
        "folds": folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"{dataset.calibration_arm}: native CV={baseline:.4f}px")
    for result in results:
        print(
            f"  {result['feature_model']}: "
            f"{result['corrected_radial_rms_px']:.4f}px, "
            f"squared reduction={result['relative_squared_error_reduction']:.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
