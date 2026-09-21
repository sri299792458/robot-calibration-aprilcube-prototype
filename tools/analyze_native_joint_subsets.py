#!/usr/bin/env python3
"""Cross-validate every static arm-joint-offset subset with Ferguson/Ceres."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

from g1_aprilcube_calibration.authored_collection import (
    modeled_hand_T_target_from_hardware,
    validate_hardware_target_profile,
)
from g1_aprilcube_calibration.calibration_evaluation import (
    NativeCalibrationProjection,
)
from g1_aprilcube_calibration.dataset_builder import CalibrationDataset
from g1_aprilcube_calibration.joint_map import arm_joint_names
from g1_aprilcube_calibration.robot_calibration_bridge import (
    solve_robot_calibration_dataset,
)
from g1_aprilcube_calibration.transforms import transform_to_pose_vector
from g1_aprilcube_calibration.urdf_model import URDFModel


def _fold_pose_ids(
    dataset: CalibrationDataset, folds: int, seed: int, mode: str
) -> tuple[tuple[str, ...], ...]:
    pose_ids = sorted({sample.pose_id for sample in dataset.samples})
    if not 2 <= folds <= len(pose_ids):
        raise ValueError("fold count must lie between two and the pose count")
    if mode == "hash":
        ranked = sorted(
            pose_ids,
            key=lambda pose_id: hashlib.sha256(f"{seed}:{pose_id}".encode()).digest(),
        )
        return tuple(tuple(ranked[index::folds]) for index in range(folds))
    if mode != "chronological":
        raise ValueError(f"unsupported fold mode: {mode}")
    receipt_by_pose = {
        sample.pose_id: str(sample.pairing["image_receipt_utc"])
        for sample in dataset.samples
    }
    ranked = sorted(pose_ids, key=lambda pose_id: receipt_by_pose[pose_id])
    base, remainder = divmod(len(ranked), folds)
    result = []
    start = 0
    for index in range(folds):
        stop = start + base + (1 if index < remainder else 0)
        result.append(tuple(ranked[start:stop]))
        start = stop
    return tuple(result)


def _joint_subsets(
    names: tuple[str, ...], maximum_size: int
) -> tuple[tuple[str, ...], ...]:
    if not 0 <= maximum_size <= len(names):
        raise ValueError("maximum subset size is outside the arm joint count")
    return tuple(
        subset
        for size in range(maximum_size + 1)
        for subset in combinations(names, size)
    )


def _transform_delta(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    vector = transform_to_pose_vector(np.linalg.inv(first) @ second)
    return float(np.linalg.norm(vector[:3]) * 1000.0), float(
        np.degrees(np.linalg.norm(vector[3:]))
    )


def _run_task(task: dict) -> dict:
    dataset = CalibrationDataset.from_json(task["dataset"])
    excluded = set(task["exclude_capture_ids"])
    if excluded:
        dataset = replace(
            dataset,
            samples=tuple(
                sample
                for sample in dataset.samples
                if sample.capture_id not in excluded
            ),
        )
    model = URDFModel(task["urdf"])
    held_out = set(task["holdout_pose_ids"])
    training = tuple(
        sample for sample in dataset.samples if sample.pose_id not in held_out
    )
    holdout = tuple(sample for sample in dataset.samples if sample.pose_id in held_out)
    training_dataset = replace(dataset, samples=training)
    output = Path(task["temporary_root"]) / task["task_id"]
    try:
        solution = solve_robot_calibration_dataset(
            training_dataset,
            model,
            output,
            initial_hand_T_target=np.asarray(task["initial_hand_T_target"]),
            initial_target_source="configured_nominal_palm_T_marker",
            optimize_hand_target=task["target_mode"] == "optimize",
            free_joint_offsets=tuple(task["joint_subset"]),
            joint_offset_prior_sigma_deg=task["joint_offset_prior_sigma_deg"],
            robot_calibration_directory=task["robot_calibration_directory"],
            runner_path=task["runner"],
            timeout_s=task["timeout_s"],
        )
        projection = NativeCalibrationProjection(
            model,
            calibration_arm=dataset.calibration_arm,
            fixed_hand_T_target=(
                None
                if task["target_mode"] == "optimize"
                else np.asarray(task["initial_hand_T_target"])
            ),
        )
        train_xy = projection.pixel_residuals(solution.offsets, training).reshape(-1, 2)
        holdout_xy = projection.pixel_residuals(solution.offsets, holdout).reshape(
            -1, 2
        )
        per_capture = []
        for sample in holdout:
            xy = projection.pixel_residuals(solution.offsets, (sample,)).reshape(-1, 2)
            per_capture.append(
                {
                    "capture_id": sample.capture_id,
                    "pose_id": sample.pose_id,
                    "rms_px": float(np.sqrt(np.mean(np.sum(np.square(xy), axis=1)))),
                }
            )
        return {
            "task_id": task["task_id"],
            "target_mode": task["target_mode"],
            "joint_subset": task["joint_subset"],
            "joint_count": len(task["joint_subset"]),
            "fold": task["fold"],
            "training_corner_count": len(train_xy),
            "training_squared_radial_error": float(np.sum(np.square(train_xy))),
            "holdout_corner_count": len(holdout_xy),
            "holdout_squared_radial_error": float(np.sum(np.square(holdout_xy))),
            "holdout_per_capture": per_capture,
            "rank": solution.observability.rank,
            "parameter_count": solution.observability.parameter_count,
            "condition_number": solution.observability.condition_number,
            "observable": solution.observability.observable,
            "offsets": dict(solution.offsets),
            "torso_T_camera": solution.torso_T_camera.tolist(),
            "hand_T_target": solution.hand_T_target.tolist(),
        }
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return {
            "task_id": task["task_id"],
            "target_mode": task["target_mode"],
            "joint_subset": task["joint_subset"],
            "joint_count": len(task["joint_subset"]),
            "fold": task["fold"],
            "error": f"{type(error).__name__}: {error}",
        }


def _summarize(
    tasks: list[dict], results: list[dict], dataset: CalibrationDataset
) -> dict:
    failures = [result for result in results if "error" in result]
    grouped: dict[tuple[str, tuple[str, ...]], list[dict]] = {}
    for result in results:
        if "error" in result:
            continue
        key = (result["target_mode"], tuple(result["joint_subset"]))
        grouped.setdefault(key, []).append(result)
    models = []
    for (target_mode, subset), folds in grouped.items():
        holdout_sum = sum(item["holdout_squared_radial_error"] for item in folds)
        holdout_count = sum(item["holdout_corner_count"] for item in folds)
        train_sum = sum(item["training_squared_radial_error"] for item in folds)
        train_count = sum(item["training_corner_count"] for item in folds)
        cameras = [np.asarray(item["torso_T_camera"]) for item in folds]
        camera_deltas = [
            _transform_delta(cameras[first], cameras[second])
            for first in range(len(cameras))
            for second in range(first + 1, len(cameras))
        ]
        conditions = [
            item["condition_number"]
            for item in folds
            if item["condition_number"] is not None
        ]
        models.append(
            {
                "target_mode": target_mode,
                "joint_subset": list(subset),
                "joint_count": len(subset),
                "parameter_count": folds[0]["parameter_count"],
                "successful_folds": len(folds),
                "all_folds_observable": all(item["observable"] for item in folds),
                "minimum_rank": min(item["rank"] for item in folds),
                "cv_training_radial_rms_px": float(np.sqrt(train_sum / train_count)),
                "cv_holdout_radial_rms_px": float(np.sqrt(holdout_sum / holdout_count)),
                "maximum_condition_number": max(conditions) if conditions else None,
                "camera_pairwise_translation_max_mm": (
                    max(item[0] for item in camera_deltas) if camera_deltas else 0.0
                ),
                "camera_pairwise_rotation_max_deg": (
                    max(item[1] for item in camera_deltas) if camera_deltas else 0.0
                ),
                "folds": sorted(folds, key=lambda item: item["fold"]),
            }
        )
    models.sort(
        key=lambda item: (
            not item["all_folds_observable"],
            item["cv_holdout_radial_rms_px"],
            item["joint_count"],
        )
    )
    return {
        "schema_version": 1,
        "dataset_path": tasks[0]["dataset"],
        "dataset_sha256": dataset.content_sha256,
        "source_dataset_sha256": tasks[0]["source_dataset_sha256"],
        "excluded_capture_ids": tasks[0]["exclude_capture_ids"],
        "calibration_arm": dataset.calibration_arm,
        "sample_count": len(dataset.samples),
        "pose_count": len({sample.pose_id for sample in dataset.samples}),
        "fold_count": len({task["fold"] for task in tasks}),
        "fold_mode": tasks[0]["fold_mode"],
        "fold_seed": tasks[0]["fold_seed"],
        "target_modes": sorted({task["target_mode"] for task in tasks}),
        "model_count": len(grouped),
        "task_count": len(tasks),
        "failure_count": len(failures),
        "failures": failures,
        "models": models,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--hardware-config", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot-calibration-directory", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument(
        "--fold-mode", choices=("hash", "chronological"), default="hash"
    )
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 6))
    parser.add_argument("--maximum-subset-size", type=int, default=7)
    parser.add_argument(
        "--only-joint-subset",
        action="append",
        help=(
            "evaluate only this comma-separated set of full joint names; "
            "repeat for multiple selected models"
        ),
    )
    parser.add_argument("--target-mode", choices=("fixed", "optimize"), action="append")
    parser.add_argument("--joint-offset-prior-sigma-deg", type=float, default=5.0)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--exclude-capture-id", action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"analysis output already exists: {args.output}")
    source_dataset = CalibrationDataset.from_json(args.dataset)
    excluded = tuple(dict.fromkeys(args.exclude_capture_id))
    unknown_excluded = sorted(
        set(excluded) - {sample.capture_id for sample in source_dataset.samples}
    )
    if unknown_excluded:
        raise ValueError(
            "excluded capture IDs are absent from the dataset: "
            + ", ".join(unknown_excluded)
        )
    dataset = replace(
        source_dataset,
        samples=tuple(
            sample
            for sample in source_dataset.samples
            if sample.capture_id not in set(excluded)
        ),
    )
    if len(dataset.samples) < 2:
        raise ValueError("capture exclusion left fewer than two samples")
    model = URDFModel(args.urdf)
    if dataset.urdf_sha256 != model.sha256:
        raise ValueError("dataset and requested URDF hashes do not match")
    hardware = yaml.safe_load(args.hardware_config.read_text(encoding="utf-8"))
    target = json.loads(args.target_config.read_text(encoding="utf-8"))
    validate_hardware_target_profile(hardware, target)
    if hardware["robot"]["calibration_arm"] != dataset.calibration_arm:
        raise ValueError("hardware profile arm differs from dataset")
    if (
        hashlib.sha256(args.target_config.read_bytes()).hexdigest()
        != dataset.target_artifact_sha256
    ):
        raise ValueError("target profile hash differs from dataset")
    initial_target = modeled_hand_T_target_from_hardware(hardware)
    folds = _fold_pose_ids(dataset, args.folds, args.seed, args.fold_mode)
    allowed_joints = arm_joint_names(dataset.calibration_arm)
    if args.only_joint_subset:
        subsets = tuple(
            tuple(item for item in value.split(",") if item)
            for value in args.only_joint_subset
        )
        invalid = sorted(
            {
                name
                for subset in subsets
                for name in subset
                if name not in allowed_joints
            }
        )
        if invalid:
            raise ValueError(
                "selected offsets do not belong to the calibration arm: "
                + ", ".join(invalid)
            )
        if any(len(subset) != len(set(subset)) for subset in subsets):
            raise ValueError("selected joint subsets cannot contain duplicates")
        subsets = tuple(dict.fromkeys(subsets))
    else:
        subsets = _joint_subsets(allowed_joints, args.maximum_subset_size)
    modes = tuple(dict.fromkeys(args.target_mode or ("fixed", "optimize")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"g1_{dataset.calibration_arm}_sweep_"
    ) as temporary:
        tasks = []
        for mode in modes:
            for subset_index, subset in enumerate(subsets):
                for fold_index, holdout_pose_ids in enumerate(folds):
                    tasks.append(
                        {
                            "task_id": f"{mode}_{subset_index:03d}_fold_{fold_index:02d}",
                            "dataset": str(args.dataset.resolve()),
                            "source_dataset_sha256": source_dataset.content_sha256,
                            "exclude_capture_ids": list(excluded),
                            "urdf": str(args.urdf.resolve()),
                            "temporary_root": temporary,
                            "robot_calibration_directory": str(
                                args.robot_calibration_directory.resolve()
                            ),
                            "runner": str(args.runner.resolve()),
                            "target_mode": mode,
                            "joint_subset": list(subset),
                            "fold": fold_index,
                            "fold_mode": args.fold_mode,
                            "fold_seed": args.seed,
                            "holdout_pose_ids": list(holdout_pose_ids),
                            "initial_hand_T_target": initial_target.tolist(),
                            "joint_offset_prior_sigma_deg": args.joint_offset_prior_sigma_deg,
                            "timeout_s": args.timeout_s,
                        }
                    )
        print(
            f"running {len(tasks)} native solves: arm={dataset.calibration_arm}, "
            f"models={len(subsets) * len(modes)}, folds={args.folds}, workers={args.workers}",
            flush=True,
        )
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_run_task, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if completed % 50 == 0 or completed == len(tasks):
                    print(
                        f"completed {completed}/{len(tasks)} native solves", flush=True
                    )
        document = _summarize(tasks, results, dataset)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "models": document["model_count"],
                "failures": document["failure_count"],
                "best_models": [
                    {
                        "target_mode": item["target_mode"],
                        "joint_subset": item["joint_subset"],
                        "holdout_rms_px": item["cv_holdout_radial_rms_px"],
                        "observable": item["all_folds_observable"],
                    }
                    for item in document["models"][:10]
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
