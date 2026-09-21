#!/usr/bin/env python3
"""Cross-validate a shared camera against left- and right-arm datasets.

Mike Ferguson's robot_calibration/Ceres remains the only optimizer.  This tool
only constructs a two-arm calibration bag/configuration and evaluates the
native offsets on held-out observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.authored_collection import (
    modeled_hand_T_target_from_hardware,
    validate_hardware_target_profile,
)
from g1_aprilcube_calibration.calibration_evaluation import (
    CAMERA_MOUNT_JOINT,
    CAMERA_OFFSET_NAMES,
    NativeCalibrationProjection,
)
from g1_aprilcube_calibration.dataset_builder import (
    CalibrationDataset,
    CalibrationSample,
)
from g1_aprilcube_calibration.joint_map import (
    G1_29_JOINT_NAMES,
    arm_hand_link,
    arm_joint_names,
)
from g1_aprilcube_calibration.robot_calibration_bridge import (
    ROBOT_CALIBRATION_REVISION,
    _fill_camera_info,
    _parse_solver_output,
    _sample_stamp_ns,
    _verify_robot_calibration_revision,
    add_color_optical_frame,
)
from g1_aprilcube_calibration.transforms import (
    pose_vector_to_transform,
    transform_points,
    transform_to_pose_vector,
)
from g1_aprilcube_calibration.urdf_model import URDFModel

CAMERA_MODEL = "camera"
CAMERA_FRAME = "camera_color_optical_frame"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    target_mode: str
    camera_mode: str
    intrinsics_mode: str
    left_joints: tuple[str, ...]
    right_joints: tuple[str, ...]


def _parse_model_spec(value: str) -> ModelSpec:
    fields = value.split(":")
    if len(fields) == 4:
        name, target_mode, left_text, right_text = fields
        camera_mode = "all"
        intrinsics_mode = "fixed"
    elif len(fields) == 5:
        name, target_mode, camera_mode, left_text, right_text = fields
        intrinsics_mode = "fixed"
    elif len(fields) == 6:
        (
            name,
            target_mode,
            camera_mode,
            intrinsics_mode,
            left_text,
            right_text,
        ) = fields
    else:
        raise argparse.ArgumentTypeError(
            "model must be NAME:fixed|optimize[:CAMERA_MODE]:LEFT_SUFFIXES:RIGHT_SUFFIXES"
        )
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise argparse.ArgumentTypeError("model name contains invalid characters")
    if target_mode not in {"fixed", "optimize"}:
        raise argparse.ArgumentTypeError("target mode must be fixed or optimize")
    if camera_mode not in {"nominal", "pitch", "rotation", "translation_pitch", "all"}:
        raise argparse.ArgumentTypeError(
            "camera mode must be nominal, pitch, rotation, translation_pitch, or all"
        )
    if intrinsics_mode not in {"fixed", "focal", "all"}:
        raise argparse.ArgumentTypeError("intrinsics mode must be fixed, focal, or all")

    def joints(side: str, text: str) -> tuple[str, ...]:
        suffixes = tuple(item for item in text.split(",") if item)
        names = tuple(f"{side}_{suffix}_joint" for suffix in suffixes)
        invalid = sorted(set(names) - set(arm_joint_names(side)))
        if invalid:
            raise argparse.ArgumentTypeError(
                f"invalid {side} joint suffixes: {', '.join(invalid)}"
            )
        return tuple(dict.fromkeys(names))

    return ModelSpec(
        name,
        target_mode,
        camera_mode,
        intrinsics_mode,
        joints("left", left_text),
        joints("right", right_text),
    )


def _camera_components(mode: str) -> dict[str, bool]:
    enabled = {
        "nominal": (),
        "pitch": ("pitch",),
        "rotation": ("roll", "pitch", "yaw"),
        "translation_pitch": ("x", "y", "z", "pitch"),
        "all": ("x", "y", "z", "roll", "pitch", "yaw"),
    }[mode]
    return {name: name in enabled for name in ("x", "y", "z", "roll", "pitch", "yaw")}


def _camera_offset_names(mode: str) -> tuple[str, ...]:
    mapping = {
        "x": "d435_joint_x",
        "y": "d435_joint_y",
        "z": "d435_joint_z",
        "roll": "d435_joint_a",
        "pitch": "d435_joint_b",
        "yaw": "d435_joint_c",
    }
    return tuple(
        mapping[name] for name, enabled in _camera_components(mode).items() if enabled
    )


def _intrinsic_offset_names(mode: str) -> tuple[str, ...]:
    return {
        "fixed": (),
        "focal": ("camera_fx", "camera_fy"),
        "all": ("camera_fx", "camera_fy", "camera_cx", "camera_cy"),
    }[mode]


def _fold_pose_ids(
    dataset: CalibrationDataset, folds: int, seed: int
) -> tuple[tuple[str, ...], ...]:
    pose_ids = sorted({sample.pose_id for sample in dataset.samples})
    ranked = sorted(
        pose_ids,
        key=lambda pose_id: hashlib.sha256(f"{seed}:{pose_id}".encode()).digest(),
    )
    return tuple(tuple(ranked[index::folds]) for index in range(folds))


def _target_frame(side: str) -> str:
    return f"{side}_calibration_target"


def _target_offset_names(side: str) -> tuple[str, ...]:
    return tuple(
        f"{_target_frame(side)}_{suffix}" for suffix in ("x", "y", "z", "a", "b", "c")
    )


def _target_initial_values(transform: np.ndarray) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*")
        rpy = Rotation.from_matrix(transform[:3, :3].copy()).as_euler("xyz")
    return {
        "x": float(transform[0, 3]),
        "y": float(transform[1, 3]),
        "z": float(transform[2, 3]),
        "roll": float(rpy[0]),
        "pitch": float(rpy[1]),
        "yaw": float(rpy[2]),
    }


def _optimizer_config(
    spec: ModelSpec,
    sample_count: int,
    initial_targets: dict[str, np.ndarray],
    prior_sigma_deg: float,
) -> dict[str, Any]:
    models = ["left_arm", "right_arm", CAMERA_MODEL]
    step: dict[str, Any] = {
        "max_num_iterations": 1000,
        "models": models,
        "left_arm": {"type": "chain3d", "frame": arm_hand_link("left")},
        "right_arm": {"type": "chain3d", "frame": arm_hand_link("right")},
        CAMERA_MODEL: {
            "type": "camera2d",
            "frame": CAMERA_FRAME,
            "param_name": CAMERA_MODEL,
        },
        "free_frames": [],
        "error_blocks": ["left_reprojection", "right_reprojection"],
        "left_reprojection": {
            "type": "chain3d_to_camera2d",
            "model_3d": "left_arm",
            "model_2d": CAMERA_MODEL,
            "scale": 1.0,
        },
        "right_reprojection": {
            "type": "chain3d_to_camera2d",
            "model_3d": "right_arm",
            "model_2d": CAMERA_MODEL,
            "scale": 1.0,
        },
    }
    if spec.camera_mode != "nominal":
        step["free_frames"].append(CAMERA_MOUNT_JOINT)
        step[CAMERA_MOUNT_JOINT] = _camera_components(spec.camera_mode)
    if spec.target_mode == "optimize":
        step["free_frames_initial_values"] = []
        for side in ("left", "right"):
            frame = _target_frame(side)
            step["free_frames"].append(frame)
            step[frame] = {
                "x": True,
                "y": True,
                "z": True,
                "roll": True,
                "pitch": True,
                "yaw": True,
            }
            step["free_frames_initial_values"].append(frame)
            step[f"{frame}_initial_values"] = _target_initial_values(
                initial_targets[side]
            )
    free_joints = (*spec.left_joints, *spec.right_joints)
    free_parameters = (*free_joints, *_intrinsic_offset_names(spec.intrinsics_mode))
    if free_parameters:
        step["free_params"] = list(free_parameters)
    if free_joints:
        sigma_rad = math.radians(prior_sigma_deg)
        joint_scale = 1.0 / (sigma_rad * math.sqrt(sample_count))
        for index, joint_name in enumerate(free_joints):
            name = f"joint_offset_prior_{index:02d}"
            step["error_blocks"].append(name)
            step[name] = {
                "type": "outrageous",
                "param": joint_name,
                "joint_scale": joint_scale,
                "position_scale": 0.0,
                "rotation_scale": 0.0,
            }
    return {
        "robot_calibration": {
            "ros__parameters": {
                "verbose": True,
                "base_link": "torso_link",
                "calibration_steps": ["dual_arm_calibration"],
                "dual_arm_calibration": step,
            }
        }
    }


def _write_bag(
    samples: tuple[tuple[str, CalibrationSample], ...],
    robot_description: str,
    bag_directory: Path,
    spec: ModelSpec,
    initial_targets: dict[str, np.ndarray],
) -> None:
    import rosbag2_py
    from geometry_msgs.msg import PointStamped
    from rclpy.serialization import serialize_message
    from robot_calibration_msgs.msg import CalibrationData, Observation
    from std_msgs.msg import String

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(bag_directory), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            name="/robot_description",
            type="std_msgs/msg/String",
            serialization_format="cdr",
        )
    )
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            name="/calibration_data",
            type="robot_calibration_msgs/msg/CalibrationData",
            serialization_format="cdr",
        )
    )
    first_stamp_ns = _sample_stamp_ns(samples[0][1], fallback=1)
    writer.write(
        "/robot_description",
        serialize_message(String(data=robot_description)),
        first_stamp_ns,
    )
    previous_stamp_ns = first_stamp_ns
    for index, (side, sample) in enumerate(samples, start=1):
        position = np.asarray(sample.measured_state["position"], dtype=np.float64)
        object_points = np.asarray(sample.object_points_m, dtype=np.float64)
        if spec.target_mode == "fixed":
            object_points = transform_points(initial_targets[side], object_points)
            feature_frame = arm_hand_link(side)
        else:
            feature_frame = _target_frame(side)
        message = CalibrationData()
        message.joint_states.name = list(G1_29_JOINT_NAMES)
        message.joint_states.position = position.tolist()
        arm = Observation(sensor_name=f"{side}_arm")
        camera = Observation(sensor_name=CAMERA_MODEL)
        for object_point, image_point in zip(
            object_points, sample.image_points_px, strict=True
        ):
            arm_feature = PointStamped()
            arm_feature.header.frame_id = feature_frame
            arm_feature.point.x = float(object_point[0])
            arm_feature.point.y = float(object_point[1])
            arm_feature.point.z = float(object_point[2])
            arm.features.append(arm_feature)
            camera_feature = PointStamped()
            camera_feature.header.frame_id = CAMERA_FRAME
            camera_feature.point.x = float(image_point[0])
            camera_feature.point.y = float(image_point[1])
            camera.features.append(camera_feature)
        _fill_camera_info(camera.ext_camera_info.camera_info, sample.camera_info)
        message.observations = [arm, camera]
        stamp_ns = max(
            _sample_stamp_ns(sample, fallback=index + 1), previous_stamp_ns + 1
        )
        writer.write("/calibration_data", serialize_message(message), stamp_ns)
        previous_stamp_ns = stamp_ns


def _side_offsets(
    offsets: dict[str, float], side: str, target_mode: str
) -> dict[str, float]:
    result = {
        name: value
        for name, value in offsets.items()
        if name in CAMERA_OFFSET_NAMES or name.startswith(f"{side}_")
    }
    if target_mode == "optimize":
        for suffix, source in zip(
            ("x", "y", "z", "a", "b", "c"),
            _target_offset_names(side),
            strict=True,
        ):
            result[f"calibration_target_{suffix}"] = offsets[source]
            result.pop(source)
    return result


def _evaluate(
    model: URDFModel,
    datasets: dict[str, CalibrationDataset],
    samples: dict[str, tuple[CalibrationSample, ...]],
    offsets: dict[str, float],
    spec: ModelSpec,
    initial_targets: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    vectors = []
    summary: dict[str, Any] = {}
    for side in ("left", "right"):
        projection = NativeCalibrationProjection(
            model,
            calibration_arm=side,
            fixed_hand_T_target=(
                initial_targets[side] if spec.target_mode == "fixed" else None
            ),
        )
        side_offsets = _side_offsets(offsets, side, spec.target_mode)
        residual_rows = []
        per_capture = []
        for sample in samples[side]:
            predicted, depths = projection.project_sample_with_offsets(
                sample, side_offsets
            )
            if np.any(depths <= 0):
                raise ValueError("native solution projects points behind camera")
            camera_matrix = np.asarray(
                sample.camera_info["p"], dtype=np.float64
            ).reshape(3, 4)[:, :3]
            predicted[:, 0] = (1.0 + offsets.get("camera_fx", 0.0)) * (
                predicted[:, 0] - camera_matrix[0, 2]
            ) + (1.0 + offsets.get("camera_cx", 0.0)) * camera_matrix[0, 2]
            predicted[:, 1] = (1.0 + offsets.get("camera_fy", 0.0)) * (
                predicted[:, 1] - camera_matrix[1, 2]
            ) + (1.0 + offsets.get("camera_cy", 0.0)) * camera_matrix[1, 2]
            item = predicted - np.asarray(sample.image_points_px, dtype=np.float64)
            residual_rows.append(item.reshape(-1))
            per_capture.append(
                {
                    "capture_id": sample.capture_id,
                    "pose_id": sample.pose_id,
                    "rms_px": float(np.sqrt(np.mean(np.sum(np.square(item), axis=1)))),
                }
            )
        residual = np.concatenate(residual_rows)
        vectors.append(residual)
        summary[side] = {
            "corner_count": len(residual) // 2,
            "squared_radial_error": float(residual @ residual),
            "radial_rms_px": float(
                np.sqrt(np.mean(np.sum(np.square(residual.reshape(-1, 2)), axis=1)))
            ),
            "per_capture": per_capture,
        }
    return np.concatenate(vectors), summary


def _observability(
    model: URDFModel,
    datasets: dict[str, CalibrationDataset],
    samples: dict[str, tuple[CalibrationSample, ...]],
    offsets: dict[str, float],
    spec: ModelSpec,
    initial_targets: dict[str, np.ndarray],
) -> dict[str, Any]:
    names = tuple(offsets)
    center = np.asarray([offsets[name] for name in names])
    columns = []
    for index in range(len(names)):
        plus = center.copy()
        minus = center.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        plus_residual, _ = _evaluate(
            model,
            datasets,
            samples,
            dict(zip(names, plus, strict=True)),
            spec,
            initial_targets,
        )
        minus_residual, _ = _evaluate(
            model,
            datasets,
            samples,
            dict(zip(names, minus, strict=True)),
            spec,
            initial_targets,
        )
        columns.append((plus_residual - minus_residual) / 2e-6)
    singular = np.linalg.svd(np.column_stack(columns), compute_uv=False)
    threshold = 1e-7 * singular[0]
    rank = int(np.count_nonzero(singular > threshold))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    return {
        "rank": rank,
        "parameter_count": len(names),
        "condition_number": condition,
        "observable": rank == len(names) and condition is not None and condition <= 1e8,
        "singular_values": singular.tolist(),
    }


def _run_fit(
    *,
    output: Path,
    spec: ModelSpec,
    model: URDFModel,
    datasets: dict[str, CalibrationDataset],
    training: dict[str, tuple[CalibrationSample, ...]],
    holdout: dict[str, tuple[CalibrationSample, ...]],
    initial_targets: dict[str, np.ndarray],
    prior_sigma_deg: float,
    runner: Path,
    timeout_s: float,
) -> dict[str, Any]:
    output.mkdir(parents=True)
    robot_description = add_color_optical_frame(model.path.read_text(encoding="utf-8"))
    (output / "robot_description.urdf").write_text(robot_description, encoding="utf-8")
    sample_count = sum(len(items) for items in training.values())
    config = _optimizer_config(spec, sample_count, initial_targets, prior_sigma_deg)
    config_path = output / "calibrate.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    bag_path = output / "calibration_data"
    combined = tuple(
        (side, sample) for side in ("left", "right") for sample in training[side]
    )
    _write_bag(combined, robot_description, bag_path, spec, initial_targets)
    command = [
        str(runner),
        "ros2",
        "run",
        "robot_calibration",
        "calibrate",
        "--from-bag",
        str(bag_path),
        "--ros-args",
        "--params-file",
        str(config_path),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout_s, check=False
    )
    log = completed.stdout + (
        "\n--- stderr ---\n" + completed.stderr if completed.stderr else ""
    )
    (output / "solver.log").write_text(log, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Ferguson optimizer failed with status {completed.returncode}: {output / 'solver.log'}"
        )
    offsets, iterations, final_cost, termination = _parse_solver_output(
        completed.stdout
    )
    expected = [
        *spec.left_joints,
        *spec.right_joints,
        *_camera_offset_names(spec.camera_mode),
        *_intrinsic_offset_names(spec.intrinsics_mode),
    ]
    if spec.target_mode == "optimize":
        expected.extend(_target_offset_names("left"))
        expected.extend(_target_offset_names("right"))
    if set(offsets) != set(expected):
        raise RuntimeError(
            f"unexpected native parameters: expected={expected}, actual={sorted(offsets)}"
        )
    offsets = {name: offsets[name] for name in expected}
    train_vector, train_summary = _evaluate(
        model, datasets, training, offsets, spec, initial_targets
    )
    result: dict[str, Any] = {
        "model": spec.name,
        "target_mode": spec.target_mode,
        "camera_mode": spec.camera_mode,
        "intrinsics_mode": spec.intrinsics_mode,
        "left_joints": list(spec.left_joints),
        "right_joints": list(spec.right_joints),
        "parameter_count": len(offsets),
        "offsets": offsets,
        "iterations": iterations,
        "final_cost": final_cost,
        "termination": termination,
        "training": train_summary,
        "training_radial_rms_px": float(
            np.sqrt(np.mean(np.sum(np.square(train_vector.reshape(-1, 2)), axis=1)))
        ),
        "observability": _observability(
            model, datasets, training, offsets, spec, initial_targets
        ),
    }
    if all(holdout.values()):
        holdout_vector, holdout_summary = _evaluate(
            model, datasets, holdout, offsets, spec, initial_targets
        )
        result["holdout"] = holdout_summary
        result["holdout_radial_rms_px"] = float(
            np.sqrt(np.mean(np.sum(np.square(holdout_vector.reshape(-1, 2)), axis=1)))
        )
    camera_projection = NativeCalibrationProjection(
        model,
        calibration_arm="left",
        fixed_hand_T_target=initial_targets["left"],
    )
    camera, _ = camera_projection.transforms(_side_offsets(offsets, "left", "fixed"))
    result["torso_T_camera"] = camera.tolist()
    for side in ("left", "right"):
        if spec.target_mode == "fixed":
            target = initial_targets[side]
        else:
            target = pose_vector_to_transform(
                [offsets[name] for name in _target_offset_names(side)]
            )
        result[f"{side}_hand_T_target"] = target.tolist()
    return result


def _transform_delta(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    vector = transform_to_pose_vector(np.linalg.inv(first) @ second)
    return float(np.linalg.norm(vector[:3]) * 1000), float(
        np.degrees(np.linalg.norm(vector[3:]))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    for side in ("left", "right"):
        parser.add_argument(f"--{side}-dataset", type=Path, required=True)
        parser.add_argument(f"--{side}-hardware-config", type=Path, required=True)
        parser.add_argument(f"--{side}-target-config", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot-calibration-directory", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument(
        "--model", action="append", type=_parse_model_spec, required=True
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--joint-offset-prior-sigma-deg", type=float, default=5.0)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"analysis output already exists: {args.output}")
    _verify_robot_calibration_revision(args.robot_calibration_directory)
    model = URDFModel(args.urdf)
    datasets: dict[str, CalibrationDataset] = {}
    initial_targets: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        dataset = CalibrationDataset.from_json(getattr(args, f"{side}_dataset"))
        if dataset.calibration_arm != side or dataset.urdf_sha256 != model.sha256:
            raise ValueError(f"{side} dataset arm or URDF does not match")
        hardware_path = getattr(args, f"{side}_hardware_config")
        target_path = getattr(args, f"{side}_target_config")
        hardware = yaml.safe_load(hardware_path.read_text(encoding="utf-8"))
        target = json.loads(target_path.read_text(encoding="utf-8"))
        validate_hardware_target_profile(hardware, target)
        if (
            hashlib.sha256(target_path.read_bytes()).hexdigest()
            != dataset.target_artifact_sha256
        ):
            raise ValueError(f"{side} target profile hash differs from dataset")
        datasets[side] = dataset
        initial_targets[side] = modeled_hand_T_target_from_hardware(hardware)
    fold_ids = {
        side: _fold_pose_ids(dataset, args.folds, args.seed)
        for side, dataset in datasets.items()
    }
    results = []
    with tempfile.TemporaryDirectory(prefix="g1_dual_native_") as temporary:
        root = Path(temporary)
        for spec in args.model:
            folds = []
            for fold in range(args.folds):
                holdout_ids = {side: set(fold_ids[side][fold]) for side in datasets}
                training = {
                    side: tuple(
                        sample
                        for sample in dataset.samples
                        if sample.pose_id not in holdout_ids[side]
                    )
                    for side, dataset in datasets.items()
                }
                holdout = {
                    side: tuple(
                        sample
                        for sample in dataset.samples
                        if sample.pose_id in holdout_ids[side]
                    )
                    for side, dataset in datasets.items()
                }
                folds.append(
                    _run_fit(
                        output=root / spec.name / f"fold_{fold:02d}",
                        spec=spec,
                        model=model,
                        datasets=datasets,
                        training=training,
                        holdout=holdout,
                        initial_targets=initial_targets,
                        prior_sigma_deg=args.joint_offset_prior_sigma_deg,
                        runner=args.runner.resolve(),
                        timeout_s=args.timeout_s,
                    )
                )
            full = _run_fit(
                output=root / spec.name / "full",
                spec=spec,
                model=model,
                datasets=datasets,
                training={side: dataset.samples for side, dataset in datasets.items()},
                holdout={side: () for side in datasets},
                initial_targets=initial_targets,
                prior_sigma_deg=args.joint_offset_prior_sigma_deg,
                runner=args.runner.resolve(),
                timeout_s=args.timeout_s,
            )
            holdout_squared = sum(
                fold["holdout"][side]["squared_radial_error"]
                for fold in folds
                for side in ("left", "right")
            )
            holdout_count = sum(
                fold["holdout"][side]["corner_count"]
                for fold in folds
                for side in ("left", "right")
            )
            cameras = [np.asarray(fold["torso_T_camera"]) for fold in folds]
            camera_deltas = [
                _transform_delta(cameras[first], cameras[second])
                for first in range(len(cameras))
                for second in range(first + 1, len(cameras))
            ]
            summary = {
                "model": spec.name,
                "target_mode": spec.target_mode,
                "camera_mode": spec.camera_mode,
                "intrinsics_mode": spec.intrinsics_mode,
                "left_joints": list(spec.left_joints),
                "right_joints": list(spec.right_joints),
                "parameter_count": full["parameter_count"],
                "cv_holdout_radial_rms_px": float(
                    np.sqrt(holdout_squared / holdout_count)
                ),
                "cv_left_radial_rms_px": float(
                    np.sqrt(
                        sum(
                            fold["holdout"]["left"]["squared_radial_error"]
                            for fold in folds
                        )
                        / sum(fold["holdout"]["left"]["corner_count"] for fold in folds)
                    )
                ),
                "cv_right_radial_rms_px": float(
                    np.sqrt(
                        sum(
                            fold["holdout"]["right"]["squared_radial_error"]
                            for fold in folds
                        )
                        / sum(
                            fold["holdout"]["right"]["corner_count"] for fold in folds
                        )
                    )
                ),
                "all_folds_observable": all(
                    fold["observability"]["observable"] for fold in folds
                ),
                "maximum_condition_number": max(
                    fold["observability"]["condition_number"] for fold in folds
                ),
                "camera_pairwise_translation_max_mm": max(
                    delta[0] for delta in camera_deltas
                ),
                "camera_pairwise_rotation_max_deg": max(
                    delta[1] for delta in camera_deltas
                ),
                "folds": folds,
                "full_fit": full,
            }
            results.append(summary)
            print(
                f"{spec.name}: CV={summary['cv_holdout_radial_rms_px']:.4f}px "
                f"(L={summary['cv_left_radial_rms_px']:.4f}, "
                f"R={summary['cv_right_radial_rms_px']:.4f}), "
                f"observable={summary['all_folds_observable']}",
                flush=True,
            )
    results.sort(
        key=lambda item: (
            not item["all_folds_observable"],
            item["cv_holdout_radial_rms_px"],
        )
    )
    document = {
        "schema_version": 1,
        "optimizer_backend": "mikeferguson/robot_calibration:Ceres",
        "robot_calibration_revision": ROBOT_CALIBRATION_REVISION,
        "urdf_sha256": model.sha256,
        "fold_count": args.folds,
        "fold_seed": args.seed,
        "joint_offset_prior_sigma_deg": args.joint_offset_prior_sigma_deg,
        "datasets": {
            side: {
                "path": str(getattr(args, f"{side}_dataset")),
                "sha256": dataset.content_sha256,
                "sample_count": len(dataset.samples),
            }
            for side, dataset in datasets.items()
        },
        "models": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
