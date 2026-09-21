#!/usr/bin/env python3
"""Diagnose one native Ferguson joint-subset sweep without fitting parameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.stats import spearmanr

from g1_aprilcube_calibration.authored_collection import (
    modeled_hand_T_target_from_hardware,
)
from g1_aprilcube_calibration.calibration_evaluation import (
    NativeCalibrationProjection,
)
from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.dataset_builder import (
    CalibrationDataset,
    CalibrationSample,
)
from g1_aprilcube_calibration.joint_map import G1_29_JOINT_NAMES, arm_hand_link
from g1_aprilcube_calibration.transforms import (
    invert_transform,
    transform_to_pose_vector,
)
from g1_aprilcube_calibration.urdf_model import URDFModel


def _pnp_pose(sample: CalibrationSample) -> tuple[np.ndarray, dict[str, float]]:
    object_points = np.asarray(sample.object_points_m, dtype=np.float64)
    image_points = np.asarray(sample.image_points_px, dtype=np.float64)
    camera = RectifiedCameraInfo.from_dict(sample.camera_info)
    success, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        object_points,
        image_points,
        camera.rectified_camera_matrix,
        np.zeros(5),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        raise RuntimeError(f"PnP failed for {sample.capture_id}")
    candidates = []
    for rvec, tvec in zip(rvecs, tvecs, strict=True):
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            camera.rectified_camera_matrix,
            np.zeros(5),
        )
        delta = projected.reshape(-1, 2) - image_points
        radial_rms = float(np.sqrt(np.mean(np.sum(np.square(delta), axis=1))))
        candidates.append(
            (float(np.asarray(tvec).reshape(3)[2]) <= 0, radial_rms, rvec, tvec)
        )
    candidates.sort(key=lambda item: item[:2])
    _, radial_rms, rvec, tvec = candidates[0]
    rotation, _ = cv2.Rodrigues(rvec)
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(tvec).reshape(3)
    diagnostics = {"pnp_reprojection_rms_px": radial_rms}
    if len(candidates) < 2:
        raise RuntimeError(f"IPPE returned no ambiguity pair for {sample.capture_id}")
    _, second_rms, second_rvec, second_tvec = candidates[1]
    second_rotation, _ = cv2.Rodrigues(second_rvec)
    second = np.eye(4)
    second[:3, :3] = second_rotation
    second[:3, 3] = np.asarray(second_tvec).reshape(3)
    ambiguity = transform_to_pose_vector(invert_transform(result) @ second)
    diagnostics.update(
        {
            "pnp_second_minus_first_rms_px": second_rms - radial_rms,
            "pnp_solution_separation_mm": float(np.linalg.norm(ambiguity[:3]) * 1000.0),
            "pnp_solution_separation_deg": float(
                np.degrees(np.linalg.norm(ambiguity[3:]))
            ),
        }
    )
    return result, diagnostics


def _polygon_area(points: np.ndarray) -> float:
    return float(
        0.5
        * abs(
            np.dot(points[:, 0], np.roll(points[:, 1], 1))
            - np.dot(points[:, 1], np.roll(points[:, 0], 1))
        )
    )


def _sample_features(
    sample: CalibrationSample, index: int, arm_indices: tuple[int, ...]
) -> tuple[dict[str, float], np.ndarray, float]:
    image_points = np.asarray(sample.image_points_px)
    camera_T_target, pnp_diagnostics = _pnp_pose(sample)
    normal = camera_T_target[:3, 2]
    incidence = float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0))))
    position = np.asarray(sample.measured_state["position"], dtype=np.float64)
    velocity = np.asarray(sample.measured_state["velocity"], dtype=np.float64)
    torque = np.asarray(sample.measured_state["estimated_torque"], dtype=np.float64)
    pairing = sample.pairing
    features = {
        "collection_order": float(index),
        "nearest_pairing_delta_ms": float(pairing["nearest_delta_s"]) * 1000.0,
        "state_bracket_span_ms": float(pairing["bracket_span_s"]) * 1000.0,
        "marker_center_x_px": float(np.mean(image_points[:, 0])),
        "marker_center_y_px": float(np.mean(image_points[:, 1])),
        "marker_area_px2": _polygon_area(image_points),
        **pnp_diagnostics,
        "pnp_depth_mm": float(camera_T_target[2, 3]) * 1000.0,
        "pnp_range_mm": float(np.linalg.norm(camera_T_target[:3, 3])) * 1000.0,
        "pnp_incidence_deg": incidence,
        "maximum_arm_velocity_rad_s": float(
            np.max(np.abs(velocity[list(arm_indices)]))
        ),
        "maximum_arm_torque_nm": float(np.max(np.abs(torque[list(arm_indices)]))),
    }
    for joint_index in arm_indices:
        features[f"q_{G1_29_JOINT_NAMES[joint_index]}_rad"] = float(
            position[joint_index]
        )
    return features, camera_T_target


def _bh_qvalues(pvalues: list[float]) -> list[float]:
    count = len(pvalues)
    order = np.argsort(pvalues)
    result = np.ones(count)
    running = 1.0
    for reverse_index in range(count - 1, -1, -1):
        original = int(order[reverse_index])
        rank = reverse_index + 1
        running = min(running, pvalues[original] * count / rank)
        result[original] = running
    return result.tolist()


def _correlations(rows: list[dict]) -> list[dict]:
    dependent = (
        "cv_rms_px",
        "cv_mean_x_px",
        "cv_mean_y_px",
        "metric_translation_error_mm",
    )
    features = sorted(
        set(rows[0])
        - {
            "capture_id",
            "pose_id",
            "cv_squared_radial_error",
            "cv_corner_count",
            "image_plane_equivalent_error_mm",
            "metric_rotation_error_deg",
            *dependent,
        }
    )
    raw = []
    for outcome in dependent:
        y = np.asarray([row[outcome] for row in rows])
        for feature in features:
            x = np.asarray([row[feature] for row in rows])
            if np.ptp(x) == 0:
                continue
            result = spearmanr(x, y)
            if np.isfinite(result.statistic) and np.isfinite(result.pvalue):
                raw.append(
                    {
                        "outcome": outcome,
                        "feature": feature,
                        "spearman_rho": float(result.statistic),
                        "p_value": float(result.pvalue),
                    }
                )
    qvalues = _bh_qvalues([item["p_value"] for item in raw])
    for item, qvalue in zip(raw, qvalues, strict=True):
        item["bh_q_value"] = qvalue
    raw.sort(key=lambda item: (item["bh_q_value"], -abs(item["spearman_rho"])))
    return raw


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--hardware-config", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"diagnostic output already exists: {args.output}")
    dataset = CalibrationDataset.from_json(args.dataset)
    sweep = json.loads(args.sweep.read_text(encoding="utf-8"))
    if sweep["dataset_sha256"] != dataset.content_sha256:
        raise ValueError("sweep belongs to a different dataset")
    model = URDFModel(args.urdf)
    if model.sha256 != dataset.urdf_sha256:
        raise ValueError("URDF differs from dataset")
    hardware = yaml.safe_load(args.hardware_config.read_text(encoding="utf-8"))
    initial_target = modeled_hand_T_target_from_hardware(hardware)
    selected = next(
        item
        for item in sweep["models"]
        if item["all_folds_observable"]
        and item["successful_folds"] == sweep["fold_count"]
    )
    projection = NativeCalibrationProjection(
        model,
        calibration_arm=dataset.calibration_arm,
        fixed_hand_T_target=(
            initial_target if selected["target_mode"] == "fixed" else None
        ),
    )
    sample_by_pose = {sample.pose_id: sample for sample in dataset.samples}
    fold_by_pose = {
        entry["pose_id"]: fold
        for fold in selected["folds"]
        for entry in fold["holdout_per_capture"]
    }
    if set(sample_by_pose) != set(fold_by_pose):
        raise ValueError("cross-validation folds do not cover each pose exactly once")
    arm_start = 15 if dataset.calibration_arm == "left" else 22
    arm_indices = tuple(range(arm_start, arm_start + 7))
    rows = []
    camera_transforms = []
    target_transforms = []
    for index, sample in enumerate(dataset.samples):
        fold = fold_by_pose[sample.pose_id]
        offsets = fold["offsets"]
        predicted, _ = projection.project_sample_with_offsets(sample, offsets)
        residual_xy = predicted - np.asarray(sample.image_points_px)
        torso_T_camera, hand_T_target = projection.transforms(offsets)
        position = np.asarray(sample.measured_state["position"], dtype=np.float64)
        positions = {
            name: float(value) + float(offsets.get(name, 0.0))
            for name, value in zip(G1_29_JOINT_NAMES, position, strict=True)
        }
        torso_T_hand = model.transform(
            "torso_link", arm_hand_link(dataset.calibration_arm), positions
        )
        predicted_camera_T_target = (
            invert_transform(torso_T_camera) @ torso_T_hand @ hand_T_target
        )
        features, observed_camera_T_target = _sample_features(
            sample, index, arm_indices
        )
        delta = transform_to_pose_vector(
            invert_transform(predicted_camera_T_target) @ observed_camera_T_target
        )
        squared = float(np.sum(np.square(residual_xy)))
        radial_rms_px = float(np.sqrt(np.mean(np.sum(np.square(residual_xy), axis=1))))
        camera_projection = np.asarray(sample.camera_info["p"], dtype=np.float64)
        focal_px = float(np.sqrt(camera_projection[0] * camera_projection[5]))
        row = {
            "capture_id": sample.capture_id,
            "pose_id": sample.pose_id,
            "cv_rms_px": radial_rms_px,
            "cv_mean_x_px": float(np.mean(residual_xy[:, 0])),
            "cv_mean_y_px": float(np.mean(residual_xy[:, 1])),
            "cv_squared_radial_error": squared,
            "cv_corner_count": len(residual_xy),
            "metric_translation_error_mm": float(np.linalg.norm(delta[:3]) * 1000.0),
            "metric_rotation_error_deg": float(np.degrees(np.linalg.norm(delta[3:]))),
            "image_plane_equivalent_error_mm": (
                radial_rms_px * features["pnp_depth_mm"] / focal_px
            ),
            **features,
        }
        rows.append(row)
        camera_transforms.append(torso_T_camera)
        target_transforms.append(hand_T_target)
    ordered = sorted(rows, key=lambda item: item["cv_rms_px"], reverse=True)
    total_squared = sum(item["cv_squared_radial_error"] for item in rows)
    total_corners = sum(item["cv_corner_count"] for item in rows)
    removal_curve = []
    for remove_count in (0, 1, 2, 3, 5, max(1, round(len(rows) * 0.1))):
        retained = ordered[remove_count:]
        removal_curve.append(
            {
                "removed_worst_capture_count": remove_count,
                "retained_capture_count": len(retained),
                "radial_rms_px": float(
                    np.sqrt(
                        sum(item["cv_squared_radial_error"] for item in retained)
                        / sum(item["cv_corner_count"] for item in retained)
                    )
                ),
            }
        )
    document = {
        "schema_version": 1,
        "dataset_path": str(args.dataset),
        "dataset_sha256": dataset.content_sha256,
        "calibration_arm": dataset.calibration_arm,
        "selected_model": {
            key: selected[key]
            for key in (
                "target_mode",
                "joint_subset",
                "parameter_count",
                "cv_training_radial_rms_px",
                "cv_holdout_radial_rms_px",
                "camera_pairwise_translation_max_mm",
                "camera_pairwise_rotation_max_deg",
            )
        },
        "cross_validated_radial_rms_px": float(np.sqrt(total_squared / total_corners)),
        "metric_translation_error_mm": _summary(
            [item["metric_translation_error_mm"] for item in rows]
        ),
        "metric_rotation_error_deg": _summary(
            [item["metric_rotation_error_deg"] for item in rows]
        ),
        "image_plane_equivalent_error_mm": _summary(
            [item["image_plane_equivalent_error_mm"] for item in rows]
        ),
        "pnp_reprojection_rms_px": _summary(
            [item["pnp_reprojection_rms_px"] for item in rows]
        ),
        "pnp_second_minus_first_rms_px": _summary(
            [item["pnp_second_minus_first_rms_px"] for item in rows]
        ),
        "pnp_solution_separation_mm": _summary(
            [item["pnp_solution_separation_mm"] for item in rows]
        ),
        "pnp_solution_separation_deg": _summary(
            [item["pnp_solution_separation_deg"] for item in rows]
        ),
        "pairing_nearest_delta_ms": _summary(
            [item["nearest_pairing_delta_ms"] for item in rows]
        ),
        "pairing_bracket_span_ms": _summary(
            [item["state_bracket_span_ms"] for item in rows]
        ),
        "outlier_removal_curve": removal_curve,
        "worst_captures": ordered[:10],
        "correlations": _correlations(rows),
        "rows": rows,
    }
    if selected["target_mode"] == "optimize":
        target_deltas = [
            transform_to_pose_vector(invert_transform(initial_target) @ transform)
            for transform in target_transforms
        ]
        document["optimized_target_vs_cad"] = {
            "translation_mm": _summary(
                [float(np.linalg.norm(delta[:3]) * 1000.0) for delta in target_deltas]
            ),
            "rotation_deg": _summary(
                [
                    float(np.degrees(np.linalg.norm(delta[3:])))
                    for delta in target_deltas
                ]
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"{dataset.calibration_arm}: CV={document['cross_validated_radial_rms_px']:.4f}px, "
        f"metric median={document['metric_translation_error_mm']['median']:.2f}mm/"
        f"{document['metric_rotation_error_deg']['median']:.2f}deg, "
        f"PnP median={document['pnp_reprojection_rms_px']['median']:.3f}px"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
