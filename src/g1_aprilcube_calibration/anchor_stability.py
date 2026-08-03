"""Joint-compensated repeated-anchor stability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.dataset_builder import CalibrationDataset
from g1_aprilcube_calibration.joint_map import G1_29_JOINT_NAMES, arm_hand_link
from g1_aprilcube_calibration.transforms import invert_transform, validate_transform
from g1_aprilcube_calibration.urdf_model import URDFModel


@dataclass(frozen=True, slots=True)
class AnchorStabilityConfig:
    minimum_captures: int = 3
    maximum_pairwise_translation_m: float = 0.002
    maximum_pairwise_rotation_deg: float = 0.5

    def __post_init__(self) -> None:
        if self.minimum_captures < 2:
            raise ValueError("minimum anchor captures must be at least two")
        if self.maximum_pairwise_translation_m <= 0:
            raise ValueError("anchor translation threshold must be positive")
        if self.maximum_pairwise_rotation_deg <= 0:
            raise ValueError("anchor rotation threshold must be positive")


@dataclass(frozen=True, slots=True)
class AnchorPoseStability:
    pose_id: str
    capture_ids: tuple[str, ...]
    maximum_pairwise_translation_m: float
    rms_pairwise_translation_m: float
    maximum_pairwise_rotation_deg: float
    rms_pairwise_rotation_deg: float
    passed: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "pose_id": self.pose_id,
            "capture_ids": list(self.capture_ids),
            "maximum_pairwise_translation_m": self.maximum_pairwise_translation_m,
            "rms_pairwise_translation_m": self.rms_pairwise_translation_m,
            "maximum_pairwise_rotation_deg": self.maximum_pairwise_rotation_deg,
            "rms_pairwise_rotation_deg": self.rms_pairwise_rotation_deg,
            "passed": self.passed,
            "failures": list(self.failures),
        }


@dataclass(frozen=True, slots=True)
class AnchorStabilityReport:
    config: AnchorStabilityConfig
    poses: tuple[AnchorPoseStability, ...]

    @property
    def passed(self) -> bool:
        return bool(self.poses) and all(pose.passed for pose in self.poses)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "config": {
                name: getattr(self.config, name)
                for name in self.config.__dataclass_fields__
            },
            "poses": [pose.to_dict() for pose in self.poses],
        }


def analyze_dataset_anchors(
    dataset: CalibrationDataset,
    model: URDFModel,
    *,
    torso_T_camera: np.ndarray,
    pose_ids: tuple[str, ...] = (),
    config: AnchorStabilityConfig | None = None,
) -> AnchorStabilityReport:
    if dataset.urdf_sha256 != model.sha256:
        raise ValueError("anchor dataset and URDF hashes do not match")
    camera_transform = validate_transform(torso_T_camera)
    selected = set(pose_ids)
    grouped: dict[str, list] = {}
    for sample in dataset.samples:
        if selected and sample.pose_id not in selected:
            continue
        grouped.setdefault(sample.pose_id, []).append(sample)
    resolved_config = config or AnchorStabilityConfig()
    results: list[AnchorPoseStability] = []
    for pose_id, samples in sorted(grouped.items()):
        if len(samples) < resolved_config.minimum_captures:
            if pose_id in selected:
                raise ValueError(
                    f"anchor {pose_id} has {len(samples)} captures; need "
                    f"{resolved_config.minimum_captures}"
                )
            continue
        transforms: dict[str, np.ndarray] = {}
        for sample in samples:
            camera_T_target = _pnp_camera_T_target(sample)
            position = np.asarray(sample.measured_state["position"], dtype=np.float64)
            if position.shape != (29,) or not np.all(np.isfinite(position)):
                raise ValueError("anchor sample has an invalid measured state")
            torso_T_hand = model.transform(
                "torso_link",
                arm_hand_link(dataset.calibration_arm),
                dict(zip(G1_29_JOINT_NAMES, position, strict=True)),
            )
            transforms[sample.capture_id] = (
                invert_transform(torso_T_hand) @ camera_transform @ camera_T_target
            )
        results.append(
            analyze_anchor_transforms(
                pose_id,
                transforms,
                config=resolved_config,
            )
        )
    if selected - set(grouped):
        raise ValueError(
            "requested anchor pose IDs are absent: "
            + ", ".join(sorted(selected - set(grouped)))
        )
    if not results:
        raise ValueError(
            "dataset has no pose with enough repeated captures for anchor stability"
        )
    return AnchorStabilityReport(resolved_config, tuple(results))


def analyze_anchor_transforms(
    pose_id: str,
    transforms_by_capture: dict[str, np.ndarray],
    *,
    config: AnchorStabilityConfig | None = None,
) -> AnchorPoseStability:
    resolved_config = config or AnchorStabilityConfig()
    if len(transforms_by_capture) < resolved_config.minimum_captures:
        raise ValueError(
            f"anchor requires at least {resolved_config.minimum_captures} transforms"
        )
    transforms = {
        capture_id: validate_transform(transform)
        for capture_id, transform in transforms_by_capture.items()
    }
    translation: list[float] = []
    rotation_deg: list[float] = []
    for first, second in combinations(sorted(transforms), 2):
        delta = invert_transform(transforms[first]) @ transforms[second]
        translation.append(float(np.linalg.norm(delta[:3, 3])))
        rotation_deg.append(
            float(np.degrees(Rotation.from_matrix(delta[:3, :3]).magnitude()))
        )
    maximum_translation = max(translation)
    maximum_rotation = max(rotation_deg)
    failures: list[str] = []
    if maximum_translation > resolved_config.maximum_pairwise_translation_m:
        failures.append(
            f"pairwise translation {maximum_translation:.6f}m exceeds "
            f"{resolved_config.maximum_pairwise_translation_m:.6f}m"
        )
    if maximum_rotation > resolved_config.maximum_pairwise_rotation_deg:
        failures.append(
            f"pairwise rotation {maximum_rotation:.4f}deg exceeds "
            f"{resolved_config.maximum_pairwise_rotation_deg:.4f}deg"
        )
    return AnchorPoseStability(
        pose_id=pose_id,
        capture_ids=tuple(sorted(transforms)),
        maximum_pairwise_translation_m=maximum_translation,
        rms_pairwise_translation_m=float(np.sqrt(np.mean(np.square(translation)))),
        maximum_pairwise_rotation_deg=maximum_rotation,
        rms_pairwise_rotation_deg=float(np.sqrt(np.mean(np.square(rotation_deg)))),
        passed=not failures,
        failures=tuple(failures),
    )


def _pnp_camera_T_target(sample) -> np.ndarray:
    info = RectifiedCameraInfo.from_dict(sample.camera_info)
    success, rotation_vector, translation_vector = cv2.solvePnP(
        np.asarray(sample.object_points_m, dtype=np.float64),
        np.asarray(sample.image_points_px, dtype=np.float64),
        info.rectified_camera_matrix,
        np.zeros(5),
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not success:
        raise RuntimeError(f"anchor PnP failed for capture {sample.capture_id}")
    transform = np.eye(4)
    transform[:3, :3], _ = cv2.Rodrigues(rotation_vector)
    transform[:3, 3] = translation_vector.reshape(3)
    if transform[2, 3] <= 0:
        raise ValueError(f"anchor PnP is behind camera for {sample.capture_id}")
    return transform
