"""Export immutable AprilCube datasets for Mike Ferguson's robot_calibration."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.camera_initialization import (
    estimate_hand_T_target_from_sample,
    nominal_torso_T_color_optical,
)
from g1_aprilcube_calibration.dataset_builder import (
    CalibrationDataset,
    CalibrationSample,
)
from g1_aprilcube_calibration.joint_map import (
    G1_29_JOINT_NAMES,
    arm_hand_link,
)
from g1_aprilcube_calibration.urdf_model import URDFModel

ROBOT_CALIBRATION_REVISION = "db991b040d1dc28af09d8865fc72f09720e12b73"
CAMERA_MODEL_NAME = "camera"
ARM_MODEL_NAME = "arm"
TARGET_FRAME = "aprilcube_target"
CAMERA_OPTICAL_FRAME = "camera_color_optical_frame"
CAMERA_OPTICAL_JOINT = "camera_color_optical_joint"
CAMERA_MOUNT_JOINT = "d435_joint"


@dataclass(frozen=True, slots=True)
class RobotCalibrationArtifacts:
    output_directory: Path
    bag_directory: Path
    robot_description_path: Path
    extrinsics_config_path: Path
    shoulder_roll_config_path: Path
    provenance_path: Path
    sample_count: int


def add_color_optical_frame(urdf_text: str) -> str:
    """Add the REP-103 color optical frame missing from the official G1 URDF."""

    root = ET.fromstring(urdf_text)
    if root.tag != "robot":
        raise ValueError("URDF root must be <robot>")
    link_names = {element.attrib["name"] for element in root.findall("link")}
    joint_names = {element.attrib["name"] for element in root.findall("joint")}
    if "d435_link" not in link_names:
        raise ValueError("G1 URDF does not contain d435_link")
    if CAMERA_OPTICAL_FRAME in link_names or CAMERA_OPTICAL_JOINT in joint_names:
        raise ValueError("G1 URDF already contains the configured color optical frame")

    ET.SubElement(root, "link", {"name": CAMERA_OPTICAL_FRAME})
    joint = ET.SubElement(
        root,
        "joint",
        {"name": CAMERA_OPTICAL_JOINT, "type": "fixed"},
    )
    ET.SubElement(
        joint,
        "origin",
        {
            "xyz": "0 0 0",
            "rpy": f"{-math.pi / 2:.17g} 0 {-math.pi / 2:.17g}",
        },
    )
    ET.SubElement(joint, "parent", {"link": "d435_link"})
    ET.SubElement(joint, "child", {"link": CAMERA_OPTICAL_FRAME})
    return ET.tostring(root, encoding="unicode") + "\n"


def sample_to_observation_record(sample: CalibrationSample) -> dict[str, Any]:
    """Return the lossless fields used to construct one CalibrationData message."""

    position = np.asarray(sample.measured_state["position"], dtype=np.float64)
    if position.shape != (len(G1_29_JOINT_NAMES),) or not np.all(
        np.isfinite(position)
    ):
        raise ValueError(f"sample {sample.frame_id} has an invalid G1 joint state")
    image_points = np.asarray(sample.image_points_px, dtype=np.float64)
    object_points = np.asarray(sample.object_points_m, dtype=np.float64)
    if len(image_points) != len(object_points):
        raise ValueError(f"sample {sample.frame_id} has mismatched observations")
    return {
        "capture_id": sample.capture_id,
        "frame_id": sample.frame_id,
        "joint_names": list(G1_29_JOINT_NAMES),
        "joint_positions": position.tolist(),
        "arm_sensor_name": ARM_MODEL_NAME,
        "arm_feature_frame": TARGET_FRAME,
        "object_points_m": object_points.tolist(),
        "camera_sensor_name": CAMERA_MODEL_NAME,
        "camera_feature_frame": CAMERA_OPTICAL_FRAME,
        "image_points_px": image_points.tolist(),
        "camera_info": sample.camera_info,
    }


def build_optimizer_config(
    *,
    hand_T_target: np.ndarray,
    calibration_arm: str,
    sample_count: int,
    calibrate_shoulder_roll: bool,
    shoulder_roll_prior_sigma_deg: float = 5.0,
) -> dict[str, Any]:
    """Build one native robot_calibration ROS parameter document."""

    if sample_count <= 0:
        raise ValueError("robot_calibration export requires at least one sample")
    if not np.isfinite(shoulder_roll_prior_sigma_deg) or (
        shoulder_roll_prior_sigma_deg <= 0
    ):
        raise ValueError("shoulder-roll prior sigma must be positive")
    target = np.asarray(hand_T_target, dtype=np.float64)
    if target.shape != (4, 4) or not np.all(np.isfinite(target)):
        raise ValueError("hand_T_target must be a finite 4x4 transform")
    target_rpy = Rotation.from_matrix(target[:3, :3]).as_euler("xyz")
    step: dict[str, Any] = {
        "max_num_iterations": 1000,
        "models": [ARM_MODEL_NAME, CAMERA_MODEL_NAME],
        ARM_MODEL_NAME: {
            "type": "chain3d",
            "frame": arm_hand_link(calibration_arm),
        },
        CAMERA_MODEL_NAME: {
            "type": "camera2d",
            "frame": CAMERA_OPTICAL_FRAME,
            "param_name": CAMERA_MODEL_NAME,
        },
        "free_frames": [CAMERA_MOUNT_JOINT, TARGET_FRAME],
        CAMERA_MOUNT_JOINT: {
            "x": True,
            "y": True,
            "z": True,
            "roll": True,
            "pitch": True,
            "yaw": True,
        },
        TARGET_FRAME: {
            "x": True,
            "y": True,
            "z": True,
            "roll": True,
            "pitch": True,
            "yaw": True,
        },
        "free_frames_initial_values": [TARGET_FRAME],
        f"{TARGET_FRAME}_initial_values": {
            "x": float(target[0, 3]),
            "y": float(target[1, 3]),
            "z": float(target[2, 3]),
            "roll": float(target_rpy[0]),
            "pitch": float(target_rpy[1]),
            "yaw": float(target_rpy[2]),
        },
        "error_blocks": ["aprilcube_reprojection"],
        "aprilcube_reprojection": {
            "type": "chain3d_to_camera2d",
            "model_3d": ARM_MODEL_NAME,
            "model_2d": CAMERA_MODEL_NAME,
            "scale": 1.0,
        },
    }
    if calibrate_shoulder_roll:
        shoulder_joint = f"{calibration_arm}_shoulder_roll_joint"
        sigma_rad = math.radians(shoulder_roll_prior_sigma_deg)
        # robot_calibration adds each configured error block to every sample.
        # Dividing by sqrt(N) makes the aggregate prior equal offset / sigma.
        joint_scale = 1.0 / (sigma_rad * math.sqrt(sample_count))
        step["free_params"] = [shoulder_joint]
        step["error_blocks"].append("shoulder_roll_prior")
        step["shoulder_roll_prior"] = {
            "type": "outrageous",
            "param": shoulder_joint,
            "joint_scale": joint_scale,
            "position_scale": 0.0,
            "rotation_scale": 0.0,
        }
    return {
        "robot_calibration": {
            "ros__parameters": {
                "verbose": False,
                "base_link": "torso_link",
                "calibration_steps": ["aprilcube_calibration"],
                "aprilcube_calibration": step,
            }
        }
    }


def export_robot_calibration_dataset(
    dataset: CalibrationDataset,
    model: URDFModel,
    output_directory: str | Path,
    *,
    pnp_sample_index: int = 0,
    shoulder_roll_prior_sigma_deg: float = 5.0,
    robot_calibration_directory: str | Path | None = None,
) -> RobotCalibrationArtifacts:
    """Write a native bag, URDF overlay, optimizer YAML, and provenance."""

    if dataset.urdf_sha256 != model.sha256:
        raise ValueError("dataset URDF hash does not match the requested G1 URDF")
    if not 0 <= pnp_sample_index < len(dataset.samples):
        raise ValueError("PnP sample index is outside the dataset")
    if robot_calibration_directory is not None:
        _verify_robot_calibration_revision(Path(robot_calibration_directory))
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"robot_calibration output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    nominal_camera = nominal_torso_T_color_optical(model)
    initial_target = estimate_hand_T_target_from_sample(
        model,
        dataset.samples[pnp_sample_index],
        initial_torso_T_camera=nominal_camera,
        calibration_arm=dataset.calibration_arm,
    )
    robot_description = add_color_optical_frame(model.path.read_text(encoding="utf-8"))
    robot_description_path = output / "robot_description.urdf"
    robot_description_path.write_text(robot_description, encoding="utf-8")

    extrinsics_config_path = output / "calibrate_extrinsics.yaml"
    shoulder_roll_config_path = output / "calibrate_shoulder_roll.yaml"
    _write_yaml(
        extrinsics_config_path,
        build_optimizer_config(
            hand_T_target=initial_target,
            calibration_arm=dataset.calibration_arm,
            sample_count=len(dataset.samples),
            calibrate_shoulder_roll=False,
            shoulder_roll_prior_sigma_deg=shoulder_roll_prior_sigma_deg,
        ),
    )
    _write_yaml(
        shoulder_roll_config_path,
        build_optimizer_config(
            hand_T_target=initial_target,
            calibration_arm=dataset.calibration_arm,
            sample_count=len(dataset.samples),
            calibrate_shoulder_roll=True,
            shoulder_roll_prior_sigma_deg=shoulder_roll_prior_sigma_deg,
        ),
    )

    bag_directory = output / "calibration_data"
    _write_rosbag(dataset, robot_description, bag_directory)

    provenance = {
        "schema_version": 1,
        "dataset_sha256": dataset.content_sha256,
        "dataset_session_id": dataset.session_id,
        "urdf_sha256": dataset.urdf_sha256,
        "augmented_urdf_sha256": hashlib.sha256(
            robot_description.encode("utf-8")
        ).hexdigest(),
        "robot_calibration_revision": ROBOT_CALIBRATION_REVISION,
        "sample_count": len(dataset.samples),
        "pnp_initialization_sample_index": pnp_sample_index,
        "pnp_initialization_frame_id": dataset.samples[pnp_sample_index].frame_id,
        "shoulder_roll_prior_sigma_deg": shoulder_roll_prior_sigma_deg,
        "observation_mapping": {
            "arm_model": ARM_MODEL_NAME,
            "arm_tip": arm_hand_link(dataset.calibration_arm),
            "target_frame": TARGET_FRAME,
            "camera_model": CAMERA_MODEL_NAME,
            "camera_frame": CAMERA_OPTICAL_FRAME,
            "camera_mount_free_frame": CAMERA_MOUNT_JOINT,
        },
    }
    provenance_path = output / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return RobotCalibrationArtifacts(
        output_directory=output,
        bag_directory=bag_directory,
        robot_description_path=robot_description_path,
        extrinsics_config_path=extrinsics_config_path,
        shoulder_roll_config_path=shoulder_roll_config_path,
        provenance_path=provenance_path,
        sample_count=len(dataset.samples),
    )


def _write_rosbag(
    dataset: CalibrationDataset,
    robot_description: str,
    bag_directory: Path,
) -> None:
    try:
        import rosbag2_py
        from geometry_msgs.msg import PointStamped
        from rclpy.serialization import serialize_message
        from robot_calibration_msgs.msg import CalibrationData, Observation
        from std_msgs.msg import String
    except ImportError as error:
        raise RuntimeError(
            "ROS 2 and the built robot_calibration_msgs package must be sourced "
            "before exporting a robot_calibration bag"
        ) from error

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
    description = String(data=robot_description)
    first_stamp_ns = _sample_stamp_ns(dataset.samples[0], fallback=1)
    writer.write(
        "/robot_description", serialize_message(description), first_stamp_ns
    )
    previous_stamp_ns = first_stamp_ns
    for index, sample in enumerate(dataset.samples, start=1):
        record = sample_to_observation_record(sample)
        message = CalibrationData()
        message.joint_states.name = record["joint_names"]
        message.joint_states.position = record["joint_positions"]

        arm = Observation(sensor_name=record["arm_sensor_name"])
        camera = Observation(sensor_name=record["camera_sensor_name"])
        for object_point, image_point in zip(
            record["object_points_m"], record["image_points_px"], strict=True
        ):
            arm_feature = PointStamped()
            arm_feature.header.frame_id = record["arm_feature_frame"]
            arm_feature.point.x = object_point[0]
            arm_feature.point.y = object_point[1]
            arm_feature.point.z = object_point[2]
            arm.features.append(arm_feature)

            camera_feature = PointStamped()
            camera_feature.header.frame_id = record["camera_feature_frame"]
            camera_feature.point.x = image_point[0]
            camera_feature.point.y = image_point[1]
            camera.features.append(camera_feature)
        _fill_camera_info(camera.ext_camera_info.camera_info, record["camera_info"])
        message.observations = [arm, camera]
        stamp_ns = max(
            _sample_stamp_ns(sample, fallback=index + 1), previous_stamp_ns + 1
        )
        writer.write(
            "/calibration_data", serialize_message(message), stamp_ns
        )
        previous_stamp_ns = stamp_ns


def _fill_camera_info(message: Any, data: dict[str, Any]) -> None:
    message.header.frame_id = str(data["frame_id"])
    message.height = int(data["height"])
    message.width = int(data["width"])
    message.distortion_model = str(data["distortion_model"])
    message.d = [float(value) for value in data["d"]]
    message.k = [float(value) for value in data["k"]]
    message.r = [float(value) for value in data["r"]]
    message.p = [float(value) for value in data["p"]]


def _sample_stamp_ns(sample: CalibrationSample, *, fallback: int) -> int:
    value = sample.pairing.get("image", {}).get("header_stamp_ns")
    if value is None:
        value = sample.pairing.get("image_header_stamp_ns")
    if value is None:
        return fallback
    stamp = int(value)
    return stamp if stamp > 0 else fallback


def _write_yaml(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )


def _verify_robot_calibration_revision(directory: Path) -> None:
    if not (directory / ".git").is_dir():
        raise FileNotFoundError(
            f"robot_calibration is not a Git checkout: {directory.resolve()}"
        )
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip()
    if actual != ROBOT_CALIBRATION_REVISION:
        raise ValueError(
            "robot_calibration revision mismatch: "
            f"expected {ROBOT_CALIBRATION_REVISION}, got {actual}"
        )
