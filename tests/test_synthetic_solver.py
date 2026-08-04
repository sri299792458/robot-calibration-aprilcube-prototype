import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.calibration_pipeline import (
    CalibrationPipeline,
    PipelineConfig,
)
from g1_aprilcube_calibration.calibration_solver import (
    DegenerateCalibrationError,
    ExtrinsicsSolver,
)
from g1_aprilcube_calibration.camera_initialization import (
    estimate_hand_T_target_from_sample,
    nominal_torso_T_color_optical,
    realsense_link_T_color_optical,
)
from g1_aprilcube_calibration.dataset_builder import (
    CalibrationDataset,
    deterministic_holdout_split,
)
from g1_aprilcube_calibration.residual_report import load_exported_result
from g1_aprilcube_calibration.synthetic import (
    make_synthetic_dataset,
    perturbed_initial_transforms,
)
from g1_aprilcube_calibration.transforms import (
    invert_transform,
    pose_vector_to_transform,
    transform_points,
    transform_to_pose_vector,
)
from g1_aprilcube_calibration.urdf_model import URDFModel

ROOT = Path(__file__).parents[1]
URDF = ROOT / "unitree_ros" / "robots" / "g1_description" / "g1_29dof_rev_1_0.urdf"
TARGET = ROOT / "aprilcube" / "models" / "dex3_safe_cube" / "config.json"


def transform_error(estimated: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    translation = float(np.linalg.norm(estimated[:3, 3] - truth[:3, 3]))
    rotation = float(
        Rotation.from_matrix((estimated[:3, :3].T @ truth[:3, :3]).copy()).magnitude()
    )
    return translation, np.degrees(rotation)


def test_transform_convention_round_trip_and_optical_axes() -> None:
    vector = np.asarray([0.1, -0.2, 0.3, 0.2, -0.1, 0.05])
    transform = pose_vector_to_transform(vector)
    assert np.allclose(transform_to_pose_vector(transform), vector)
    points = np.asarray([[1.0, 2.0, 3.0]])
    assert np.allclose(
        transform_points(
            invert_transform(transform), transform_points(transform, points)
        ),
        points,
    )
    optical = realsense_link_T_color_optical()
    assert np.allclose(optical[:3, 2], [1.0, 0.0, 0.0])
    nominal = nominal_torso_T_color_optical(URDFModel(URDF))
    assert np.allclose(nominal[:3, 3], [0.0576235, 0.01753, 0.42987])


def test_noiseless_twelve_parameter_recovery() -> None:
    model = URDFModel(URDF)
    dataset, truth = make_synthetic_dataset(model, TARGET, pose_count=40, seed=5)
    training, holdout = deterministic_holdout_split(dataset)
    initial_camera, initial_target = perturbed_initial_transforms(truth)
    solver = ExtrinsicsSolver(model)

    result = solver.solve(
        training,
        initial_torso_T_camera=initial_camera,
        initial_hand_T_target=initial_target,
    )

    assert result.observability.observable
    assert result.observability.rank == 12
    assert result.optimization_rms_px < 1e-8
    assert transform_error(result.torso_T_camera, truth.torso_T_camera)[0] < 1e-9
    assert transform_error(result.hand_T_target, truth.hand_T_target)[0] < 1e-9
    holdout_residual = solver.pixel_residuals(np.asarray(result.parameters), holdout)
    assert np.sqrt(np.mean(np.square(holdout_residual))) < 1e-8

    pnp_target = estimate_hand_T_target_from_sample(
        model,
        training[0],
        initial_torso_T_camera=truth.torso_T_camera,
    )
    pnp_translation, pnp_rotation = transform_error(pnp_target, truth.hand_T_target)
    assert pnp_translation < 1e-6
    assert pnp_rotation < 1e-4


def test_repeated_single_pose_is_rejected_as_degenerate() -> None:
    model = URDFModel(URDF)
    dataset, truth = make_synthetic_dataset(model, TARGET, pose_count=4, seed=5)
    initial_camera, initial_target = perturbed_initial_transforms(truth)
    with pytest.raises(DegenerateCalibrationError, match="rank=6/12"):
        ExtrinsicsSolver(model).solve(
            (dataset.samples[0],) * 4,
            initial_torso_T_camera=initial_camera,
            initial_hand_T_target=initial_target,
        )


def test_rectified_projection_uses_p_even_when_k_differs() -> None:
    model = URDFModel(URDF)
    dataset, truth = make_synthetic_dataset(model, TARGET, pose_count=4, seed=9)
    sample = dataset.samples[0]
    camera_info = dict(sample.camera_info)
    camera_info["k"] = [500.0, 0.0, 100.0, 0.0, 510.0, 120.0, 0.0, 0.0, 1.0]
    rectified_sample = replace(sample, camera_info=camera_info)
    parameters = np.concatenate(
        (
            transform_to_pose_vector(truth.torso_T_camera),
            transform_to_pose_vector(truth.hand_T_target),
        )
    )
    residual = ExtrinsicsSolver(model).pixel_residuals(parameters, (rectified_sample,))
    assert np.max(np.abs(residual)) < 1e-9
    initialized = estimate_hand_T_target_from_sample(
        model,
        rectified_sample,
        initial_torso_T_camera=truth.torso_T_camera,
    )
    assert transform_error(initialized, truth.hand_T_target)[0] < 1e-6


def test_holdout_split_keeps_repeated_pose_captures_together() -> None:
    model = URDFModel(URDF)
    dataset, _ = make_synthetic_dataset(model, TARGET, pose_count=5, seed=13)
    first = dataset.samples[0]
    repeated = replace(
        first,
        capture_id="repeated_capture",
        frame_id="repeated_frame",
        raw_image_path="synthetic/repeated.png",
    )
    with_repeat = CalibrationDataset(
        session_id=dataset.session_id,
        session_manifest_sha256=dataset.session_manifest_sha256,
        target_artifact_sha256=dataset.target_artifact_sha256,
        pose_set_sha256=dataset.pose_set_sha256,
        urdf_sha256=dataset.urdf_sha256,
        calibration_arm=dataset.calibration_arm,
        observation_phase=dataset.observation_phase,
        samples=(*dataset.samples, repeated),
    )
    training, holdout = deterministic_holdout_split(with_repeat)
    assert {sample.pose_id for sample in training}.isdisjoint(
        sample.pose_id for sample in holdout
    )
    repeated_locations = {
        "training" if sample in training else "holdout" for sample in (first, repeated)
    }
    assert len(repeated_locations) == 1


def test_noisy_pipeline_bootstrap_holdout_and_export(tmp_path) -> None:
    model = URDFModel(URDF)
    dataset, truth = make_synthetic_dataset(
        model,
        TARGET,
        pose_count=30,
        pixel_noise_stddev=0.25,
        seed=11,
    )
    initial_camera, initial_target = perturbed_initial_transforms(truth)
    output = tmp_path / "run_001"
    pipeline = CalibrationPipeline(
        ExtrinsicsSolver(model),
        config=PipelineConfig(
            holdout_fraction=0.2, bootstrap_trials=3, bootstrap_seed=2
        ),
    )
    result = pipeline.run(
        dataset,
        initial_torso_T_camera=initial_camera,
        initial_hand_T_target=initial_target,
        output_directory=output,
        provenance={"urdf_sha256": model.sha256, "synthetic": True},
    )

    camera_translation, camera_rotation = transform_error(
        result.solution.torso_T_camera, truth.torso_T_camera
    )
    target_translation, target_rotation = transform_error(
        result.solution.hand_T_target, truth.hand_T_target
    )
    assert camera_translation < 0.002
    assert target_translation < 0.002
    assert camera_rotation < 0.3
    assert target_rotation < 0.3
    assert result.residuals.training.rms_px < 0.7
    assert result.residuals.holdout.rms_px < 0.8
    assert result.bootstrap.successful_trials == 3
    assert dataset.calibration_arm == "left"
    assert set(result.residuals.calibration_joint_correlations) == {
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    }

    assert {path.name for path in output.iterdir()} == {
        "result.json",
        "provenance.json",
        "calibrated_extrinsics.yaml",
        "corner_residuals.csv",
        "report.md",
    }
    exported = json.loads((output / "result.json").read_text())
    extrinsics = yaml.safe_load((output / "calibrated_extrinsics.yaml").read_text())
    assert exported["dataset_sha256"] == dataset.content_sha256
    assert exported["solution"]["observability"]["rank"] == 12
    assert extrinsics["result_sha256"] == exported["content_sha256"]
    assert "Residual-first decision" in (output / "report.md").read_text()
    assert (
        load_exported_result(output / "result.json")["content_sha256"]
        == exported["content_sha256"]
    )

    exported["solution"]["cost"] += 1.0
    (output / "result.json").write_text(json.dumps(exported))
    with pytest.raises(ValueError, match="SHA-256"):
        load_exported_result(output / "result.json")
