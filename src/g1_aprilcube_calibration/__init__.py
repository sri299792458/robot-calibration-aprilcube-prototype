"""G1 RealSense-to-AprilCube calibration prototype."""

from g1_aprilcube_calibration.calibration_pipeline import (
    CalibrationPipeline,
    PipelineConfig,
    PipelineResult,
)
from g1_aprilcube_calibration.calibration_solver import (
    DegenerateCalibrationError,
    ExtrinsicsSolver,
    SolverConfig,
    SolveResult,
)
from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.collision import (
    AttachedBox,
    CollisionConfig,
    FCLCollisionChecker,
)
from g1_aprilcube_calibration.config import QualityThresholds
from g1_aprilcube_calibration.dataset_builder import (
    CalibrationDataset,
    CalibrationSample,
    DatasetBuilder,
)
from g1_aprilcube_calibration.executor_state_machine import (
    ExecutorConfig,
    ExecutorState,
    PoseExecutor,
    TransitionApproval,
)
from g1_aprilcube_calibration.joint_map import (
    G1_29_JOINT_NAMES,
    G1_MODE_MACHINE,
    LEFT_ARM_INDICES,
    RIGHT_ARM_INDICES,
)
from g1_aprilcube_calibration.models import RobotStateSample
from g1_aprilcube_calibration.pose_recorder import (
    PoseRecorder,
    PoseRecordingAssessment,
    PoseRecordingRequest,
)
from g1_aprilcube_calibration.pose_schema import PoseRecord, PoseSet
from g1_aprilcube_calibration.pose_store import PoseStore
from g1_aprilcube_calibration.pose_validator import (
    PathValidationConfig,
    PosePathValidator,
    ValidationReport,
)
from g1_aprilcube_calibration.quality import (
    CameraIntrinsics,
    PoseQualityEvaluator,
    QualityGrade,
    QualityReport,
    ViewSignature,
)
from g1_aprilcube_calibration.readiness import (
    ReadinessReport,
    RecordingGateConfig,
    StateSampleBuffer,
)
from g1_aprilcube_calibration.residual_report import (
    CalibrationRunExporter,
    ResidualReport,
)
from g1_aprilcube_calibration.session_store import CaptureFrameInput, SessionStore
from g1_aprilcube_calibration.timestamp_pairing import (
    ImageTiming,
    PairingConfig,
    PairingResult,
)
from g1_aprilcube_calibration.urdf_model import URDFModel

__all__ = [
    "G1_29_JOINT_NAMES",
    "G1_MODE_MACHINE",
    "LEFT_ARM_INDICES",
    "RIGHT_ARM_INDICES",
    "AttachedBox",
    "CalibrationDataset",
    "CalibrationPipeline",
    "CalibrationRunExporter",
    "CalibrationSample",
    "CameraIntrinsics",
    "CaptureFrameInput",
    "CollisionConfig",
    "DatasetBuilder",
    "DegenerateCalibrationError",
    "ExecutorConfig",
    "ExecutorState",
    "ExtrinsicsSolver",
    "FCLCollisionChecker",
    "ImageTiming",
    "PairingConfig",
    "PairingResult",
    "PathValidationConfig",
    "PipelineConfig",
    "PipelineResult",
    "PoseExecutor",
    "PosePathValidator",
    "PoseQualityEvaluator",
    "PoseRecord",
    "PoseRecorder",
    "PoseRecordingAssessment",
    "PoseRecordingRequest",
    "PoseSet",
    "PoseStore",
    "QualityGrade",
    "QualityReport",
    "QualityThresholds",
    "ReadinessReport",
    "RecordingGateConfig",
    "RectifiedCameraInfo",
    "ResidualReport",
    "RobotStateSample",
    "SessionStore",
    "SolveResult",
    "SolverConfig",
    "StateSampleBuffer",
    "TransitionApproval",
    "URDFModel",
    "ValidationReport",
    "ViewSignature",
]
