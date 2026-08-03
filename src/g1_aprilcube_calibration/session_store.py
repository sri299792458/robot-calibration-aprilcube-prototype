"""Crash-safe raw session writer with immutable finalized artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import yaml

from aprilcube import CorrespondenceResult
from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.collision import CollisionConfig
from g1_aprilcube_calibration.correspondence import correspondence_sha256
from g1_aprilcube_calibration.models import (
    RobotStateSample,
    utc_now_iso,
    validate_utc_iso,
)
from g1_aprilcube_calibration.pose_schema import PoseSet
from g1_aprilcube_calibration.pose_validator import ValidationReport
from g1_aprilcube_calibration.quality import QualityGrade, QualityReport
from g1_aprilcube_calibration.readiness import (
    RecordingGateConfig,
    evaluate_recording_window,
)
from g1_aprilcube_calibration.session_models import (
    CAPTURE_OUTCOMES,
    CaptureRecord,
    RawFrameRecord,
    SessionManifest,
)
from g1_aprilcube_calibration.timestamp_pairing import (
    ImageTiming,
    PairingConfig,
    PairingResult,
    pair_state_to_image,
)

REQUIRED_SESSION_ARTIFACTS = frozenset(
    {
        "pose_set.yaml",
        "hardware.yaml",
        "target.json",
        "collision_pairs.yaml",
        "validation_report.json",
    }
)
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(data) -> bytes:
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


@dataclass(frozen=True, slots=True)
class CaptureFrameInput:
    frame_id: str
    image_bgr: np.ndarray
    image_timing: ImageTiming
    camera_info: RectifiedCameraInfo
    state_window: tuple[RobotStateSample, ...]
    pairing: PairingResult
    correspondences: CorrespondenceResult
    quality: QualityReport

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.frame_id):
            raise ValueError(f"invalid capture frame ID: {self.frame_id!r}")
        image = np.asarray(self.image_bgr)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("raw rectified image must be uint8 BGR")
        if image.shape[:2] != (self.camera_info.height, self.camera_info.width):
            raise ValueError("raw image dimensions do not match CameraInfo")
        if self.correspondences.image_size_wh != (
            self.camera_info.width,
            self.camera_info.height,
        ):
            raise ValueError("correspondence dimensions do not match CameraInfo")
        if self.pairing.image != self.image_timing:
            raise ValueError("pairing result belongs to a different image")
        if not self.state_window:
            raise ValueError("capture frame requires its complete state window")
        object.__setattr__(self, "image_bgr", image.copy())
        self.image_bgr.setflags(write=False)
        object.__setattr__(self, "state_window", tuple(self.state_window))


class SessionStore:
    def __init__(self, session_directory: str | Path) -> None:
        self.directory = Path(session_directory)
        self.manifest_path = self.directory / "manifest.json"

    def create(
        self,
        *,
        session_id: str,
        created_at_utc: str,
        camera_info: RectifiedCameraInfo,
        pose_set_content_sha256: str,
        artifacts: Mapping[str, bytes],
        pairing_config: PairingConfig,
        recording_gate_config: RecordingGateConfig,
        provenance: dict,
    ) -> SessionManifest:
        if self.directory.exists():
            raise FileExistsError(f"session directory already exists: {self.directory}")
        if set(artifacts) != REQUIRED_SESSION_ARTIFACTS:
            raise ValueError(
                "session artifacts must be exactly: "
                + ", ".join(sorted(REQUIRED_SESSION_ARTIFACTS))
            )
        self._validate_source_artifacts(
            artifacts,
            expected_pose_set_content_sha256=pose_set_content_sha256,
        )
        self.directory.mkdir(parents=True)
        (self.directory / "raw" / "images").mkdir(parents=True)
        (self.directory / "raw" / "states").mkdir(parents=True)
        (self.directory / "preview").mkdir(parents=True)
        artifact_hashes: dict[str, str] = {}
        for name, content in artifacts.items():
            if not isinstance(content, bytes):
                raise TypeError(f"session artifact {name} must be bytes")
            self._write_new_file(self.directory / name, content)
            artifact_hashes[name] = _sha256(content)
        manifest = SessionManifest(
            session_id=session_id,
            created_at_utc=created_at_utc,
            camera_profile_sha256=camera_info.profile_sha256,
            pose_set_content_sha256=pose_set_content_sha256,
            artifact_sha256=artifact_hashes,
            pairing_config={
                "maximum_nearest_delta_s": pairing_config.maximum_nearest_delta_s,
                "maximum_bracket_span_s": pairing_config.maximum_bracket_span_s,
            },
            recording_gate_config={
                name: getattr(recording_gate_config, name)
                for name in recording_gate_config.__dataclass_fields__
            },
            provenance=provenance,
        )
        self._write_manifest(manifest, first=True)
        return manifest

    def load(self) -> SessionManifest:
        with self.manifest_path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        return SessionManifest.from_dict(data)

    def append_capture(
        self,
        *,
        capture_id: str,
        pose_id: str,
        outcome: str,
        reason: str,
        frames: Sequence[CaptureFrameInput] = (),
        recorded_at_utc: str | None = None,
    ) -> SessionManifest:
        manifest = self.load()
        if manifest.finalized:
            raise RuntimeError("cannot append to a finalized session")
        if any(item.capture_id == capture_id for item in manifest.captures):
            raise ValueError(f"duplicate capture ID: {capture_id}")
        if not _ID_PATTERN.fullmatch(capture_id) or not _ID_PATTERN.fullmatch(pose_id):
            raise ValueError("capture and pose IDs must use safe identifier characters")
        if outcome not in CAPTURE_OUTCOMES:
            raise ValueError(f"unsupported capture outcome: {outcome}")
        if not reason:
            raise ValueError("capture reason must be non-empty")
        capture_utc = recorded_at_utc or utc_now_iso()
        validate_utc_iso(capture_utc)
        frame_inputs = tuple(frames)
        if len({frame.frame_id for frame in frame_inputs}) != len(frame_inputs):
            raise ValueError("capture input contains duplicate frame IDs")
        if any(
            frame.camera_info.profile_sha256 != manifest.camera_profile_sha256
            for frame in frame_inputs
        ):
            raise ValueError("camera profile changed during the session")
        pairing_config = PairingConfig(**manifest.pairing_config)
        gate_config = RecordingGateConfig(**manifest.recording_gate_config)
        for frame in frame_inputs:
            rebuilt_pairing = pair_state_to_image(
                frame.image_timing,
                frame.state_window,
                config=pairing_config,
            )
            if rebuilt_pairing.to_dict() != frame.pairing.to_dict():
                raise ValueError("provided image/state pairing is not reproducible")
            readiness = evaluate_recording_window(
                frame.state_window,
                now_monotonic_s=frame.state_window[-1].receipt_monotonic_s,
                config=gate_config,
            )
            if not readiness.ready:
                raise ValueError(
                    "capture frame state window is not stationary: "
                    + "; ".join(readiness.hard_failures)
                )
        selected_frame_id = None
        if outcome == "accepted":
            if not frame_inputs:
                raise ValueError("accepted capture requires raw frames")
            if any(
                frame.quality.grade is QualityGrade.RED
                or not frame.correspondences.valid
                for frame in frame_inputs
            ):
                raise ValueError("accepted capture contains an invalid/red raw frame")
            selected_frame_id = self._select_medoid_frame(frame_inputs).frame_id

        existing_frame_ids = {
            frame.frame_id for capture in manifest.captures for frame in capture.frames
        }
        if any(frame.frame_id in existing_frame_ids for frame in frame_inputs):
            raise ValueError("session already contains one of the raw frame IDs")

        records: list[RawFrameRecord] = []
        for frame in frame_inputs:
            records.append(self._write_raw_frame(frame))
        capture = CaptureRecord(
            capture_id=capture_id,
            pose_id=pose_id,
            outcome=outcome,
            reason=reason,
            recorded_at_utc=capture_utc,
            frames=tuple(records),
            selected_frame_id=selected_frame_id,
        )
        updated = replace(manifest, captures=(*manifest.captures, capture))
        self._write_manifest(updated)
        return updated

    def finalize(self) -> SessionManifest:
        manifest = self.load()
        if manifest.finalized:
            return manifest
        updated = replace(manifest, finalized=True)
        self._write_manifest(updated)
        for path in self.directory.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        return updated

    def find_orphans(self) -> tuple[str, ...]:
        manifest = self.load()
        referenced = {
            frame.image_path
            for capture in manifest.captures
            for frame in capture.frames
        } | {
            frame.states_path
            for capture in manifest.captures
            for frame in capture.frames
        }
        actual = {
            path.relative_to(self.directory).as_posix()
            for path in (self.directory / "raw").rglob("*")
            if path.is_file()
        }
        return tuple(sorted(actual - referenced))

    def verify_artifacts(self) -> None:
        manifest = self.load()
        for name, expected in manifest.artifact_sha256.items():
            content = (self.directory / name).read_bytes()
            if _sha256(content) != expected:
                raise ValueError(f"frozen session artifact hash mismatch: {name}")

    def _write_raw_frame(self, frame: CaptureFrameInput) -> RawFrameRecord:
        success, encoded = cv2.imencode(".png", frame.image_bgr)
        if not success:
            raise RuntimeError("OpenCV failed to encode raw PNG")
        image_bytes = encoded.tobytes()
        states_bytes = _canonical_json(
            [sample.to_dict() for sample in frame.state_window]
        )
        image_path = f"raw/images/{frame.frame_id}.png"
        states_path = f"raw/states/{frame.frame_id}.json"
        self._write_new_file(self.directory / image_path, image_bytes)
        # If the second write fails, the image intentionally remains as a
        # detectable orphan; recovery never silently overwrites it.
        self._write_new_file(self.directory / states_path, states_bytes)
        return RawFrameRecord(
            frame_id=frame.frame_id,
            image_path=image_path,
            image_sha256=_sha256(image_bytes),
            states_path=states_path,
            states_sha256=_sha256(states_bytes),
            image_timing={
                "receipt_monotonic_s": frame.image_timing.receipt_monotonic_s,
                "receipt_utc": frame.image_timing.receipt_utc,
                "header_stamp_ns": frame.image_timing.header_stamp_ns,
            },
            camera_info=frame.camera_info.to_dict(),
            pairing=frame.pairing.to_dict(),
            quality=frame.quality.to_dict(),
            live_correspondence_sha256=correspondence_sha256(frame.correspondences),
        )

    @staticmethod
    def _validate_source_artifacts(
        artifacts: Mapping[str, bytes],
        *,
        expected_pose_set_content_sha256: str,
    ) -> None:
        try:
            pose_document = yaml.safe_load(artifacts["pose_set.yaml"])
            pose_set = PoseSet.from_dict(pose_document)
        except Exception as error:
            raise ValueError(
                "pose_set.yaml is not a valid content-hashed pose set"
            ) from error
        if pose_set.content_sha256 != expected_pose_set_content_sha256:
            raise ValueError("pose-set content hash does not match pose_set.yaml")
        try:
            hardware = yaml.safe_load(artifacts["hardware.yaml"])
            target = json.loads(artifacts["target.json"])
            collision_document = yaml.safe_load(artifacts["collision_pairs.yaml"])
            validation_document = json.loads(artifacts["validation_report.json"])
        except Exception as error:
            raise ValueError(
                "one or more frozen session artifacts cannot be parsed"
            ) from error
        if not isinstance(hardware, dict) or not isinstance(target, dict):
            raise TypeError("hardware and target artifacts must contain mappings")
        if not isinstance(validation_document, dict):
            raise TypeError("validation report artifact must contain a mapping")
        validation = ValidationReport.from_dict(validation_document)
        collision = CollisionConfig.from_mapping(collision_document)
        if validation.pose_set_sha256 != pose_set.content_sha256:
            raise ValueError("validation report belongs to a different pose set")
        if validation.urdf_sha256 != pose_set.urdf_sha256:
            raise ValueError("validation report belongs to a different URDF")
        if not validation.passed:
            raise ValueError("session cannot freeze a failed transition report")
        if not collision.hardware_ready:
            raise ValueError("session cannot freeze an unready collision config")
        if validation.collision_config_sha256 != collision.content_sha256:
            raise ValueError(
                "validation report belongs to a different collision configuration"
            )

    @staticmethod
    def _select_medoid_frame(
        frames: tuple[CaptureFrameInput, ...],
    ) -> CaptureFrameInput:
        signatures: dict[tuple[int, ...], list[CaptureFrameInput]] = {}
        for frame in frames:
            signatures.setdefault(frame.correspondences.tag_ids, []).append(frame)
        candidates = max(
            signatures.values(),
            key=lambda group: (len(group), len(group[0].correspondences.tag_ids)),
        )
        vectors = np.asarray(
            [
                np.vstack(
                    [
                        observation.image_corners_px
                        for observation in frame.correspondences.observations
                    ]
                ).reshape(-1)
                for frame in candidates
            ]
        )
        median = np.median(vectors, axis=0)
        distances = np.linalg.norm(vectors - median, axis=1)
        index = min(range(len(candidates)), key=lambda item: (distances[item], item))
        return candidates[index]

    def _write_manifest(
        self, manifest: SessionManifest, *, first: bool = False
    ) -> None:
        data = json.dumps(manifest.to_dict(), indent=2, sort_keys=True).encode() + b"\n"
        if first:
            self._write_new_file(self.manifest_path, data)
            return
        self._replace_file(self.manifest_path, data)

    @staticmethod
    def _write_new_file(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                raise FileExistsError(
                    f"immutable artifact already exists: {path}"
                ) from None
            SessionStore._fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _replace_file(path: Path, data: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            SessionStore._fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
