"""Read-only manual-pose recorder built on measured state and visual gates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from g1_aprilcube_calibration.joint_map import arm_indices, opposite_arm
from g1_aprilcube_calibration.pose_schema import PoseRecord, PoseSet
from g1_aprilcube_calibration.pose_store import PoseStore
from g1_aprilcube_calibration.quality import QualityGrade, QualityReport
from g1_aprilcube_calibration.readiness import (
    ReadinessReport,
    RecordingGateConfig,
    StateSampleBuffer,
    evaluate_recording_window,
)
from g1_aprilcube_calibration.timestamp_pairing import (
    ImageTiming,
    PairingConfig,
    PairingResult,
    pair_state_to_image,
)


@dataclass(frozen=True, slots=True)
class PoseRecordingRequest:
    pose_id: str
    group: str
    image_timing: ImageTiming
    visual_report: QualityReport
    head_witness_ack: bool
    anchor: bool = False
    preview_path: str | Path | None = None
    yellow_override_reason: str | None = None


@dataclass(frozen=True, slots=True)
class PoseRecordingAssessment:
    allowed: bool
    failures: tuple[str, ...]
    warnings: tuple[str, ...]
    readiness: ReadinessReport | None
    pairing: PairingResult | None
    candidate: PoseRecord | None


class PoseRecorder:
    """Assesses and atomically stores poses; it has no command-transport API."""

    def __init__(
        self,
        *,
        store: PoseStore,
        state_buffer: StateSampleBuffer,
        gate_config: RecordingGateConfig | None = None,
        pairing_config: PairingConfig | None = None,
    ) -> None:
        self.store = store
        self.state_buffer = state_buffer
        self.gate_config = gate_config or RecordingGateConfig()
        self.pairing_config = pairing_config or PairingConfig()

    def assess(
        self,
        request: PoseRecordingRequest,
        *,
        now_monotonic_s: float,
    ) -> PoseRecordingAssessment:
        failures: list[str] = []
        warnings: list[str] = []
        if not request.head_witness_ack:
            failures.append("head-pitch witness mark has not been acknowledged")
        if request.visual_report.grade is QualityGrade.RED:
            failures.extend(
                f"visual: {reason}" for reason in request.visual_report.hard_failures
            )
            if not request.visual_report.hard_failures:
                failures.append("visual quality is red")
        elif request.visual_report.grade is QualityGrade.YELLOW:
            reason = request.yellow_override_reason
            if reason is None or not reason.strip():
                failures.append("yellow visual quality requires an override reason")
            else:
                warnings.append(f"yellow visual override: {reason.strip()}")
        warnings.extend(
            f"visual: {warning}" for warning in request.visual_report.warnings
        )

        samples = self.state_buffer.snapshot()
        pairing = None
        try:
            pairing = pair_state_to_image(
                request.image_timing,
                samples,
                config=self.pairing_config,
            )
        except ValueError as error:
            failures.append(str(error))

        window = self.state_buffer.centered_window(
            center_monotonic_s=request.image_timing.receipt_monotonic_s,
            duration_s=self.gate_config.stationary_duration_s,
        )
        readiness = evaluate_recording_window(
            window,
            now_monotonic_s=now_monotonic_s,
            config=self.gate_config,
        )
        failures.extend(readiness.hard_failures)
        pose_set = self.store.load()
        if pose_set.calibration_arm != self.gate_config.calibration_arm:
            failures.append(
                "recording configuration arm does not match the pose set: "
                f"{self.gate_config.calibration_arm} != {pose_set.calibration_arm}"
            )
        if window:
            hold_arm = opposite_arm(pose_set.calibration_arm)
            hold_positions = np.vstack([sample.arm_q(hold_arm) for sample in window])
            hold_spread = float(np.max(np.ptp(hold_positions, axis=0)))
            if hold_spread > self.gate_config.maximum_hold_position_spread_rad:
                failures.append(
                    f"held {hold_arm}-arm position spread is {hold_spread:.4f}rad; "
                    "limit is "
                    f"{self.gate_config.maximum_hold_position_spread_rad:.4f}rad"
                )
            hold_error = float(
                np.max(
                    np.abs(
                        np.median(hold_positions, axis=0) - np.asarray(pose_set.hold_q)
                    )
                )
            )
            if hold_error > self.gate_config.maximum_hold_position_spread_rad:
                failures.append(
                    f"measured {hold_arm} arm differs from pose-set hold by "
                    f"{hold_error:.4f}rad"
                )
        if failures:
            return PoseRecordingAssessment(
                allowed=False,
                failures=tuple(dict.fromkeys(failures)),
                warnings=tuple(warnings),
                readiness=readiness,
                pairing=pairing,
                candidate=None,
            )

        full_positions = np.vstack([sample.position for sample in window])
        calibration_indices = np.asarray(arm_indices(pose_set.calibration_arm))
        calibration_positions = full_positions[:, calibration_indices]
        measured_full_q = np.median(full_positions, axis=0)
        measured_calibration_q = measured_full_q[calibration_indices]
        calibration_spread = np.ptp(calibration_positions, axis=0)
        visual_quality = request.visual_report.to_dict()
        visual_quality["state_readiness"] = readiness.to_dict()
        visual_quality["timestamp_pairing"] = pairing.to_dict() if pairing else None
        visual_quality["yellow_override_reason"] = (
            None
            if request.yellow_override_reason is None
            else request.yellow_override_reason.strip()
        )
        candidate = PoseRecord(
            id=request.pose_id,
            group=request.group,
            measured_calibration_q=tuple(measured_calibration_q),
            measured_full_q=tuple(measured_full_q),
            calibration_q_spread=tuple(calibration_spread),
            recorded_at_utc=request.image_timing.receipt_utc,
            recorded_monotonic_s=request.image_timing.receipt_monotonic_s,
            anchor=request.anchor,
            head_witness_ack=request.head_witness_ack,
            preview_path=(
                None if request.preview_path is None else str(request.preview_path)
            ),
            visual_quality=visual_quality,
        )
        return PoseRecordingAssessment(
            allowed=True,
            failures=(),
            warnings=tuple(warnings),
            readiness=readiness,
            pairing=pairing,
            candidate=candidate,
        )

    def record(
        self,
        request: PoseRecordingRequest,
        *,
        now_monotonic_s: float,
    ) -> PoseSet:
        assessment = self.assess(request, now_monotonic_s=now_monotonic_s)
        if not assessment.allowed or assessment.candidate is None:
            reasons = "; ".join(assessment.failures)
            raise ValueError(f"pose is not recordable: {reasons}")
        return self.store.append(
            assessment.candidate,
            details={
                "visual_grade": request.visual_report.grade.value,
                "yellow_override_reason": (
                    None
                    if request.yellow_override_reason is None
                    else request.yellow_override_reason.strip()
                ),
            },
        )
