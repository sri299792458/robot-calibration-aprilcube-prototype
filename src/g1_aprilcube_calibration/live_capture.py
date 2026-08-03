"""Hardware-neutral construction of stationary raw bursts from live buffers."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from aprilcube import CorrespondenceDetector
from g1_aprilcube_calibration.clock import MonotonicClock, SystemClock
from g1_aprilcube_calibration.quality import (
    CameraIntrinsics,
    PoseQualityEvaluator,
    QualityGrade,
    ViewSignature,
)
from g1_aprilcube_calibration.readiness import (
    RecordingGateConfig,
    StateSampleBuffer,
    evaluate_recording_window,
)
from g1_aprilcube_calibration.ros.camera_adapter import ROSFrameBuffer, ROSImageFrame
from g1_aprilcube_calibration.session_store import CaptureFrameInput
from g1_aprilcube_calibration.timestamp_pairing import (
    PairingConfig,
    pair_state_to_image,
)


@dataclass(frozen=True, slots=True)
class LiveBurstConfig:
    frame_count: int = 7
    timeout_s: float = 15.0
    poll_interval_s: float = 0.01

    def __post_init__(self) -> None:
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if self.timeout_s <= 0 or self.poll_interval_s <= 0:
            raise ValueError("live burst time values must be positive")


class LiveBurstFrameSource:
    """Collect only new, reproducibly paired, stationary, visually valid frames."""

    def __init__(
        self,
        *,
        camera_frames: ROSFrameBuffer,
        robot_states: StateSampleBuffer,
        detector: CorrespondenceDetector,
        quality_evaluator: PoseQualityEvaluator,
        recording_config: RecordingGateConfig,
        pairing_config: PairingConfig,
        config: LiveBurstConfig | None = None,
        clock: MonotonicClock | None = None,
        wait_once: Callable[[float], None] = time.sleep,
        accept_yellow: Callable[[ROSImageFrame], bool] | None = None,
        preview: Callable[[ROSImageFrame, object, object], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        self.camera_frames = camera_frames
        self.robot_states = robot_states
        self.detector = detector
        self.quality_evaluator = quality_evaluator
        self.recording_config = recording_config
        self.pairing_config = pairing_config
        self.config = config or LiveBurstConfig()
        self.clock = clock or SystemClock()
        self.wait_once = wait_once
        self.accept_yellow = accept_yellow or (lambda _frame: False)
        self.preview = preview
        self.cancelled = cancelled or (lambda: False)
        self._history: list[ViewSignature] = []

    def capture_burst(
        self, *, pose_id: str, capture_id: str
    ) -> tuple[CaptureFrameInput, ...]:
        del pose_id
        seen = {self._frame_key(frame) for frame in self.camera_frames.snapshot()}
        accepted: list[CaptureFrameInput] = []
        deadline = self.clock.monotonic() + self.config.timeout_s
        last_rejection = "no new rectified image"
        while len(accepted) < self.config.frame_count:
            if self.cancelled():
                raise RuntimeError("operator cancelled live burst")
            if self.clock.monotonic() >= deadline:
                raise RuntimeError(
                    f"timed out with {len(accepted)}/{self.config.frame_count} "
                    f"valid frames; last rejection: {last_rejection}"
                )
            for frame in self.camera_frames.snapshot():
                key = self._frame_key(frame)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    candidate = self._evaluate_frame(
                        frame,
                        frame_id=f"{capture_id}_{len(accepted):03d}",
                    )
                except (RuntimeError, TypeError, ValueError) as error:
                    last_rejection = str(error)
                    continue
                accepted.append(candidate)
                if len(accepted) >= self.config.frame_count:
                    break
            if len(accepted) < self.config.frame_count:
                self.wait_once(self.config.poll_interval_s)
        signature = accepted[-1].quality.signature
        if signature is not None:
            self._history.append(signature)
        return tuple(accepted)

    def _evaluate_frame(
        self, frame: ROSImageFrame, *, frame_id: str
    ) -> CaptureFrameInput:
        correspondences = self.detector.detect(frame.image_bgr)
        intrinsics = CameraIntrinsics(
            frame.camera_info.rectified_camera_matrix,
            np.asarray(frame.camera_info.d),
        )
        quality = self.quality_evaluator.evaluate(
            correspondences,
            intrinsics=intrinsics,
            history=self._history,
        )
        if self.preview is not None:
            self.preview(frame, correspondences, quality)
        if quality.grade is QualityGrade.RED:
            raise ValueError(
                "visual quality is red: " + "; ".join(quality.hard_failures)
            )
        if quality.grade is QualityGrade.YELLOW and not self.accept_yellow(frame):
            raise ValueError("visual quality is yellow and was not confirmed")
        state_window = self.robot_states.centered_window(
            center_monotonic_s=frame.timing.receipt_monotonic_s,
            duration_s=self.recording_config.stationary_duration_s,
        )
        readiness = evaluate_recording_window(
            state_window,
            now_monotonic_s=(
                state_window[-1].receipt_monotonic_s
                if state_window
                else frame.timing.receipt_monotonic_s
            ),
            config=self.recording_config,
        )
        if not readiness.ready:
            raise ValueError(
                "robot state is not stationary: " + "; ".join(readiness.hard_failures)
            )
        pairing = pair_state_to_image(
            frame.timing, state_window, config=self.pairing_config
        )
        return CaptureFrameInput(
            frame_id=frame_id,
            image_bgr=frame.image_bgr,
            image_timing=frame.timing,
            camera_info=frame.camera_info,
            state_window=state_window,
            pairing=pairing,
            correspondences=correspondences,
            quality=quality,
        )

    @staticmethod
    def _frame_key(frame: ROSImageFrame) -> tuple[float, int | None]:
        return (
            frame.timing.receipt_monotonic_s,
            frame.timing.header_stamp_ns,
        )
