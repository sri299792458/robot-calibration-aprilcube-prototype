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
from g1_aprilcube_calibration.session_store import CaptureFrameInput, SessionStore
from g1_aprilcube_calibration.timestamp_pairing import (
    PairingConfig,
    pair_state_to_image,
)


@dataclass(frozen=True, slots=True)
class LiveBurstConfig:
    frame_count: int = 7
    timeout_s: float = 15.0
    poll_interval_s: float = 0.01
    maximum_inter_frame_gap_s: float = 0.5
    maximum_duration_s: float = 2.0

    def __post_init__(self) -> None:
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if (
            self.timeout_s <= 0
            or self.poll_interval_s <= 0
            or self.maximum_inter_frame_gap_s <= 0
            or self.maximum_duration_s <= 0
        ):
            raise ValueError("live burst time values must be positive")
        if self.maximum_duration_s < self.maximum_inter_frame_gap_s:
            raise ValueError(
                "maximum burst duration must be at least the inter-frame gap"
            )


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
        history: tuple[ViewSignature, ...] = (),
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
        self._history = list(history)

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
                latest_state = self.robot_states.latest
                required_state_time = (
                    frame.timing.receipt_monotonic_s
                    + self.recording_config.stationary_duration_s / 2.0
                )
                if (
                    latest_state is None
                    or latest_state.receipt_monotonic_s < required_state_time
                ):
                    # Keep the image eligible until its centered state window
                    # has a post-exposure bracket.  Marking it seen here would
                    # reject every genuinely live frame before future states
                    # can arrive.
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
                rejection_reason = self._burst_rejection_reason(accepted, candidate)
                if rejection_reason is not None:
                    raise RuntimeError("burst rejected: " + rejection_reason)
                accepted.append(candidate)
                if len(accepted) >= self.config.frame_count:
                    break
            if len(accepted) < self.config.frame_count:
                self.wait_once(self.config.poll_interval_s)
        selected = SessionStore.select_medoid_frame(tuple(accepted))
        signature = selected.quality.signature
        if signature is not None:
            self._history.append(signature)
        return tuple(accepted)

    def _burst_rejection_reason(
        self,
        accepted: list[CaptureFrameInput],
        candidate: CaptureFrameInput,
    ) -> str | None:
        if not accepted:
            return None
        candidate_time = candidate.image_timing.receipt_monotonic_s
        previous_time = accepted[-1].image_timing.receipt_monotonic_s
        frame_gap = candidate_time - previous_time
        if frame_gap > self.config.maximum_inter_frame_gap_s + 1e-12:
            return (
                f"image gap is {frame_gap:.3f}s; limit is "
                f"{self.config.maximum_inter_frame_gap_s:.3f}s"
            )
        burst_duration = (
            candidate_time - accepted[0].image_timing.receipt_monotonic_s
        )
        if burst_duration > self.config.maximum_duration_s + 1e-12:
            return (
                f"burst duration is {burst_duration:.3f}s; limit is "
                f"{self.config.maximum_duration_s:.3f}s"
            )

        samples_by_time = {
            sample.receipt_monotonic_s: sample
            for frame in (*accepted, candidate)
            for sample in frame.state_window
        }
        combined_window = tuple(
            samples_by_time[timestamp]
            for timestamp in sorted(samples_by_time)
        )
        readiness = evaluate_recording_window(
            combined_window,
            now_monotonic_s=combined_window[-1].receipt_monotonic_s,
            config=self.recording_config,
        )
        if not readiness.ready:
            return "; ".join(readiness.hard_failures)
        return None

    def undo_last_signature(self) -> None:
        if not self._history:
            raise ValueError("cannot undo an empty live-capture history")
        self._history.pop()

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
