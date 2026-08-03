"""Capture lifecycle and approved pose-plan orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from g1_aprilcube_calibration.executor_state_machine import (
    ExecutorState,
    PoseExecutor,
)
from g1_aprilcube_calibration.pose_validator import ValidationReport
from g1_aprilcube_calibration.session_store import CaptureFrameInput


class CaptureStore(Protocol):
    def append_capture(
        self,
        *,
        capture_id: str,
        pose_id: str,
        outcome: str,
        reason: str,
        frames: Sequence[CaptureFrameInput] = (),
        recorded_at_utc: str | None = None,
    ): ...

    def finalize(self): ...


class FrameSource(Protocol):
    def capture_burst(
        self, *, pose_id: str, capture_id: str
    ) -> tuple[CaptureFrameInput, ...]: ...


@dataclass(frozen=True, slots=True)
class SessionExecutionPlan:
    home_pose_id: str
    capture_pose_ids: tuple[str, ...]
    maximum_control_steps_per_transition: int = 10_000

    def __post_init__(self) -> None:
        if not self.home_pose_id:
            raise ValueError("home_pose_id must be non-empty")
        if not self.capture_pose_ids:
            raise ValueError("capture plan must contain at least one pose")
        if self.maximum_control_steps_per_transition <= 0:
            raise ValueError("maximum control steps must be positive")


class ScheduledFrameSource:
    """One-shot deterministic frame bursts used by offline integration runs."""

    def __init__(
        self, frames_by_pose: Mapping[str, Sequence[CaptureFrameInput]]
    ) -> None:
        self._frames = {
            pose_id: tuple(frames) for pose_id, frames in frames_by_pose.items()
        }
        self._consumed: set[str] = set()

    def capture_burst(
        self, *, pose_id: str, capture_id: str
    ) -> tuple[CaptureFrameInput, ...]:
        del capture_id
        if pose_id in self._consumed:
            raise RuntimeError(f"frame burst for {pose_id} was already consumed")
        if pose_id not in self._frames:
            raise ValueError(f"no scheduled frame burst for pose {pose_id}")
        self._consumed.add(pose_id)
        return self._frames[pose_id]


class CaptureSessionRunner:
    """Tie raw writes to the executor's stationary capture interlock."""

    def __init__(self, *, executor: PoseExecutor, store: CaptureStore) -> None:
        self.executor = executor
        self.store = store

    def capture(
        self,
        *,
        capture_id: str,
        pose_id: str,
        frames: Sequence[CaptureFrameInput],
        reason: str = "stationary burst passed",
    ) -> None:
        if self.executor.current_pose_id != pose_id:
            raise ValueError("capture pose does not match executor's current pose")
        self.executor.begin_capture()
        try:
            self.store.append_capture(
                capture_id=capture_id,
                pose_id=pose_id,
                outcome="accepted",
                reason=reason,
                frames=frames,
            )
        except Exception:
            self.executor.finish_capture(outcome="raw capture failed")
            raise
        self.executor.finish_capture(outcome="accepted")

    def reject(self, *, capture_id: str, pose_id: str, reason: str) -> None:
        if self.executor.current_pose_id != pose_id:
            raise ValueError("capture pose does not match executor's current pose")
        self.executor.begin_capture()
        try:
            self.store.append_capture(
                capture_id=capture_id,
                pose_id=pose_id,
                outcome="rejected",
                reason=reason,
            )
        except Exception:
            self.executor.finish_capture(outcome="rejection write failed")
            raise
        self.executor.finish_capture(outcome="rejected")


class ApprovedSessionOrchestrator:
    """Run an already validated pose plan; every physical move is confirmed."""

    def __init__(
        self,
        *,
        executor: PoseExecutor,
        validation_report: ValidationReport,
        capture_runner: CaptureSessionRunner,
        frame_source: FrameSource,
        control_step: Callable[[], None],
        confirm_move: Callable[[str, str], bool],
        plan: SessionExecutionPlan,
    ) -> None:
        if validation_report.pose_set_sha256 != executor.pose_set.content_sha256:
            raise ValueError("validation report belongs to a different pose set")
        if validation_report.content_sha256 != (
            executor.approved_validation_report_sha256
        ):
            raise ValueError("executor was not bound to this validation report")
        if not validation_report.passed:
            raise ValueError("cannot execute a failed validation report")
        pose_ids = {pose.id for pose in executor.pose_set.poses}
        missing = set(plan.capture_pose_ids) | {plan.home_pose_id}
        missing -= pose_ids
        if missing:
            raise ValueError(
                "execution plan references unknown poses: " + ", ".join(sorted(missing))
            )
        self.executor = executor
        self.validation_report = validation_report
        self.capture_runner = capture_runner
        self.frame_source = frame_source
        self.control_step = control_step
        self.confirm_move = confirm_move
        self.plan = plan

    def run(self, *, confirm_acquisition: bool, confirm_release: bool) -> None:
        try:
            self.executor.acquire(
                operator_confirmed=confirm_acquisition,
                initial_pose_id=self.plan.home_pose_id,
            )
            self._drive_until(ExecutorState.READY)
            for index, pose_id in enumerate(self.plan.capture_pose_ids, start=1):
                if self.executor.current_pose_id != pose_id:
                    self._move_to(pose_id)
                capture_id = f"capture_{index:03d}"
                frames = self.frame_source.capture_burst(
                    pose_id=pose_id, capture_id=capture_id
                )
                self.capture_runner.capture(
                    capture_id=capture_id,
                    pose_id=pose_id,
                    frames=frames,
                )
            if self.executor.current_pose_id != self.plan.home_pose_id:
                self._move_to(self.plan.home_pose_id)
            self.executor.begin_clean_release(
                approved_home_pose_id=self.plan.home_pose_id,
                operator_confirmed=confirm_release,
            )
            self._drive_until(ExecutorState.STOPPED)
            self.capture_runner.store.finalize()
        except Exception:
            self._emergency_release()
            raise

    def _move_to(self, pose_id: str) -> None:
        source = self.executor.current_pose_id
        if source is None:
            raise RuntimeError("executor has no current pose")
        self.executor.start_pose(
            pose_id,
            approval=self.validation_report.approval(source, pose_id),
            operator_confirmed=self.confirm_move(source, pose_id),
        )
        self._drive_until(ExecutorState.READY)

    def _drive_until(self, desired: ExecutorState) -> None:
        for _ in range(self.plan.maximum_control_steps_per_transition):
            if self.executor.state is desired:
                return
            if self.executor.state is ExecutorState.STOPPED:
                raise RuntimeError(
                    f"executor stopped before reaching {desired.value}: "
                    f"{self.executor.fault_reason or 'unknown reason'}"
                )
            self.control_step()
        raise RuntimeError(f"executor did not reach {desired.value} within step limit")

    def _emergency_release(self) -> None:
        if self.executor.state is ExecutorState.STOPPED:
            return
        self.executor.emergency_stop("session orchestration failed")
        for _ in range(self.plan.maximum_control_steps_per_transition):
            if self.executor.state is ExecutorState.STOPPED:
                return
            try:
                self.control_step()
            except (RuntimeError, TypeError, ValueError):
                return
