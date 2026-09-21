"""Command-line entry point for the G1 calibration prototype."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from aprilcube import CorrespondenceDetector
from g1_aprilcube_calibration.config import QualityThresholds
from g1_aprilcube_calibration.hardware_cli import add_hardware_subparsers
from g1_aprilcube_calibration.preview import render_operator_preview
from g1_aprilcube_calibration.quality import (
    CameraIntrinsics,
    PoseQualityEvaluator,
    QualityGrade,
    ViewSignature,
)
from g1_aprilcube_calibration.workflow_cli import add_workflow_subparsers

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET_CONFIG = (
    WORKSPACE_ROOT / "config" / "dex3_dorsal_aruco_target.json"
)
DEFAULT_QUALITY_CONFIG = (
    WORKSPACE_ROOT / "config" / "capture_quality_dex3_aruco.yaml"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="g1-calib",
        description="Unitree G1 RealSense-AprilCube calibration prototype",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preview = subparsers.add_parser(
        "preview",
        help="show stateless AprilCube detections and visual pose quality",
    )
    source = preview.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="process one saved image")
    source.add_argument("--video", type=Path, help="preview a video file")
    source.add_argument("--camera", type=int, help="OpenCV camera index")
    preview.add_argument(
        "--target-config",
        type=Path,
        default=DEFAULT_TARGET_CONFIG,
        help="hand-target config (default: right Dex3 dorsal ArUco plate)",
    )
    preview.add_argument(
        "--quality-config",
        type=Path,
        default=DEFAULT_QUALITY_CONFIG,
    )
    preview.add_argument("--intrinsics-json", type=Path)
    preview.add_argument("--fx", type=float)
    preview.add_argument("--fy", type=float)
    preview.add_argument("--cx", type=float)
    preview.add_argument("--cy", type=float)
    preview.add_argument(
        "--output",
        type=Path,
        help="write the annotated result (single-image mode)",
    )
    preview.add_argument("--report-json", type=Path)
    preview.add_argument(
        "--save-directory",
        type=Path,
        help="write visual snapshots accepted with the S key",
    )
    preview.add_argument(
        "--no-window",
        action="store_true",
        help="headless single-image processing",
    )
    preview.set_defaults(handler=run_preview)
    add_workflow_subparsers(
        subparsers,
        workspace_root=WORKSPACE_ROOT,
        default_target=DEFAULT_TARGET_CONFIG,
    )
    add_hardware_subparsers(
        subparsers,
        workspace_root=WORKSPACE_ROOT,
        default_target=DEFAULT_TARGET_CONFIG,
        default_quality=DEFAULT_QUALITY_CONFIG,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("interrupted by operator", file=sys.stderr)
        return 130
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def run_preview(args: argparse.Namespace) -> int:
    if args.no_window and args.image is None:
        raise ValueError("--no-window currently requires --image")
    if args.output is not None and args.image is None:
        raise ValueError("--output currently requires --image")

    detector = CorrespondenceDetector(args.target_config)
    thresholds = QualityThresholds.from_yaml(args.quality_config)
    evaluator = PoseQualityEvaluator(thresholds)

    if args.image is not None:
        frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"could not read image: {args.image}")
        intrinsics = _load_intrinsics(args, frame.shape)
        report, rendered = _evaluate_frame(
            frame,
            detector=detector,
            evaluator=evaluator,
            intrinsics=intrinsics,
            history=(),
        )
        _emit_report(report.to_dict(), args.report_json)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.output), rendered):
                raise RuntimeError(f"failed to write preview: {args.output}")
        if args.no_window:
            return 0 if report.save_allowed else 1
        cv2.imshow("G1 AprilCube calibration preview", rendered)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0 if report.save_allowed else 1

    source: int | str = 0
    if args.video is not None:
        source = str(args.video)
    elif args.camera is not None:
        source = args.camera
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"could not open video source: {source}")

    saved_views: list[ViewSignature] = []
    pending_yellow = False
    latest_report = None
    latest_rendered = None
    intrinsics = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if intrinsics is None:
                intrinsics = _load_intrinsics(args, frame.shape)
            report, rendered = _evaluate_frame(
                frame,
                detector=detector,
                evaluator=evaluator,
                intrinsics=intrinsics,
                history=saved_views,
            )
            latest_report = report
            latest_rendered = rendered
            cv2.imshow("G1 AprilCube calibration preview", rendered)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("u"):
                pending_yellow = False
                if saved_views:
                    saved_views.pop()
                    print(f"removed visual view; {len(saved_views)} remain")
            elif key == ord("s"):
                if report.grade is QualityGrade.RED or report.signature is None:
                    pending_yellow = False
                    print("visual view not saved: red quality gate")
                elif report.grade is QualityGrade.YELLOW and not pending_yellow:
                    pending_yellow = True
                    print("yellow view: press S again to confirm visual save")
                else:
                    saved_views.append(report.signature)
                    pending_yellow = False
                    print(
                        f"saved visual view {len(saved_views)} ({report.grade.value})"
                    )
                    if args.save_directory is not None:
                        _write_snapshot(args.save_directory, len(saved_views), rendered)
            elif key != 255:
                pending_yellow = False
    finally:
        capture.release()
        cv2.destroyAllWindows()

    if latest_report is not None:
        _emit_report(latest_report.to_dict(), args.report_json)
    if args.save_directory is not None and latest_rendered is not None:
        _write_snapshot(args.save_directory, 0, latest_rendered, name="latest")
    return 0


def _evaluate_frame(
    frame: np.ndarray,
    *,
    detector: CorrespondenceDetector,
    evaluator: PoseQualityEvaluator,
    intrinsics: CameraIntrinsics | None,
    history: Sequence[ViewSignature],
):
    detections = detector.detect(frame)
    report = evaluator.evaluate(
        detections,
        intrinsics=intrinsics,
        history=history,
    )
    rendered = render_operator_preview(
        frame,
        detections,
        report,
        intrinsics=intrinsics,
        saved_view_count=len(history),
    )
    return report, rendered


def _load_intrinsics(
    args: argparse.Namespace,
    image_shape: Sequence[int],
) -> CameraIntrinsics | None:
    if args.intrinsics_json is not None:
        with args.intrinsics_json.open(encoding="utf-8") as stream:
            data = json.load(stream)
        if "camera_matrix" in data:
            raw_matrix = data["camera_matrix"]
            if isinstance(raw_matrix, dict):
                raw_matrix = raw_matrix["data"]
            matrix = np.asarray(raw_matrix, dtype=np.float64).reshape(3, 3)
            distortion = data.get(
                "dist_coeffs", data.get("distortion_coefficients", [])
            )
            if isinstance(distortion, dict):
                distortion = distortion.get("data", [])
            return CameraIntrinsics(matrix, np.asarray(distortion or [0] * 5))
        return CameraIntrinsics.from_parameters(
            fx=float(data["fx"]),
            fy=float(data.get("fy", data["fx"])),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            dist_coeffs=data.get("dist_coeffs", [0] * 5),
        )

    provided = [args.fx, args.fy, args.cx, args.cy]
    if all(value is None for value in provided):
        return None
    if args.fx is None:
        raise ValueError("--fx is required when providing inline intrinsics")
    height, width = image_shape[:2]
    return CameraIntrinsics.from_parameters(
        fx=args.fx,
        fy=args.fx if args.fy is None else args.fy,
        cx=(width - 1) / 2.0 if args.cx is None else args.cx,
        cy=(height - 1) / 2.0 if args.cy is None else args.cy,
    )


def _emit_report(report: dict, output: Path | None) -> None:
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    print(encoded)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")


def _write_snapshot(
    directory: Path,
    index: int,
    image: np.ndarray,
    *,
    name: str | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    stem = name if name is not None else f"view_{index:03d}"
    path = directory / f"{stem}.png"
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write snapshot: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
