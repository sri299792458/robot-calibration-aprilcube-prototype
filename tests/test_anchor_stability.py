import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.anchor_stability import (
    AnchorStabilityConfig,
    analyze_anchor_transforms,
)


def transform(x_m=0.0, angle_deg=0.0):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("z", angle_deg, degrees=True).as_matrix()
    result[0, 3] = x_m
    return result


def test_anchor_pairwise_stability_passes_small_repeatability_error():
    report = analyze_anchor_transforms(
        "home",
        {
            "capture_001": transform(),
            "capture_010": transform(0.0004, 0.1),
            "capture_020": transform(-0.0003, -0.1),
        },
    )
    assert report.passed
    assert report.maximum_pairwise_translation_m == pytest.approx(0.0007)
    assert report.maximum_pairwise_rotation_deg < 0.21


def test_anchor_pairwise_stability_reports_mount_motion():
    report = analyze_anchor_transforms(
        "home",
        {
            "before": transform(),
            "middle": transform(0.0002, 0.1),
            "after": transform(0.004, 1.2),
        },
        config=AnchorStabilityConfig(
            maximum_pairwise_translation_m=0.002,
            maximum_pairwise_rotation_deg=0.5,
        ),
    )
    assert not report.passed
    assert len(report.failures) == 2
