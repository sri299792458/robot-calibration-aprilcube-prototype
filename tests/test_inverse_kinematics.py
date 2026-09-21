from pathlib import Path

import numpy as np
import pytest

from g1_aprilcube_calibration.inverse_kinematics import (
    IKConfig,
    solve_arm_ik_candidates,
)
from g1_aprilcube_calibration.joint_map import G1_29_JOINT_NAMES
from g1_aprilcube_calibration.urdf_model import URDFModel

ROOT = Path(__file__).parents[1]
URDF = ROOT / "unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"


def _hand_transform(model: URDFModel, q: np.ndarray) -> np.ndarray:
    full = np.zeros(29)
    full[15:22] = q
    return model.transform(
        "torso_link",
        "left_rubber_hand",
        dict(zip(G1_29_JOINT_NAMES, full, strict=True)),
    )


def test_bounded_multistart_ik_recovers_reachable_hand_transform() -> None:
    model = URDFModel(URDF)
    target_q = np.asarray([0.35, 0.2, -0.25, 0.7, 0.3, -0.2, 0.15])

    solutions = solve_arm_ik_candidates(
        model,
        desired_torso_T_hand=_hand_transform(model, target_q),
        reference_full_q=np.zeros(29),
        calibration_arm="left",
        config=IKConfig(restart_count=12),
    )

    assert solutions
    assert solutions[0].translation_error_m < 0.0005
    assert solutions[0].rotation_error_deg < 0.25
    assert np.allclose(
        solutions[0].torso_T_hand,
        _hand_transform(model, np.asarray(solutions[0].calibration_q)),
    )
    limits = model.joint_limits(tuple(G1_29_JOINT_NAMES[15:22]))
    for value, limit in zip(solutions[0].calibration_q, limits, strict=True):
        assert value >= limit.lower + 0.03 - 1e-9
        assert value <= limit.upper - 0.03 + 1e-9


def test_ik_rejects_unreachable_cartesian_target() -> None:
    model = URDFModel(URDF)
    unreachable = np.eye(4)
    unreachable[:3, 3] = [5.0, 5.0, 5.0]

    with pytest.raises(ValueError, match="no bounded IK solution"):
        solve_arm_ik_candidates(
            model,
            desired_torso_T_hand=unreachable,
            reference_full_q=np.zeros(29),
            calibration_arm="left",
            config=IKConfig(restart_count=4, maximum_function_evaluations=100),
        )


def test_ik_configuration_rejects_nonpositive_limits() -> None:
    with pytest.raises(ValueError, match="positive"):
        IKConfig(joint_limit_margin_rad=0.0)
