import pytest

from g1_aprilcube_calibration.process_lock import CommandOwnerLock


def test_only_one_local_command_owner_can_hold_lock(tmp_path) -> None:
    first = CommandOwnerLock(tmp_path / "arm_sdk.lock")
    second = CommandOwnerLock(tmp_path / "arm_sdk.lock")
    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="another"):
            second.acquire()
    finally:
        first.release()

    second.acquire()
    second.release()
