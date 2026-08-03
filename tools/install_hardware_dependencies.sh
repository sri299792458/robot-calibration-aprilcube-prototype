#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${G1_CALIB_ROS_PREFIX:-}" ]]; then
    ros_prefix="${G1_CALIB_ROS_PREFIX}"
elif [[ -f /opt/ros/jazzy/setup.bash ]]; then
    ros_prefix="/opt/ros/jazzy"
elif [[ -f /opt/ros/humble/setup.bash ]]; then
    ros_prefix="/opt/ros/humble"
else
    ros_prefix="/opt/ros/jazzy"
fi
cyclone_prefix="${workspace_root}/deps/cyclonedds_python_prefix"

"${workspace_root}/tools/bootstrap_hardware_dependencies.sh"

if [[ ! -f "${ros_prefix}/setup.bash" ]]; then
    echo "ROS installation not found: ${ros_prefix}" >&2
    exit 1
fi
if [[ -d "${ros_prefix}/include/CycloneDDS" ]]; then
    cyclone_include="${ros_prefix}/include/CycloneDDS"
elif [[ -f "${ros_prefix}/include/dds/dds.h" ]]; then
    cyclone_include="${ros_prefix}/include"
else
    echo "ROS CycloneDDS headers not found under ${ros_prefix}" >&2
    exit 1
fi

cyclone_lib="${ros_prefix}/lib"
if [[ -f "${ros_prefix}/lib/x86_64-linux-gnu/libddsc.so" ]]; then
    cyclone_lib="${ros_prefix}/lib/x86_64-linux-gnu"
fi
mkdir -p "${cyclone_prefix}"
ln -sfn "${cyclone_include}" "${cyclone_prefix}/include"
ln -sfn "${ros_prefix}/bin" "${cyclone_prefix}/bin"
ln -sfn "${cyclone_lib}" "${cyclone_prefix}/lib"

export CYCLONEDDS_HOME="${cyclone_prefix}"
export CMAKE_PREFIX_PATH="${cyclone_prefix}:${ros_prefix}:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${cyclone_prefix}/lib:${ros_prefix}/lib:${ros_prefix}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${cyclone_prefix}/lib:${LIBRARY_PATH:-}"

uv pip install "cyclonedds==0.10.2" pygame
uv pip install --no-deps -e "${workspace_root}/external/unitree_sdk2_python"

"${workspace_root}/.venv/bin/python" - <<'PY'
from g1_aprilcube_calibration.transports.unitree_arm_sdk import UnitreeSDKBindings

bindings = UnitreeSDKBindings.load()
message = bindings.make_low_command()
assert len(message.motor_cmd) == 35
assert bindings.calculate_crc(message) >= 0
print("verified Unitree HG LowCmd: 35 motor slots and working CRC")
PY
