#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ros_prefix="${G1_CALIB_ROS_PREFIX:-/opt/ros/jazzy}"
cyclone_prefix="${workspace_root}/deps/cyclonedds_python_prefix"

if [[ ! -f "${ros_prefix}/setup.bash" || ! -d "${cyclone_prefix}/lib" ]]; then
    echo "hardware environment is not installed; run tools/install_hardware_dependencies.sh" >&2
    exit 1
fi

# ROS setup scripts are not guaranteed to be nounset-clean.
set +u
source "${ros_prefix}/setup.bash"
set -u

export CYCLONEDDS_HOME="${cyclone_prefix}"
export CMAKE_PREFIX_PATH="${cyclone_prefix}:${ros_prefix}:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${cyclone_prefix}/lib:${ros_prefix}/lib:${ros_prefix}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${cyclone_prefix}/lib:${LIBRARY_PATH:-}"

cd "${workspace_root}"
exec uv run g1-calib "$@"
