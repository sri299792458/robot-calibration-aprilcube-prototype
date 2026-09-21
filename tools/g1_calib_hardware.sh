#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
command_args=("$@")
network_interface=""
domain_id="0"
for ((index = 0; index < ${#command_args[@]}; index++)); do
    case "${command_args[index]}" in
        --network-interface)
            if ((index + 1 < ${#command_args[@]})); then
                network_interface="${command_args[index + 1]}"
            fi
            ;;
        --network-interface=*)
            network_interface="${command_args[index]#*=}"
            ;;
        --domain-id)
            if ((index + 1 < ${#command_args[@]})); then
                domain_id="${command_args[index + 1]}"
            fi
            ;;
        --domain-id=*)
            domain_id="${command_args[index]#*=}"
            ;;
    esac
done

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

if [[ ! -f "${ros_prefix}/setup.bash" || ! -d "${cyclone_prefix}/lib" ]]; then
    echo "hardware environment is not installed; run tools/install_hardware_dependencies.sh" >&2
    exit 1
fi
if [[ ! -x "${workspace_root}/.venv/bin/g1-calib" ]]; then
    echo "project environment is not installed; run uv sync --python /usr/bin/python3 --group dev" >&2
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

# Use the same explicit wired interface for ROS camera traffic and Unitree DDS.
# The latter also receives this value through the CLI's --network-interface.
if [[ -n "${network_interface}" ]]; then
    if [[ ! "${network_interface}" =~ ^[[:alnum:]_.:-]+$ ]] || \
        [[ ! -e "/sys/class/net/${network_interface}" ]]; then
        echo "invalid local network interface: ${network_interface}" >&2
        exit 1
    fi
    if [[ ! "${domain_id}" =~ ^[0-9]+$ ]]; then
        echo "invalid ROS domain ID: ${domain_id}" >&2
        exit 1
    fi
    export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
    export ROS_DOMAIN_ID="${domain_id}"
    export ROS_LOCALHOST_ONLY=0
    if [[ -z "${CYCLONEDDS_URI:-}" ]]; then
        export CYCLONEDDS_URI="<CycloneDDS><Domain Id=\"any\"><General><Interfaces><NetworkInterface name=\"${network_interface}\" priority=\"default\" multicast=\"default\" /></Interfaces></General></Domain></CycloneDDS>"
    fi
fi

cd "${workspace_root}"
exec "${workspace_root}/.venv/bin/g1-calib" "$@"
