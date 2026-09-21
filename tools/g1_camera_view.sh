#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
network_interface="${G1_CALIB_NETWORK_INTERFACE:-enp134s0}"
domain_id="${G1_CALIB_ROS_DOMAIN_ID:-0}"
image_topic="${G1_CALIB_CAMERA_TOPIC:-/camera/color/image_raw}"
restart_camera=1

usage() {
    cat <<'EOF'
Usage: ./tools/g1_camera_view.sh [options]

Open the PC2 RealSense color stream in the standard ROS rqt image viewer.

Options:
  --network-interface NAME  Laptop robot-network interface (default: enp134s0)
  --domain-id ID            ROS domain ID (default: 0)
  --topic NAME              ROS image topic (default: /camera/color/image_raw)
  --no-restart              Keep an already-running tracked RealSense process
  -h, --help                Show this help

Environment overrides:
  G1_CALIB_NETWORK_INTERFACE
  G1_CALIB_ROS_DOMAIN_ID
  G1_CALIB_CAMERA_TOPIC
  G1_CALIB_ROS_PREFIX

By default the command cleanly restarts only the temporary RealSense process
tracked by tools/g1_realsense_pc2.sh. Closing the viewer leaves that stream
running for subsequent calibration commands.
EOF
}

while (($#)); do
    case "$1" in
        --network-interface)
            [[ $# -ge 2 ]] || {
                echo "--network-interface requires a value" >&2
                exit 2
            }
            network_interface="$2"
            shift 2
            ;;
        --domain-id)
            [[ $# -ge 2 ]] || {
                echo "--domain-id requires a value" >&2
                exit 2
            }
            domain_id="$2"
            shift 2
            ;;
        --topic)
            [[ $# -ge 2 ]] || {
                echo "--topic requires a value" >&2
                exit 2
            }
            image_topic="$2"
            shift 2
            ;;
        --no-restart)
            restart_camera=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! "${network_interface}" =~ ^[[:alnum:]_.:-]+$ ]] || \
    [[ ! -e "/sys/class/net/${network_interface}" ]]; then
    echo "invalid local network interface: ${network_interface}" >&2
    exit 1
fi
if [[ ! "${domain_id}" =~ ^[0-9]+$ ]]; then
    echo "invalid ROS domain ID: ${domain_id}" >&2
    exit 1
fi
if [[ "${image_topic}" != /* ]] || [[ "${image_topic}" =~ [[:space:]] ]]; then
    echo "invalid absolute ROS image topic: ${image_topic}" >&2
    exit 1
fi
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    echo "no graphical display is available for rqt_image_view" >&2
    exit 1
fi

if [[ -n "${G1_CALIB_ROS_PREFIX:-}" ]]; then
    ros_prefix="${G1_CALIB_ROS_PREFIX}"
elif [[ -f /opt/ros/humble/setup.bash ]]; then
    ros_prefix="/opt/ros/humble"
elif [[ -f /opt/ros/jazzy/setup.bash ]]; then
    ros_prefix="/opt/ros/jazzy"
else
    echo "ROS 2 environment not found under /opt/ros" >&2
    exit 1
fi
if [[ ! -f "${ros_prefix}/setup.bash" ]]; then
    echo "ROS 2 setup file not found: ${ros_prefix}/setup.bash" >&2
    exit 1
fi

# ROS setup scripts are not guaranteed to be nounset-clean.
set +u
source "${ros_prefix}/setup.bash"
set -u

if ! ros2 pkg prefix rqt_image_view >/dev/null 2>&1; then
    echo "ROS package rqt_image_view is not installed" >&2
    exit 1
fi

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_DOMAIN_ID="${domain_id}"
export ROS_LOCALHOST_ONLY=0
if [[ -z "${CYCLONEDDS_URI:-}" ]]; then
    export CYCLONEDDS_URI="<CycloneDDS><Domain Id=\"any\"><General><Interfaces><NetworkInterface name=\"${network_interface}\" priority=\"default\" multicast=\"default\" /></Interfaces></General></Domain></CycloneDDS>"
fi

if ((restart_camera)); then
    "${workspace_root}/tools/g1_realsense_pc2.sh" stop
fi
"${workspace_root}/tools/g1_realsense_pc2.sh" start

echo "opening ${image_topic} on ${network_interface} (ROS domain ${domain_id})"
exec ros2 run rqt_image_view rqt_image_view "${image_topic}"
