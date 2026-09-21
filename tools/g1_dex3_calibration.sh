#!/usr/bin/env bash
set -Eeuo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
network_interface="${G1_CALIB_NETWORK_INTERFACE:-enp134s0}"
image_topic="${G1_CALIB_CAMERA_TOPIC:-/camera/color/image_raw}"
camera_info_topic="${G1_CALIB_CAMERA_INFO_TOPIC:-/camera/color/camera_info}"
camera_name="${G1_CALIB_CAMERA_NAME:-g1_head_color}"
camera_serial="${G1_CALIB_CAMERA_SERIAL:-348522074178}"
sessions_root="${G1_DEX3_CALIBRATION_SESSIONS_ROOT:-${workspace_root}/sessions}"
motion_ack="I CONFIRM THE G1 IS SECURED BY THE LOAD-BEARING HARNESS AND THE WORKSPACE IS CLEAR"

usage() {
    cat <<'EOF'
Usage: ./tools/g1_dex3_calibration.sh [right|left]

Run one complete Dex3 automatic calibration collection. The default arm is
right. The script
starts the tracked PC2 RealSense stream, chooses a unique session directory,
and launches the existing guarded collect-auto workflow. It does not bypass
the read-only IK/FCL preflight or the final SPACE confirmation.

The normal G1 setup needs no arguments. Advanced overrides are available via
G1_CALIB_* and G1_DEX3_CALIBRATION_* environment variables.
EOF
}

if [[ $# -eq 1 && ("$1" == "-h" || "$1" == "--help") ]]; then
    usage
    exit 0
fi
if (( $# > 1 )); then
    usage >&2
    exit 2
fi

arm="${1:-right}"
case "${arm}" in
right)
    default_plan="${workspace_root}/work/dex3_auto_collection_g1pilot_002/authored_plan.json"
    hardware_config="${workspace_root}/config/hardware_dex3_aruco.yaml"
    target_config="${workspace_root}/config/dex3_dorsal_aruco_target.json"
    ;;
left)
    default_plan="${workspace_root}/work/dex3_left_auto_collection_g1pilot_002/authored_plan.json"
    hardware_config="${workspace_root}/config/hardware_dex3_left_aruco_id5.yaml"
    target_config="${workspace_root}/config/dex3_left_dorsal_aruco_id5_target.json"
    ;;
*)
    echo "arm must be exactly right or left: ${arm}" >&2
    usage >&2
    exit 2
    ;;
esac
plan="${G1_DEX3_CALIBRATION_PLAN:-${default_plan}}"
collision_config="${workspace_root}/config/collision_pairs_dex3_aruco.yaml"

validation_report="$(dirname "${plan}")/validation_report.json"
for required_file in "${plan}" "${validation_report}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "required Dex3 calibration input is missing: ${required_file}" >&2
        exit 1
    fi
done

run_id="dex3_${arm}_auto_$(date -u +%Y%m%dT%H%M%SZ)"
session_directory="${sessions_root}/${run_id}"
suffix=1
while [[ -e "${session_directory}" ]]; do
    session_directory="${sessions_root}/${run_id}_${suffix}"
    ((suffix += 1))
done
session_id="$(basename "${session_directory}")"

on_error() {
    status=$?
    if [[ -e "${session_directory}" ]]; then
        echo "Dex3 calibration stopped; retained session: ${session_directory}" >&2
    fi
    exit "${status}"
}
trap on_error ERR

cd "${workspace_root}"
echo "Dex3 calibration session: ${session_directory}"
"${workspace_root}/tools/g1_realsense_pc2.sh" start

"${workspace_root}/tools/g1_calib_hardware.sh" collect-auto \
    --network-interface "${network_interface}" \
    --plan "${plan}" \
    --session-directory "${session_directory}" \
    --session-id "${session_id}" \
    --hardware-config "${hardware_config}" \
    --target-config "${target_config}" \
    --collision-config "${collision_config}" \
    --image-topic "${image_topic}" \
    --camera-info-topic "${camera_info_topic}" \
    --camera-name "${camera_name}" \
    --camera-serial "${camera_serial}" \
    --confirm "${motion_ack}"

trap - ERR
echo "Dex3 calibration complete: ${session_directory}"
