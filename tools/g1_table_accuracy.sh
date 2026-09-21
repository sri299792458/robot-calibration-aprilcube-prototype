#!/usr/bin/env bash
set -Eeuo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
network_interface="${G1_CALIB_NETWORK_INTERFACE:-enp134s0}"
image_topic="${G1_CALIB_CAMERA_TOPIC:-/camera/color/image_raw}"
camera_info_topic="${G1_CALIB_CAMERA_INFO_TOPIC:-/camera/color/camera_info}"
camera_name="${G1_CALIB_CAMERA_NAME:-g1_head_color}"
camera_serial="${G1_CALIB_CAMERA_SERIAL:-348522074178}"
runs_root="${G1_TABLE_ACCURACY_RUNS_ROOT:-${workspace_root}/work/table_accuracy_runs}"
dataset="${G1_TABLE_ACCURACY_DATASET:-${workspace_root}/work/auto_run_004_latest/dataset.json}"
result_json="${G1_TABLE_ACCURACY_RESULT:-${workspace_root}/work/auto_run_004_latest/solve/result.json}"
hand_cube_config="${G1_TABLE_ACCURACY_HAND_CUBE_CONFIG:-${workspace_root}/aprilcube/models/dex3_safe_cube/config.json}"
motion_ack="I CONFIRM THE G1 IS SECURED BY THE LOAD-BEARING HARNESS AND THE WORKSPACE IS CLEAR"

usage() {
    cat <<'EOF'
Usage: ./tools/g1_table_accuracy.sh

Run one complete tabletop-accuracy trial: ensure the tracked PC2 RealSense
stream is running, capture a fresh supported-start intent burst, and launch the
guarded seated execution. After SPACE, the executor acquires lowcmd, settles,
captures a new loaded-state burst, and only then builds the IK/FCL route. A
unique artifact directory is chosen automatically. No changing arm target is
issued until that post-takeover route passes every validation.

The normal G1 setup needs no arguments. Advanced overrides are available via
G1_CALIB_* and G1_TABLE_ACCURACY_* environment variables.
EOF
}

if (($#)); then
    if [[ $# -eq 1 && ("$1" == "-h" || "$1" == "--help") ]]; then
        usage
        exit 0
    fi
    usage >&2
    exit 2
fi

for required_file in "${dataset}" "${result_json}" "${hand_cube_config}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "required tabletop input is missing: ${required_file}" >&2
        exit 1
    fi
done

run_id="$(date -u +%Y%m%dT%H%M%SZ)"
run_directory="${runs_root}/${run_id}"
suffix=1
while [[ -e "${run_directory}" ]]; do
    run_directory="${runs_root}/${run_id}_${suffix}"
    ((suffix += 1))
done
planning_directory="${run_directory}/planning"
plan_path="${run_directory}/plan.json"
execution_directory="${run_directory}/execution"

on_error() {
    status=$?
    if [[ -e "${run_directory}" ]]; then
        echo "tabletop trial stopped; retained artifacts: ${run_directory}" >&2
    fi
    exit "${status}"
}
trap on_error ERR

cd "${workspace_root}"
echo "tabletop trial artifacts: ${run_directory}"
"${workspace_root}/tools/g1_realsense_pc2.sh" start

"${workspace_root}/tools/g1_calib_hardware.sh" capture-table-images \
    --network-interface "${network_interface}" \
    --image-topic "${image_topic}" \
    --camera-info-topic "${camera_info_topic}" \
    --camera-name "${camera_name}" \
    --camera-serial "${camera_serial}" \
    --output-directory "${planning_directory}"

"${workspace_root}/.venv/bin/g1-calib" plan-table-accuracy \
    --dataset "${dataset}" \
    --result-json "${result_json}" \
    --image-directory "${planning_directory}" \
    --hand-cube-config "${hand_cube_config}" \
    --lift-mm 100 \
    --board-x-mm 0 \
    --board-y-mm 150 \
    --output "${plan_path}"

"${workspace_root}/tools/g1_calib_hardware.sh" execute-table-accuracy \
    --network-interface "${network_interface}" \
    --image-topic "${image_topic}" \
    --camera-info-topic "${camera_info_topic}" \
    --camera-name "${camera_name}" \
    --camera-serial "${camera_serial}" \
    --plan "${plan_path}" \
    --output-directory "${execution_directory}" \
    --confirm "${motion_ack}"

trap - ERR
echo "tabletop trial complete: ${execution_directory}"
