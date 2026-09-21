#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

clone_pin() {
    local destination="$1"
    local url="$2"
    local revision="$3"

    if [[ ! -d "${destination}/.git" ]]; then
        if [[ -e "${destination}" ]]; then
            echo "refusing to replace non-Git path: ${destination}" >&2
            return 1
        fi
        git clone "${url}" "${destination}"
    fi

    git -C "${destination}" fetch --quiet origin "${revision}"
    git -C "${destination}" checkout --quiet --detach "${revision}"
    local actual
    actual="$(git -C "${destination}" rev-parse HEAD)"
    if [[ "${actual}" != "${revision}" ]]; then
        echo "pin verification failed for ${destination}" >&2
        return 1
    fi
    echo "ready: ${destination} @ ${actual}"
}

mkdir -p "${workspace_root}/external"
clone_pin \
    "${workspace_root}/xr_teleoperate" \
    "https://github.com/unitreerobotics/xr_teleoperate.git" \
    "845b25a32f7febedf220e830952a7134897adb9d"
git -C "${workspace_root}/xr_teleoperate" submodule update --init --depth 1
clone_pin \
    "${workspace_root}/external/unitree_sdk2_python_official" \
    "https://github.com/unitreerobotics/unitree_sdk2_python.git" \
    "65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5"
clone_pin \
    "${workspace_root}/external/unitree_sdk2_python" \
    "https://github.com/lnotspotl/unitree_sdk2_python.git" \
    "7c661d27f4ae064ffd0dd633fd9d5b518ef0b508"
