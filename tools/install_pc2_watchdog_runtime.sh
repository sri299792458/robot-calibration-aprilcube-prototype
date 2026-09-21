#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pc2_host="${G1_PC2_HOST:-unitree@192.168.123.164}"
identity="${G1_PC2_SSH_IDENTITY:-${HOME}/.ssh/g1_pc2_ed25519}"
runtime_base="/home/unitree/.local/share/g1-aprilcube-watchdog"
sdk_revision="7c661d27f4ae064ffd0dd633fd9d5b518ef0b508"
cyclonedds_version="0.10.2"
cyclonedds_sha256="f834962eabbdcdf4e9cd75cf87222f3c5ef22d9cb9e7ed651d9a8710fe984a30"
dependency_specs=(
    "typing_extensions==4.12.2"
    "rich-click==1.8.9"
    "click==8.1.8"
    "rich==13.9.4"
    "markdown-it-py==3.0.0"
    "mdurl==0.1.2"
    "Pygments==2.19.1"
)
declare -A dependency_sha256=(
    ["click-8.1.8-py3-none-any.whl"]="63c132bbbed01578a06712a2d1f497bb62d9c1c0d329b7903a866228027263b2"
    ["markdown_it_py-3.0.0-py3-none-any.whl"]="355216845c60bd96232cd8d8c40e8f9765cc86f46880e43a8fd22dc1a1a8cab1"
    ["mdurl-0.1.2-py3-none-any.whl"]="84008a41e51615a49fc9966191ff91509e3c40b939176e643fd50a5c2196b8f8"
    ["pygments-2.19.1-py3-none-any.whl"]="9ea1544ad55cecf4b8242fab6dd35a93bbce657034b0611ee383099054ab6d8c"
    ["rich-13.9.4-py3-none-any.whl"]="6049d5e6ec054bf2779ab3358186963bac2ea89175919d699e378b99738c2a90"
    ["rich_click-1.8.9-py3-none-any.whl"]="c3fa81ed8a671a10de65a9e20abf642cfdac6fdb882db1ef465ee33919fbcfe2"
    ["typing_extensions-4.12.2-py3-none-any.whl"]="04e5ca0351e0f3f85c6853954072df659d0d13fac324d0072316b67d7794700d"
)
sdk_source="${workspace_root}/external/unitree_sdk2_python"
cyclonedds_home="${workspace_root}/deps/cyclonedds_python_prefix"

if [[ ! -f "${identity}" ]]; then
    echo "PC2 SSH identity does not exist: ${identity}" >&2
    exit 1
fi
if [[ ! "${pc2_host}" =~ ^[A-Za-z0-9_.-]+@[A-Za-z0-9_.:-]+$ ]]; then
    echo "invalid PC2 SSH destination: ${pc2_host}" >&2
    exit 1
fi
if [[ ! -d "${sdk_source}/.git" ]]; then
    echo "pinned Unitree SDK checkout is missing: ${sdk_source}" >&2
    exit 1
fi
actual_sdk_revision="$(git -C "${sdk_source}" rev-parse HEAD)"
if [[ "${actual_sdk_revision}" != "${sdk_revision}" ]]; then
    echo "Unitree SDK revision mismatch: ${actual_sdk_revision}" >&2
    exit 1
fi
if [[ -n "$(git -C "${sdk_source}" status --porcelain)" ]]; then
    echo "Unitree SDK checkout has local changes; refusing deployment" >&2
    exit 1
fi
if [[ ! -d "${cyclonedds_home}/lib" ]]; then
    echo "local CycloneDDS build prefix is missing: ${cyclonedds_home}" >&2
    exit 1
fi

ssh_options=(
    -i "${identity}"
    -o BatchMode=yes
    -o ConnectTimeout=5
    -o ServerAliveInterval=1
    -o ServerAliveCountMax=2
)

local_stage="$(mktemp -d)"
remote_stage=""
cleanup() {
    if [[ -n "${remote_stage}" && \
          "${remote_stage}" == /tmp/g1-aprilcube-watchdog-install.* ]]; then
        ssh -T "${ssh_options[@]}" "${pc2_host}" \
            "rm -rf -- '${remote_stage}'" >/dev/null 2>&1 || true
    fi
    rm -rf -- "${local_stage}"
}
trap cleanup EXIT

export CYCLONEDDS_HOME="${cyclonedds_home}"
export PIP_NO_CACHE_DIR=1
python3 -m pip download \
    --no-deps \
    --no-binary=:all: \
    --no-build-isolation \
    --dest "${local_stage}" \
    "cyclonedds==${cyclonedds_version}" >/dev/null
cyclonedds_sdist="${local_stage}/cyclonedds-${cyclonedds_version}.tar.gz"
if [[ ! -f "${cyclonedds_sdist}" ]]; then
    echo "CycloneDDS source archive was not downloaded" >&2
    exit 1
fi
actual_cyclonedds_sha256="$(sha256sum "${cyclonedds_sdist}" | awk '{print $1}')"
if [[ "${actual_cyclonedds_sha256}" != "${cyclonedds_sha256}" ]]; then
    echo "CycloneDDS source checksum mismatch" >&2
    exit 1
fi

python3 -m pip download \
    --only-binary=:all: \
    --no-deps \
    --dest "${local_stage}" \
    "${dependency_specs[@]}" >/dev/null
dependency_checksums="${local_stage}/dependency-checksums.sha256"
for dependency_file in "${!dependency_sha256[@]}"; do
    dependency_path="${local_stage}/${dependency_file}"
    if [[ ! -f "${dependency_path}" ]]; then
        echo "required offline dependency was not downloaded: ${dependency_file}" >&2
        exit 1
    fi
    actual_dependency_sha256="$(sha256sum "${dependency_path}" | awk '{print $1}')"
    if [[ "${actual_dependency_sha256}" != "${dependency_sha256[${dependency_file}]}" ]]; then
        echo "offline dependency checksum mismatch: ${dependency_file}" >&2
        exit 1
    fi
    printf '%s  %s\n' \
        "${dependency_sha256[${dependency_file}]}" \
        "${dependency_file}"
done | sort -k2 > "${dependency_checksums}"
dependency_set_sha256="$(sha256sum "${dependency_checksums}" | awk '{print $1}')"

sdk_archive="${local_stage}/unitree_sdk2_python-${sdk_revision}.tar.gz"
git -C "${sdk_source}" archive --format=tar "${sdk_revision}" | gzip -n > "${sdk_archive}"
sdk_sha256="$(sha256sum "${sdk_archive}" | awk '{print $1}')"
runtime_id="$({
    printf 'schema=3\n'
    printf 'cyclonedds=%s:%s\n' "${cyclonedds_version}" "${cyclonedds_sha256}"
    printf 'unitree_sdk=%s:%s\n' "${sdk_revision}" "${sdk_sha256}"
    printf 'dependency_set=%s\n' "${dependency_set_sha256}"
} | sha256sum | cut -c1-16)"

remote_stage="$(ssh -T "${ssh_options[@]}" "${pc2_host}" \
    'mktemp -d /tmp/g1-aprilcube-watchdog-install.XXXXXXXX')"
if [[ "${remote_stage}" != /tmp/g1-aprilcube-watchdog-install.* ]]; then
    echo "PC2 returned an invalid staging directory: ${remote_stage}" >&2
    exit 1
fi

scp "${ssh_options[@]}" \
    "${cyclonedds_sdist}" \
    "${sdk_archive}" \
    "${dependency_checksums}" \
    "${local_stage}/"*.whl \
    "${pc2_host}:${remote_stage}/"

ssh -T "${ssh_options[@]}" "${pc2_host}" bash -s -- \
    "${remote_stage}" \
    "${runtime_base}" \
    "${runtime_id}" \
    "${cyclonedds_version}" \
    "${cyclonedds_sha256}" \
    "${sdk_revision}" \
    "${sdk_sha256}" \
    "${dependency_set_sha256}" <<'REMOTE'
set -euo pipefail

remote_stage="$1"
runtime_base="$2"
runtime_id="$3"
cyclonedds_version="$4"
cyclonedds_sha256="$5"
sdk_revision="$6"
sdk_sha256="$7"
dependency_set_sha256="$8"
cyclonedds_home="/home/unitree/cyclonedds_ws/install/cyclonedds"

if [[ "${remote_stage}" != /tmp/g1-aprilcube-watchdog-install.* ]]; then
    echo "invalid remote staging directory" >&2
    exit 1
fi
if [[ "${runtime_base}" != "/home/unitree/.local/share/g1-aprilcube-watchdog" ]]; then
    echo "invalid runtime base" >&2
    exit 1
fi
if [[ ! "${runtime_id}" =~ ^[0-9a-f]{16}$ ]]; then
    echo "invalid runtime ID" >&2
    exit 1
fi
if [[ ! -f "${cyclonedds_home}/lib/libddsc.so" ]]; then
    echo "PC2 CycloneDDS library is missing" >&2
    exit 1
fi
cyclonedds_library="$(basename "$(readlink -f "${cyclonedds_home}/lib/libddsc.so")")"
if [[ "${cyclonedds_library}" != "libddsc.so.0.10.2" ]]; then
    echo "unexpected PC2 CycloneDDS library: ${cyclonedds_library}" >&2
    exit 1
fi
if [[ ! -f "/usr/include/python3.8/Python.h" ]]; then
    echo "PC2 Python 3.8 development headers are missing" >&2
    exit 1
fi

cyclonedds_sdist="${remote_stage}/cyclonedds-${cyclonedds_version}.tar.gz"
sdk_archive="${remote_stage}/unitree_sdk2_python-${sdk_revision}.tar.gz"
printf '%s  %s\n' "${cyclonedds_sha256}" "${cyclonedds_sdist}" | sha256sum --check --status
printf '%s  %s\n' "${sdk_sha256}" "${sdk_archive}" | sha256sum --check --status
actual_dependency_set_sha256="$(sha256sum "${remote_stage}/dependency-checksums.sha256" | awk '{print $1}')"
if [[ "${actual_dependency_set_sha256}" != "${dependency_set_sha256}" ]]; then
    echo "offline dependency-set checksum mismatch" >&2
    exit 1
fi
(cd "${remote_stage}" && sha256sum --check --status dependency-checksums.sha256)

mkdir -p "${runtime_base}/runtimes"
target="${runtime_base}/runtimes/${runtime_id}"
current="${runtime_base}/current"
if [[ -e "${current}" && ! -L "${current}" ]]; then
    echo "refusing to replace non-symlink runtime selector: ${current}" >&2
    exit 1
fi

activate_runtime() {
    link_tmp="${runtime_base}/.current-${runtime_id}-$$"
    ln -s "runtimes/${runtime_id}" "${link_tmp}"
    mv -Tf "${link_tmp}" "${current}"
}

if [[ -d "${target}" ]]; then
    if [[ ! -x "${target}/venv/bin/python" || ! -f "${target}/manifest.json" ]]; then
        echo "existing runtime is incomplete: ${target}" >&2
        exit 1
    fi
    activate_runtime
    echo "PC2 watchdog runtime already installed: ${target}"
    exit 0
fi

staging="${runtime_base}/runtimes/.${runtime_id}.staging-$$"
cleanup_staging() {
    if [[ "${staging}" == "${runtime_base}/runtimes/.${runtime_id}.staging-"* ]]; then
        rm -rf -- "${staging}"
    fi
}
trap cleanup_staging EXIT
mkdir "${staging}"

mkdir "${staging}/wheels" "${staging}/sdk-source"
tar -xzf "${sdk_archive}" -C "${staging}/sdk-source"

export CYCLONEDDS_HOME="${cyclonedds_home}"
export CMAKE_PREFIX_PATH="${cyclonedds_home}:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${cyclonedds_home}/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"
export PIP_NO_CACHE_DIR=1
python3 -m pip wheel \
    --no-index --no-deps --no-build-isolation \
    --wheel-dir "${staging}/wheels" \
    "${cyclonedds_sdist}"
python3 -m pip wheel \
    --no-index --no-deps --no-build-isolation \
    --wheel-dir "${staging}/wheels" \
    "${staging}/sdk-source"

python3 -m venv "${staging}/venv"
"${staging}/venv/bin/python" -m pip install \
    --no-index --no-deps \
    "${remote_stage}/"*.whl \
    "${staging}/wheels/"*.whl

PYTHONNOUSERSITE=1 "${staging}/venv/bin/python" - <<PY
from importlib.metadata import version
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

assert version("cyclonedds") == "${cyclonedds_version}"
assert version("unitree-sdk2py") == "1.0.1"
assert callable(MotionSwitcherClient.CheckMode)
assert callable(MotionSwitcherClient.SelectMode)
print("validated private raw Unitree MotionSwitcher imports")
PY

PYTHONNOUSERSITE=1 "${staging}/venv/bin/python" - \
    "${runtime_id}" \
    "${cyclonedds_version}" \
    "${cyclonedds_sha256}" \
    "${sdk_revision}" \
    "${sdk_sha256}" \
    "${dependency_set_sha256}" \
    "${cyclonedds_home}" \
    "${cyclonedds_library}" \
    "${target}" \
    "${staging}/manifest.json" <<'PY'
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    runtime_id,
    cyclonedds_version,
    cyclonedds_sha256,
    sdk_revision,
    sdk_sha256,
    dependency_set_sha256,
    cyclonedds_home,
    cyclonedds_library,
    target,
    manifest_path,
) = sys.argv[1:]
manifest = {
    "schema_version": 3,
    "runtime_id": runtime_id,
    "installed_at": datetime.now(timezone.utc).isoformat(),
    "install_scope": "unitree-user-private-versioned-venv",
    "python": platform.python_version(),
    "cyclonedds_python": {
        "version": cyclonedds_version,
        "source_sha256": cyclonedds_sha256,
        "linked_home": cyclonedds_home,
        "linked_library": cyclonedds_library,
    },
    "unitree_sdk2_python": {
        "package_version": "1.0.1",
        "revision": sdk_revision,
        "source_sha256": sdk_sha256,
    },
    "offline_dependency_set_sha256": dependency_set_sha256,
    "installed_distributions": {
        name: __import__("importlib.metadata", fromlist=["version"]).version(name)
        for name in (
            "click",
            "markdown-it-py",
            "mdurl",
            "Pygments",
            "rich",
            "rich-click",
            "typing_extensions",
        )
    },
    "runtime_path": target,
    "runtime_installer_modified_global_environment": False,
    "system_prerequisite": "python3.8-venv installed separately with apt",
    "boot_service_installed": False,
    "validation": "venv-imports-only-no-dds-participant-no-robot-command",
}
Path(manifest_path).write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

rm -rf -- "${staging}/wheels" "${staging}/sdk-source"
mv "${staging}" "${target}"
trap - EXIT
activate_runtime

echo "installed PC2 watchdog runtime: ${target}"
echo "active PC2 watchdog runtime: ${current} -> runtimes/${runtime_id}"
REMOTE

echo "PC2 watchdog runtime installation complete: ${runtime_id}"
