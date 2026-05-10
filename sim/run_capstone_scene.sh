#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOWNLOADS_DIR="${ROBOT_CAPSTONE_DOWNLOADS_DIR:-${HOME}/Downloads}"
XR_CONTENT_ROOT="${ROBOT_CAPSTONE_XR_CONTENT_ROOT:-${DOWNLOADS_DIR}/XR_Content_NVD@10010}"
STAGES_DIR="${XR_CONTENT_ROOT}/Assets/XR/Stages"
SCENES_DIR="${SCRIPT_DIR}/scenes"
ISAACSIM_DIR="${PROJECT_ROOT}/isaacsim"

if [[ ! -x "${ISAACSIM_DIR}/isaac-sim.sh" ]]; then
  echo "Isaac Sim launcher not found: ${ISAACSIM_DIR}/isaac-sim.sh" >&2
  exit 1
fi

mkdir -p "${STAGES_DIR}"

if [[ -f "${SCENES_DIR}/robot_capstone.usd" ]]; then
  cp "${SCENES_DIR}/robot_capstone.usd" "${STAGES_DIR}/robot_capstone.usd"
fi

if [[ -f "${SCENES_DIR}/robot_capstone_scene.usd" ]]; then
  cp "${SCENES_DIR}/robot_capstone_scene.usd" "${STAGES_DIR}/robot_capstone_scene.usd"
fi

# Isaac Sim 5.1 ships ROS 2 Jazzy Python 3.11 bindings internally.
# If the shell already sourced /opt/ros/jazzy (Python 3.12 on Ubuntu 24.04),
# the bridge tries to load incompatible rclpy binaries and startup breaks.
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset PYTHONPATH
unset ROS_DISTRO
unset ROS_PYTHON_VERSION
unset ROS_VERSION
unset RMW_IMPLEMENTATION

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  CLEAN_LD_LIBRARY_PATH="$(
    printf '%s' "${LD_LIBRARY_PATH}" | awk -v RS=: -v ORS=: '$0 !~ "^/opt/ros/" {print}'
  )"
  CLEAN_LD_LIBRARY_PATH="${CLEAN_LD_LIBRARY_PATH%:}"
  export LD_LIBRARY_PATH="${CLEAN_LD_LIBRARY_PATH}"
fi

cd "${ISAACSIM_DIR}"
./isaac-sim.sh --enable omni.activity.ui --exec "${SCRIPT_DIR}/setup_initial_scene.py"
