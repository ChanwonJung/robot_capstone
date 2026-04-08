#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOME_DIR="${HOME}"
DOWNLOADS_DIR="${ROBOT_CAPSTONE_DOWNLOADS_DIR:-${HOME_DIR}/Downloads}"
XR_CONTENT_ROOT="${ROBOT_CAPSTONE_XR_CONTENT_ROOT:-${DOWNLOADS_DIR}/XR_Content_NVD@10010}"
STAGES_DIR="${XR_CONTENT_ROOT}/Assets/XR/Stages"
SCENES_DIR="${SCRIPT_DIR}/scenes"

mkdir -p "${STAGES_DIR}"

if [[ -f "${SCENES_DIR}/robot_capstone.usd" ]]; then
  cp "${SCENES_DIR}/robot_capstone.usd" "${STAGES_DIR}/robot_capstone.usd"
fi

if [[ -f "${SCENES_DIR}/robot_capstone_scene.usd" ]]; then
  cp "${SCENES_DIR}/robot_capstone_scene.usd" "${STAGES_DIR}/robot_capstone_scene.usd"
fi

cd "${HOME_DIR}/isaacsim"
./isaac-sim.sh --enable omni.activity.ui --exec "${SCRIPT_DIR}/setup_initial_scene.py"
