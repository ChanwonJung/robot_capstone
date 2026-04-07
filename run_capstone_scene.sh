#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/isaacsim"
./isaac-sim.sh --enable omni.activity.ui --exec "${SCRIPT_DIR}/setup_initial_scene.py"
