#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/isaacsim"
./isaac-sim.sh --enable omni.activity.ui --exec "$HOME/robot_capstone/setup_initial_scene.py"
