#!/usr/bin/env bash
set -euo pipefail

cd /home/chanwonjung/robot_capstone/isaacsim
./isaac-sim.sh --enable omni.activity.ui --exec "/home/chanwonjung/robot_capstone/setup_initial_scene.py"
