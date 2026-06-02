#!/usr/bin/env bash
# Hazard replan demo wrapper.
#
# Parks a bottle directly above the book so MoveIt's global planner has to
# re-plan around it (Phase 2b ReplanInvalidatedTrajectory verification).
# Wraps run_capstone_scene.sh — every hazard env var is pre-exported here so
# the manual command line stays short and typo-resistant. To tweak a value,
# edit it once in this file and re-run.
#
# Usage:
#   ./run_hazard_demo.sh
#
# Override on the command line (highest precedence — overrides exports here):
#   ROBOT_CAPSTONE_HAZARD_PARK_X=0.50 ./run_hazard_demo.sh
set -eu

# ── Hazard scenario ────────────────────────────────────────────────────────
export ROBOT_CAPSTONE_HAZARD_OBJECT="${ROBOT_CAPSTONE_HAZARD_OBJECT:-bottle}"
export ROBOT_CAPSTONE_HAZARD_MODE="${ROBOT_CAPSTONE_HAZARD_MODE:-park}"
export ROBOT_CAPSTONE_BOTTLE_VX="${ROBOT_CAPSTONE_BOTTLE_VX:--0.5}"

# ── Park position — directly above the book ───────────────────────────────
# Book centroid (from /world_map_result diagnostic): (0.584, -0.162, 0.093).
# Bottle spawns at x=0.85 with gravity disabled, flies at -x, halts when its
# x reaches PARK_X. Aligning PARK_X + SPAWN_Y to the book centroid puts the
# parked bottle exactly above it.  SPAWN_Z=0.35 leaves the bottle bottom
# ≈0.24 m above the table, just inside the arm's descent path to pre-grasp
# (pre_grasp z = book z + 0.12 = 0.21 m).
export ROBOT_CAPSTONE_HAZARD_PARK_X="${ROBOT_CAPSTONE_HAZARD_PARK_X:-0.58}"
export ROBOT_CAPSTONE_BOTTLE_SPAWN_Y="${ROBOT_CAPSTONE_BOTTLE_SPAWN_Y:--0.16}"
export ROBOT_CAPSTONE_BOTTLE_SPAWN_Z="${ROBOT_CAPSTONE_BOTTLE_SPAWN_Z:-0.35}"

# ── Auto-launch — fires when the arm actually starts moving ───────────────
# Threshold 0.05 rad is loose enough for goal-directed motion, tight enough
# that hybrid startup jitter doesn't trip it after the AUTO_ARM_SEC warm-up.
export ROBOT_CAPSTONE_HAZARD_AUTO_LAUNCH="${ROBOT_CAPSTONE_HAZARD_AUTO_LAUNCH:-1}"
export ROBOT_CAPSTONE_AUTO_TRIGGER_RAD="${ROBOT_CAPSTONE_AUTO_TRIGGER_RAD:-0.05}"
# ROBOT_CAPSTONE_AUTO_ARM_SEC defaults to 5 s — raise if hybrid takes longer
# to settle on slow machines.

# ── Launch the scene ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[run_hazard_demo] HAZARD env:"
env | grep -E "^ROBOT_CAPSTONE_(HAZARD|BOTTLE|AUTO)" | sort | sed 's/^/  /'
exec "${SCRIPT_DIR}/run_capstone_scene.sh" "$@"
