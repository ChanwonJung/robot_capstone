#!/usr/bin/env bash
# Hazard demo wrapper — chooses bottle behaviour by scenario arg so the rest
# of the launch pipeline (T2..T9) stays identical across the two demo modes.
#
# Usage:
#   ./run_hazard_demo.sh                # default: replan  (persistent hazard)
#   ./run_hazard_demo.sh replan         # Phase 2b: bottle parks, planner re-routes
#   ./run_hazard_demo.sh stop_resume    # Phase 2a: bottle flies through, arm holds + resumes
#
# Matching T2 commands:
#   ros2 launch moveit_isaac_bridge_pkg hybrid_planning.launch.py manager_logic:=replan
#   ros2 launch moveit_isaac_bridge_pkg hybrid_planning.launch.py manager_logic:=stop_resume
#
# Override on the command line (highest precedence — beats the defaults below):
#   ROBOT_CAPSTONE_HAZARD_PARK_X=0.30 ./run_hazard_demo.sh replan
set -eu

MODE="${1:-replan}"

# Strip the mode arg from "$@" so it isn't passed downstream to
# run_capstone_scene.sh as a stray positional.
if [ $# -gt 0 ]; then
    shift
fi

case "${MODE}" in
    replan)
        # ── Phase 2b: persistent hazard → global planner re-routes around it ──
        # Bottle parks above the book and stays put. Local planner detects the
        # invalidated trajectory, manager triggers a global re-plan, and OMPL
        # finds a path around the parked bottle. Tuned alongside the negative
        # xy_margin in hazard_collision_injector.launch.py and the gripper
        # link_padding shrink in hybrid_planning.launch.py — those three sets
        # of values together keep the arm from crashing the manager on the
        # mm-scale fingertip overlap that would otherwise trip
        # CheckStartStateCollision at every replan attempt.
        : "${ROBOT_CAPSTONE_HAZARD_MODE:=park}"
        : "${ROBOT_CAPSTONE_BOTTLE_VX:=-2.0}"
        : "${ROBOT_CAPSTONE_HAZARD_PARK_X:=0.05}"
        : "${ROBOT_CAPSTONE_BOTTLE_SPAWN_Y:=0.01}"
        : "${ROBOT_CAPSTONE_BOTTLE_SPAWN_Z:=0.35}"
        ;;
    stop_resume)
        # ── Phase 2a: transient hazard → arm halts and resumes once it clears ─
        # Cardbox flies through the workspace at a steady speed; ForwardTrajectory
        # halts the arm when the box enters its near-future path, and resumes
        # forward execution once the box leaves the scene (clear_timeout_sec in
        # the injector). No re-plan happens — the global trajectory is the same
        # before and after the box passes.
        #
        # Object switched from bottle → box for stop_resume to keep visual
        # parity with the Phase 2a regression runs (the cardbox USD is what
        # those were captured with). Launch trigger uses BOX_VX since
        # _apply_bottle_launch_velocity picks per-object velocity based on
        # HAZARD_OBJECT.
        : "${ROBOT_CAPSTONE_HAZARD_OBJECT:=box}"
        : "${ROBOT_CAPSTONE_HAZARD_MODE:=flythrough}"
        # Fast -X crossing — short physical contact with the arm body keeps
        # the Isaac collision impulse tiny, and the planning-scene
        # collision_object lifetime (~box transit + clear_timeout_sec)
        # stays inside the local planner's ~500 ms abort budget. At
        # -2.2 m/s a 0.5 m workspace takes ~230 ms; YOLO at 30 Hz still
        # gets 7 detection frames, plenty for the injector to register.
        : "${ROBOT_CAPSTONE_BOX_VX:=-2.2}"
        : "${ROBOT_CAPSTONE_BOX_VY:=0.0}"
        # PARK_X is unused in flythrough but exporting a value keeps the env
        # snapshot tidy and lets users flip to park mid-session without
        # editing the wrapper.
        : "${ROBOT_CAPSTONE_HAZARD_PARK_X:=0.05}"
        # Spawn back at the off-frame edge (x=0.85) so the box is OUTSIDE
        # the top camera's FOV at scene start; otherwise YOLO sees it
        # immediately, the injector floods the planning scene with a
        # collision_object before the BT can even send its first goal,
        # and the hybrid manager bounces the goal as SERVER_UNREACHABLE.
        : "${ROBOT_CAPSTONE_BOTTLE_SPAWN_X:=0.85}"
        : "${ROBOT_CAPSTONE_BOTTLE_SPAWN_Y:=-0.08}"
        # Lowered from 0.35 to 0.25 — keeps the box closer to the table
        # so its top edge (centre + half-height) sits just below
        # panda_link2..3's home-pose z-range (~0.30 m and up). Less arm-
        # body intersection during the flythrough while the planning
        # scene still catches the trajectory (target pre_grasp at z=0.21,
        # well inside the inflated collision_object).
        : "${ROBOT_CAPSTONE_BOTTLE_SPAWN_Z:=0.25}"
        # Halfway between the 0.003 default and the 0.002 aggressively-small
        # variant — gives ~12.5 cm box, big enough that YOLO catches it
        # cleanly during the fast -2.2 m/s flythrough but still smaller
        # than the original 0.15 m so arm-body intersection stays modest.
        : "${ROBOT_CAPSTONE_BOX_SCALE:=0.0025}"
        ;;
    *)
        echo "[run_hazard_demo] unknown mode: '${MODE}' (expected 'replan' or 'stop_resume')" >&2
        exit 2
        ;;
esac

# ── Shared scenario knobs ─────────────────────────────────────────────────
# Default to bottle for any unset path (replan branch above relies on bottle).
export ROBOT_CAPSTONE_HAZARD_OBJECT="${ROBOT_CAPSTONE_HAZARD_OBJECT:-bottle}"
export ROBOT_CAPSTONE_HAZARD_MODE
# Per-object launch velocities — _apply_bottle_launch_velocity in
# setup_initial_scene.py picks whichever matches HAZARD_OBJECT, so we export
# both with sane defaults; the unused one is harmless. The box has Y/Z too
# (for the stop+resume lateral cross); bottle uses VX only.
export ROBOT_CAPSTONE_BOTTLE_VX="${ROBOT_CAPSTONE_BOTTLE_VX:--0.3}"
export ROBOT_CAPSTONE_BOX_VX="${ROBOT_CAPSTONE_BOX_VX:--0.4}"
export ROBOT_CAPSTONE_BOX_VY="${ROBOT_CAPSTONE_BOX_VY:-0.0}"
export ROBOT_CAPSTONE_BOX_VZ="${ROBOT_CAPSTONE_BOX_VZ:-0.0}"
export ROBOT_CAPSTONE_HAZARD_PARK_X
# Spawn position env vars — X defaults to 0.85 (off-frame for replan).
export ROBOT_CAPSTONE_BOTTLE_SPAWN_X="${ROBOT_CAPSTONE_BOTTLE_SPAWN_X:-0.85}"
export ROBOT_CAPSTONE_BOTTLE_SPAWN_Y
export ROBOT_CAPSTONE_BOTTLE_SPAWN_Z
# Box visual scale — default matches setup_initial_scene.py's 0.003 baseline.
export ROBOT_CAPSTONE_BOX_SCALE="${ROBOT_CAPSTONE_BOX_SCALE:-0.003}"

# ── Auto-launch — fires when the arm actually starts moving ───────────────
# Threshold 0.02 rad is loose enough for the slow goal-directed motion this
# pipeline uses (0.01 m/s in the replan demo) and tight enough that hybrid
# startup jitter doesn't trip it after the AUTO_ARM_SEC warm-up.
export ROBOT_CAPSTONE_HAZARD_AUTO_LAUNCH="${ROBOT_CAPSTONE_HAZARD_AUTO_LAUNCH:-1}"
export ROBOT_CAPSTONE_AUTO_TRIGGER_RAD="${ROBOT_CAPSTONE_AUTO_TRIGGER_RAD:-0.02}"
# ROBOT_CAPSTONE_AUTO_ARM_SEC defaults to 5 s — raise if hybrid takes longer
# to settle on slow machines.

# ── Launch the scene ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[run_hazard_demo] mode=${MODE}"
echo "[run_hazard_demo] HAZARD env:"
env | grep -E "^ROBOT_CAPSTONE_(HAZARD|BOTTLE|AUTO)" | sort | sed 's/^/  /'
exec "${SCRIPT_DIR}/run_capstone_scene.sh" "$@"
