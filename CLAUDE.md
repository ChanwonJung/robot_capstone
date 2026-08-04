# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Specifications

Ubuntu 24.04 · ROS Jazzy · IsaacSim 5.1.0

## End Goal
Language-directed robotic manipulator that interprets ambiguous natural language commands, visually grounds targets in cluttered scenes, and executes safe grasps while avoiding dynamic obstacles. Validated within NVIDIA Isaac Sim.

## Architecture: Slow Brain / Fast Brain

**Slow Brain** runs once per command. The user types a natural-language instruction; a grounding model (GSAM) generates labeled bounding boxes + segmentation masks from both cameras; a VLM (Qwen) selects the target and destination from the annotated detections; a projection node fuses both depth streams into a labeled 3D point cloud; a grasp generator (GraspGen, on the remote A100) infers 6-DOF grasp candidates from the target point cloud.

**Fast Brain** runs at >30 FPS. A YOLO-based detector (`yolo_hazard_pkg`) monitors both cameras for hazards simultaneously. Detected hazards are injected as dynamic collision objects into the MoveIt planning scene, which uses hybrid planning for long-range trajectory + low-latency local reactions.

**Behavior Tree** (`bt_pkg`) closes the loop: it waits for Slow Brain results, selects grasp candidates, drives MoveIt2 for pick-and-place, and suspends the arm on E-stop hazard signals.

## Hardware
Two machines over plain SSH — no VPN or overlay network. One local workstation (RTX 5070, 12 GB) runs Isaac Sim, ROS 2, MoveIt, the YOLO tracking loop, the behavior tree, and RViz. A remote **A100** (`tta@123.37.28.208`) serves all heavy inference: Qwen vLLM on :8000, GraspGen on :5556, SwinDRNet on :5557, all loopback-only and reached through the tunnels `launch_env.bash` opens. The local GPU is capped at 12 GB, so anything larger offloads. All three remote services share one A100, so they contend under load.

## Key Constraints
- Avoidance loop must sustain >30 FPS
- Heavy inference (GSAM, Qwen VLM) always offloads to the remote cluster
- Current scope is Isaac Sim validation only; sim-to-real is a future goal
- ROS 2 Jazzy, Python 3.10+, C++20
- Do **not** run `hybrid_pose_client_node` alongside `bt_executor_node` — both submit goals to `/run_hybrid_planning`, causing double-goal / "Unknown event" crashes in the hybrid planning manager

---

## Build & Environment

### Initial setup (once)
```bash
# 1. Create and populate the venv
python3 -m venv gsam_venv
source gsam_venv/bin/activate
pip install -r ros_pkgs/src/grounded_sam_pkg/requirements.txt

# 2. Download GSAM model weights
mkdir -p models/g-sam
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
     -O models/g-sam/groundingdino_swint_ogc.pth
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
     -O models/g-sam/sam_vit_b_01ec64.pth

# 3. System deps. libzmq3-dev/libsqlite3-dev/libtinyxml2-dev are for
#    BehaviorTree.CPP (the library source is already vendored in-tree at
#    ros_pkgs/src/behavior_tree/ — nothing to clone). xterm is required by the
#    Slow Brain launch files, which wrap instruction_prompt_node in one because
#    `ros2 launch` does not forward stdin; without it the launch dies with
#    FileNotFoundError before you can type anything.
sudo apt install -y libzmq3-dev libsqlite3-dev libtinyxml2-dev xterm

# 4. YOLO venv (separate from gsam_venv — yolo_hazard_pkg launch files hardcode it)
python3 -m venv .venv-yolo
.venv-yolo/bin/pip install ultralytics
```

**This repo has no git submodules.** `.gitmodules` is empty and no gitlinks are
tracked; `BehaviorTree.CPP`, `BehaviorTree.ROS2`, and everything else are
committed as ordinary files. Any instruction to run `git submodule update
--init` is stale — there is nothing to initialise.

### Per-session environment
```bash
# Always run this from the repo root before any ROS 2 commands
source launch_env.bash
```
`launch_env.bash` sources `/opt/ros/jazzy/setup.bash`, the workspace install overlay at `ros_pkgs/install/setup.bash`, injects `gsam_venv/lib/python3.12/site-packages` into `PYTHONPATH`, and exports `ROBOT_CAPSTONE_ROOT` (repo root) for use by all launch files loading `config/robot_defaults.yaml`.

**SSH tunnel side effect**: `launch_env.bash` opens **four** tunnels, skipping any whose port is already bound. All terminate on the A100 (`tta@123.37.28.208`), loopback-only on the server:

| Port | Serves |
|---|---|
| 8000 | Qwen3.5-27B via vLLM, served as `qwen35-local` (OpenAI-compatible API) |
| 5556 | GraspGen inference (ZMQ) |
| 5557 | SwinDRNet depth restoration (ZMQ) |
| 5558 | SAM 2.1 segmentation (ZMQ) |

Qwen previously lived on `aurora-g6` behind the `aurora.khu.ac.kr:30080` jump host; that block is retained commented-out in `launch_env.bash` in case it moves back.

Suppress the tunnels when off-network or they will hang on connect:
```bash
source launch_env.bash --no-tunnel   # or: SKIP_A100_TUNNEL=1 source launch_env.bash
```

### Build ROS packages
```bash
cd ros_pkgs && colcon build --symlink-install && source install/setup.bash
```

Single package:
```bash
cd ros_pkgs && colcon build --symlink-install --packages-select bt_pkg
```

**Post-rebuild shebang fix (grounded_sam_pkg only)**: After every rebuild of `grounded_sam_pkg`, colcon resets entry-script shebangs to system Python, which lacks `torch`. Fix before launching:
```bash
VENV_PY="$PWD/gsam_venv/bin/python"
for f in $(find ros_pkgs/install/grounded_sam_pkg/lib -maxdepth 3 -type f -executable); do
  head -1 "$f" | grep -q "^#!/usr/bin/python3$" && \
    sed -i "1s|^#!/usr/bin/python3$|#!${VENV_PY}|" "$f"
done
```

### Run tests
```bash
cd ros_pkgs
colcon test --packages-select grounded_sam_pkg
colcon test-result --verbose
# Single file:
python3 -m pytest ros_pkgs/src/grounded_sam_pkg/test/test_flake8.py -v
```

---

## Launch

### Isaac Sim scene
```bash
./run_capstone_scene.sh   # delegates to sim/run_capstone_scene.sh
```

### Full system — teammate node (bt_pkg + moveit bridge)
```bash
# Terminal 1 — MoveIt hybrid planner + gripper server
source launch_env.bash
ros2 launch moveit_isaac_bridge_pkg hybrid_planning.launch.py

# Terminal 2 — YOLO hazard detection (both cameras)
source launch_env.bash
ros2 launch yolo_hazard_pkg yolo_hazard_both.launch.py

# Terminal 3 — Hazard → MoveIt collision object injector
source launch_env.bash
ros2 launch moveit_isaac_bridge_pkg hazard_collision_injector.launch.py

# Terminal 4 — Behavior tree (waits 5 s for action servers to be ready)
source launch_env.bash
ros2 launch bt_pkg bt_system.launch.py
```

**Extrinsics path**: the only extrinsics file in the repo is
`ros_pkgs/src/mask_projection_pkg/config/camera_extrinsics.yaml`, which every
launch file already falls back to. There is **no** `camera_extrinsics_isaac.yaml`
and no `config/` copy — pass `extrinsics_config:=` only when overriding with a
file you generated yourself.

### Hazard demo (scripted Isaac scene + planner mode)

`run_hazard_demo.sh` sets the `ROBOT_CAPSTONE_*` scene env vars and launches the
Isaac scene; the remaining terminals are unchanged. The mode must match
`manager_logic` on the hybrid planner:

```bash
./run_hazard_demo.sh replan        # persistent hazard — bottle parks, OMPL re-routes
./run_hazard_demo.sh stop_resume   # transient hazard — bottle passes, arm holds then resumes
```
```bash
ros2 launch moveit_isaac_bridge_pkg hybrid_planning.launch.py manager_logic:=replan
```

Scene geometry/velocity overrides take precedence over the script defaults, e.g.
`ROBOT_CAPSTONE_HAZARD_PARK_X=0.30 ./run_hazard_demo.sh replan`. The `replan`
tuning is a three-way coupling — `local_planner.yaml` frequency, the negative
`xy_margin` in `hazard_collision_injector.launch.py`, and the gripper
`link_padding` shrink in `hybrid_planning.launch.py` — retune together or
`CheckStartStateCollision` trips on mm-scale fingertip overlap at every replan.

### Full Slow Brain pipeline (Isaac Sim)
```bash
# Terminal 1 — GSAM (dual-view: EE + Top cameras)
source launch_env.bash
ros2 launch grounded_sam_pkg grounded_sam_dual.launch.py

# Terminal 2a — Real Qwen VLM (requires SSH tunnel or cluster access)
source launch_env.bash
ros2 launch qwen_pkg inst_input_qwen.launch.py \
  vllm_endpoint_url:=http://localhost:8000/v1 \
  model_name:=qwen35-local
# Opens an xterm for typing user instructions — ros2 launch doesn't forward stdin

# Terminal 2b — OR: Qwen stub (hardcoded LABEL_TO_CATEGORY, no VLM call needed)
source launch_env.bash
ros2 run grounded_sam_pkg qwen_stub_node

# Terminal 3 — Multi-view projection → /world_map + /world_map_result
source launch_env.bash
ros2 launch mask_projection_pkg multi_view_projector.launch.py \
  ee_depth_topic:=/ee_rgbd_camera/depth_image \
  ee_camera_info_topic:=/ee_rgbd_camera/camera_info \
  top_depth_topic:=/rgbd_camera/depth_image \
  top_camera_info_topic:=/rgbd_camera/camera_info

# Terminal 4 — GraspGen (remote A100 via the 5556 tunnel) → /grasp_candidates
source launch_env.bash
ros2 launch graspgen_pkg graspgen.launch.py

# Terminal 4, transparent objects (glass) — adds SwinDRNet depth restoration
ros2 launch graspgen_pkg graspgen.launch.py \
  swindrnet_enabled:=true transparent_reconstruct_enabled:=true transparent_force:=true
```

Everything from GSAM through GraspGen also runs from one launch file:
```bash
ros2 launch graspgen_pkg full_pipeline_graspgen.launch.py prompt:="cup, table, object"
```
Its `num_grasps`/`topk_num_grasps` defaults (50/5) are stale relative to
`graspgen.launch.py` (200/100) — pass them explicitly if you care.

### Test Qwen endpoint directly (SSH tunnel)
```bash
# Requires: ssh -L 8000:localhost:8000 user@cluster -N
python models/qwen3.5/qwen_ssh_client.py \
  --image demo/resources/ee_raw.png \
  --text "what objects do you see?"
```

---

## ROS Package Architecture

```
ros_pkgs/src/
├── grounded_sam_pkg/       Slow Brain perception — GroundingDINO + SAM
├── qwen_pkg/               Slow Brain VLM — Qwen grounding + instruction input
├── mask_projection_pkg/    2D mask + depth → labeled 3D PointCloud2
├── graspgen_pkg/           Slow Brain grasp detection — ACTIVE path, ZMQ → remote A100
├── target_pose_bridge_pkg/ /world_map_result → MoveIt goal poses (centroid-based, legacy)
├── moveit_isaac_bridge_pkg/MoveIt + Isaac Sim joint bridge + gripper action server
├── yolo_hazard_pkg/        Fast Brain hazard detection — YOLO on both cameras >30 FPS
├── bt_pkg/                 BehaviorTree.ROS2 pick-and-place executor
├── behavior_tree/          BehaviorTree.CPP + BehaviorTree.ROS2, vendored (do not modify)
└── moveit2/                Empty .gitkeep placeholder — MoveIt comes from apt, not source
```

See `ROS_NODES.md` for a full per-node topic/action reference, and
`A100_GRASPGEN_QUICK_START.md` / `SWINDRNET_INTEGRATION.md` for the remote
inference server runbooks.

### Unified parameter file

`config/robot_defaults.yaml` (repo root) is the single source of truth for shared robot identity parameters. All launch files load it first; package YAMLs override only what is package-specific. `$ROBOT_CAPSTONE_ROOT` (set by `launch_env.bash`) points to the repo root.

**Grasp pool vs BT retry budget — decoupled.** `robot_defaults.yaml` carries both:

- `max_grasp_candidates` (**100**) — grasp pool size, seeded to the BT blackboard. GraspGen has its own `topk_num_grasps` in `graspgen_pkg/config/graspgen_params.yaml` (also 100, paper-aligned).
- `bt_pick_retries` (**5**) — BT `RetryUntilSuccessful` num_attempts in `pick_and_place.xml`. The BT only attempts the top-N published candidates, regardless of pool size. Keep small to avoid SIGABRT-prone goal flooding into the hybrid planner.

GraspGen narrows the pool a third time: after the top-down and IK filters,
`max_published_grasps` (**10**) caps what actually reaches `/grasp_candidates`.
So the real chain is `topk 100 → filters → 10 published → BT tries 5`.
`ROS_NODES.md` still documents `max_grasp_candidates` as 5 — that is the stale
C++ `declare_parameter` fallback, always overridden by the launch file.

### Full data flow

```
/ee_camera/image_raw  →  grounded_sam_node  →  /grounded_sam/detections_json
/camera/image_raw     →  (top, cached)      →  /grounded_sam/mask_image
                                             →  /grounded_sam/annotated_image

/user_instruction             →  qwen_bridge_node (cached)
/grounded_sam/detections_json →  qwen_bridge_node (trigger)
/grounded_sam/mask_image      →  qwen_bridge_node (cached)
    │
    │  [VLM inference — remote cluster ~1–5 s]
    │
    ├──► /qwen/labeled_detections  (JSON + "category" field: TARGET/DESTINATION/OBSTACLE)
    ├──► /qwen/grounding_result    (structured JSON: target_id, destination type+relation)
    └──► /qwen/mask_image          (pass-through — published LAST to trigger projector)

/ee_camera/depth_image   →  multi_view_projector_node  →  /world_map        (PointCloud2, labeled)
/top_camera/depth_image  →                             →  /world_map_result (JSON centroid+bbox)
/qwen/mask_image (trigger) →                           →  /world_cloud_raw  (PointCloud2, unlabeled)

/world_map_result (trigger) ─┐
/ee_camera/{depth,image_raw,camera_info} ─┤→ graspgen_node → /grasp_candidates (JSON, latched)
/sam/mask_image, /qwen/labeled_detections ┘   │  [ZMQ → A100 :5556]  → /grasp_markers (RViz)
                                              └─ optional [ZMQ → A100 :5557 SwinDRNet]
                                                 → /graspgen/target_cloud (debug)

/world_map_result  ─┐
/grasp_candidates  ─┤→  bt_executor_node  →  /run_hybrid_planning  →  MoveIt hybrid planner
/qwen/grounding_result ┘                  →  /gripper_command       →  gripper_action_server
/yolo/world_map    ─┘                     →  /bt/replan_request     (triggers new Slow Brain scan)
/bt/hazard_level   ─┘

MoveIt  →  hybrid_command_bridge_node  →  /joint_command  →  Isaac Sim
        →  joint_state_restamp_node    ←  /joint_states_isaac  (re-stamps to wall time)
```

**Grasp path choice** — three nodes consume `/world_map_result` and they are mutually exclusive; running two publishes duplicate `/grasp_candidates`:

| Node | Status | Notes |
|---|---|---|
| `graspgen_node` | **Use this one.** | 6-DOF candidates from the remote A100 over ZMQ. |
| `target_pose_bridge_node` | Legacy | Simple centroid-offset pose for `capstone_pick_pipeline.launch.py`. |

Both grasp publishers use **latched QoS** (`transient_local`, depth 1) on `/grasp_candidates`, as does `/world_map_result` — `bt_executor_node` therefore picks up the last scan even if it starts late.

### Key design decisions

**No timestamp synchronization** — GSAM inference takes 30–40 s on CPU; by the time `mask_image` arrives the depth queue has advanced far past the matching frame. `multi_view_projector_node` caches the latest depth/camera_info and treats each incoming mask as the trigger, rather than using `ApproximateTimeSynchronizer`.

**Qwen publish order is load-bearing** — `qwen_bridge_node` always publishes `/qwen/labeled_detections` before `/qwen/mask_image`. (In the new `slow_brain/` path the mask comes from `sam_mask_node` on **`/sam/mask_image`** instead, triggered by `/qwen/source_image` — the two pipelines therefore no longer collide on one topic.) The mask is the trigger for `multi_view_projector_node`, so labels must already be in cache when it fires. Never reorder these publishes.

**Dual-view with one model instance** — `grounded_sam_dual.launch.py` runs a single `GroundedSAMNode` subscribed to both EE and Top cameras. Top images are cached; the EE callback drives both views sequentially through the same pipeline.

**Stub vs real Qwen** — `qwen_stub_node` (in `grounded_sam_pkg`) uses a hardcoded `LABEL_TO_CATEGORY` dict and requires no cluster. `qwen_bridge_node` (in `qwen_pkg`) calls the real VLM. They are **not** drop-in replacements for `bt_pkg`: the stub only publishes `/qwen/labeled_detections` and `/qwen/mask_image` — it does **not** publish `/qwen/grounding_result`, so `bt_executor_node`'s `destination_spec` (`SceneData`) is never populated and `UpdateTargetPose` will fail. The stub is sufficient to test the Slow Brain pipeline up to the projection step; use the real `qwen_bridge_node` when running the full BT pipeline. `mask_projection_pkg` itself requires no changes to swap between them.

**`projection_engine.py` is ROS-free** — all numpy projection/filter math lives there. `multi_view_projector_node.py` only handles ROS message decode/encode. Keep it that way.

**Point cloud categories** — `FREE=0` (EE non-detections), `TARGET=1`, `DESTINATION=2`, `OBSTACLE=3`, `UNKNOWN=4` (all Top-view points). When feeding octomap, use only `OBSTACLE + UNKNOWN`; `FREE` points cause background to be marked occupied. The authoritative name is `DESTINATION` (see `label_mapper.py`) — some older comments say `WORKSPACE`.

**Isaac sim time** — Isaac Sim does not publish `/clock`. All nodes run on wall time (`use_sim_time: False`). `joint_state_restamp_node` re-stamps Isaac's `/joint_states_isaac` to wall time before forwarding to `/joint_states`.

---

### bt_pkg internals

See `ros_pkgs/src/bt_pkg/README.md` for full detail. Key points:

- `SceneData` (mutex-guarded struct) is the only shared state between the ROS subscription callbacks and BT node `tick()` calls — nothing goes through the blackboard except computed poses and indices
- `RequestReplan` returns `FAILURE` intentionally to restart the pipeline from `WaitForScene` via `RepeatForever` — this is not a bug
- `MoveAction` must populate `start_state.joint_state` from `SceneData::latest_joint_state` before sending the hybrid planner goal
- `behaviortree_ros2` and `behaviortree_cpp` (BT.cpp v4 core) are both vendored as ordinary tracked files under `ros_pkgs/src/behavior_tree/` — **not** submodules, and not at `ros_pkgs/src/BehaviorTree.CPP/`. Both are built by colcon; no apt install beyond the system deps in Initial setup
- **Hazard levels** (`/bt/hazard_level`, `std_msgs/Int8`) are integer literals with no enum: `3` = HALT (a `halt_class_ids` detection at conf ≥ 0.6), `1` = SLOW (any detection at conf ≥ 0.5), `0` = CLEAR (or `decay_sec` 0.3 s elapsed). **`2` is never produced.** Only level ≥ 3 is acted on by the BT — `EmergencyStopClear` returns FAILURE and the wrapping `ReactiveSequence` suspends the tree. Levels 0–1 are handled by the hybrid planner and `hazard_collision_injector` instead
- The **destination/place phase of `pick_and_place.xml` is commented out**, and the `TargetVisible` visibility guard is stubbed to `<AlwaysSuccess/>`. The tree currently ends after the retreat move. `TargetVisible` depends on `/yolo/target_centroid`, which is never published — `yolo_world_map_node` only publishes it when the `target_search_label`/`target_seed_centroid` parameters are set, and nothing in the repo ever sets them
- `bt_executor_node.cpp` writes `behavior_trees/bt_models.xml` (a generated Groot2 artifact) on **every startup** via a hardcoded absolute `/home/hj1/...` path — it will fail or write to the wrong place on any other machine

#### Required — handle pick-only tasks (no destination)

`qwen_a100` now **omits** `destination` from `/qwen/grounding_result` when the
instruction says nothing about where the object goes ("pick up the book").
Previously the guided-decoding schema forced the field, so the VLM invented a
destination and the BT acted on a hallucination.

The key is omitted rather than set to null on purpose:
`parse_grounding_result` does `j.contains("destination")` and then calls
`.value()` on it, which would throw inside the parser on a null.

**Two changes bt_pkg needs:**

1. **Clear `destination_spec` when the key is absent.**
   [bt_executor_node.cpp:113](ros_pkgs/src/bt_pkg/src/bt_executor_node.cpp:113)
   only assigns the four strings *inside* `if (j.contains("destination"))`, so a
   pick-only scan leaves the **previous scan's spec in place**. Place the book in
   the box, then say "pick up the cup", and the stale box destination is still
   sitting in `SceneData`. Add an `else` that resets it, and set a
   `has_destination` flag alongside.

2. **Make the place phase conditional — and gate on the CENTROID, not the spec.**
   There are two independent failure modes and they need different checks:

   | Case | `destination_spec` | `destination` key in `/world_map_result` |
   |---|---|---|
   | pick-only ("pick up the book") | empty | absent |
   | named but not in scene ("put it in the box", no box visible) | **fully populated** | **absent** |

   A guard keyed on the spec passes in the second case, so the place would still
   execute — with `SceneData::destination_centroid` left at its zero-initialised
   `{}` ([scene_data.hpp:52](ros_pkgs/src/bt_pkg/include/bt_pkg/scene_data.hpp:52)).
   `compute_place_pose` with `type=container` then returns `(0, 0, 0.03)`: the arm
   dives at its own base. Collision risk, not just a wrong place.

   So the condition must be **"did `/world_map_result` carry a destination
   centroid"**, which means `parse_world_map_result` needs a
   `destination_centroid_valid` flag set only inside its
   `if (d.contains("centroid"))` branch. Guarding on the spec alone is not enough.

   Also note the empty-spec case is separately dangerous: `compute_place_pose`
   takes **no branch at all** for an unknown `type` and returns the raw centroid
   with zero offsets, placing the object *inside* the destination.

A `Fallback` around the place subtree with a `HasDestinationCentroid` condition
node is the smallest change; a `SubTree` guarded by a blackboard flag also works.

Related: `SceneData::destination_label` is written and never read, and
`grounding_result_fresh` is set and never read — so `/qwen/grounding_result`
gates nothing. `ParseScene` just snapshots whatever spec is present when it
ticks. That is safe today only because Qwen publishes grounding well before the
mask → projector → graspgen chain completes.

#### Planned — direction-aware `near` in the place stage

`destination_calculator.cpp` currently resolves every spatial relation to a
**fixed axis offset in `panda_link0`**, ignoring scene geometry entirely:

| relation | offset applied |
|---|---|
| `left_of` / `right_of` | `x ∓ side_offset_m` |
| `in_front_of` / `behind` | `y ∓ side_offset_m` |
| `on_top_of` | `z += place_height_m * 2` |
| `near` | `z += place_height_m; x += near_offset_m` |
| unknown / empty | `z += place_height_m` only |

`near` is the weakest of these — it always shifts **+X by 8 cm**, whatever the
actual layout. If the target sits on the −X side of the destination, the robot
carries it across and places it on the far side, which is both a longer
transport and more likely to clip the destination object on the way.

**Planned change**: make `near` direction-aware. Instead of a constant axis
offset, compute the horizontal unit vector from the destination centroid toward
the target's current centroid and offset along that by `near_offset_m`. The
target then lands on the side of the destination nearest to where it already
is — shortest transport, and the approach never crosses over the destination.

Both inputs are already available and currently unused:

- `SceneData::target_centroid` and `destination_centroid` (from `/world_map_result`)
- `bbox_3d_world.{min,max}` for both categories — also in `/world_map_result`,
  read today only by `graspgen_node`, ignored entirely by the BT

The bbox opens a second refinement: rather than a fixed 8 cm, offset to just
outside the destination's XY footprint plus a clearance margin, so the distance
scales with the destination's actual size instead of assuming one.

This matters because `near` is also the **fallback** the whole relation vocabulary
degrades to — an empty or unrecognised `relation` from the VLM resolves here (see
`qwen_a100/schema.py`). Making it geometrically sensible upgrades the worst case
from "arbitrary +X shove" to "nearest sensible free point", which is a reasonable
reading of an under-specified instruction.

Do this when re-enabling the commented-out place block in `pick_and_place.xml`.

### grounded_sam_pkg internals

| Module | Role |
|---|---|
| `ros_node.py` | ROS 2 wiring, frame-rate throttle, bbox-area filter, dual-view logic |
| `pipeline.py` | Orchestrates GroundingDINO → SAM with a single `run(image, prompt)` call |
| `gdino_runner.py` | GroundingDINO inference wrapper |
| `sam_runner.py` | SAM inference wrapper |
| `prompt_adapter.py` | Converts comma-separated input → GroundingDINO period-separated noun phrase |
| `postprocess.py` | Raw detections → JSON-serializable list |
| `visualizer.py` | Draws bboxes/masks onto BGR images |
| `qwen_stub_node.py` | Stub replacement for `qwen_bridge_node` — no cluster needed |

### qwen_pkg internals

| Module | Role |
|---|---|
| `qwen_bridge.py` | ROS 2 node — caches inputs, spawns VLM call on a daemon thread, enforces publish order |
| `qwen_call.py` | ROS-free — OpenAI client, prompt construction, `GroundingResult` Pydantic model, guided-JSON schema |
| `instruction_prompt_node.py` | Reads stdin in a daemon thread, publishes to `/user_instruction` |

`qwen_call.py` uses vLLM's `guided_json` + outlines backend for structured output. The flat JSON schema in `_SCHEMA` avoids `oneOf`/`anyOf` (outlines backend limitation) — destination subtypes are merged into a single object with optional fields.

### mask_projection_pkg internals

| Module | Role |
|---|---|
| `projection_engine.py` | Pure numpy: `project_labeled` (EE→world+labels), `project_unknown` (Top→UNKNOWN with Pass 1 filter), `filter_free_by_unknown` (Pass 2) |
| `label_mapper.py` | Category IDs/colors, mask pixel→category mapping. **Do not change category IDs** — they are part of the downstream API |
| `cloud_builder.py` | `CategoryPoints` → `PointCloud2` message |
| `ply_utils.py` | PLY save + `build_result_json` (centroid + 3D bbox per category) |
| `back_projection.py` | Depth image → (N,3) camera-frame points |
| `multi_view_projector_node.py` | ROS wiring only |
| `projector_node.py` | Single-camera Gazebo demo — do not modify |

Two-pass filtering: Pass 1 removes Top UNKNOWN points within 1.5 cm XY + 10 cm Z of EE-segmented points. Pass 2 removes EE FREE points that overlap Top UNKNOWN (UNKNOWN > FREE priority).

### graspgen_pkg internals

The active grasp path. All heavy inference is remote; the node only extracts the
target cloud, ships it over ZMQ, and filters what comes back.

| Module | Role |
|---|---|
| `graspgen_node.py` | The only ROS node. Caches EE depth/RGB/CameraInfo + GSAM mask, triggers on `/world_map_result`, extracts the TARGET cloud, calls GraspGen, filters and re-poses the grasps |
| `zmq_client.py` | `GraspGenClient` — ZMQ REQ/REP + msgpack, with REQ-socket reset-on-timeout recovery |
| `swindrnet_client.py` | `SwinDRNetClient` — sends (RGB, broken depth, K), gets restored depth back |
| `grasp_filter.py` | `top_down_filter` (approach angle), `confidence_top_n`, and `IKFeasibilityChecker` (MoveIt `/compute_ik`, **fail-open**) |
| `cloud_extractor.py` | TARGET detection → mask value; masked depth → world-frame `(N,3)` cloud |
| `depth_utils.py` | ROS-free depth/mask decode, `extract_K`, extrinsics YAML load |
| `marker_publisher.py` | Grasp → 4-CUBE Panda gripper `MarkerArray`. Namespace is still `vgn_grasps` — it matches `rviz/grasp_demo.rviz`, rescued when `vgn_grasp_pkg` was retired. Rename both together or neither |
| `graspgen_cli.py` | Offline CLI to hit the ZMQ server directly. **Not** an installed entry point and uses bare imports — run it from inside `graspgen_pkg/graspgen_pkg/` |
| `transparent_reconstruct.py` | **Dead code** — analytic cylinder prior, no longer imported by anything |

**`MultiThreadedExecutor` is required**, not optional — the IK checker polls
futures on a `ReentrantCallbackGroup` and will deadlock on a single-threaded spin.

**SwinDRNet is opt-in and fails loudly.** It only runs when
`transparent_reconstruct_enabled` **and** the TARGET label is in
`transparent_labels` (or `transparent_force`). On failure the node logs and falls
back to **raw depth** — deliberately, so a bad restoration is visible rather than
silently papered over by a geometric prior. `SWINDRNET_INTEGRATION.md` still
documents an analytic-cylinder fallback stage that no longer exists.

### moveit_isaac_bridge_pkg additions

`gripper_action_server.py` serves `control_msgs/action/GripperCommand` on `/gripper_command`. Uses MoveIt `panda_hand` group for motion; polls `/joint_states` at 20 Hz to detect contact (finger stall > 8 mm = grasped → `stalled=True`). Launched by `hybrid_planning.launch.py`.

---

### Model weights (not in git)

Place under `models/g-sam/` (paths in `ros_pkgs/src/grounded_sam_pkg/config/model_paths.yaml`):
- `groundingdino_swint_ogc.pth` (~662 MB)
- `sam_vit_b_01ec64.pth` (~375 MB)

YOLO weights: `yolo_hazard_pkg/config/model_paths.yaml` points at
`models/yolo26/release/capstone-hazard-seg-best.pt`, which **does not exist in
the working tree**. The resolver only rewrites relative paths when the file is
present, so a missing weight silently degrades to a bare relative string handed
to ultralytics, which then fails at load. What is actually on disk lives under
`models/yolo26/training/` (`yolo26m-seg.pt`, plus `runs/segment/*/weights/`).
Note `models/yolo26/ultralytics/` is an untracked 71 MB upstream clone and is
not gitignored.

**YOLO class taxonomy is inconsistent across three sources** — reconcile before
trusting `halt_class_ids`:

| Source | Taxonomy |
|---|---|
| `config/robot_defaults.yaml` (effective) | `0=arm/person, 1=bottle, 2=box` |
| `models/yolo26/training/config.yaml` + pkg README (trained model) | `0=hand, 1=forearm, 2=pet_bottle, 3=small_box` |
| `yolo_hazard_pkg/ros_node.py` default (overridden) | `[0, 39]` — COCO person/bottle |

`halt_class_ids: [0]` means "arm/person" under the first and "hand" under the
second; the 4-class model's `3=small_box` is not in `class_allowlist` at all.

Qwen SSH client for manual testing: `models/qwen3.5/qwen_ssh_client.py`
Qwen FastAPI wrapper (cluster deploy): `models/qwen3.5/qwenapi.py` — fill in `MODEL_NAME` and `VLLM_BASE_URL` before deploying.

### Isaac Sim bridge scripts (`sim/`)

| File | Role |
|---|---|
| `setup_initial_scene.py` | Loads USD stage, places objects (apple, glass, cube, book), attaches cameras |
| `isaac_ros_camera_bridge.py` | Wires Isaac Sim cameras to ROS 2 image/depth topics |
| `isaac_ros_joint_bridge.py` | Wires Panda joint states/commands between Isaac Sim and ROS 2 |
| `diag_depth.py` | Depth-stream diagnostic — inspect raw Isaac depth for holes/units |
| `import_downloaded_assets.py` | GLB → USD conversion into `sim/assets/` (`run_import_assets.sh`) |
| `image_capture_helper.py` | Dumps camera frames to PNG for offline VLM testing |

**Asset paths**: USD stage files live in `~/Downloads/XR_Content_NVD@10010/Assets/XR/Stages/` by default. Override with `ROBOT_CAPSTONE_XR_CONTENT_ROOT`. Downloaded GLBs default to `~/Downloads`; override with `ROBOT_CAPSTONE_DOWNLOADS_DIR`.

**Camera extrinsics** (`ros_pkgs/src/mask_projection_pkg/config/camera_extrinsics.yaml` — there is no copy under `config/`): extrinsics are captured at the robot's start pose. If the start pose changes, regenerate from Isaac Sim Script Editor by dumping `panda_link0`, `EEViewCamera`, and `TopViewCamera` world transforms, then converting from USD world into `panda_link0` frame. The conversion procedure is documented at the top of the YAML file.
