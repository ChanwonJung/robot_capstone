# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Specifications

Ubuntu 24.04 · ROS Jazzy · IsaacSim 5.1.0

## End Goal
Language-directed robotic manipulator that interprets ambiguous natural language commands, visually grounds targets in cluttered scenes, and executes safe grasps while avoiding dynamic obstacles. Validated within NVIDIA Isaac Sim.

## Architecture: Slow Brain / Fast Brain

**Slow Brain** runs once per command. The user types a natural-language instruction; a grounding model (GSAM) generates labeled bounding boxes + segmentation masks from both cameras; a VLM (Qwen) selects the target and destination from the annotated detections; a projection node fuses both depth streams into a labeled 3D point cloud; VGN infers grasp candidates from a signed TSDF.

**Fast Brain** runs at >30 FPS. A YOLO-based detector (`yolo_hazard_pkg`) monitors both cameras for hazards simultaneously. Detected hazards are injected as dynamic collision objects into the MoveIt planning scene, which uses hybrid planning for long-range trajectory + low-latency local reactions.

**Behavior Tree** (`bt_pkg`) closes the loop: it waits for Slow Brain results, selects VGN grasp candidates, drives MoveIt2 for pick-and-place, and suspends the arm on E-stop hazard signals.

## Hardware
Three-node distributed setup over Tailscale. The simulation node runs Isaac Sim, ROS 2, and MoveIt. A remote GPU cluster (`aurora-g6` via `aurora.khu.ac.kr`) handles all heavy Slow Brain inference via vLLM (OpenAI-compatible API). The development node runs the YOLO tracking loop, behavior tree, and RViz. Each local node is constrained to 12GB VRAM — anything heavier offloads to the cluster.

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

# 3. VGN submodule + weights (needed only for vgn_grasp_pkg)
git submodule add https://github.com/ethz-asl/vgn external/vgn
pip install pytorch-ignite tqdm
# Download data.zip from ethz-asl/vgn GitHub → extract → copy:
cp data/models/vgn_conv.pth models/vgn_conv.pth
# Filename convention: vgn_<network>.pth — load_network() parses the second field.
# Wrong filename → KeyError at startup.

# 4. BT.cpp v4 core library — tracked as a git submodule, built by colcon
git submodule update --init ros_pkgs/src/BehaviorTree.CPP
# System deps required by BehaviorTree.CPP:
sudo apt install -y libzmq3-dev libsqlite3-dev libtinyxml2-dev
```

### Per-session environment
```bash
# Always run this from the repo root before any ROS 2 commands
source launch_env.bash
```
`launch_env.bash` sources `/opt/ros/jazzy/setup.bash`, the workspace install overlay at `ros_pkgs/install/setup.bash`, injects `gsam_venv/lib/python3.12/site-packages` into `PYTHONPATH`, and exports `ROBOT_CAPSTONE_ROOT` (repo root) for use by all launch files loading `config/robot_defaults.yaml`.

**SSH tunnel side effect**: `launch_env.bash` also automatically opens an SSH tunnel `localhost:8000 → aurora-g6:8000` via the jump host `aurora.khu.ac.kr:30080` using the username `jaewonheo1101`. If port 8000 is already bound it skips silently.

### Build ROS packages
```bash
cd ros_pkgs && colcon build --symlink-install && source install/setup.bash
```

Single package:
```bash
cd ros_pkgs && colcon build --symlink-install --packages-select bt_pkg
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
ros2 launch bt_pkg bt_system.launch.py \
  extrinsics_config:=/abs/path/camera_extrinsics_isaac.yaml
```

### Full Slow Brain pipeline (Isaac Sim)
```bash
# Terminal 1 — GSAM (dual-view: EE + Top cameras)
source launch_env.bash
ros2 launch grounded_sam_pkg grounded_sam_dual.launch.py

# Terminal 2a — Real Qwen VLM (requires SSH tunnel or cluster access)
source launch_env.bash
ros2 launch qwen_pkg inst_input_qwen.launch.py \
  vllm_endpoint_url:=http://localhost:8000/v1 \
  model_name:=Qwen/Qwen3.5-VL-9B-Instruct
# Opens an xterm for typing user instructions — ros2 launch doesn't forward stdin

# Terminal 2b — OR: Qwen stub (hardcoded LABEL_TO_CATEGORY, no VLM call needed)
source launch_env.bash
ros2 run grounded_sam_pkg qwen_stub_node

# Terminal 3 — Multi-view projection → /world_map + /world_map_result
source launch_env.bash
ros2 launch mask_projection_pkg multi_view_projector.launch.py \
  extrinsics_config:=<path>/camera_extrinsics_isaac.yaml \
  ee_depth_topic:=/isaac/ee/depth_image \
  top_depth_topic:=/isaac/top/depth_image

# Terminal 4 (optional) — VGN grasp detection → /grasp_candidates
source launch_env.bash
ros2 launch vgn_grasp_pkg vgn_grasp.launch.py \
  extrinsics_config:=<path>/camera_extrinsics_isaac.yaml
```

### Test Qwen endpoint directly (SSH tunnel)
```bash
# Requires: ssh -L 8000:localhost:8000 user@cluster -N
python models/qwen3.5/qwen_ssh_client.py \
  --image ee_raw.png \
  --text "what objects do you see?"
```

---

## ROS Package Architecture

```
ros_pkgs/src/
├── grounded_sam_pkg/       Slow Brain perception — GroundingDINO + SAM
├── qwen_pkg/               Slow Brain VLM — Qwen grounding + instruction input
├── mask_projection_pkg/    2D mask + depth → labeled 3D PointCloud2
├── vgn_grasp_pkg/          Slow Brain grasp detection — TSDF + VGN inference
├── target_pose_bridge_pkg/ /world_map_result → MoveIt goal poses (centroid-based)
├── moveit_isaac_bridge_pkg/MoveIt + Isaac Sim joint bridge + gripper action server
├── bt_pkg/                 BehaviorTree.ROS2 pick-and-place executor  ← NEW
└── behavior_tree/          BehaviorTree.ROS2 vendored library (do not modify)
```

### Unified parameter file

`config/robot_defaults.yaml` (repo root) is the single source of truth for shared robot identity parameters. All launch files load it first; package YAMLs override only what is package-specific. `$ROBOT_CAPSTONE_ROOT` (set by `launch_env.bash`) points to the repo root.

**`max_grasp_candidates` is coupled** — it lives only in `robot_defaults.yaml` and controls both VGN's Top-K output count and the BT's `RetryUntilSuccessful` retry budget. Change it in one place only.

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

/world_map_result  →  vgn_grasp_node      →  /grasp_candidates (JSON, NMS filtered)
                                          →  /grasp_markers    (MarkerArray, RViz)

/world_map_result  ─┐
/grasp_candidates  ─┤→  bt_executor_node  →  /run_hybrid_planning  →  MoveIt hybrid planner
/qwen/grounding_result ┘                  →  /gripper_command       →  gripper_action_server
/yolo/world_map    ─┘                     →  /bt/replan_request     (triggers new Slow Brain scan)
/bt/hazard_level   ─┘

MoveIt  →  hybrid_command_bridge_node  →  /joint_command  →  Isaac Sim
        →  joint_state_restamp_node    ←  /joint_states_isaac  (re-stamps to wall time)
```

**Grasp path choice**: `vgn_grasp_node` and `target_pose_bridge_node` both consume `/world_map_result`. VGN provides ranked grasp candidates with 6-DOF poses used by `bt_pkg`; `target_pose_bridge` provides a simple centroid-offset pose used by the legacy `capstone_pick_pipeline.launch.py`. Do not run both simultaneously.

### Key design decisions

**No timestamp synchronization** — GSAM inference takes 30–40 s on CPU; by the time `mask_image` arrives the depth queue has advanced far past the matching frame. `multi_view_projector_node` caches the latest depth/camera_info and treats each incoming mask as the trigger, rather than using `ApproximateTimeSynchronizer`.

**Qwen publish order is load-bearing** — `qwen_bridge_node` always publishes `/qwen/labeled_detections` before `/qwen/mask_image`. The mask is the trigger for `multi_view_projector_node`, so labels must already be in cache when it fires. Never reorder these publishes.

**Dual-view with one model instance** — `grounded_sam_dual.launch.py` runs a single `GroundedSAMNode` subscribed to both EE and Top cameras. Top images are cached; the EE callback drives both views sequentially through the same pipeline.

**Stub vs real Qwen** — `qwen_stub_node` (in `grounded_sam_pkg`) uses a hardcoded `LABEL_TO_CATEGORY` dict and requires no cluster. `qwen_bridge_node` (in `qwen_pkg`) calls the real VLM. Both publish identical topic schemas — `mask_projection_pkg` requires no changes to swap between them.

**`projection_engine.py` is ROS-free** — all numpy projection/filter math lives there. `multi_view_projector_node.py` only handles ROS message decode/encode. Keep it that way.

**Point cloud categories** — `FREE=0` (EE non-detections), `TARGET=1`, `DESTINATION=2`, `OBSTACLE=3`, `UNKNOWN=4` (all Top-view points). When feeding octomap, use only `OBSTACLE + UNKNOWN`; `FREE` points cause background to be marked occupied. The authoritative name is `DESTINATION` (see `label_mapper.py`) — some older comments say `WORKSPACE`.

**VGN filename convention** — `load_network()` parses the network type from the weight filename (`vgn_conv.pth` → `conv`). A misnamed file causes a `KeyError` at startup.

**Isaac sim time** — Isaac Sim does not publish `/clock`. All nodes run on wall time (`use_sim_time: False`). `joint_state_restamp_node` re-stamps Isaac's `/joint_states_isaac` to wall time before forwarding to `/joint_states`.

---

### bt_pkg internals

See `ros_pkgs/src/bt_pkg/README.md` for full detail. Key points:

- `SceneData` (mutex-guarded struct) is the only shared state between the ROS subscription callbacks and BT node `tick()` calls — nothing goes through the blackboard except computed poses and indices
- `RequestReplan` returns `FAILURE` intentionally to restart the pipeline from `WaitForScene` via `RepeatForever` — this is not a bug
- `MoveAction` must populate `start_state.joint_state` from `SceneData::latest_joint_state` before sending the hybrid planner goal
- `behaviortree_ros2` is vendored at `ros_pkgs/src/behavior_tree/`; `behaviortree_cpp` (core C++ lib) must be apt-installed separately

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

### vgn_grasp_pkg internals

| Module | Role |
|---|---|
| `vgn_grasp_node.py` | Full pipeline: depth → signed TSDF (40×40×40) → VGN inference → NMS → semantic bbox filter → `/grasp_candidates` + `/grasp_markers` |

External dependency: `external/vgn` git submodule (ethz-asl/vgn). Semantic filtering keeps only grasps whose centre falls inside the target `bbox_3d_world` from `/world_map_result`. The node does **not** subscribe to `/world_map` — that topic is RViz-only.

The parameter is `max_grasp_candidates` (renamed from `max_candidates` in `feature/bt_trunk`). Override via launch arg:
```bash
ros2 launch vgn_grasp_pkg vgn_grasp.launch.py \
  vgn_model_path:=/abs/path/to/vgn_conv.pth \
  min_quality:=0.4 \
  max_grasp_candidates:=3 \
  use_top_depth:=false
```

### moveit_isaac_bridge_pkg additions

`gripper_action_server.py` serves `control_msgs/action/GripperCommand` on `/gripper_command`. Uses MoveIt `panda_hand` group for motion; polls `/joint_states` at 20 Hz to detect contact (finger stall > 8 mm = grasped → `stalled=True`). Launched by `hybrid_planning.launch.py`.

---

### Model weights (not in git)

Place under `models/g-sam/` (paths in `ros_pkgs/src/grounded_sam_pkg/config/model_paths.yaml`):
- `groundingdino_swint_ogc.pth` (~662 MB)
- `sam_vit_b_01ec64.pth` (~375 MB)

VGN weights: `models/vgn_conv.pth` (filename must follow `vgn_<network>.pth` convention)

YOLO weights: `models/yolo26/`

Qwen SSH client for manual testing: `models/qwen3.5/qwen_ssh_client.py`
Qwen FastAPI wrapper (cluster deploy): `models/qwen3.5/qwenapi.py` — fill in `MODEL_NAME` and `VLLM_BASE_URL` before deploying.

### Isaac Sim bridge scripts (`sim/`)

| File | Role |
|---|---|
| `setup_initial_scene.py` | Loads USD stage, places objects (apple, glass, cube, book), attaches cameras |
| `isaac_ros_camera_bridge.py` | Wires Isaac Sim cameras to ROS 2 image/depth topics |
| `isaac_ros_joint_bridge.py` | Wires Panda joint states/commands between Isaac Sim and ROS 2 |

**Asset paths**: USD stage files live in `~/Downloads/XR_Content_NVD@10010/Assets/XR/Stages/` by default. Override with `ROBOT_CAPSTONE_XR_CONTENT_ROOT`. Downloaded GLBs default to `~/Downloads`; override with `ROBOT_CAPSTONE_DOWNLOADS_DIR`.

**Camera extrinsics** (`config/camera_extrinsics.yaml`): extrinsics are captured at the robot's start pose. If the start pose changes, regenerate from Isaac Sim Script Editor by dumping `panda_link0`, `EEViewCamera`, and `TopViewCamera` world transforms, then converting from USD world into `panda_link0` frame. The conversion procedure is documented at the top of the YAML file.
