# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Specifications

Ubuntu 24.04 · ROS Jazzy · IsaacSim 5.1.0

## End Goal
Language-directed robotic manipulator that interprets ambiguous natural language commands, visually grounds targets in cluttered scenes, and executes safe grasps while avoiding dynamic obstacles. Validated within NVIDIA Isaac Sim.

## Architecture: Slow Brain / Fast Brain

**Slow Brain** runs once per command. The user types a natural-language instruction; a grounding model (GSAM) generates labeled bounding boxes + segmentation masks from both cameras; a VLM (Qwen) selects the target and destination from the annotated detections; a projection node fuses both depth streams into a labeled 3D point cloud.

**Fast Brain** runs at >30 FPS. A YOLO-based detector (`yolo_hazard_pkg`) monitors both cameras for hazards simultaneously. Detected hazards are injected as dynamic collision objects into the MoveIt planning scene, which uses hybrid planning for long-range trajectory + low-latency local reactions.

## Hardware
Three-node distributed setup over Tailscale. The simulation node runs Isaac Sim, ROS 2, and MoveIt. A remote GPU cluster handles all heavy Slow Brain inference via vLLM (OpenAI-compatible API). The development node runs the YOLO tracking loop and RViz. Each local node is constrained to 12GB VRAM — anything heavier offloads to the cluster.

## Key Constraints
- Avoidance loop must sustain >30 FPS
- Heavy inference (GSAM, Qwen VLM) always offloads to the remote cluster
- Current scope is Isaac Sim validation only; sim-to-real is a future goal
- ROS 2 Jazzy, Python 3.10+, C++20

---

## Build & Environment

### Initial setup (once)
```bash
# 1. Create and populate the venv
python3 -m venv gsam_venv
source gsam_venv/bin/activate
pip install -r ros_pkgs/src/grounded_sam_pkg/requirements.txt

# 2. Download model weights
mkdir -p models/g-sam
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
     -O models/g-sam/groundingdino_swint_ogc.pth
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
     -O models/g-sam/sam_vit_b_01ec64.pth
```

### Per-session environment
```bash
# Always run this from the repo root before any ROS 2 commands
source launch_env.bash
```
`launch_env.bash` sources `/opt/ros/jazzy/setup.bash`, the workspace install overlay at `ros_pkgs/install/setup.bash`, and injects `gsam_venv/lib/python3.12/site-packages` into `PYTHONPATH` so torch/groundingdino are visible alongside ROS 2 Python.

### Build ROS packages
```bash
cd ros_pkgs && colcon build --symlink-install && source install/setup.bash
```

Single package:
```bash
cd ros_pkgs && colcon build --symlink-install --packages-select grounded_sam_pkg
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
```

### Full motion pipeline
```bash
source launch_env.bash
ros2 launch moveit_isaac_bridge_pkg capstone_pick_pipeline.launch.py
```
Composes `target_pose_bridge_pkg` (`/world_map_result` → `/pre_grasp_target_pose` + `/grasp_target_pose`) with `moveit_isaac_bridge_pkg` (MoveIt + RViz).

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
├── target_pose_bridge_pkg/ /world_map_result → MoveIt goal poses
├── moveit_isaac_bridge_pkg/MoveIt + Isaac Sim joint bridge
└── behavior_tree/          BehaviorTree.ROS2 (vendored)
```

### Full data flow (Slow Brain)

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

/ee_camera/depth_image   →  multi_view_projector_node  →  /world_map        (PointCloud2)
/top_camera/depth_image  →                             →  /world_map_result (JSON centroid+bbox)
/qwen/mask_image (trigger) →

/world_map_result  →  target_pose_bridge_node  →  /pre_grasp_target_pose (PoseStamped)
                                               →  /grasp_target_pose      (PoseStamped)

/grasp_target_pose  →  target_pose_executor_node  →  [/move_action]  →  MoveIt
MoveIt  →  [/panda_arm_controller/follow_joint_trajectory]  →  joint_trajectory_bridge_node
        →  /joint_command  →  Isaac Sim
```

### Key design decisions

**No timestamp synchronization** — GSAM inference takes 30–40 s on CPU; by the time `mask_image` arrives the depth queue has advanced far past the matching frame. `multi_view_projector_node` caches the latest depth/camera_info and treats each incoming mask as the trigger, rather than using `ApproximateTimeSynchronizer`.

**Qwen publish order is load-bearing** — `qwen_bridge_node` always publishes `/qwen/labeled_detections` before `/qwen/mask_image`. The mask is the trigger for `multi_view_projector_node`, so labels must already be in cache when it fires. Never reorder these publishes.

**Dual-view with one model instance** — `grounded_sam_dual.launch.py` runs a single `GroundedSAMNode` subscribed to both EE and Top cameras. Top images are cached; the EE callback drives both views sequentially through the same pipeline.

**Stub vs real Qwen** — `qwen_stub_node` (in `grounded_sam_pkg`) uses a hardcoded `LABEL_TO_CATEGORY` dict and requires no cluster. `qwen_bridge_node` (in `qwen_pkg`) calls the real VLM. Both publish identical topic schemas (`/qwen/labeled_detections`, `/qwen/mask_image`) — `mask_projection_pkg` requires no changes to swap between them.

**`projection_engine.py` is ROS-free** — all numpy projection/filter math lives there. `multi_view_projector_node.py` only handles ROS message decode/encode. Keep it that way.

**Point cloud categories** — `FREE=0` (EE non-detections), `TARGET=1`, `DESTINATION=2`, `OBSTACLE=3`, `UNKNOWN=4` (all Top-view points). When feeding octomap, use only `OBSTACLE + UNKNOWN`; `FREE` points cause background to be marked occupied.

---

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

Two-pass filtering in `projection_engine.py`: Pass 1 removes Top UNKNOWN points within 1.5 cm XY + 10 cm Z of EE-segmented points (avoids double-counting). Pass 2 removes EE FREE points that overlap Top UNKNOWN (UNKNOWN > FREE priority).

### Model weights (not in git)

Place under `models/g-sam/` (paths in `ros_pkgs/src/grounded_sam_pkg/config/model_paths.yaml`):
- `groundingdino_swint_ogc.pth` (~662 MB)
- `sam_vit_b_01ec64.pth` (~375 MB)

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
