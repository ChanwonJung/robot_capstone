# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Specifications

Ubuntu 24.04 · ROS Jazzy · IsaacSim 5.1.0

## End Goal
Language-directed robotic manipulator that interprets ambiguous natural language commands, visually grounds targets in cluttered scenes, and executes safe grasps while avoiding dynamic obstacles. Validated within NVIDIA Isaac Sim.

## Architecture: Slow Brain / Fast Brain

**Slow Brain** runs once per command. An LLM parses the user's input into noun candidates, a grounding model generates labeled bounding boxes from the top-view RGB-D feed, and a VLM selects the correct target from the annotated image.

**Fast Brain** runs at >30 FPS. A YOLO-based tracker locks onto the target and detects hazards from both cameras simultaneously. Detected hazards are injected as dynamic collision objects into the MoveIt planning scene, which uses hybrid planning to separate the long-range trajectory from low-latency local reactions.

## Hardware
Three-node distributed setup over Tailscale. The simulation node runs Isaac Sim, ROS 2, and MoveIt. A remote GPU cluster handles all heavy Slow Brain inference via FastAPI. The development node runs the YOLO tracking loop and RViz. Each local node is constrained to 12GB VRAM — anything heavier offloads to the cluster.

## Key Constraints
- Avoidance loop must sustain >30 FPS
- Heavy inference (LLM, grounding, VLM) always offloads to the remote cluster
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
cd ros_pkgs
colcon build --symlink-install
source install/setup.bash
```

Build a single package:
```bash
cd ros_pkgs
colcon build --symlink-install --packages-select grounded_sam_pkg
```

### Run tests
```bash
cd ros_pkgs
colcon test --packages-select grounded_sam_pkg
colcon test-result --verbose
```

Single test file:
```bash
python3 -m pytest ros_pkgs/src/grounded_sam_pkg/test/test_flake8.py -v
```

---

## Launch

### Isaac Sim scene
```bash
./run_capstone_scene.sh   # delegates to sim/run_capstone_scene.sh
```

### Full Slow Brain perception pipeline (Isaac Sim)
```bash
# Terminal 1 — GSAM (dual-view: EE + Top cameras)
source launch_env.bash
ros2 launch grounded_sam_pkg grounded_sam_dual.launch.py

# Terminal 2 — Qwen stub (label-based category assignment until real VLM is wired)
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
This composes `target_pose_bridge_pkg` (converts `/world_map_result` → `/pre_grasp_target_pose` + `/grasp_target_pose`) with `moveit_isaac_bridge_pkg` (MoveIt + RViz).

---

## ROS Package Architecture

```
ros_pkgs/src/
├── grounded_sam_pkg/       Slow Brain perception — GroundingDINO + SAM
├── mask_projection_pkg/    2D mask + depth → labeled 3D PointCloud2
├── target_pose_bridge_pkg/ /world_map_result → MoveIt goal poses
├── moveit_isaac_bridge_pkg/MoveIt + Isaac Sim joint bridge
└── behavior_tree/          BehaviorTree.ROS2 (vendored)
```

### Data flow (Slow Brain)

```
Camera RGB   →  grounded_sam_node  →  /grounded_sam/detections_json (JSON array)
                                   →  /grounded_sam/mask_image      (mono8, 1-based idx)
                                   →  /grounded_sam/annotated_image

/grounded_sam/detections_json  →  qwen_stub_node (or real Qwen)
/grounded_sam/mask_image       →                 →  /qwen/labeled_detections (adds "category" field)
                                                 →  /qwen/mask_image (pass-through)

EE depth + CameraInfo         →  multi_view_projector_node
Top depth + CameraInfo        →                          →  /world_map        (PointCloud2)
/qwen/mask_image (trigger)    →                          →  /world_map_result (JSON: centroid + bbox per category)

/world_map_result  →  target_pose_bridge_node  →  /pre_grasp_target_pose (PoseStamped)
                                               →  /grasp_target_pose      (PoseStamped)

/pre_grasp_target_pose  →  MoveIt              →  /joint_command (JointState)
/grasp_target_pose      →                      →  → Isaac Sim via joint_trajectory_bridge_node
```

### Key design decisions to know

**No timestamp synchronization** — GSAM inference takes 30–40 s on CPU; by the time `mask_image` arrives the depth queue has advanced far past the matching frame. `multi_view_projector_node` caches the latest depth/camera_info and treats each incoming mask as the trigger, rather than using `ApproximateTimeSynchronizer`.

**Dual-view with one model instance** — `grounded_sam_dual.launch.py` runs a single `GroundedSAMNode` that subscribes to both EE and Top cameras. Top images are cached; the EE callback is the trigger that drives both views sequentially through the same pipeline instance.

**Qwen stub** — `qwen_stub_node` is the placeholder for the real Qwen VLM. It uses a hardcoded `LABEL_TO_CATEGORY` dict. The downstream topic names (`/qwen/mask_image`, `/qwen/labeled_detections`) and JSON schema are stable — replacing the stub with the real VLM requires no changes to `mask_projection_pkg`.

**Point cloud categories** — FREE=0 (EE non-detections), TARGET=1, WORKSPACE=2, OBSTACLE=3, UNKNOWN=4 (all Top-view points). When feeding octomap, use only OBSTACLE + UNKNOWN; FREE points cause background to be marked occupied.

**`projection_engine.py` is ROS-free** — all numpy projection/filter math lives there. `multi_view_projector_node.py` only handles ROS message decode/encode. Keep it that way.

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

### Model weights (not in git)

Place under `models/g-sam/` (paths configured in `ros_pkgs/src/grounded_sam_pkg/config/model_paths.yaml`):
- `groundingdino_swint_ogc.pth` (~662 MB)
- `sam_vit_b_01ec64.pth` (~375 MB)

### Isaac Sim bridge scripts (`sim/`)

| File | Role |
|---|---|
| `setup_initial_scene.py` | Loads USD stage, places objects (apple, glass, cube, book), attaches cameras |
| `isaac_ros_camera_bridge.py` | Wires Isaac Sim cameras to ROS 2 image/depth topics |
| `isaac_ros_joint_bridge.py` | Wires Panda joint states/commands between Isaac Sim and ROS 2 |

**Asset paths**: USD stage files live in `~/Downloads/XR_Content_NVD@10010/Assets/XR/Stages/` by default. Override with `ROBOT_CAPSTONE_XR_CONTENT_ROOT`. Downloaded GLBs default to `~/Downloads`; override with `ROBOT_CAPSTONE_DOWNLOADS_DIR`.

### Qwen VLM API (`models/qwen3.5/qwenapi.py`)

FastAPI wrapper for a remote vLLM OpenAI-compatible endpoint. `MODEL_NAME` and `VLLM_BASE_URL` are placeholders — fill them in before deploying to the cluster. Uses `guided_json` for structured JSON output.
