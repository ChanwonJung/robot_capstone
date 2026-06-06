# Capstone Project: Language-Directed Manipulator with Dynamic Tracking and Obstacle Avoidance

This repository hosts the implementation of an autonomous robotic manipulation system. The project integrates **Vision-Language Models (VLM)** for semantic reasoning and **High-Frequency Reflexive Control** for real-time execution and safety.

---

## 1. Project Objective

The system is designed to interpret high-level, ambiguous natural language commands (e.g., *"Remove the hazardous object"*) and translate them into precise robotic actions. By employing a **"Slow Brain / Fast Brain"** architecture, the system maintains complex reasoning capabilities without sacrificing the low-latency requirements of dynamic obstacle avoidance.

> **Scope:** Developed and validated within NVIDIA Isaac Sim using a dual-camera RGB-D configuration.

---

## 2. System Architecture

The pipeline is bifurcated to optimize computational resources and response time:

### Phase 1 — The Slow Brain (Reasoning & Grounding)

Handles high-level cognition and target identification. This phase runs **once per user command**.

| Component | Role |
|---|---|
| **LLM Parser** | Extracts target-related nouns from the prompt |
| **Grounding SAM** | Identifies potential bounding boxes in the Top-View RGB-D feed |
| **Qwen VLM** | Analyzes the annotated image and original prompt to select the correct target index |

### Phase 2 — The Fast Brain (Tracking & Reflexive Control)

Maintains a **>30 FPS** control loop for execution and safety.

| Component | Role |
|---|---|
| **YOLO26 Segmentation** | Continuously tracks the target and monitors for dynamic hazards (e.g., human hands) |
| **MoveIt Hybrid Planning** | Global planner for initial trajectory + local planner for real-time adjustments based on detected "danger factors" |

```text
robot_capstone/
├── sim/                  # Project-owned Isaac Sim scripts and scene setup entrypoints
│   ├── assets/           # USD assets converted from downloaded GLBs
│   ├── scenes/           # USD stage files synced into XR_Content before launch
│   ├── run_capstone_scene.sh
│   ├── run_import_assets.sh
│   └── setup_initial_scene.py
├── isaacsim/             # Local Isaac Sim installation used by the launch scripts
├── models/               # Model weights and configs
│   ├── fast_brain/       # YOLO model for real-time tracking and hazard detection
│   └── slow_brain/       # LLM + grounding models for parsing and visual grounding
└── ros_pkgs/             # ROS 2 packages and planning integration
```

## 3. Distributed Hardware Configuration

To ensure simulation stability, the system uses a distributed ROS 2 network over **Tailscale**:

| Node | Hardware | Primary Role |
|---|---|---|
| **Simulation Node** | RTX 5070 (12 GB) | Isaac Sim, ROS 2 Orchestration, MoveIt |
| **AI Inference Node** | 16× RTX 3090  | LLM, Grounding DINO, Qwen VLM API, GraspGen API |
| **Development Node** | RTX 4070 Super (12 GB) | YOLO Tracking, RViz Visualization |

---

## 4. Dependencies & External Models

| Dependency | Purpose |
|---|---|
| [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) | Simulation environment |
| [MoveIt 2](https://moveit.ros.org/) | Motion planning |
| [Qwen-VL](https://github.com/QwenLM/Qwen-VL) | Vision-Language foundation model |
| [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) | Spatial grounding |
| [YOLO26 (Segmentation)](https://github.com/ultralytics/ultralytics) | Object detection & tracking |

### Expected File Paths

- Isaac Sim is expected at `robot_capstone/isaacsim`
- Scene scripts are expected at `robot_capstone/sim`
- Runtime stages are copied into `~/Downloads/XR_Content_NVD@10010/Assets/XR/Stages` unless `ROBOT_CAPSTONE_XR_CONTENT_ROOT` is set
- Downloaded GLB assets are read from `~/Downloads` unless `ROBOT_CAPSTONE_DOWNLOADS_DIR` is set

---

## 5. How to Run — Slow Brain Motion Validation

> Snapshot of the validated run sequence at the time of this commit. Update when the launch graph changes.

### Prerequisites (one-time)

- ROS 2 workspace built: `cd ros_pkgs && colcon build --symlink-install`
- `gsam_venv` prepared at the repo root with Grounded-SAM dependencies (`torch`, `segment-anything`, etc.)
- ROS 2 + workspace sourced in every terminal:
  ```bash
  source /opt/ros/jazzy/setup.bash
  source ros_pkgs/install/setup.bash
  ```
- **After every rebuild of `grounded_sam_pkg`**, patch entry-script shebangs back to the venv Python (colcon resets them to system Python, which lacks `torch`):
  ```bash
  VENV_PY="$PWD/gsam_venv/bin/python"
  for f in $(find ros_pkgs/install/grounded_sam_pkg/lib -maxdepth 3 -type f -executable); do
    head -1 "$f" | grep -q "^#!/usr/bin/python3$" && \
      sed -i "1s|^#!/usr/bin/python3$|#!${VENV_PY}|" "$f"
  done
  ```

### Run sequence (each command in its own terminal)

1. **T1 — MoveIt hybrid planner** (venv OFF)
   ```bash
   source ~/robot_capstone/launch_env.bash
   ros2 launch moveit_isaac_bridge_pkg hybrid_planning.launch.py
   ```

2. **T2 — Static TF (`panda_link0 = world`)** (venv OFF)
   ```bash
   source ~/robot_capstone/launch_env.bash
   ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 panda_link0 world
   ```

3. **T3 — Grounded-SAM** (venv ON)
   ```bash
   source ~/robot_capstone/launch_env.bash
   source ~/robot_capstone/gsam_venv/bin/activate
   ros2 launch grounded_sam_pkg grounded_sam_dual.launch.py prompt:="book"
   ```

4. **T4 — Qwen stub labeling** (venv OFF)
   ```bash
   source ~/robot_capstone/launch_env.bash
   ros2 run grounded_sam_pkg qwen_stub_node
   ```

5. **T5 — Mask projector** (venv ON)
   ```bash
   source ~/robot_capstone/launch_env.bash
   source ~/robot_capstone/gsam_venv/bin/activate
   ros2 launch mask_projection_pkg multi_view_projector.launch.py \
     ee_depth_topic:=/ee_rgbd_camera/depth_image \
     ee_camera_info_topic:=/ee_rgbd_camera/camera_info \
     top_depth_topic:=/rgbd_camera/depth_image \
     top_camera_info_topic:=/rgbd_camera/camera_info \
     freeze_after_first_publish:=true
   ```

6. **T6 — GraspGen client** (venv OFF)
   ```bash
   source ~/robot_capstone/launch_env.bash
   ros2 run graspgen_pkg graspgen_node --ros-args \
     --params-file $ROBOT_CAPSTONE_ROOT/ros_pkgs/install/graspgen_pkg/share/graspgen_pkg/config/graspgen_params.yaml \
     -p extrinsics_config:=$ROBOT_CAPSTONE_ROOT/ros_pkgs/src/mask_projection_pkg/config/camera_extrinsics.yaml \
     -p ee_depth_topic:=/ee_rgbd_camera/depth_image \
     -p ee_camera_info_topic:=/ee_rgbd_camera/camera_info \
     -p zmq_port:=8000 \
     -p top_down_angle_deg:=45.0 \
     -p ik_filter_enabled:=false \
     -p override_xy_with_bbox_center:=true \
     -p force_top_down_orientation:=true \
     -p freeze_after_first_publish:=true
   ```

7. **T7 — Behavior Tree** (venv OFF)
   ```bash
   source ~/robot_capstone/launch_env.bash
   ros2 run bt_pkg bt_executor_node --ros-args \
     -p extrinsics_config:=$ROBOT_CAPSTONE_ROOT/ros_pkgs/src/mask_projection_pkg/config/camera_extrinsics.yaml \
     -p bt_pick_retries:=1 \
     -p pre_grasp_z_offset:=0.08
   ```

### Re-arm for the next command

The executor latches after a successful motion (Slow Brain semantics: "1 command = 1 motion"). To accept a new target without restarting:

```bash
ros2 topic pub --once /target_pose_executor/reset std_msgs/msg/Empty "{}"
```

---

## 6. Contributors

| Name | Role |
|---|---|
| **Chanwon Jeong** | System Integration & MoveIt Motion Planning & Yolo |
| **Sanghyun Park** | G-SAM & GraspGen & Sensor Fusion |
| **Jaewon Heo** | QWEN & Behavior Tree |
