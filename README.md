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
| **YOLO v11 Tracking** | Continuously tracks the target and monitors for dynamic hazards (e.g., human hands) |
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
| **AI Inference Node** | 16× RTX 4090 Cluster | LLM, Grounding DINO, Qwen VLM API |
| **Development Node** | RTX 4070 Super (12 GB) | YOLO Tracking, RViz Visualization |

---

## 4. Logic Implementation

The following snippet demonstrates the state transition from target identification to high-speed reflexive tracking:

```python
def fast_brain_loop(self):
    """
    Main loop for reflexive tracking and dynamic obstacle avoidance.
    Fuses Top-view and End-effector data for sub-centimeter precision.
    """
    while rclpy.ok():
        # Get latest detections from YOLO
        detections = self.yolo_node.get_latest_frame()

        for item in detections:
            # Map detection to 3D world coordinates via depth frame
            point_3d = self.transform_to_world(item.centroid, self.depth_buffer)

            if item.label == 'target':
                # Update MoveIt Planning Scene
                self.moveit.update_target(point_3d)
            elif item.label in self.hazard_list:
                # Inject dynamic collision object with buffer zone
                self.moveit.add_dynamic_obstacle(point_3d, radius=0.15)

        self.rate.sleep()
```

---

## 5. Dependencies & External Models

| Dependency | Purpose |
|---|---|
| [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) | Simulation environment |
| [MoveIt 2](https://moveit.ros.org/) | Motion planning |
| [Qwen-VL](https://github.com/QwenLM/Qwen-VL) | Vision-Language foundation model |
| [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) | Spatial grounding |
| [YOLOv11](https://github.com/ultralytics/ultralytics) | Object detection & tracking |

### Expected File Paths

- Isaac Sim is expected at `robot_capstone/isaacsim`
- Scene scripts are expected at `robot_capstone/sim`
- Runtime stages are copied into `~/Downloads/XR_Content_NVD@10010/Assets/XR/Stages` unless `ROBOT_CAPSTONE_XR_CONTENT_ROOT` is set
- Downloaded GLB assets are read from `~/Downloads` unless `ROBOT_CAPSTONE_DOWNLOADS_DIR` is set

### Launch

```bash
./run_capstone_scene.sh
```

or equivalently:

```bash
./sim/run_capstone_scene.sh
```
---

## 7. Contributors

| Name | Role |
|---|---|
| **Chanwon Jeong** | Perception Pipeline & VLM Integration |
| **Sanghyun Park** | Simulation Environment & Sensor Fusion |
| **Jaewon Heo** | ROS 2 Architecture & MoveIt Motion Planning |
