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

```
robot_capstone/
├── sim/                  # Isaac Sim environment and scene setup
│   ├── assets/           # USD assets converted from downloaded GLBs
│   └── scenes/           # USD stage files loaded into Isaac Sim
├── models/               # Model weights and configs
│   ├── fast_brain/       # YOLO model for real-time tracking and hazard detection
│   └── slow_brain/       # LLM + grounding models for parsing and visual grounding
└── moveit/               # MoveIt 2 config and planning setup for the Franka arm
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

### Expected File Paths:

assumes isaacsim is installed as a folder on ~/isaacsim
assumes moveit is installed on ~~
---

## 7. Contributors

| Name | Role |
|---|---|
| **Chanwon Jeong** | Perception Pipeline & VLM Integration |
| **Sanghyun Park** | Simulation Environment & Sensor Fusion |
| **Jaewon Heo** | ROS 2 Architecture & MoveIt Motion Planning |
