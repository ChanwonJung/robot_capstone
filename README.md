# robot_capstone

Language-Directed Manipulator with Dynamic Tracking and Obstacle Avoidance

This repository contains the implementation of an intelligent robotic manipulation system developed for a university capstone project. The system integrates High-Level Reasoning (VLMs) with Low-Level Reflexive Control (YOLO/MoveIt) to execute language-directed tasks in a dynamic, simulated environment.
1. Project Objective

The goal is to bridge the gap between semantic understanding and real-time physical execution. The system processes natural language commands, identifies targets in a cluttered scene using a dual-camera setup (Top-view and End-effector), and performs safe grasping with continuous obstacle avoidance.

Development Status: Completed through the high-fidelity simulation phase in NVIDIA Isaac Sim.
2. System Architecture: "Slow Brain / Fast Brain"

The pipeline bifurcates perception to optimize for both reasoning depth and execution latency:
Phase 1: The Slow Brain (Semantic & Spatial Grounding)

Triggered once per command to handle high-level logic and target identification.

    LLM Parser: Extracts target candidates from raw human prompts.

    Grounding SAM: Performs zero-shot detection on the Top-View RGB-D feed.

    Qwen VLM: Final decision-maker that selects the optimal target box based on semantic context.

Phase 2: The Fast Brain (Reflexive Tracking & Safety)

Runs in a continuous high-frequency loop (>30 FPS) for motion execution.

    YOLO Tracking: Locks onto the target and monitors for dynamic hazards (e.g., human hands).

    MoveIt Hybrid Planning: Combines global trajectory generation with local, low-latency collision avoidance.

3. Perception & Kinematics

The system utilizes dual RGB-D streams to generate point clouds and calculate precise spatial coordinates. For a pixel (u,v) with depth Z and camera intrinsics (fx​,fy​,cx​,cy​), the 3D distance d in the camera frame is calculated as:
d=(fx​(u−cx​)Z​)2+(fy​(v−cy​)Z​)2+Z2​

This depth data is projected into the world frame and published to the MoveIt Planning Scene as dynamic collision objects.
4. Distributed Hardware Resources

We utilize a distributed ROS 2 network over Tailscale to manage computational load:
Node Type	Primary Hardware	Responsibilities
Simulation Node	RTX 5070 (12GB)	Isaac Sim, ROS 2 Core, MoveIt Physics
AI Inference Node	16x RTX 4090 Cluster	LLM, Grounding DINO, Qwen VLM API
Development Node	RTX 4070 Super (12GB)	YOLO Tracking, Rviz Visualization, Node Handoffs
5. Implementation Details
Python

# Logic Snippet: Fast Brain Target Update Loop
def target_tracking_callback(self, yolo_data, depth_frame):
    # 1. Extract 2D Centroid from YOLO
    u, v = self.get_centroid(yolo_data)
    
    # 2. Get depth and transform to 3D World Coordinates
    target_3d = self.project_2d_to_3d(u, v, depth_frame)
    
    # 3. Update MoveIt Planning Scene if delta exceeds threshold
    if self.euclidean_dist(target_3d, self.prev_target) > self.threshold:
        self.moveit_interface.update_target(target_3d)
        self.moveit_interface.replan()

6. Dependencies & Model Links

    Simulation: NVIDIA Isaac Sim

    Planning: MoveIt 2 (Humble/Iron)

    Perception:

        Grounding SAM (Grounded-Segment-Anything)

        Qwen-VL (Vision-Language Model)

        YOLOv11 (Ultralytics)

7. Contributors

    Chanwon Jeong – Perception Pipeline & VLM Integration

    Sanghyun Park – Simulation Environment & Sensor Fusion

    Jaewon Heo – ROS 2 Architecture & MoveIt Motion Planning
