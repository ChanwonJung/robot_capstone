# robot_capstone

 Gemini said
Capstone Project: Language-Directed Manipulator with Dynamic Tracking and Obstacle Avoidance

This repository hosts the implementation of an autonomous robotic manipulation system. The project integrates Vision-Language Models (VLM) for semantic reasoning and High-Frequency Reflexive Control for real-time execution and safety.
1. Project Objective

The system is designed to interpret high-level, ambiguous natural language commands (e.g., "Remove the hazardous object") and translate them into precise robotic actions. By employing a "Slow Brain / Fast Brain" architecture, the system maintains complex reasoning capabilities without sacrificing the low-latency requirements of dynamic obstacle avoidance.

Scope: Developed and validated within NVIDIA Isaac Sim using a dual-camera RGB-D configuration.
2. System Architecture

The pipeline is bifurcated to optimize computational resources and response time:
Phase 1: The Slow Brain (Reasoning & Grounding)

Handles high-level cognition and target identification. This phase runs once per user command.

    LLM Parser: Extracts target-related nouns from the prompt.

    Grounding SAM: Identifies potential bounding boxes in the Top-View RGB-D feed.

    Qwen VLM: Analyzes the annotated image and the original prompt to select the correct target index.

Phase 2: The Fast Brain (Tracking & Reflexive Control)

Maintains a >30 FPS control loop for execution and safety.

    YOLO v11 Tracking: Continuously tracks the target and monitors for dynamic hazards (e.g., human hands).

    MoveIt Hybrid Planning: Utilizes a global planner for the initial trajectory and a local planner for real-time adjustments based on "danger factors" detected in the workspace.

3. Perception & Kinematics

The system utilizes synchronized Top-View and End-Effector RGB-D streams. Given a pixel coordinate (u,v) with a corresponding depth value Z, and camera intrinsics (fx​,fy​,cx​,cy​), the 3D position in the camera frame is calculated as:
x=fx​(u−cx​)Z​
y=fy​(v−cy​)Z​
z=Z

The Euclidean distance d to the target centroid is derived as:
d=(fx​(u−cx​)Z​)2+(fy​(v−cy​)Z​)2+Z2
