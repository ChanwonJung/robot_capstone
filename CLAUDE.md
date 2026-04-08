# Specifications

Ubuntu 24.04
ROS Jazzy
IsaacSim 5.1.0

## End Goal
Language-directed robotic manipulator that interprets ambiguous natural language commands, visually grounds targets in cluttered scenes, and executes safe grasps while avoiding dynamic obstacles. Validated within NVIDIA Isaac Sim.

## Architecture: Slow Brain / Fast Brain

**Slow Brain** runs once per command. An LLM parses the user's input into noun candidates, a grounding model generates labeled bounding boxes from the top-view RGB-D feed, and a VLM selects the correct target from the annotated image.

**Fast Brain** runs at >30 FPS. A YOLO-based tracker locks onto the target and detects hazards from both cameras simultaneously. Detected hazards are injected as dynamic collision objects into the MoveIt planning scene, which uses hybrid planning to separate the long-range trajectory from low-latency local reactions.

## Perception
Two synchronized RGB-D streams: a static top-view camera for global scene context, and an eye-in-hand camera on the end-effector for local precision. Both publish image and point cloud topics in Isaac Sim.

## Hardware
Three-node distributed setup over Tailscale. The simulation node runs Isaac Sim, ROS 2, and MoveIt. A remote GPU cluster handles all heavy Slow Brain inference via FastAPI. The development node runs the YOLO tracking loop and RViz. Each local node is constrained to 12GB VRAM — anything heavier offloads to the cluster.

## Key Constraints
- Avoidance loop must sustain >30 FPS
- Heavy inference (LLM, grounding, VLM) always offloads to the remote cluster
- Current scope is Isaac Sim validation only; sim-to-real is a future goal
- ROS 2 Jazzy, Python 3.10+, C++20
