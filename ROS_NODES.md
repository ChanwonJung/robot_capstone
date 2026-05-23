# ROS 2 Packages, Nodes & Topics

> **Workspace:** `ros_pkgs/src/` · **ROS Distro:** Jazzy

---

## Package Overview

| Package | Role |
|---|---|
| `grounded_sam_pkg` | Visual grounding — GroundingDINO + SAM inference |
| `qwen_pkg` | Real Qwen VLM bridge — classifies detections, extracts grounding result |
| `yolo_hazard_pkg` | Fast Brain hazard detector — YOLO on EE + Top cameras at >30 FPS |
| `mask_projection_pkg` | 2D mask + depth → labeled 3D PointCloud2 |
| `target_pose_bridge_pkg` | `/world_map_result` → MoveIt goal poses |
| `moveit_isaac_bridge_pkg` | MoveIt planning + Isaac Sim joint bridge |

---

## grounded_sam_pkg

### `grounded_sam_node`

Runs GroundingDINO + SAM on EE camera (and optionally Top camera in dual-view mode).

| Direction | Topic | Type |
|---|---|---|
| SUB | `/ee_camera/image_raw` | `sensor_msgs/Image` |
| SUB | `/camera/image_raw` *(top, dual-view only)* | `sensor_msgs/Image` |
| PUB | `/grounded_sam/annotated_image` | `sensor_msgs/Image` |
| PUB | `/grounded_sam/mask_image` | `sensor_msgs/Image` (mono8, 1-based idx) |
| PUB | `/grounded_sam/detections_json` | `std_msgs/String` (JSON array) |
| PUB | `/top/grounded_sam/annotated_image` *(dual-view only)* | `sensor_msgs/Image` |
| PUB | `/top/grounded_sam/mask_image` *(dual-view only)* | `sensor_msgs/Image` |
| PUB | `/top/grounded_sam/detections_json` *(dual-view only)* | `std_msgs/String` |

**Launch:** `grounded_sam_dual.launch.py` (dual-view), `grounded_sam_ee.launch.py` (EE only)

---

### `qwen_stub_node`

Placeholder for the real Qwen VLM. Uses a hardcoded `LABEL_TO_CATEGORY` dict.

| Direction | Topic | Type |
|---|---|---|
| SUB | `/grounded_sam/detections_json` | `std_msgs/String` |
| SUB | `/grounded_sam/mask_image` | `sensor_msgs/Image` |
| PUB | `/qwen/labeled_detections` | `std_msgs/String` (JSON + `"category"` field) |
| PUB | `/qwen/mask_image` | `sensor_msgs/Image` (pass-through) |

---

### `test_image_pub`

Test utility — publishes a static image to simulate a camera feed.

| Direction | Topic | Type |
|---|---|---|
| PUB | `/camera/image_raw` | `sensor_msgs/Image` |

---

## qwen_pkg

### `instruction_prompt_node`

Reads natural-language commands from stdin and publishes them.

| Direction | Topic | Type |
|---|---|---|
| PUB | `/user_instruction` | `std_msgs/String` |

---

### `qwen_bridge_node`

Calls the real Qwen VLM endpoint. Classifies detections and extracts a structured grounding result (target + destination relation). Triggered by `/grounded_sam/detections_json`.

| Direction | Topic | Type |
|---|---|---|
| SUB | `/grounded_sam/detections_json` *(trigger)* | `std_msgs/String` |
| SUB | `/grounded_sam/mask_image` | `sensor_msgs/Image` |
| SUB | `/user_instruction` | `std_msgs/String` |
| PUB | `/qwen/labeled_detections` | `std_msgs/String` (JSON + `"category"` field) |
| PUB | `/qwen/grounding_result` | `std_msgs/String` (`GroundingResult` JSON) |
| PUB | `/qwen/mask_image` | `sensor_msgs/Image` (pass-through, published **last** to trigger projector) |

**Parameters:** `vllm_endpoint_url` (default `http://localhost:8000/v1`), `model_name`

---

## yolo_hazard_pkg

### `yolo_hazard_node`

YOLO-based hazard detector running on both cameras. Part of the Fast Brain — runs at >30 FPS.

| Direction | Topic | Type | Notes |
|---|---|---|---|
| SUB | `/camera/image_raw` | `sensor_msgs/Image` | Top-view RGB |
| SUB | `/ee_camera/image_raw` | `sensor_msgs/Image` | Eye-in-hand RGB |
| PUB | `/yolo_hazard/top/detections_json` | `std_msgs/String` | JSON detection payload |
| PUB | `/yolo_hazard/ee/detections_json` | `std_msgs/String` | JSON detection payload |
| PUB | `/yolo_hazard/top/annotated_image` *(optional)* | `sensor_msgs/Image` | Only when `publish_annotated: true` |
| PUB | `/yolo_hazard/ee/annotated_image` *(optional)* | `sensor_msgs/Image` | Only when `publish_annotated: true` |

---

## mask_projection_pkg

### `multi_view_projector_node`

Fuses EE + Top depth streams into a single labeled world-frame point cloud. Trigger-based — fires on each incoming mask.

| Direction | Topic | Type | Notes |
|---|---|---|---|
| SUB | `/qwen/mask_image` *(trigger)* | `sensor_msgs/Image` | Configurable via `mask_topic` |
| SUB | `/qwen/labeled_detections` | `std_msgs/String` | Configurable via `detections_topic` |
| SUB | `/ee_camera/depth_image` | `sensor_msgs/Image` (32FC1) | Configurable |
| SUB | `/ee_camera/camera_info` | `sensor_msgs/CameraInfo` | Configurable |
| SUB | `/top_camera/depth_image` | `sensor_msgs/Image` (32FC1) | Configurable |
| SUB | `/top_camera/camera_info` | `sensor_msgs/CameraInfo` | Configurable |
| PUB | `/world_map` | `sensor_msgs/PointCloud2` | XYZRGB + category label |
| PUB | `/world_map_result` | `std_msgs/String` | JSON: centroid + bbox per category |

**Isaac Sim overrides:**
```
top_depth_topic:=/isaac/top/depth_image
ee_depth_topic:=/isaac/ee/depth_image
```

**Category values in point cloud:** `FREE=0`, `TARGET=1`, `WORKSPACE=2`, `OBSTACLE=3`, `UNKNOWN=4`

**Launch:** `multi_view_projector.launch.py`

---

### `mask_projector_node` *(single-view variant)*

| Direction | Topic | Type |
|---|---|---|
| SUB | `/rgbd_camera/depth_image` *(trigger via mask)* | `sensor_msgs/Image` |
| SUB | `/rgbd_camera/camera_info` | `sensor_msgs/CameraInfo` |
| SUB | `/grounded_sam/mask_image` *(trigger)* | `sensor_msgs/Image` |
| SUB | `/grounded_sam/detections_json` | `std_msgs/String` |
| PUB | `/labeled_points` | `sensor_msgs/PointCloud2` |
| PUB | `/projection_result` | `std_msgs/String` (JSON centroid summary) |

---

## target_pose_bridge_pkg

### `target_pose_bridge_node`

Converts `/world_map_result` JSON into MoveIt-ready pose stamped messages.

| Direction | Topic | Type |
|---|---|---|
| SUB | `/world_map_result` *(trigger)* | `std_msgs/String` |
| PUB | `/pre_grasp_target_pose` | `geometry_msgs/PoseStamped` |
| PUB | `/grasp_target_pose` | `geometry_msgs/PoseStamped` |

---

## moveit_isaac_bridge_pkg

### `target_pose_executor_node`

Subscribes to grasp poses and calls MoveIt via action.

| Direction | Topic / Action | Type |
|---|---|---|
| SUB | `/grasp_target_pose` *(trigger)* | `geometry_msgs/PoseStamped` |
| SUB | `/target_pose_executor/reset` | `std_msgs/Empty` |
| ACTION CLIENT | `/move_action` | `moveit_msgs/MoveGroup` |

---

### `joint_trajectory_bridge_node`

Bridges MoveIt trajectory execution to Isaac Sim joint commands.

| Direction | Topic / Action | Type |
|---|---|---|
| SUB | `/joint_states` | `sensor_msgs/JointState` |
| PUB | `/joint_command` | `sensor_msgs/JointState` |
| ACTION SERVER | `/panda_arm_controller/follow_joint_trajectory` | `control_msgs/FollowJointTrajectory` |

**Launch:** `capstone_pick_pipeline.launch.py` (composes target_pose_bridge + MoveIt + RViz)

---

## Full Data Flow

```
/ee_camera/image_raw ──► grounded_sam_node ──► /grounded_sam/detections_json ──► qwen_bridge_node (or qwen_stub_node)
                                           ──► /grounded_sam/mask_image      ──►
/camera/image_raw    ──► (top view, cached)     /user_instruction             ──►

                                                    qwen_bridge_node ──► /qwen/labeled_detections ──► multi_view_projector_node
                                                                     ──► /qwen/grounding_result
                                                                     ──► /qwen/mask_image (trigger) ──►

/ee_camera/depth_image ──► multi_view_projector_node ──► /world_map
/top_camera/depth_image ──►                          ──► /world_map_result ──► target_pose_bridge_node

                                                                               target_pose_bridge_node ──► /pre_grasp_target_pose
                                                                                                       ──► /grasp_target_pose ──► target_pose_executor_node

                                                                                                           target_pose_executor_node ──► [/move_action] ──► MoveIt

MoveIt ──► [/panda_arm_controller/follow_joint_trajectory] ──► joint_trajectory_bridge_node ──► /joint_command ──► Isaac Sim
```
