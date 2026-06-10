# ROS 2 Packages, Nodes & Topics

> **Workspace:** `ros_pkgs/src/` · **ROS Distro:** Jazzy
>
> Topic names below are the node defaults; most are configurable via parameters
> and remapped per-launch (e.g. Isaac Sim depth topics, `/qwen/*` triggers).

---

## Package Overview

| Package | Role |
|---|---|
| `grounded_sam_pkg` | Slow Brain visual grounding — GroundingDINO + SAM inference |
| `qwen_pkg` | Slow Brain VLM — Qwen detection classification + grounding result |
| `mask_projection_pkg` | 2D mask + depth → labeled 3D PointCloud2 |
| `vgn_grasp_pkg` | Slow Brain grasp detection — TSDF + VGN inference |
| `yolo_hazard_pkg` | Fast Brain hazard detector — YOLO on EE + Top cameras at >30 FPS |
| `bt_pkg` | BehaviorTree.ROS2 pick-and-place executor + hazard/YOLO support nodes |
| `moveit_isaac_bridge_pkg` | MoveIt planning + Isaac Sim joint bridge + gripper server |
| `target_pose_bridge_pkg` | *(legacy)* `/world_map_result` → centroid-offset MoveIt goal poses |
| `graspgen_pkg` | *(legacy)* ZMQ GraspGen client — superseded by `vgn_grasp_pkg` |

---

## grounded_sam_pkg

### `grounded_sam_node`

Runs GroundingDINO + SAM. In dual-view mode (`grounded_sam_dual.launch.py`) a single node instance handles both cameras: Top images are cached, the EE callback drives both views sequentially.

| Direction | Topic | Type | Notes |
|---|---|---|---|
| SUB | `/camera/image_raw` | `sensor_msgs/Image` | EE trigger (`image_topic`; throttled by `process_every_n_frames`=30 + `min_process_interval_sec`=1.0) |
| SUB | *(top image topic)* | `sensor_msgs/Image` | Cached; dual-view only (`top_image_topic`, empty = disabled) |
| SUB | `/dino_prompt` | `std_msgs/String` | Cached; runtime prompt update |
| PUB | `/grounded_sam/annotated_image` | `sensor_msgs/Image` | bbox/mask overlay |
| PUB | `/grounded_sam/mask_image` | `sensor_msgs/Image` | mono8, 1-based detection index |
| PUB | `/grounded_sam/detections_json` | `std_msgs/String` | JSON array |
| PUB | `/top/grounded_sam/annotated_image` | `sensor_msgs/Image` | dual-view only |
| PUB | `/top/grounded_sam/mask_image` | `sensor_msgs/Image` | dual-view only |
| PUB | `/top/grounded_sam/detections_json` | `std_msgs/String` | dual-view only |

Key params: `model_config` (required), `prompt`, `max_bbox_area_ratio`=0.4 (drops oversized boxes), `top_min_depth`/`top_max_depth` (depth masking when `top_depth_topic` set).

**Launch:** `grounded_sam_dual.launch.py` (dual-view), `grounded_sam_ee.launch.py` (EE only), `grounded_sam.launch.py`

---

### `qwen_stub_node`

Placeholder for the real Qwen VLM — hardcoded `LABEL_TO_CATEGORY` dict, no cluster required. **Does not publish `/qwen/grounding_result`**, so it cannot drive `bt_executor_node`'s destination logic; sufficient for testing the pipeline up to projection.

| Direction | Topic | Type |
|---|---|---|
| SUB | `/grounded_sam/detections_json` | `std_msgs/String` |
| SUB | `/grounded_sam/mask_image` *(trigger)* | `sensor_msgs/Image` |
| PUB | `/qwen/labeled_detections` | `std_msgs/String` (JSON + `"category"` field) |
| PUB | `/qwen/mask_image` | `sensor_msgs/Image` (pass-through) |

---

### `instruction_parser_node`

Parses free-form `/instruction` text into GroundingDINO noun phrases via the Gemini API (`GEMINI_API_KEY` env var or `api_key` param).

| Direction | Topic | Type |
|---|---|---|
| SUB | `/instruction` *(trigger)* | `std_msgs/String` |
| PUB | `/dino_prompt` | `std_msgs/String` |

---

### `test_image_pub`

Test utility — publishes a static image to simulate a camera feed (path hardcoded in source).

| Direction | Topic | Type |
|---|---|---|
| PUB | `/camera/image_raw` | `sensor_msgs/Image` |

---

## qwen_pkg

### `qwen_bridge_node`

Calls the real Qwen VLM endpoint (vLLM, OpenAI-compatible). Classifies detections and extracts a structured grounding result. Inference runs on a daemon thread (~1–5 s). **Publish order is load-bearing:** `mask_image` is always published last because it triggers the projector.

| Direction | Topic | Type | Notes |
|---|---|---|---|
| SUB | `/grounded_sam/detections_json` *(trigger)* | `std_msgs/String` | |
| SUB | `/grounded_sam/mask_image` | `sensor_msgs/Image` | cached |
| SUB | `/user_instruction` | `std_msgs/String` | cached |
| PUB | `/qwen/labeled_detections` | `std_msgs/String` | JSON + `"category"` field |
| PUB | `/qwen/grounding_result` | `std_msgs/String` | `GroundingResult` JSON: target_id + destination spec |
| PUB | `/qwen/mask_image` | `sensor_msgs/Image` | pass-through, published **last** |

**Parameters:** `vllm_endpoint_url` (default `http://localhost:8000/v1`), `model_name` (default `qwen-vl`), `instruction` (optional seed)

**Launch:** `inst_input_qwen.launch.py` (opens xterm for instruction input)

---

### `instruction_prompt_node`

Reads natural-language commands from stdin (daemon thread) and publishes them.

| Direction | Topic | Type |
|---|---|---|
| PUB | `/user_instruction` | `std_msgs/String` |

---

## mask_projection_pkg

### `multi_view_projector_node`

Fuses EE + Top depth streams into a single labeled world-frame point cloud. Trigger-based — caches depth/camera_info, fires on each incoming mask (no timestamp sync; GSAM latency makes it impossible). In the full pipeline the mask/detections topics are remapped to `/qwen/mask_image` and `/qwen/labeled_detections`.

| Direction | Topic | Type | Notes |
|---|---|---|---|
| SUB | `/grounded_sam/mask_image` *(trigger)* | `sensor_msgs/Image` | `mask_topic` — remap to `/qwen/mask_image` in full pipeline |
| SUB | `/grounded_sam/detections_json` | `std_msgs/String` | `detections_topic` — remap to `/qwen/labeled_detections` |
| SUB | `/ee_camera/depth_image` | `sensor_msgs/Image` (32FC1) | cached, required |
| SUB | `/ee_camera/camera_info` | `sensor_msgs/CameraInfo` | cached, required |
| SUB | `/top_camera/depth_image` | `sensor_msgs/Image` (32FC1) | cached, optional |
| SUB | `/top_camera/camera_info` | `sensor_msgs/CameraInfo` | cached, optional |
| PUB | `/world_map` | `sensor_msgs/PointCloud2` | XYZRGB + category label (RViz-only downstream) |
| PUB | `/world_map_result` | `std_msgs/String` | JSON: centroid + 3D bbox per category |
| PUB | `/world_cloud_raw` | `sensor_msgs/PointCloud2` | geometry-only, for OctoMap |

Key params: `extrinsics_config` (camera extrinsics YAML, `p_world = R @ p_cam + t`), `ee_seg_filter_radius`=0.015 / `ee_seg_z_margin`=0.10 (Pass 1 filter), `free_unknown_xy_radius`=0.05 / `free_unknown_z_margin`=0.10 (Pass 2 filter), `min_depth`=0.05, `max_depth`=15.0.

**Isaac Sim overrides:** `ee_depth_topic:=/isaac/ee/depth_image top_depth_topic:=/isaac/top/depth_image`

**Category values in point cloud:** `FREE=0`, `TARGET=1`, `DESTINATION=2`, `OBSTACLE=3`, `UNKNOWN=4`

**Launch:** `multi_view_projector.launch.py`

---

### `mask_projector_node` *(legacy single-view variant — do not modify)*

| Direction | Topic | Type |
|---|---|---|
| SUB | `/rgbd_camera/depth_image` | `sensor_msgs/Image` (cached) |
| SUB | `/rgbd_camera/camera_info` | `sensor_msgs/CameraInfo` (cached) |
| SUB | `/grounded_sam/mask_image` *(trigger)* | `sensor_msgs/Image` |
| SUB | `/grounded_sam/detections_json` | `std_msgs/String` |
| PUB | `/labeled_points` | `sensor_msgs/PointCloud2` |
| PUB | `/projection_result` | `std_msgs/String` (JSON centroid summary) |

---

## vgn_grasp_pkg

### `vgn_grasp_node`

Depth → signed TSDF (40×40×40) → VGN inference → NMS → semantic bbox filter. Keeps only grasps whose centre falls inside the target `bbox_3d_world` from `/world_map_result`. Does **not** subscribe to `/world_map` (RViz-only topic).

| Direction | Topic | Type | Notes |
|---|---|---|---|
| SUB | `/world_map_result` *(trigger)* | `std_msgs/String` | target centroid + bbox |
| SUB | `/ee_camera/depth_image` | `sensor_msgs/Image` | cached, required |
| SUB | `/ee_camera/camera_info` | `sensor_msgs/CameraInfo` | cached, required |
| SUB | `/top_camera/depth_image` | `sensor_msgs/Image` | cached, if `use_top_depth=True` |
| SUB | `/top_camera/camera_info` | `sensor_msgs/CameraInfo` | cached, if `use_top_depth=True` |
| PUB | `/grasp_candidates` | `std_msgs/String` | JSON: ranked candidates + target_centroid + stamp |
| PUB | `/grasp_markers` | `visualization_msgs/MarkerArray` | RViz gripper markers |
| PUB | `/tsdf_debug` | `sensor_msgs/PointCloud2` | TSDF voxel debug cloud |

Key params: `vgn_model_path` (filename must follow `vgn_<network>.pth`), `min_quality`=0.5, `max_grasp_candidates`=5 (coupled with BT retry budget via `config/robot_defaults.yaml`), `roi_size_m`=0.30, `min_point_count`=50, `use_top_depth`=True, `extrinsics_config`.

TF2: looks up `panda_link0` → `world` to transform grasp poses.

**Launch:** `vgn_grasp.launch.py`, `full_pipeline.launch.py`, `grasp_debug.launch.py`

---

### `vgn_grasp_4cam_node` *(4-camera variant)*

Same pipeline as `vgn_grasp_node` plus optional `/right_camera/*` and `/left_camera/*` depth streams (`use_side_depth=True`) with per-camera TSDF weights and `camera_extrinsics_4cam.yaml`. Same pub topics.

**Launch:** `vgn_grasp_4cam.launch.py`, `full_pipeline_4cam.launch.py`

---

## yolo_hazard_pkg

### `yolo_hazard_node`

YOLO hazard detector — one instance per camera (Fast Brain, >30 FPS). `yolo_hazard_both.launch.py` starts two instances with namespaced topics.

| Direction | Topic | Type | Notes |
|---|---|---|---|
| SUB | `/camera/image_raw` | `sensor_msgs/Image` | trigger (`image_topic`, per-instance) |
| PUB | `/yolo_hazard/detections_json` | `std_msgs/String` | `/yolo_hazard/{top,ee}/detections_json` in dual launch |
| PUB | `/yolo_hazard/annotated_image` | `sensor_msgs/Image` | only when `publish_annotated: true` |

Key params: `model_config` (required), `conf_threshold`=0.35, `device`=`cuda:0`, `half`=True (fp16), `class_allowlist` (when `filter_by_class: true`).

**Launch:** `yolo_hazard_both.launch.py` (both cameras), `yolo_hazard_top.launch.py`, `yolo_hazard_ee.launch.py`

---

## bt_pkg

### `bt_executor_node` (C++)

Owns all ROS subscriptions, writes into mutex-guarded `SceneData`, ticks the behavior tree at 10 Hz. See `ros_pkgs/src/bt_pkg/README.md` for tree structure and blackboard keys. **Do not run alongside `hybrid_pose_client_node`** — double goals to `/run_hybrid_planning` crash the hybrid planning manager.

| Direction | Topic / Action | Type | SceneData field |
|---|---|---|---|
| SUB | `/world_map_result` | `std_msgs/String` | `world_map_fresh`, centroids, `target_label` |
| SUB | `/grasp_candidates` | `std_msgs/String` | `grasp_candidates_fresh`, `grasp_candidates` |
| SUB | `/qwen/grounding_result` | `std_msgs/String` | `destination_spec` |
| SUB | `/yolo/world_map` | `std_msgs/String` | `yolo_objects` |
| SUB | `/yolo/target_centroid` | `geometry_msgs/PointStamped` | `target_centroid_live` |
| SUB | `/bt/hazard_level` | `std_msgs/Int8` | `hazard_level` |
| SUB | `/joint_states` | `sensor_msgs/JointState` | `latest_joint_state` |
| PUB | `/bt/replan_request` | — | triggers new Slow Brain scan |
| ACTION CLIENT | `/run_hybrid_planning` | `moveit_msgs/action/HybridPlanner` | `MoveAction` |
| ACTION CLIENT | `/move_action` | `moveit_msgs/action/MoveGroup` | `MoveToHome` |
| ACTION CLIENT | `/gripper_command` | `control_msgs/action/GripperCommand` | `GripperAction` |

Key params: `tree_file` (`behavior_trees/pick_and_place.xml`), `max_grasp_candidates`=5 (seeded to blackboard, from `robot_defaults.yaml`), `pre_grasp_z_offset`=0.12, `retreat_z_offset`=0.15, `home_joint_values`, destination offsets (`side_offset_m`, `place_height_m`, `container_drop_z`, `near_offset_m`).

**Launch:** `bt_system.launch.py` (delays this node 5 s so the hybrid planner + gripper server finish startup)

---

### `hazard_level_translator_node` (Python)

Aggregates YOLO detections from both cameras into a single hazard level at 30 Hz. 300 ms decay back to level 0.

| Direction | Topic | Type | Notes |
|---|---|---|---|
| SUB | `/yolo_hazard/top/detections_json` | `std_msgs/String` | cached |
| SUB | `/yolo_hazard/ee/detections_json` | `std_msgs/String` | cached |
| PUB | `/bt/hazard_level` | `std_msgs/Int8` | 0=clear, 1=slow (planner reacts), 3=halt (BT suspends) |

Key params: `halt_class_ids`=[0], `halt_conf_threshold`=0.6, `slow_conf_threshold`=0.5, `decay_sec`=0.3.

**Launch:** `bt_system.launch.py`

---

### `yolo_world_map_node` (Python)

Projects YOLO bbox centres through top-camera depth (median over `depth_sample_window`=5) into world-frame 3D positions at 10 Hz. Extrinsics: `p_world = R @ p_cam + t`.

| Direction | Topic | Type | Notes |
|---|---|---|---|
| SUB | `/yolo_hazard/top/detections_json` | `std_msgs/String` | cached |
| SUB | `/top_camera/depth_image` | `sensor_msgs/Image` (32FC1) | cached |
| SUB | `/top_camera/camera_info` | `sensor_msgs/CameraInfo` | cached |
| PUB | `/yolo/world_map` | `std_msgs/String` | JSON `{objects: [{class_name, centroid, confidence}]}` |
| PUB | `/yolo/target_centroid` | `geometry_msgs/PointStamped` | nearest match within `target_search_radius_m`=0.2 |

**Launch:** `bt_system.launch.py`

---

## moveit_isaac_bridge_pkg

### `hybrid_command_bridge_node`

Forwards hybrid-planner joint commands to Isaac Sim.

| Direction | Topic | Type |
|---|---|---|
| SUB | `/hybrid/joint_position_command` *(trigger)* | `std_msgs/Float64MultiArray` |
| PUB | `/joint_command` | `sensor_msgs/JointState` |

**Launch:** `hybrid_planning.launch.py`

---

### `joint_state_restamp_node`

Re-stamps Isaac's joint states to wall time (Isaac Sim publishes no `/clock`).

| Direction | Topic | Type |
|---|---|---|
| SUB | `/joint_states_isaac` *(trigger)* | `sensor_msgs/JointState` |
| PUB | `/joint_states` | `sensor_msgs/JointState` |

---

### `gripper_action_server`

Serves gripper goals via MoveIt `panda_hand` group; polls `/joint_states` at 20 Hz for contact detection (finger stall > `contact_threshold_m`=0.008 → `stalled=True` = grasped).

| Direction | Topic / Action | Type |
|---|---|---|
| SUB | `/joint_states` | `sensor_msgs/JointState` (cached) |
| ACTION SERVER | `/gripper_command` | `control_msgs/action/GripperCommand` |
| ACTION CLIENT | `/move_action` | `moveit_msgs/action/MoveGroup` |

**Launch:** `hybrid_planning.launch.py`

---

### `hazard_collision_injector_node`

Projects YOLO hazard bboxes through depth into MoveIt collision objects. Stale objects are removed by a cleanup timer (`clear_timeout_sec`=0.5).

| Direction | Topic | Type | Notes |
|---|---|---|---|
| SUB | `/yolo_hazard/top/detections_json` *(trigger)* | `std_msgs/String` | |
| SUB | `/rgbd_camera/depth_image` | `sensor_msgs/Image` | cached (`depth_topic`) |
| SUB | `/rgbd_camera/camera_info` | `sensor_msgs/CameraInfo` | cached |
| PUB | `/collision_object` | `moveit_msgs/CollisionObject` | ADD/REMOVE |

Key params: `trigger_class_ids`=[0,1,2], `conf_threshold`=0.5, `extrinsics_config` + `extrinsics_key`=`top_camera`, `default_obstacle_height`=0.25, `object_id_prefix`=`hazard_`.

**Launch:** `hazard_collision_injector.launch.py`, `hazard_collision_injector_ee.launch.py`

---

### `hybrid_pose_client_node`

Standalone hybrid-planner client for single-pose demos. **Never run alongside `bt_executor_node`** (both submit to `/run_hybrid_planning`).

| Direction | Topic / Action | Type |
|---|---|---|
| SUB | `/grasp_target_pose` | `geometry_msgs/PoseStamped` (cached, dispatched when server ready) |
| SUB | `/hybrid_pose_client/reset` | `std_msgs/Empty` (resets one-shot latch) |
| SUB | `/joint_states` | `sensor_msgs/JointState` (cached, start_state) |
| PUB | `/hazard/launch_bottle` | `std_msgs/Empty` (Isaac hazard-launch sync) |
| ACTION CLIENT | `/run_hybrid_planning` | `moveit_msgs/action/HybridPlanner` |

**Launch:** `hybrid_planning.launch.py`

---

### `target_pose_executor_node` *(legacy pipeline)*

One-shot MoveGroup executor — latches after a successful motion; reset to re-arm.

| Direction | Topic / Action | Type |
|---|---|---|
| SUB | `/grasp_target_pose` *(trigger)* | `geometry_msgs/PoseStamped` |
| SUB | `/target_pose_executor/reset` | `std_msgs/Empty` |
| ACTION CLIENT | `/move_action` | `moveit_msgs/action/MoveGroup` |

**Launch:** `target_pose_executor.launch.py`, `capstone_pick_pipeline.launch.py`

---

### `joint_trajectory_bridge_node` *(legacy pipeline)*

Bridges MoveIt trajectory execution to Isaac Sim joint commands at 50 Hz.

| Direction | Topic / Action | Type |
|---|---|---|
| SUB | `/joint_states` | `sensor_msgs/JointState` |
| PUB | `/joint_command` | `sensor_msgs/JointState` |
| ACTION SERVER | `/panda_arm_controller/follow_joint_trajectory` | `control_msgs/FollowJointTrajectory` |

**Launch:** `joint_trajectory_bridge.launch.py`, `capstone_pick_pipeline.launch.py`

---

### `hazard_monitor_node` *(legacy — superseded by hazard_collision_injector + hybrid planning)*

Cancels the active MoveGroup goal when a hazard is detected.

| Direction | Topic / Service | Type |
|---|---|---|
| SUB | `/yolo_hazard/top/detections_json` | `std_msgs/String` |
| SUB | `/yolo_hazard/ee/detections_json` | `std_msgs/String` |
| SRV CLIENT | `/move_action/_action/cancel_goal` | `action_msgs/srv/CancelGoal` |

**Launch:** `hazard_monitor.launch.py`

---

## target_pose_bridge_pkg *(legacy)*

### `target_pose_bridge_node`

Converts `/world_map_result` into a fixed-orientation centroid-offset grasp pose. Used by the legacy `capstone_pick_pipeline.launch.py` — do **not** run alongside `vgn_grasp_node` (both consume `/world_map_result` to produce competing grasp goals).

| Direction | Topic | Type |
|---|---|---|
| SUB | `/world_map_result` *(trigger)* | `std_msgs/String` |
| PUB | `/pre_grasp_target_pose` | `geometry_msgs/PoseStamped` |
| PUB | `/grasp_target_pose` | `geometry_msgs/PoseStamped` |

Key params: `grasp_z_offset`=0.03, `pre_grasp_z_offset`=0.12, fixed approach orientation (roll=π).

---

## graspgen_pkg *(legacy)*

### `graspgen_node`

GraspGen inference via remote ZMQ server (`tcp://{zmq_host}:{zmq_port}`, default `127.0.0.1:5556`). Drop-in alternative to `vgn_grasp_node` — identical `/grasp_candidates` JSON schema. Superseded by `vgn_grasp_pkg` in the current pipeline.

| Direction | Topic | Type | Notes |
|---|---|---|---|
| SUB | `/world_map_result` *(trigger)* | `std_msgs/String` | |
| SUB | `/ee_camera/depth_image` | `sensor_msgs/Image` | cached |
| SUB | `/ee_camera/camera_info` | `sensor_msgs/CameraInfo` | cached |
| SUB | `/qwen/mask_image` | `sensor_msgs/Image` | cached |
| SUB | `/qwen/labeled_detections` | `std_msgs/String` | cached |
| PUB | `/grasp_candidates` | `std_msgs/String` | same schema as VGN |
| PUB | `/grasp_markers` | `visualization_msgs/MarkerArray` | |
| PUB | `/graspgen/target_cloud` | `sensor_msgs/PointCloud2` | debug |

---

## Full Data Flow (current BT pipeline)

```
/ee_camera/image_raw  →  grounded_sam_node  →  /grounded_sam/detections_json
/camera/image_raw     →  (top, cached)      →  /grounded_sam/mask_image

/user_instruction             →  qwen_bridge_node (cached)
/grounded_sam/detections_json →  qwen_bridge_node (trigger)
    ├──► /qwen/labeled_detections
    ├──► /qwen/grounding_result
    └──► /qwen/mask_image  (published LAST — projector trigger)

/ee_camera/depth_image    →  multi_view_projector_node  →  /world_map         (RViz)
/top_camera/depth_image   →                             →  /world_map_result
/qwen/mask_image (trigger)→                             →  /world_cloud_raw   (OctoMap)

/world_map_result  →  vgn_grasp_node  →  /grasp_candidates
                                      →  /grasp_markers (RViz)

/yolo_hazard/{top,ee}/detections_json  →  hazard_level_translator_node  →  /bt/hazard_level
/yolo_hazard/top/detections_json       →  yolo_world_map_node           →  /yolo/world_map
                                                                        →  /yolo/target_centroid
/yolo_hazard/top/detections_json       →  hazard_collision_injector     →  /collision_object

/world_map_result      ─┐
/grasp_candidates      ─┤
/qwen/grounding_result ─┤→  bt_executor_node  →  /run_hybrid_planning  (hybrid planner)
/yolo/world_map        ─┤                     →  /gripper_command      (gripper server)
/yolo/target_centroid  ─┤                     →  /bt/replan_request    (new Slow Brain scan)
/bt/hazard_level       ─┘

MoveIt hybrid planner  →  /hybrid/joint_position_command  →  hybrid_command_bridge_node  →  /joint_command  →  Isaac Sim
Isaac Sim  →  /joint_states_isaac  →  joint_state_restamp_node  →  /joint_states
```

## Legacy Data Flow (`capstone_pick_pipeline.launch.py`)

```
/world_map_result  →  target_pose_bridge_node  →  /grasp_target_pose  →  target_pose_executor_node
                                                                            │  [/move_action]
MoveIt  →  [/panda_arm_controller/follow_joint_trajectory]  →  joint_trajectory_bridge_node  →  /joint_command  →  Isaac Sim
```

Re-arm the legacy executor after a successful motion:
```bash
ros2 topic pub --once /target_pose_executor/reset std_msgs/msg/Empty "{}"
```
