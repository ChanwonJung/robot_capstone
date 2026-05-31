# bt_pkg — BehaviorTree.ROS2 Pick-and-Place Executor

Behavior tree package that closes the loop between the Slow Brain perception pipeline and MoveIt2 hybrid planning. Runs on the **teammate's node** alongside `yolo_hazard_pkg` and the MoveIt bridge.

---

## Package layout

```
bt_pkg/
├── src/
│   ├── bt_executor_node.cpp          main — owns all ROS subs, builds factory, ticks tree
│   ├── destination_calculator.cpp    pure geometry: destination_spec → PoseStamped
│   ├── action_nodes/
│   │   ├── wait_for_scene.cpp
│   │   ├── parse_scene.cpp
│   │   ├── select_grasp_candidate.cpp
│   │   ├── move_action.cpp
│   │   ├── move_to_home.cpp
│   │   ├── gripper_action.cpp
│   │   ├── update_target_pose.cpp
│   │   └── request_replan.cpp
│   └── condition_nodes/
│       ├── emergency_stop_clear.cpp
│       └── target_visible.cpp
├── include/bt_pkg/
│   ├── scene_data.hpp                shared state struct (mutex-guarded)
│   ├── action_nodes.hpp              all BT node class declarations
│   ├── condition_nodes.hpp
│   └── destination_calculator.hpp
├── scripts/
│   ├── hazard_level_translator_node.py   YOLO detections → /bt/hazard_level
│   └── yolo_world_map_node.py            YOLO + depth → 3D world positions
├── behavior_trees/
│   └── pick_and_place.xml
├── config/
│   └── bt_params.yaml
└── launch/
    └── bt_system.launch.py
```

---

## How it works

### Shared state: `SceneData`

All BT nodes receive a `shared_ptr<SceneData>` injected at construction time (no blackboard polling for sensor data). `bt_executor_node` owns the ROS subscriptions and writes into `SceneData` under `scene->mtx`. BT node `tick()` calls lock the mutex to read.

```
ROS topics → bt_executor_node subscriptions → SceneData (mutex-guarded)
                                                    ↓
                                            BT node tick() reads SceneData
                                            BT blackboard  holds computed poses
```

**SceneData fields by source topic:**

| Field | Source |
|---|---|
| `world_map_fresh`, `target_centroid`, `destination_centroid`, `target_label` | `/world_map_result` |
| `grasp_candidates_fresh`, `grasp_candidates` | `/grasp_candidates` |
| `destination_spec` | `/qwen/grounding_result` |
| `yolo_objects` | `/yolo/world_map` |
| `target_centroid_live`, `target_centroid_stamp` | `/yolo/target_centroid` |
| `hazard_level` | `/bt/hazard_level` |
| `latest_joint_state` | `/joint_states` |
| `last_processed_stamp` | set by `ParseScene` to prevent re-trigger |

### BT blackboard keys

Keys written by sync nodes and read by action nodes:

| Key | Written by | Read by |
|---|---|---|
| `grasp_pose` | `ParseScene`, `SelectGraspCandidate` | `MoveAction` |
| `pre_grasp_pose` | `ParseScene`, `SelectGraspCandidate` | `MoveAction` |
| `retreat_pose` | `ParseScene` | `MoveAction` |
| `place_pose` | `UpdateTargetPose` | `MoveAction` |
| `destination_centroid_init` | `ParseScene` | `UpdateTargetPose` |
| `destination_spec` | `ParseScene` | `UpdateTargetPose` |
| `grasp_index` | `ParseScene` (reset to 0), `SelectGraspCandidate` (increment) | `SelectGraspCandidate` |
| `grasp_candidates` | `ParseScene` | `SelectGraspCandidate` |
| `max_grasp_candidates` | `bt_executor_node` (seeded from ROS param at startup) | `RetryUntilSuccessful` in XML |

### Tree structure

```
RepeatForever
  ReactiveSequence  ← "E-Stop Guard"
    EmergencyStopClear          ← re-checked every 100 ms; FAILURE suspends all
    Sequence  ← "Main Pipeline"
      WaitForScene              ← RUNNING until both /world_map_result + /grasp_candidates
      ParseScene                ← writes all blackboard keys, resets grasp_index

      Fallback  ← "Pick or Recover"
        ReactiveSequence  ← "Visibility Guard"
          TargetVisible         ← re-checked each tick; FAILURE aborts to Pick Recovery
          RetryUntilSuccessful  num_attempts={max_grasp_candidates}
            Sequence  ← "Pick Attempt"
              SelectGraspCandidate
              MoveAction  pose_key="pre_grasp_pose"
              MoveAction  pose_key="grasp_pose"
              GripperAction  command="close"    ← SUCCESS = stalled (contact)
              MoveAction  pose_key="retreat_pose"

        Sequence  ← "Pick Recovery"
          MoveToHome
          RequestReplan         ← resets freshness flags, returns FAILURE
                                  → propagates up → RepeatForever re-ticks from WaitForScene

      UpdateTargetPose          ← destination_spec + YOLO → place_pose on blackboard

      Fallback  ← "Place or Recover"
        RetryUntilSuccessful  num_attempts=3
          Sequence  ← "Place Attempt"
            MoveAction  pose_key="place_pose"
            GripperAction  command="open"
            MoveAction  pose_key="retreat_pose"
        MoveToHome
```

### `RequestReplan` returns FAILURE intentionally

`RequestReplan` publishes `/bt/replan_request`, resets `world_map_fresh` and `grasp_candidates_fresh` to `false`, and returns `FAILURE`. This failure propagates:

```
RequestReplan FAILURE
  → Pick Recovery Sequence FAILURE
  → Pick Fallback FAILURE
  → Main Pipeline Sequence FAILURE
  → E-Stop Guard ReactiveSequence FAILURE
  → RepeatForever re-ticks from WaitForScene
```

No extra decorator needed. `WaitForScene` blocks until the Slow Brain delivers fresh data after the replan.

### `WaitForScene` freshness check

`WaitForScene` returns `RUNNING` until **both** of these are true:
1. `world_map_fresh == true`
2. `grasp_candidates_fresh == true`
3. Both stamps are strictly newer than `last_processed_stamp`

`ParseScene` sets `last_processed_stamp = world_map_stamp` on success. This prevents a second BT cycle from consuming the same perception result without a new scan.

### `MoveAction` and the hybrid planner

`MoveAction` wraps `moveit_msgs/action/HybridPlanner` (`/run_hybrid_planning`). It:
- Reads a `PoseStamped` from the blackboard via the `pose_key` port
- Populates `start_state.joint_state` from `scene->latest_joint_state` (critical — hybrid planner needs current state)
- Builds position + orientation constraints around the target pose with tolerances from ROS params
- Scales velocity via the `speed` port

**Do not run `hybrid_pose_client_node` alongside `bt_executor_node`** — both submit goals to `/run_hybrid_planning`, which causes double-goal / "Unknown event" crashes in the hybrid planning manager.

### `GripperAction` contact detection

`GripperAction(command="close")` returns:
- `SUCCESS` if `result.stalled == true` (fingers stopped above 8 mm — object grasped)
- `FAILURE` if `result.stalled == false` (fingers fully closed — missed)

The actual contact detection happens in `gripper_action_server.py` (in `moveit_isaac_bridge_pkg`), which polls `/joint_states` at 20 Hz and reports stall.

### `max_grasp_candidates` coupling

`max_grasp_candidates` is the single control point for grasp pipeline depth:

```
config/robot_defaults.yaml
      max_grasp_candidates: 5
             ↓                           ↓
  vgn_grasp_node publishes        bt_executor_node reads param
  at most N candidates        →   seeds blackboard: {max_grasp_candidates}=5
                                         ↓
                                  pick_and_place.xml
                                  RetryUntilSuccessful num_attempts="{max_grasp_candidates}"
```

Change only the value in `config/robot_defaults.yaml`.

---

## Support nodes (Python)

### `hazard_level_translator_node.py`

Aggregates YOLO detections from both cameras into a single hazard level published at 30 Hz on `/bt/hazard_level` (Int8).

| Level | Condition | BT response |
|---|---|---|
| 0 | No qualifying detections, or decayed | Normal |
| 1 | Any detection above `slow_conf_threshold` | Hybrid planner reacts (BT ignores) |
| 3 | Class in `halt_class_ids` AND conf ≥ `halt_conf_threshold` | `EmergencyStopClear` FAILURE → tree suspends |

300 ms decay: if no qualifying detection arrives within 0.3 s, level returns to 0.

### `yolo_world_map_node.py`

Projects YOLO bounding box centres through top-camera depth to world-frame 3D positions. Publishes at 10 Hz.

- `/yolo/world_map` — JSON `{objects: [{class_name, centroid, confidence}]}` — consumed by `UpdateTargetPose` via `SceneData`
- `/yolo/target_centroid` — `PointStamped` nearest matching object within `target_search_radius_m` of the seed — consumed by `TargetVisible`

Extrinsics loaded from the same `camera_extrinsics_isaac.yaml` format as `mask_projection_pkg`: `p_world = R @ p_cam + t`.

---

## Dependency: `behaviortree_ros2`

Two libraries are needed:

| Library | Where |
|---|---|
| `behaviortree_ros2` | vendored at `ros_pkgs/src/behavior_tree/BehaviorTree.ROS2/` — do not modify |
| `behaviortree_cpp` (BT.cpp v4 core) | git submodule at `ros_pkgs/src/BehaviorTree.CPP/` |

After cloning the repo, initialise both:
```bash
git submodule update --init ros_pkgs/src/BehaviorTree.CPP
sudo apt install -y libzmq3-dev libsqlite3-dev libtinyxml2-dev
```
Then `colcon build` resolves and builds them in the correct order.

---

## Launch

```bash
# Prerequisites running elsewhere:
#   hybrid_planning.launch.py  → /run_hybrid_planning, /move_action, /gripper_command
#   yolo_hazard_both.launch.py → /yolo_hazard/top/detections_json, .../ee/...
#   hazard_collision_injector.launch.py
#   Slow Brain pipeline → /world_map_result, /grasp_candidates, /qwen/grounding_result

source launch_env.bash
ros2 launch bt_pkg bt_system.launch.py \
  extrinsics_config:=/abs/path/camera_extrinsics_isaac.yaml
```

`bt_system.launch.py` starts `hazard_level_translator_node` and `yolo_world_map_node` immediately, then delays `bt_executor_node` by 5 s to let the hybrid planner and gripper server finish their startup sequences.

Parameters load in order: `config/robot_defaults.yaml` → `config/bt_params.yaml` → inline launch-arg overrides. Last value wins.
