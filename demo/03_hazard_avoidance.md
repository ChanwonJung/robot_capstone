# 03 — Fast Brain: Hazard Avoidance

**Shows:** the >30 FPS safety loop. A YOLO-detected hazard becomes a MoveIt
collision object in real time, and the arm either holds and resumes, or re-routes
around it.

**Needs:** Isaac Sim. No A100 — this demo is entirely local.

Two scenarios, driven by `run_hazard_demo.sh`. The mode **must** match
`manager_logic` on the hybrid planner or the behaviour will not match the name.

---

## Scenario A — stop and resume (transient hazard)

A bottle flies through the workspace. The arm halts, waits for the hazard to
clear, then continues its original trajectory.

```bash
./run_hazard_demo.sh stop_resume
```
```bash
ros2 launch moveit_isaac_bridge_pkg hybrid_planning.launch.py manager_logic:=stop_resume
```

Then, each in its own terminal after `source launch_env.bash`:

```bash
ros2 launch yolo_hazard_pkg yolo_hazard_both.launch.py
```
```bash
ros2 launch moveit_isaac_bridge_pkg hazard_collision_injector.launch.py
```
```bash
ros2 launch bt_pkg bt_system.launch.py
```

## Scenario B — replan (persistent hazard)

The bottle parks over the target and stays. The local planner detects the
invalidated trajectory, the manager triggers a global replan, and OMPL finds a
path around the obstruction.

```bash
./run_hazard_demo.sh replan
```
```bash
ros2 launch moveit_isaac_bridge_pkg hybrid_planning.launch.py manager_logic:=replan
```

Remaining terminals as above.

---

## What to watch

**Hazard level** — the Fast Brain's summary signal:
```bash
ros2 topic echo /bt/hazard_level
```

| Value | Meaning |
|---|---|
| `0` | CLEAR (or 0.3 s decay elapsed since the last detection) |
| `1` | SLOW — any detection at confidence ≥ 0.5 |
| `3` | **HALT** — a `halt_class_ids` detection at confidence ≥ 0.6 |

`2` is never produced. Only `≥ 3` reaches the behavior tree: `EmergencyStopClear`
returns FAILURE and the wrapping `ReactiveSequence` suspends the whole tree.
Levels 0–1 are handled by the hybrid planner and the collision injector instead.

**Detections and collision objects:**
```bash
ros2 topic echo /yolo_hazard/top/detections_json
```
```bash
ros2 topic echo /collision_object
```

**Frame rate** — `yolo_hazard_node` logs it periodically; it should hold above 30 FPS.

**RViz** — the injected collision object appears in the planning scene as the
bottle moves.

---

## Tuning, and why it is fragile

The `replan` scenario is a three-way coupling. Change one and the others need
revisiting:

| Knob | Where |
|---|---|
| `local_planning_frequency: 1.5` | `moveit_isaac_bridge_pkg/config/hybrid/local_planner.yaml` |
| negative `xy_margin` | `hazard_collision_injector.launch.py` |
| gripper `link_padding` shrink | `hybrid_planning.launch.py` |

Together these keep `CheckStartStateCollision` from tripping on millimetre-scale
fingertip overlap at every replan attempt, which otherwise crashes the hybrid
planning manager.

Scene overrides take precedence over the script defaults:
```bash
ROBOT_CAPSTONE_HAZARD_PARK_X=0.30 ./run_hazard_demo.sh replan
```

Other useful variables: `ROBOT_CAPSTONE_BOTTLE_VX`, `ROBOT_CAPSTONE_BOTTLE_SPAWN_{X,Y,Z}`,
`ROBOT_CAPSTONE_HAZARD_MODE`, `ROBOT_CAPSTONE_BOX_V{X,Y,Z}`.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| No detections at all | YOLO weights missing — `model_paths.yaml` points at `models/yolo26/release/`, which may not exist |
| Arm never halts | Detected class not in `halt_class_ids` (default `[0]`), or confidence below 0.6 |
| Planner crashes on replan | The three-way tuning above is out of sync |
| Node dies importing `ultralytics` | `.venv-yolo` missing — the YOLO launch files hardcode that path |
| FPS below 30 | Both cameras on one GPU; try `yolo_hazard_top.launch.py` alone |

**Class taxonomy is inconsistent across three sources** — worth knowing before
you trust `halt_class_ids`:

| Source | Taxonomy |
|---|---|
| `config/robot_defaults.yaml` (effective) | `0=arm/person, 1=bottle, 2=box` |
| trained model `config.yaml` | `0=hand, 1=forearm, 2=pet_bottle, 3=small_box` |
| `ros_node.py` default (overridden) | `[0, 39]` — COCO person/bottle |

So `halt_class_ids: [0]` means "arm/person" under one reading and "hand" under
another.
