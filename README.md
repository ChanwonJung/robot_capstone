# Language-Directed Manipulator with Dynamic Tracking and Obstacle Avoidance

Autonomous robotic manipulation driven by natural language. The system interprets
ambiguous commands (*"put the book in the box"*), visually grounds the target in a
cluttered scene, and executes a safe grasp while reacting to dynamic obstacles.

Built and validated in **NVIDIA Isaac Sim** on a Franka Panda with a dual RGB-D
camera setup (end-effector + top-down).

**Stack:** Ubuntu 24.04 · ROS 2 Jazzy · Isaac Sim 5.1.0 · Python 3.12 · C++20

---

## 1. Architecture — Slow Brain / Fast Brain

The pipeline is split so that heavy reasoning never blocks the safety loop.

### Slow Brain — runs once per command

| Stage | Component | Role |
|---|---|---|
| Grounding | **Qwen3.5-27B** (remote A100) | Instruction + EE image → labeled boxes, target/destination, spatial relation |
| Segmentation | **SAM 2.1** (remote A100) | Boxes → pixel masks for the two key objects |
| Fusion | `mask_projection_pkg` | Masks + dual depth → labeled 3D point cloud + centroids |
| Grasping | **GraspGen** (remote A100) | Target cloud → ranked 6-DOF grasp candidates |

### Fast Brain — sustains >30 FPS

| Component | Role |
|---|---|
| **YOLO26 segmentation** (`yolo_hazard_pkg`) | Monitors both cameras for hazards |
| **Hazard injector** | Publishes hazards as MoveIt collision objects |
| **MoveIt hybrid planning** | Global trajectory + low-latency local reaction |

### Behavior Tree

`bt_pkg` closes the loop: waits for Slow Brain results, selects grasp candidates,
drives MoveIt for pick-and-place, and suspends the arm on an E-stop hazard.

---

## 2. Hardware

Two machines: one local workstation running everything real-time, and a remote
A100 running all heavy inference. The local GPU is capped at 12 GB, so anything
larger offloads.

| Node | Hardware | Role |
|---|---|---|
| **Local** | RTX 5070 (12 GB) | Isaac Sim, ROS 2, MoveIt, YOLO tracking, Behavior Tree, RViz |
| **Inference** | A100 (`tta@123.37.28.208`) | Qwen3.5-27B :8000 · GraspGen :5556 · SwinDRNet :5557 · SAM 2.1 :5558 |

All four A100 services are **loopback-only** on the server and reached over
plain SSH tunnels that `launch_env.bash` opens automatically. No VPN or overlay
network is involved — if `ssh tta@123.37.28.208` works, the tunnels work.

Because they all share one GPU, a slow Qwen call and a grasp request contend
with each other. Worth watching if latency spikes during a full run.

---

## 3. Setup (one time)

### Python environments

Two venvs at the repo root. They are separate because `ultralytics` pulls its own
`torch` build that conflicts with the Grounded-SAM stack.

| venv | Used by | Notes |
|---|---|---|
| `gsam_venv` | everything except YOLO | `launch_env.bash` injects it into `PYTHONPATH` and **aborts if it is missing** |
| `.venv-yolo` | `yolo_hazard_pkg` | Path is hardcoded in all three YOLO launch files |

**Core dependencies** — required by the active pipeline (Qwen, GraspGen,
SwinDRNet, projection). These are what remain once Grounded-SAM is retired:

```bash
python3 -m venv gsam_venv && gsam_venv/bin/pip install openai pydantic pyzmq msgpack msgpack-numpy opencv-python-headless numpy pyyaml
```

**YOLO**, in its own venv:

```bash
python3 -m venv .venv-yolo && .venv-yolo/bin/pip install ultralytics
```

**Grounded-SAM** — *deprecated.* Still needed only while T5/T6 of the run
sequence use `grounded_sam_pkg`; skip it once the Qwen → SAM 2.1 slow brain
(§6.1) lands. This is the heavy one (`torch`, `torchvision`, GroundingDINO, SAM):

```bash
gsam_venv/bin/pip install -r ros_pkgs/src/grounded_sam_pkg/requirements.txt
```

### System packages and build

```bash
sudo apt install -y libzmq3-dev libsqlite3-dev libtinyxml2-dev
```
```bash
cd ros_pkgs && colcon build --symlink-install && source install/setup.bash
```

**Model weights** (not in git) go under `models/`:

| Path | File |
|---|---|
| `models/g-sam/` | `groundingdino_swint_ogc.pth` (~662 MB), `sam_vit_b_01ec64.pth` (~375 MB) |
| `models/yolo26/` | trained hazard segmentation weights |

> This repo has **no git submodules** — BehaviorTree.CPP and BehaviorTree.ROS2 are
> vendored as ordinary files under `ros_pkgs/src/behavior_tree/`.

---

## 4. Running the system

### Step 0 — environment, every terminal, always first

```bash
source launch_env.bash
```

This sources ROS 2 + the workspace overlay, injects `gsam_venv` into
`PYTHONPATH`, exports `ROBOT_CAPSTONE_ROOT`, and opens the four A100 tunnels
(8000 Qwen, 5556 GraspGen, 5557 SwinDRNet, 5558 SAM 2.1), skipping any already bound.

Off-network, skip the tunnels so they don't hang on connect:

```bash
source launch_env.bash --no-tunnel
```

**After every rebuild of `grounded_sam_pkg`**, colcon resets entry-script
shebangs to system Python, which lacks `torch`. Patch them back before launching:

```bash
VENV_PY="$PWD/gsam_venv/bin/python"; for f in $(find ros_pkgs/install/grounded_sam_pkg/lib -maxdepth 3 -type f -executable); do head -1 "$f" | grep -q "^#!/usr/bin/python3$" && sed -i "1s|^#!/usr/bin/python3$|#!${VENV_PY}|" "$f"; done
```

### Step 1 — confirm the A100 is reachable

```bash
ss -tln | grep -E ':(8000|555[678])'
```
```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

A sub-second failure from any A100 client means the tunnel is down; a slow
failure means the service is down or loaded.

If a service is down, see `A100_GRASPGEN_QUICK_START.md` and
`SWINDRNET_INTEGRATION.md` for start/restart procedures.

### Step 2 — launch order

Each command in its own terminal, `source launch_env.bash` first in all of them.

Order matters only for **T1** (Isaac must be up so the camera and joint topics
exist). Everything after that is event-driven: each stage triggers on the
previous stage's output, so it self-sequences regardless of start order. All
Slow Brain topics are latched, so a late subscriber still receives the last scan.

| # | Terminal | Command |
|---|---|---|
| **T1** | Isaac Sim scene | `./run_capstone_scene.sh` |
| **T2** | MoveIt hybrid planner + gripper | `ros2 launch moveit_isaac_bridge_pkg hybrid_planning.launch.py` |
| **T3** | YOLO hazard detection | `ros2 launch yolo_hazard_pkg yolo_hazard_both.launch.py` |
| **T4** | Hazard → collision injector | `ros2 launch moveit_isaac_bridge_pkg hazard_collision_injector.launch.py` |
| **T5** | Grounded-SAM (dual view) | `ros2 launch grounded_sam_pkg grounded_sam_dual.launch.py prompt:="book, box"` |
| **T6** | Qwen labeling | `ros2 run grounded_sam_pkg qwen_stub_node` |
| **T7** | Mask projection | `ros2 launch mask_projection_pkg multi_view_projector.launch.py ee_depth_topic:=/ee_rgbd_camera/depth_image ee_camera_info_topic:=/ee_rgbd_camera/camera_info top_depth_topic:=/rgbd_camera/depth_image top_camera_info_topic:=/rgbd_camera/camera_info` |
| **T8** | GraspGen (A100) | `ros2 launch graspgen_pkg graspgen.launch.py mask_topic:=/qwen/mask_image` |
| **T9** | Behavior Tree | `ros2 launch bt_pkg bt_system.launch.py` |

For transparent objects (glass), add SwinDRNet depth restoration to T8:

```bash
ros2 launch graspgen_pkg graspgen.launch.py mask_topic:=/qwen/mask_image swindrnet_enabled:=true transparent_reconstruct_enabled:=true transparent_force:=true
```

> **T8 note:** `mask_topic:=/qwen/mask_image` is required on this legacy path —
> GraspGen now defaults to `/sam/mask_image` (the new Slow Brain). Omit it and
> GraspGen waits for a mask forever, silently.

> **T6 note:** `qwen_stub_node` uses a hardcoded label→category table and needs no
> cluster, but it does **not** publish `/qwen/grounding_result` — so the BT's
> destination spec stays empty. It is sufficient through the grasp stage; the
> place phase needs a real grounding node.

### Hazard demo

`run_hazard_demo.sh` sets the scene env vars and launches Isaac in place of T1.
The mode must match `manager_logic` on T2:

```bash
./run_hazard_demo.sh replan
```
```bash
ros2 launch moveit_isaac_bridge_pkg hybrid_planning.launch.py manager_logic:=replan
```

| Mode | Behavior |
|---|---|
| `replan` | Bottle parks over the book; global planner re-routes around it |
| `stop_resume` | Bottle passes through; arm holds, then resumes |

### Re-arming for the next command

The BT publishes `/bt/replan_request` when it exhausts its grasp candidates,
which re-triggers a Slow Brain scan. To force one manually:

```bash
ros2 topic pub --once /bt/replan_request std_msgs/msg/Empty "{}"
```

---

## 5. Repository layout

```text
robot_capstone/
├── config/robot_defaults.yaml   # shared robot identity params, loaded by every launch file
├── launch_env.bash              # env + A100 SSH tunnels — source this first, always
├── run_capstone_scene.sh        # Isaac Sim entrypoint
├── run_hazard_demo.sh           # scripted hazard scenarios
├── models/                      # weights: g-sam/, yolo26/, qwen3.5/
├── sim/                         # Isaac Sim scene setup + ROS bridges
└── ros_pkgs/src/
    ├── grounded_sam_pkg/        # GroundingDINO + SAM (current Slow Brain perception)
    ├── qwen_pkg/                # Qwen VLM grounding (legacy — being replaced)
    ├── slow_brain/              # NEW Slow Brain — see §6
    │   ├── qwen_a100/           # instruction + image → boxes, labels, destination spec
    │   └── sam_a100/            # boxes → SAM 2.1 masks
    ├── mask_projection_pkg/     # masks + depth → labeled 3D cloud + centroids
    ├── graspgen_pkg/            # 6-DOF grasp generation via A100 (active path)
    ├── yolo_hazard_pkg/         # Fast Brain hazard detection
    ├── moveit_isaac_bridge_pkg/ # MoveIt ↔ Isaac joint bridge, gripper action server
    ├── bt_pkg/                  # BehaviorTree.ROS2 pick-and-place executor
    └── behavior_tree/           # vendored BT.CPP + BT.ROS2 (do not modify)
```

Per-node topic and action reference: `ROS_NODES.md`. Developer guidance and
design rationale: `CLAUDE.md`.

---

## 6. Roadmap

The original capstone was **completed in late June 2026**: a working
LLM → Grounded-SAM → Qwen slow brain feeding GraspGen, with the YOLO hazard
loop and MoveIt hybrid planning closing around a behavior tree that could pick.

Three expanded goals target the **AI Rookie Competition, 16 August 2026**.

### 6.1 — Collapse the Slow Brain into Qwen → SAM 2.1

*Status: in progress.*

The original chain parsed nouns out of the instruction with an LLM, fed those
nouns to Grounding DINO, and used Qwen only to pick an index from the results.
Every step discarded information: the noun parser threw away spatial and
relational context, and Grounding DINO saw only isolated nouns rather than the
instruction.

Replacing it with a single large VLM pass means Qwen sees the **full instruction
and the raw image together**, and emits boxes, categories, and the spatial
relation in one structured response. SAM 2.1 then segments those boxes directly.
Fewer stages, no noun bottleneck, and a much larger model doing the grounding.

- ✅ `qwen_a100` — grounding node, schema, A100 client, offline CLI
- ⬜ `sam_a100` — box → mask segmentation, mono8 label map
- ⬜ End-to-end validation against `mask_projection_pkg`

### 6.2 — SwinDRNet for transparent objects

*Status: integrated, pending validation.*

RGB-D sensors see through glass and return the table behind it, so transparent
objects arrive at the grasp stage as holes. SwinDRNet reconstructs plausible
depth before the point cloud is built, giving GraspGen a solid object to work
with. Runs on the A100 at port 5557; opt-in per object class.

### 6.3 — Complete the place phase

*Status: not started.*

The BT currently ends after the retreat move — the destination and place block in
`pick_and_place.xml` is commented out, because the stub grounding node never
populated a destination spec. With §6.1 emitting a real one
(`type`/`reference_label`/`relation`/`region`), `UpdateTargetPose` and
`destination_calculator` can compute a place pose and finish the task.

Alongside re-enabling it, `destination_calculator` should become
**direction-aware**: every relation currently resolves to a fixed axis offset in
`panda_link0`, so `near` always shoves the object +8 cm in X regardless of where
the target and destination actually sit. Deriving the offset direction from the
two centroids — and its magnitude from the destination's 3D bbox — places the
object on the nearest sensible free point instead. See `CLAUDE.md` →
*bt_pkg internals → Planned: direction-aware `near`*.

---

## 7. Contributors

| Name | Role |
|---|---|
| **Chanwon Jeong** | System integration, MoveIt motion planning, YOLO |
| **Sanghyun Park** | Grounded-SAM, GraspGen, sensor fusion |
| **Jaewon Heo** | Qwen VLM, Behavior Tree |
