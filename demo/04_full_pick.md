# 04 — End-to-End Pick

**Shows:** the complete system. Instruction → grounding → segmentation →
3D fusion → grasp synthesis → motion, with the hazard loop live throughout.

**Needs:** Isaac Sim + the A100 tunnels (8000, 5556; 5557 for glass).

This uses the **current** Slow Brain (Grounded-SAM + Qwen). The replacement
(Qwen → SAM 2.1) is [demo 05](05_slow_brain_v2.md).

---

## Launch order

Nine terminals. `source launch_env.bash` first in every one.

Only **T1** is genuinely order-dependent — Isaac must be up so the camera and
joint topics exist. Everything after is event-driven: each stage triggers on the
previous stage's output, and all Slow Brain topics are latched, so a late
subscriber still receives the last scan.

| # | What | Command |
|---|---|---|
| T1 | Isaac Sim scene | `./run_capstone_scene.sh` |
| T2 | MoveIt hybrid planner + gripper | `ros2 launch moveit_isaac_bridge_pkg hybrid_planning.launch.py` |
| T3 | YOLO hazard detection | `ros2 launch yolo_hazard_pkg yolo_hazard_both.launch.py` |
| T4 | Hazard → collision injector | `ros2 launch moveit_isaac_bridge_pkg hazard_collision_injector.launch.py` |
| T5 | Grounded-SAM, dual view | `ros2 launch grounded_sam_pkg grounded_sam_dual.launch.py prompt:="book, box"` |
| T6 | Qwen labeling | `ros2 run grounded_sam_pkg qwen_stub_node` |
| T7 | Mask projection | `ros2 launch mask_projection_pkg multi_view_projector.launch.py ee_depth_topic:=/ee_rgbd_camera/depth_image ee_camera_info_topic:=/ee_rgbd_camera/camera_info top_depth_topic:=/rgbd_camera/depth_image top_camera_info_topic:=/rgbd_camera/camera_info` |
| T8 | GraspGen (A100) | `ros2 launch graspgen_pkg graspgen.launch.py mask_topic:=/qwen/mask_image` |
| T9 | Behavior tree | `ros2 launch bt_pkg bt_system.launch.py` |

**Before T5**, if you rebuilt `grounded_sam_pkg`, apply the shebang patch from
the [shared prerequisites](README.md#shared-prerequisites). Without it the node
dies on `import torch`.

> **`mask_topic:=/qwen/mask_image` on T8 is required.** GraspGen now defaults to
> `/sam/mask_image` (the new Slow Brain). This legacy path publishes the mask
> from `qwen_stub_node` on `/qwen/mask_image`, so the override is what connects
> them. Omit it and GraspGen waits for a mask forever — silently, because it
> guards on `self._mask is None` and simply returns.

### Transparent objects

Swap T8 to add SwinDRNet depth restoration. Use with a glass cup in the scene:

```bash
ros2 launch graspgen_pkg graspgen.launch.py mask_topic:=/qwen/mask_image swindrnet_enabled:=true transparent_reconstruct_enabled:=true transparent_force:=true
```

Watch `/graspgen/target_cloud` in RViz: without restoration the glass is full of
holes (the sensor sees the table behind it); with it, a solid object.

---

## Watching it work

Follow the data as it flows. Each stage's output is the previous stage's proof
of correctness, so check them in order:

**1. Detections — did grounding pick the right objects?**
```bash
ros2 topic echo /qwen/labeled_detections --once
```

**2. Centroids — did the masks fuse with depth into sane 3D positions?**

This is the Slow Brain's real output, and the last place a fault is cheap to
diagnose. Values are **metres in `panda_link0`** (the frame is implicit — it is
never declared in the JSON).

```bash
ros2 topic echo --once --field data /world_map_result | python3 -m json.tool
```

```json
{
  "target":      {"label": "book", "centroid": [0.4521, 0.0183, 0.0612],
                  "bbox_3d_world": {"min": [...], "max": [...]}, "point_count": 1834},
  "destination": {"label": "box",  "centroid": [0.3104, 0.3455, 0.0498], "...": "..."}
}
```

| Check | Meaning if it fails |
|---|---|
| `target` key present | No TARGET mask reached the projector — nothing downstream will run |
| x roughly 0.2–0.8 m | Outside the Panda's reach → extrinsics wrong, not perception |
| z near table height, not ≈0 | Everything at z≈0 is the signature of identity extrinsics |
| `point_count` in the hundreds+ | Tens of points means a sliver of a mask |

If the numbers look wrong, check the projector's startup log for
`Using identity transforms — point cloud will be in camera frame.` That means
`extrinsics_config` failed to load and the node **kept publishing anyway**, in
the camera frame. The centroids will be self-consistent and completely wrong in
robot coordinates.

**3. Grasps — did the candidates land inside the target?**
```bash
ros2 topic echo /grasp_candidates --once
```

`graspgen_node` keeps only grasps whose centre falls inside
`target.bbox_3d_world`. An inflated or misplaced target bbox rejects every
grasp — which is why step 2 is worth reading before blaming the grasp stage.

In RViz: `/world_map` (labeled cloud — green target, yellow destination, red
obstacles, purple top-camera unknown), `/grasp_markers` (gripper poses), and the
planning scene. The target cloud should sit *on* the object, not floating above
or behind it.

Expected timings: GSAM 30–40 s on CPU, GraspGen 200–500 ms, SwinDRNet 20–50 ms.

---

## Important limitation of this demo

**T6 `qwen_stub_node` uses a hardcoded label→category table** and needs no
cluster — but it does **not** publish `/qwen/grounding_result`. So the behavior
tree's destination spec is never populated.

Consequence: this demo can **pick but not place**. The place block in
`pick_and_place.xml` is commented out regardless, so the tree ends after the
retreat move. Completing the place stage is roadmap item §6.3.

To drive the real VLM instead of the stub — and to get a populated destination
spec so the place phase becomes possible — replace T5/T6 with
[demo 05](05_slow_brain_v2.md), which is working now.

---

## Re-arming

The BT publishes `/bt/replan_request` when it exhausts its grasp candidates. To
force a fresh scan:

```bash
ros2 topic pub --once /bt/replan_request std_msgs/msg/Empty "{}"
```

Note that with the stub (T6) nothing subscribes to this, so it has no effect —
the new `qwen_a100` bridge is what closes that loop.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| BT sits in `WaitForScene` forever | Needs **both** `/world_map_result` and `/grasp_candidates` fresher than the last processed scan. Republishing only one blocks permanently |
| `/world_map_result` has no `target` key | Nothing was labeled TARGET — check `/qwen/labeled_detections` |
| Every grasp rejected | Semantic filter keeps only grasps inside the target's `bbox_3d_world`; check the target cloud is where you expect |
| Point cloud in the wrong frame | Extrinsics failed to load — look for `Using identity transforms` in the projector log |
| Projector never fires | It triggers on the **mask**; if detections aren't cached first, the trigger is dropped and **never retried** |
| No `/grasp_candidates`, no error | GraspGen has no mask. On this legacy path T8 needs `mask_topic:=/qwen/mask_image` |
| `grounded_sam_node` dies on `import torch` | Shebang patch not applied after rebuild |
| SIGABRT in the hybrid planner | Too many goals — keep `bt_pick_retries` small (default 5) |

**Never run `hybrid_pose_client_node` alongside `bt_executor_node`** — both
submit goals to `/run_hybrid_planning`, causing double-goal crashes in the
manager.
