# 05 — Slow Brain v2: Qwen → SAM 2.1

**Shows:** the new two-stage Slow Brain. One VLM pass turns an instruction plus a
raw frame into boxes, categories and a spatial relation; SAM 2.1 turns those
boxes into pixel-accurate masks; the projector fuses them with depth.

**Needs:** tunnels on **8000** (Qwen) and **5558** (SAM 2.1).

> **Status:** client and server both up. Tunnel 5558 confirmed open. If
> `sam_mask_node` times out per-frame, the tunnel or server is down — see
> [demo 02](02_a100_services.md).

---

## What it replaces, and why

The original chain was **LLM → Grounding DINO → Qwen**: an LLM parsed nouns out
of the instruction, Grounding DINO found boxes for those nouns, and Qwen picked
an index from the results.

Every hop lost information. The noun parser discarded spatial and relational
context ("to the left of", "the one I was reading"). Grounding DINO only ever
saw isolated nouns, never the instruction. And Qwen — the largest model in the
chain — was reduced to choosing from a list someone else produced, without ever
seeing the image.

The replacement is one VLM pass: **Qwen sees the full instruction and the raw
image together** and emits boxes, categories and the relation in a single
schema-constrained response. SAM 2.1 then segments exactly those boxes.

Two stages instead of three, no noun bottleneck, and the strongest model does
the grounding rather than the arbitration.

**Boxes are passed to SAM as prompt geometry, not drawn on the image.** SAM 2.1
has a native box prompt; painting rectangles into the pixels would corrupt the
encoder input at precisely the object boundary the mask decoder needs.

---

## Pipeline

```
/user_instruction ──┐
/ee_camera/image_raw ┴─→ qwen_bridge_node   [HTTP :8000]
                          ├─→ /qwen/labeled_detections  boxes + categories
                          ├─→ /qwen/grounding_result    destination spec → bt_pkg
                          └─→ /qwen/source_image        the exact frame, LAST
                                        ↓ (trigger)
                                  sam_mask_node          [ZMQ :5558]
                                        ├─→ /sam/mask_image       mono8 label map
                                        └─→ /sam/annotated_image  debug overlay
                                        ↓
                    /ee_camera/depth_image ─→ multi_view_projector_node
                                        ├─→ /world_map         labeled cloud
                                        └─→ /world_map_result  centroids + bboxes
```

---

## A. Offline — file in, mask out, no Isaac

Grounding and segmentation only; the projector is skipped since there is no
depth stream to pair with a still frame.

```bash
ros2 launch qwen_a100 slow_brain.launch.py enable_projector:=false image_path:=$PWD/demo/resources/ee_raw.png instruction:="put the book in the box" use_xterm:=false
```

Watch the masks in a second terminal:

```bash
ros2 run rqt_image_view rqt_image_view /sam/annotated_image
```

### What a correct overlay looks like

The **source camera frame**, recognisable as a photo, with the target tinted
green `(0,200,80)` and the destination tinted yellow `(0,220,255)` at 50% blend,
each with a small text label at its top-left corner. Obstacles are not tinted —
only TARGET and DESTINATION are segmented.

The tinted regions should hug the object silhouettes. If they do, the whole
Qwen → SAM handoff is correct.

**Nothing is written to disk.** `sam_mask_node` publishes topics only, and the
A100 stores nothing — it returns mask arrays over ZMQ and forgets them. To get a
file you can look at later, use the CLI (section E) with `--overlay`.

### If the overlay is a flat, washed-out, or uniform field

That is **not** the near-black `/sam/mask_image` (whose values are 1 and 2 — see
section D). A structureless overlay means a mask covered essentially the whole
frame, and the usual cause is the server returning **raw logits** instead of
thresholded masks.

`sam_client._binarise` now handles that: floats outside `[0,1]` are thresholded
at `> 0` (SAM's own cutoff), probabilities at `>= 0.5`, bools and 0/255 ints
directly. Previously the client did `astype(np.uint8)`, which wraps negative
floats — `-8.0` became `248` — so every pixel read as inside the mask.

`sam_mask_node` now also warns when a single mask covers more than
`max_mask_fraction` (0.6) of the frame:

```
[WARN] TARGET 'book' covers 97% of the frame (mask=1) — that is not an object.
       Check whether the server is returning raw logits instead of thresholded
       masks, and whether the box actually bounds the object.
```

If you saw a uniform overlay before this fix, rebuild and retry:

```bash
cd ros_pkgs && colcon build --symlink-install --packages-select sam_a100 && source install/setup.bash
```

## B. Full pipeline with Isaac

```bash
./run_capstone_scene.sh
```
```bash
ros2 launch qwen_a100 slow_brain.launch.py
```

An xterm opens for typing instructions. To run the complete pick, use this in
place of T5/T6 of [demo 04](04_full_pick.md), keeping T1–T4 and T8–T9 — and drop
demo 04's `mask_topic:=/qwen/mask_image` override from T8, since GraspGen already
defaults to `/sam/mask_image` for this pipeline.

## C. Centroids — the actual output of the Slow Brain

Masks are an intermediate. What the behavior tree consumes is
`/world_map_result`: a 3D centroid and bounding box per category, in **metres**,
in the **`panda_link0`** frame. Requires depth, so this needs Isaac (section B).

```bash
ros2 topic echo /world_map_result --once
```

Pretty-print just the centroids:

```bash
ros2 topic echo --once --field data /world_map_result | python3 -m json.tool
```

Expected shape:

```json
{
  "target": {
    "label": "book",
    "centroid": [0.4521, 0.0183, 0.0612],
    "bbox_3d_world": {"min": [0.41, -0.02, 0.05], "max": [0.49, 0.06, 0.07]},
    "point_count": 1834
  },
  "destination": {
    "label": "box",
    "centroid": [0.3104, 0.3455, 0.0498],
    "bbox_3d_world": {"min": [0.25, 0.29, 0.04], "max": [0.37, 0.40, 0.09]},
    "point_count": 2610
  },
  "free":    { "...": "EE pixels with no mask" },
  "unknown": { "...": "top-camera geometry" }
}
```

### Reading it

| Check | Why |
|---|---|
| `target` and `destination` keys both present | A missing key means that category's mask never reached the projector. There is no null placeholder and no error |
| `centroid` is 3 numbers in metres, `panda_link0` | The frame is **implicit** — never declared in the JSON. `bt_pkg` hardcodes `panda_link0` |
| x roughly 0.2–0.8 m | Outside the Panda's reach means the extrinsics are wrong, not the perception |
| z near table height, not 0 | z≈0 for everything is the signature of identity extrinsics |
| `point_count` in the hundreds+ | Tens of points means the mask was a sliver — usually a wrong `bbox_convention` |

**Note what is *not* there.** With the new Slow Brain only TARGET and DESTINATION
are masked, so `/world_map_result` has no per-obstacle keys — the GSAM path
produced one key per detected object. Obstacle avoidance is unaffected: it comes
from the top camera's `unknown` geometry and from the Fast Brain, not from these
labels.

### If the centroids look wrong

```bash
ros2 topic echo /world_map --once --field header
```

Frame must be `world`. Then check the projector's startup log for:

```
Using identity transforms — point cloud will be in camera frame.
```

That message means `extrinsics_config` failed to load and the node **kept
publishing anyway** in the camera frame. Centroids will be self-consistent and
completely wrong in robot coordinates.

Visual check in RViz: add `/world_map` as PointCloud2 — green is target, yellow
destination, purple the top-camera unknown geometry. The target cloud should sit
on the object, not floating above or behind it.

### Downstream

The same centroids feed two consumers:

```bash
ros2 topic echo /grasp_candidates --once
```

`graspgen_node` triggers on `/world_map_result` and semantic-filters its grasps
to those falling inside `target.bbox_3d_world` — so an inflated or misplaced
target bbox rejects every grasp. `bt_pkg` reads `target.centroid` as its
fallback grasp position and `destination.centroid` as the seed for the place
pose.

## D. Inspecting the handoff

```bash
ros2 topic echo /qwen/labeled_detections --once
```
```bash
ros2 topic echo /sam/mask_image --field encoding --once
```

`encoding` must be **`mono8`**. Anything else and the projector will silently
luma-convert it into garbage.

The mask looks almost black in an image viewer — its values are 1 and 2, not
255. That is correct. To confirm the label values:

```bash
ros2 topic echo /sam/mask_image --field data --once | tr ',' '\n' | sort -u | head
```

## E. SAM alone, without ROS

When a mask is empty or wrong and you need to isolate the cause:

```bash
python3 -m sam_a100.sam_cli --image demo/resources/ee_raw.png --box 210 150 330 260 --box 410 180 560 340 --overlay /tmp/sam.png
```

Run it from `ros_pkgs/src/slow_brain/sam_a100/`. Add `--detections` with a saved
`/qwen/labeled_detections` payload to test the realistic path.

---

## Useful launch arguments

| Argument | Default | Purpose |
|---|---|---|
| `enable_sam` | `true` | `false` runs grounding only |
| `enable_projector` | `true` | `false` stops before 3D fusion |
| `image_path` | `""` | Use a file instead of the camera |
| `instruction` | `""` | Seed; with `image_path` it auto-runs once |
| `bbox_convention` | `absolute` | **Verify this first** — see [demo 01](01_qwen_grounding.md) |
| `multimask` | `false` | 3 SAM candidates per box, keep the best — helps on mugs/handles |
| `target_priority` | `true` | TARGET wins where masks overlap |
| `min_mask_pixels` | `50` | Warn below this — usually a wrong `bbox_convention` |
| `max_mask_fraction` | `0.6` | Warn above this — usually unthresholded logits |
| `ee_camera_info_topic` | `/ee_camera/camera_info` | Sets the label-map size |

---

## The contract `sam_mask_node` satisfies

Every one of these fails **silently** if broken, which is why they are worth
knowing when debugging:

| Requirement | How it is met | Failure if wrong |
|---|---|---|
| `mono8` encoding | `cv2_to_imgmsg(..., "mono8")` | cv_bridge luma-converts, labels destroyed |
| Size = **EE depth**, not RGB | Resized to `CameraInfo` H×W, `INTER_NEAREST` | `IndexError`, or silent mislabeling |
| Pixel = 1-based array index | Value is `detection_index + 1` | Wrong category join |
| Detections before mask | Triggers on `source_image`, which Qwen publishes **last** | Trigger dropped, never retried |
| One TARGET, one DESTINATION | Enforced upstream by `order_detections` | `/world_map_result` keys collide |

Two deliberate design choices:

- **Trigger on the image, not the detections.** ROS 2 guarantees no ordering
  across topics, so triggering on Qwen's first publish could fire before the
  image lands.
- **TARGET wins on overlap**, inverting GSAM's "later detection wins". A target
  already sitting over its destination is the common case, and grasp quality
  depends on the target mask staying intact. `target_priority:=false` restores
  GSAM's behaviour.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| ZMQ timeout at 5558 | Tunnel down or server stopped — check `ss -tln \| grep 5558` |
| Overlay is a flat/uniform wash | Mask covers the whole frame — server returning logits, or a garbage box. Look for the `covers N% of the frame` warning |
| Overlay is the raw photo, no tint | Masks came back empty; check the `only N px` warning and `bbox_convention` |
| `/sam/mask_image` looks black | **Correct** — values are 1 and 2, not 255 |
| `source image arrived before detections` | Something other than `qwen_bridge` is publishing `/qwen/source_image` |
| `has only N px — suspect a wrong bbox_convention` | Boxes are in the wrong coordinate space; fix in demo 01 |
| Projector `IndexError` | Mask sized to RGB, not depth — check `ee_camera_info_topic` |
| `no CameraInfo yet` warning | Falls back to RGB size; correct only if RGB and depth match |
| Empty `/world_map_result` | No TARGET mask reached the projector |

---

## Server side

Not yet deployed. The client expects a ZMQ REQ/REP + msgpack server on **5558**:

```python
request  {"action": "segment",
          "image": (H, W, 3) uint8 RGB,
          "boxes": (N, 4) float32 [x1,y1,x2,y2] in image pixels,
          "multimask": bool}
response {"masks": (N, H, W) uint8 0/1, "scores": (N,) float32}
```

**Prefer returning thresholded 0/1 masks.** The client tolerates bools, `[0,1]`
probabilities, `0/255` ints, and raw logits, but thresholding server-side removes
the guesswork — and raw logits were the cause of the flat-overlay failure above.

Implementation note: call `predictor.set_image()` **once** per request and then
one decoder pass per box. The image encoder dominates the cost, so batching the
boxes server-side is much faster than one call per object.

Remaining work:

- ⬜ Verify `bbox_convention` against the live Qwen endpoint ([demo 01](01_qwen_grounding.md))
- ⬜ Confirm `model_name` (`qwen35-local`) matches what vLLM serves
- ⬜ End-to-end validation against `mask_projection_pkg`
