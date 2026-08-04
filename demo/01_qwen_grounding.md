# 01 — Qwen Grounding

**Shows:** a natural-language instruction plus a single camera frame becoming a
structured scene understanding — which object to pick, where to put it, and the
spatial relation between them.

**Needs:** the port-8000 tunnel. No Isaac Sim, no robot, no other ROS nodes.

This is the highest-value demo to run first: it exercises the entire remote
inference path and is the only place the bounding-box coordinate convention can
be checked visually.

---

## A. Offline — no ROS at all

The fastest feedback loop. Isolates the VLM from every piece of ROS wiring.

```bash
python3 ros_pkgs/src/slow_brain/qwen_a100/qwen_a100/qwen_cli.py --image demo/resources/ee_raw.png --instruction "put the book in the box" --annotate /tmp/boxes.png
```

Expected output:

```
==============================================================
Qwen grounding result
==============================================================
source        : 631x401
instruction   : put the book in the box
target_label  : book
destination   : box  type=container relation=- region=-
confidence    : 0.92

3 objects (array order == mask label value):
  [mask=1] TARGET      book                 box=(210,150)-(330,260) conf=0.94
  [mask=2] DESTINATION box                  box=(410,180)-(560,340) conf=0.91
  [mask=3] OBSTACLE    cup                  box=(80,200)-(150,300)  conf=0.88
```

### Then open `/tmp/boxes.png` — this is the important part

Green = target, yellow = destination, red = obstacles.

**If the boxes do not sit on the objects, the coordinate convention is wrong.**
Retry with:

```bash
python3 ros_pkgs/src/slow_brain/qwen_a100/qwen_a100/qwen_cli.py --image demo/resources/ee_raw.png --instruction "put the book in the box" --bbox-convention normalized_1000 --annotate /tmp/boxes.png
```

Whichever produces correctly-placed boxes is the right value for
`bbox_convention` in `qwen_a100_params.yaml`. This matters more than it looks:
a wrong convention produces confidently incorrect masks with **no error
anywhere downstream**.

### Worth demonstrating: ambiguity handling

Same image, vaguer instruction — shows the VLM resolving under-specification
rather than pattern-matching a noun:

```bash
python3 ros_pkgs/src/slow_brain/qwen_a100/qwen_a100/qwen_cli.py --image demo/resources/ee_raw.png --instruction "tidy up the thing I was reading"
```

And a spatial-relation case, which exercises the `relation` vocabulary the
behavior tree consumes:

```bash
python3 ros_pkgs/src/slow_brain/qwen_a100/qwen_a100/qwen_cli.py --image demo/resources/ee_raw.png --instruction "put the book to the left of the cup"
```

Expect `type=relation relation=left_of`.

---

## B. The ROS path, still without Isaac

`image_path` replaces the camera topic with a file. Pairing it with
`instruction` auto-runs once, about a second after startup.

```bash
ros2 launch qwen_a100 qwen_a100.launch.py image_path:=$PWD/demo/resources/ee_raw.png instruction:="put the book in the box" use_xterm:=false
```

In a second terminal (`source launch_env.bash` first):

```bash
ros2 topic echo /qwen/grounding_result --once
```
```bash
ros2 topic echo /qwen/labeled_detections --once
```

Both topics are latched, so you can echo them *after* the run completes — no
need to race the output.

## C. Interactive, as in the real system

Drop `image_path` and `instruction`, and an xterm opens for typing commands.
Requires a live camera on `/ee_camera/image_raw`, so run Isaac first (demo 03/04)
or supply a publisher.

```bash
ros2 launch qwen_a100 qwen_a100.launch.py
```

---

## What you should see in the log

```
[WARN] TEST MODE — using static image ... (631x401); image_topic is NOT subscribed
[INFO] qwen_bridge ready — endpoint=http://localhost:8000/v1 model=...
[INFO] offline one-shot — grounding seeded instruction
[INFO] grounded in 2.4s — 3 objects, target='book' dest='box' type=container relation=- region=-
```

## Troubleshooting

| Symptom | Cause |
|---|---|
| `Connection error` in **under 1 s** | Tunnel down — `ss -tln \| grep 8000` |
| Failure after a long pause | Model down or overloaded on the A100 |
| `404` / model not found | `model_name` mismatch — should be `qwen35-local`; verify with `curl -s localhost:8000/v1/models` |
| `no camera frame cached yet` | No `image_path` and nothing publishing on `image_topic` |
| Boxes drawn in the wrong places | `bbox_convention` — see section A |
| Boxes look plausible but clipped at the frame edge | `clamp_boxes` clipped them; a wrong convention can hide behind this. Trust the overlay, not the numbers |
| `VLM returned no TARGET` | Instruction names nothing visible in the frame |
| Import error mentioning pydantic | `launch_env.bash` not sourced in this terminal |

## Pipeline status

Every scan publishes a machine-readable state, so a failure is visible without
watching the node's console:

```bash
ros2 topic echo /slow_brain/status
```

```json
{"scan_id": 3, "stage": "qwen_grounding", "state": "ok", "reason": "",
 "target": "book", "destination": "box", "pick_only": false,
 "n_objects": 3, "confidence": 0.92, "latency_s": 2.41}
```

`state` is `running` | `ok` | `failed`. On failure `reason` carries the cause —
`no_target` plus the labels that *were* detected is the common one. The topic is
latched, so you can read the last scan's outcome at any time.

Diagnostic only: nothing in the pipeline gates on it, by design.

## Pick-only instructions

An instruction with no destination is valid, not an error:

```bash
python3 ros_pkgs/src/slow_brain/qwen_a100/qwen_a100/qwen_cli.py --image demo/resources/ee_raw.png --instruction "pick up the book"
```

`destination` is **omitted** from the grounding result and the log says
`pick-only instruction — no destination`. The target still publishes, so the BT
can pick and hold; only the place phase is unavailable.

Severity is deliberately split: **no target is fatal** (nothing to grasp, so
nothing publishes and the previous latched scan stands), **no destination is
not** (publish the target and skip placing).

## Known limits

- One call at a time. A second instruction arriving mid-call is **dropped** with
  a warning, not queued.
- No retry — a single network blip fails the scan.
- Clarification is not implemented: an ambiguous instruction gets the VLM's best
  guess with a lowered confidence, not a question.
