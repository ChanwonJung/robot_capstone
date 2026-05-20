# qwen_pkg

Slow Brain VLM pipeline for the robot capstone. Takes a natural-language user command and a list of GSAM detections, calls a remote Qwen VLM, and outputs labeled detections + a structured grounding result for downstream motion planning.

---

## Nodes

| Node | Executable | Role |
|---|---|---|
| `instruction_prompt_node` | `instruction_prompt_node` | Reads user commands from stdin, publishes to `/user_instruction` |
| `qwen_bridge_node` | `qwen_bridge_node` | Runs VLM inference, publishes labeled detections and grounding result |

---

## Launch

```bash
source launch_env.bash
ros2 launch qwen_pkg inst_input_qwen.launch.py
```

This opens an **xterm window** for `instruction_prompt_node` (stdin input requires a real terminal — `ros2 launch` does not forward stdin) and starts `qwen_bridge_node` in the same session.

Override the vLLM endpoint or model name:
```bash
ros2 launch qwen_pkg inst_input_qwen.launch.py \
  vllm_endpoint_url:=http://<cluster-host>:8000/v1 \
  model_name:=Qwen/Qwen2.5-VL-7B-Instruct
```

### Entering a user instruction

Type your command in the xterm and press Enter:
```
[robot command] > put the cup on the left side of the table
```

Each entry is published immediately to `/user_instruction`. You can update the instruction at any time — `qwen_bridge_node` always uses the most recently received one.

---

## Topic Schema

### Subscribed

| Topic | Type | Published by | Role |
|---|---|---|---|
| `/user_instruction` | `std_msgs/String` | `instruction_prompt_node` | Natural-language command — cached, used on next GSAM trigger |
| `/grounded_sam/detections_json` | `std_msgs/String` | `grounded_sam_node` | GSAM detection list — triggers VLM inference |
| `/grounded_sam/mask_image` | `sensor_msgs/Image` | `grounded_sam_node` | Segmentation mask — cached, re-published after inference |

### Published

| Topic | Type | Consumed by | Role |
|---|---|---|---|
| `/qwen/labeled_detections` | `std_msgs/String` | `multi_view_projector_node` | Detections enriched with `"category"` field |
| `/qwen/grounding_result` | `std_msgs/String` | motion planner (future) | Structured target + destination relation |
| `/qwen/mask_image` | `sensor_msgs/Image` | `multi_view_projector_node` | Mask pass-through — fires the projector |

---

## Firing Order

The bridge enforces strict ordering to guarantee the projector always has fresh labels before it is triggered:

```
grounded_sam_node
  │
  ├─► /grounded_sam/detections_json  ──► qwen_bridge (cached)
  └─► /grounded_sam/mask_image       ──► qwen_bridge (cached)

instruction_prompt_node
  └─► /user_instruction              ──► qwen_bridge (cached)

[ VLM inference — runs on remote cluster via vLLM, ~1–5 s ]

qwen_bridge publishes in this order:
  1. /qwen/labeled_detections   ← projector caches this
  2. /qwen/grounding_result     ← motion planner reads this
  3. /qwen/mask_image           ← triggers multi_view_projector_node
```

`/qwen/mask_image` is always the last message published. This is intentional — `multi_view_projector_node` uses it as the trigger to run projection, so `/qwen/labeled_detections` must already be in its cache before the trigger fires.

Inference is skipped (with a warning) if either `/user_instruction` or `/grounded_sam/mask_image` has not been received yet. Only one VLM call runs at a time — new detections arriving while a call is in progress are dropped.

---

## `/qwen/labeled_detections` format

Same schema as `/grounded_sam/detections_json` with a `"category"` field added:

```json
[
  {"idx": 0, "label": "cup",   "confidence": 0.91, "bbox_xyxy": [10, 20, 80, 90],  "category": "TARGET"},
  {"idx": 1, "label": "table", "confidence": 0.95, "bbox_xyxy": [0, 0, 640, 480],  "category": "DESTINATION"},
  {"idx": 2, "label": "book",  "confidence": 0.82, "bbox_xyxy": [150, 60, 220, 130],"category": "OBSTACLE"}
]
```

Categories: `TARGET`, `DESTINATION`, `OBSTACLE`.

---

## `/qwen/grounding_result` format

Structured JSON describing the target object and its destination relation. The `destination` field is one of three types, each carrying type-specific information:

### `container` — place the target inside the destination object

```json
{
  "target_id": 0,
  "target_label": "cup",
  "destination": {
    "type": "container",
    "reference_id": 1
  },
  "confidence": 0.95
}
```

### `surface` — place the target on a surface at an optional region

```json
{
  "target_id": 0,
  "target_label": "cup",
  "destination": {
    "type": "surface",
    "reference_id": 1,
    "region": "left_edge"
  },
  "confidence": 0.92
}
```

`region` values: `left_edge`, `right_edge`, `center`, `far_end`, `near_end`. Omitted when not specified.

### `relation` — place the target spatially relative to the destination object

```json
{
  "target_id": 0,
  "target_label": "cup",
  "destination": {
    "type": "relation",
    "reference_id": 1,
    "relation": "left_of"
  },
  "confidence": 0.88
}
```

`relation` values: `left_of`, `right_of`, `in_front_of`, `behind`, `on_top_of`, `near`.

---

### Field reference

| Field | Type | Description |
|---|---|---|
| `target_id` | int | `idx` of the TARGET detection in `labeled_detections` |
| `target_label` | str | Label string of the target object |
| `destination.type` | str | `"container"`, `"surface"`, or `"relation"` |
| `destination.reference_id` | int | `idx` of the DESTINATION detection in `labeled_detections` |
| `destination.region` | str? | Surface region — only present when `type == "surface"` |
| `destination.relation` | str | Spatial relation — only present when `type == "relation"` |
| `confidence` | float | VLM confidence in the overall grounding (0.0–1.0) |
