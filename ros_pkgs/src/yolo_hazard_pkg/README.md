# yolo_hazard_pkg

Fast Brain hazard detection node. Runs a YOLO instance-segmentation model on a camera stream and publishes detections in the `hazard_detections` JSON schema, intended for downstream injection as dynamic collision objects in MoveIt's planning scene.

## Target classes

After custom training:

| class_id | name        |
|----------|-------------|
| 0        | hand        |
| 1        | forearm     |
| 2        | pet_bottle  |
| 3        | small_box   |

Default weights point to `yolo26s-seg.pt` (COCO 80) so the pipeline can be wired up and validated before custom weights are ready. Swap `config/model_paths.yaml` once the trained `.pt` (or exported `.engine`) is available.

## Install

```bash
# in the workspace's Python environment
pip install -r src/yolo_hazard_pkg/requirements.txt
```

`ultralytics` will auto-download pretrained weights on first run.

## Build

```bash
cd ros_pkgs
colcon build --packages-select yolo_hazard_pkg
source install/setup.bash
```

## Run

Top-view camera only:

```bash
ros2 launch yolo_hazard_pkg yolo_hazard_top.launch.py
```

Eye-in-hand camera only:

```bash
ros2 launch yolo_hazard_pkg yolo_hazard_ee.launch.py
```

Both cameras simultaneously (two nodes, isolated processes):

```bash
ros2 launch yolo_hazard_pkg yolo_hazard_both.launch.py
```

## Topics

**Subscribed**

- `/camera/image_raw` — top-view RGB (sensor_msgs/Image)
- `/ee_camera/image_raw` — eye-in-hand RGB

**Published**

- `/yolo_hazard/top/detections_json` — std_msgs/String, JSON payload
- `/yolo_hazard/ee/detections_json` — std_msgs/String, JSON payload
- `/yolo_hazard/*/annotated_image` — sensor_msgs/Image (only when `publish_annotated: true`)

## JSON schema

```json
{
  "hazard_detections": {
    "timestamp": 1716000000000000000,
    "frame_id": "top_camera_link",
    "image_width": 1280,
    "image_height": 720,
    "detections": [
      {
        "class_id": 0,
        "class_name": "hand",
        "confidence": 0.91,
        "bbox": {"x": 412, "y": 188, "width": 96, "height": 132},
        "polygon": [[420, 188], [430, 192], ...]
      }
    ]
  }
}
```

Identical schema to the Roboflow workflow test output — downstream consumers can handle both sources interchangeably.

## Swapping weights after training

1. Drop the trained file (e.g. `best.pt`) into `models/yolo26/`.
2. Edit `config/model_paths.yaml`:
   ```yaml
   weights: "models/yolo26/best.pt"
   ```
3. Optional — export to TensorRT for FPS:
   ```bash
   yolo export model=models/yolo26/best.pt format=engine half=True
   # produces models/yolo26/best.engine — point model_paths.yaml at it
   ```
4. Set `filter_by_class: true` and `class_allowlist: [0, 1, 2, 3]` in `config/runtime.yaml`.
5. Rebuild and relaunch.

No node code changes required.
