#!/usr/bin/env python3
"""yolo_world_map_node.py

Continuous 3D object position tracker using YOLO detections + top-camera depth.

At each publish tick (~10 Hz):
  1. For each YOLO detection, sample a median depth window around the bbox centre.
  2. Back-project (u, v, z) → camera frame point using cached K matrix.
  3. Transform to world frame (panda_link0) using extrinsics from YAML.
  4. Publish /yolo/world_map as a JSON array of {class_name, centroid, confidence}.
  5. If target_search_label is set, find the nearest object within
     target_search_radius_m of target_seed_centroid and publish to
     /yolo/target_centroid (used by TargetVisible BT condition).

Extrinsics convention (same as mask_projection_pkg):
  p_world = R @ p_cam + t
"""
from __future__ import annotations

import json
import time
from typing import Optional

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


def _load_extrinsics(path: str, camera_key: str):
    """Return (R 3×3, t 3×1, ok) from a camera_extrinsics_*.yaml file.

    The YAML format mirrors mask_projection_pkg's camera_extrinsics.yaml, whose
    keys are `R` and `t`.  `rotation`/`translation` are accepted as aliases
    because an earlier revision of this loader only knew those names — it
    therefore threw KeyError on every real file and silently fell back to
    identity, publishing camera-frame centroids that look like robot-frame ones.

    `ok` is False when a path was given but could not be read. The caller must
    not publish in that case: a centroid in the camera frame is not a degraded
    answer, it is a wrong one, and it flows straight into the BT's place pose.
    """
    if not path:
        return np.eye(3), np.zeros(3), True
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        cam = cfg.get(camera_key)
        if not cam:
            raise KeyError(f"no '{camera_key}' block")
        rot = cam.get("R", cam.get("rotation"))
        trans = cam.get("t", cam.get("translation"))
        if rot is None or trans is None:
            raise KeyError("expected 'R' and 't' (or 'rotation'/'translation')")
        R = np.array(rot, dtype=float).reshape(3, 3)
        t = np.array(trans, dtype=float).reshape(3)
        return R, t, True
    except Exception as e:
        import rclpy.logging
        rclpy.logging.get_logger("yolo_world_map").error(
            f"Could not load extrinsics for '{camera_key}' from '{path}': {e}"
            " — /yolo/world_map will NOT be published")
        return np.eye(3), np.zeros(3), False


class YoloWorldMapNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_world_map_node")

        self.declare_parameter("extrinsics_config",      "")
        self.declare_parameter("detections_topic",       "/yolo_hazard/top/detections_json")
        # Isaac publishes the top RGBD camera under /rgbd_camera/, NOT
        # /top_camera/. The old defaults matched nothing, so _latest_depth stayed
        # None forever and this node never published a single world map.
        # robot_defaults.yaml cannot supply these either: it names them
        # top_depth_topic / top_camera_info_topic, and ROS 2 silently drops YAML
        # keys that do not match a declared parameter.
        self.declare_parameter("depth_topic",            "/rgbd_camera/depth_image")
        self.declare_parameter("camera_info_topic",      "/rgbd_camera/camera_info")
        self.declare_parameter("world_map_topic",        "/yolo/world_map")
        self.declare_parameter("target_centroid_topic",  "/yolo/target_centroid")
        self.declare_parameter("publish_rate_hz",        10.0)
        self.declare_parameter("depth_sample_window",    5)
        self.declare_parameter("target_search_radius_m", 0.2)

        ext_path   = str(self.get_parameter("extrinsics_config").value)
        det_topic  = str(self.get_parameter("detections_topic").value)
        dep_topic  = str(self.get_parameter("depth_topic").value)
        info_topic = str(self.get_parameter("camera_info_topic").value)
        wm_topic   = str(self.get_parameter("world_map_topic").value)
        tc_topic   = str(self.get_parameter("target_centroid_topic").value)
        rate       = float(self.get_parameter("publish_rate_hz").value)
        self._win  = int(self.get_parameter("depth_sample_window").value)
        self._r    = float(self.get_parameter("target_search_radius_m").value)

        # Extrinsics for the top camera
        self._R, self._t, self._extrinsics_ok = _load_extrinsics(ext_path, "top_camera")

        # Camera intrinsics (populated from /top_camera/camera_info)
        self._K: Optional[np.ndarray] = None

        # Cached sensor data (updated by callbacks)
        self._latest_detections: Optional[list] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._bridge = CvBridge()

        # For TargetVisible: set externally by the BT (via ROS parameter update)
        self._target_label: str = ""
        self._target_seed: Optional[np.ndarray] = None

        # Subscriptions
        self.create_subscription(String,     det_topic,  self._det_cb,  10)
        self.create_subscription(Image,      dep_topic,  self._depth_cb, 10)
        self.create_subscription(CameraInfo, info_topic, self._info_cb,  10)

        # Publishers
        self._pub_wm = self.create_publisher(String,       wm_topic, 10)
        self._pub_tc = self.create_publisher(PointStamped, tc_topic, 10)

        # Tick
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f"YoloWorldMapNode ready: det='{det_topic}' depth='{dep_topic}' "
            f"R={self._R.tolist()} t={self._t.tolist()}")

    # ── callbacks ──────────────────────────────────────────────────────────

    def _det_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self._latest_detections = data["hazard_detections"]["detections"]
        except Exception:
            self._latest_detections = []

    def _depth_cb(self, msg: Image) -> None:
        try:
            self._latest_depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().warn(f"depth decode failed: {e}", throttle_duration_sec=5.0)

    def _info_cb(self, msg: CameraInfo) -> None:
        if self._K is None:
            self._K = np.array(msg.k, dtype=float).reshape(3, 3)
            self.get_logger().info(f"Camera K loaded: fx={self._K[0,0]:.1f}")

    # ── main tick ──────────────────────────────────────────────────────────

    def _tick(self) -> None:
        # Refusing to publish beats publishing camera-frame centroids that the
        # BT would treat as panda_link0 metres.
        if not self._extrinsics_ok:
            self.get_logger().error(
                "extrinsics failed to load — not publishing /yolo/world_map",
                throttle_duration_sec=10.0)
            return

        if self._latest_detections is None or self._latest_depth is None or self._K is None:
            missing = [
                name for name, val in (
                    ("detections", self._latest_detections),
                    ("depth", self._latest_depth),
                    ("camera_info", self._K),
                ) if val is None
            ]
            # Without this the node looks healthy while publishing nothing —
            # exactly the failure mode that hid the wrong depth topic.
            self.get_logger().warn(
                f"waiting for {', '.join(missing)} — /yolo/world_map idle",
                throttle_duration_sec=10.0)
            return

        objects = []
        for det in self._latest_detections:
            centroid = self._backproject(det)
            if centroid is None:
                continue
            objects.append({
                "class_id":   det.get("class_id", -1),
                "class_name": det.get("class_name", "unknown"),
                "centroid":   centroid.tolist(),
                "confidence": det.get("confidence", 0.0),
            })

        # Publish world map
        out = String()
        out.data = json.dumps({"objects": objects, "stamp": time.time()})
        self._pub_wm.publish(out)

        # Publish target centroid if tracking is active
        self._maybe_publish_target(objects)

    def _backproject(self, det: dict) -> Optional[np.ndarray]:
        """Project a YOLO detection's bbox centre to world frame."""
        try:
            bbox = det["bbox"]
            u = int(bbox["x"] + bbox["width"]  / 2)
            v = int(bbox["y"] + bbox["height"] / 2)
        except (KeyError, TypeError):
            return None

        depth_map = self._latest_depth
        h, w = depth_map.shape

        # Median depth over a small window (handles noisy depth)
        hw = self._win // 2
        u0, u1 = max(0, u - hw), min(w, u + hw + 1)
        v0, v1 = max(0, v - hw), min(h, v + hw + 1)
        patch = depth_map[v0:v1, u0:u1]
        valid = patch[np.isfinite(patch) & (patch > 0.01)]
        if valid.size == 0:
            return None
        z = float(np.median(valid))

        # Back-project to camera frame
        fx, fy = self._K[0, 0], self._K[1, 1]
        cx, cy = self._K[0, 2], self._K[1, 2]
        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy
        p_cam = np.array([x_cam, y_cam, z])

        # Transform to world (panda_link0)
        p_world = self._R @ p_cam + self._t
        return p_world

    def _maybe_publish_target(self, objects: list) -> None:
        """Find and publish the nearest object to the target seed."""
        # Allow the BT to update target tracking via ROS parameter at runtime.
        # ParseScene calls set_parameters({target_search_label: target_label}).
        label_param = self.get_parameter("target_search_label") \
            if self.has_parameter("target_search_label") else None

        if label_param:
            self._target_label = str(label_param.value)

        if not self._target_label:
            return

        # Seed centroid comes from a separate parameter set by ParseScene.
        if self.has_parameter("target_seed_centroid"):
            seed_raw = self.get_parameter("target_seed_centroid").value
            if seed_raw:
                try:
                    self._target_seed = np.array(list(seed_raw), dtype=float)
                except Exception:
                    pass

        if self._target_seed is None:
            return

        best_d  = self._r
        best_pt = None

        for obj in objects:
            if obj["class_name"] != self._target_label:
                continue
            c = np.array(obj["centroid"])
            d = float(np.linalg.norm(c - self._target_seed))
            if d < best_d:
                best_d  = d
                best_pt = c

        if best_pt is None:
            return

        msg = PointStamped()
        msg.header.frame_id = "panda_link0"
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.point           = Point(x=best_pt[0], y=best_pt[1], z=best_pt[2])
        self._pub_tc.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloWorldMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # ros2 launch already shut the context down on SIGINT; calling it again
        # raises and makes every Ctrl-C print a traceback plus "process has died
        # [exit code 1]", which reads like a crash in the logs we actually care
        # about.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
