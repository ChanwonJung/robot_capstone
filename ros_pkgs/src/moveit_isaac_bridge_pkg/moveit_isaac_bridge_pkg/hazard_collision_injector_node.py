"""Inject Fast Brain hazards into the MoveIt planning scene as collision objects.

Pipeline (top-view camera only):
    YOLO hazard detection JSON (polygon, class, conf)
      + aligned depth image  + camera intrinsics (K)
      + static top-camera extrinsics (R, t)  in panda_link0 frame
    --> back-project the segmented polygon to 3D points in panda_link0
    --> box CollisionObject (XY from the points' AABB, Z extended downward
        because an overhead view only sees the object's top surface)
    --> publish on /collision_object (ADD); REMOVE after the hazard clears.

Why top-view only: the top camera is statically mounted, so its extrinsics are
a fixed YAML (see mask_projection_pkg/config/camera_extrinsics.yaml). The
eye-in-hand camera moves with the arm, so its world transform is not static and
is intentionally not used here.

This is Phase 1 of the avoidance stack — it gets the hazard *into* the planning
scene. Reacting to it (slow down / stop / resume) is the job of the hybrid
planning layer that consumes this scene.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String


def _default_extrinsics_path() -> str:
    try:
        share = get_package_share_directory("mask_projection_pkg")
        return os.path.join(share, "config", "camera_extrinsics.yaml")
    except Exception:
        return ""


class HazardCollisionInjectorNode(Node):
    def __init__(self) -> None:
        super().__init__("hazard_collision_injector_node")

        self.declare_parameter("detection_topic", "/yolo_hazard/top/detections_json")
        self.declare_parameter("depth_topic", "/rgbd_camera/depth_image")
        self.declare_parameter("camera_info_topic", "/rgbd_camera/camera_info")
        self.declare_parameter("collision_object_topic", "/collision_object")
        self.declare_parameter("extrinsics_config", _default_extrinsics_path())
        self.declare_parameter("extrinsics_key", "top_camera")
        self.declare_parameter("planning_frame", "panda_link0")
        # Fast Brain 3-class: 0=arm, 1=bottle, 2=box.
        self.declare_parameter("trigger_class_ids", [0, 1, 2])
        self.declare_parameter("conf_threshold", 0.5)
        self.declare_parameter("min_depth", 0.05)
        self.declare_parameter("max_depth", 15.0)
        self.declare_parameter("min_valid_points", 15)
        # Overhead view only sees the top surface; extend the box downward by this
        # to give the obstacle a body. Tune per expected hazard height.
        self.declare_parameter("default_obstacle_height", 0.25)
        self.declare_parameter("xy_margin", 0.02)
        self.declare_parameter("min_box_size", 0.03)
        # Clip the back-projected points to [p, 100-p] percentile per axis before
        # sizing the box. Rejects silhouette-edge pixels whose depth straddles the
        # object/background boundary and would otherwise inflate the AABB.
        self.declare_parameter("extent_percentile", 5.0)
        self.declare_parameter("object_id_prefix", "hazard_")
        # Remove a collision object if its class is not re-detected within this
        # window — lets the scene clear so the planner can resume.
        self.declare_parameter("clear_timeout_sec", 0.5)

        self._detection_topic = str(self.get_parameter("detection_topic").value)
        self._depth_topic = str(self.get_parameter("depth_topic").value)
        self._info_topic = str(self.get_parameter("camera_info_topic").value)
        self._co_topic = str(self.get_parameter("collision_object_topic").value)
        self._ext_key = str(self.get_parameter("extrinsics_key").value)
        self._frame = str(self.get_parameter("planning_frame").value)
        self._trigger = set(int(c) for c in self.get_parameter("trigger_class_ids").value)
        self._conf = float(self.get_parameter("conf_threshold").value)
        self._min_depth = float(self.get_parameter("min_depth").value)
        self._max_depth = float(self.get_parameter("max_depth").value)
        self._min_pts = int(self.get_parameter("min_valid_points").value)
        self._obs_height = float(self.get_parameter("default_obstacle_height").value)
        self._xy_margin = float(self.get_parameter("xy_margin").value)
        self._min_box = float(self.get_parameter("min_box_size").value)
        self._extent_pct = float(self.get_parameter("extent_percentile").value)
        self._id_prefix = str(self.get_parameter("object_id_prefix").value)
        self._clear_timeout = float(self.get_parameter("clear_timeout_sec").value)

        ext_path = str(self.get_parameter("extrinsics_config").value)
        self._R, self._t = self._load_extrinsics(ext_path, self._ext_key)

        # Republish-skip threshold: don't re-publish a CollisionObject when
        # its new box center has drifted less than this from the previously
        # published one (per object id). Default 1 cm — absorbs YOLO mask
        # pixel jitter (typically ±1-2 px → ±5-10 mm in world) so the
        # planning scene's box position is STABLE while the hazard sits
        # still. Critical for the replan demo: if the box jitters between
        # what local sees (when it decides to hold) and what the global
        # CheckStartStateCollision adapter sees (a few ms later), the arm
        # appears to drift in/out of contact and the manager double-aborts.
        self.declare_parameter("stable_position_threshold", 0.01)
        self._stable_thresh = float(
            self.get_parameter("stable_position_threshold").value)

        self._depth_msg: Optional[Image] = None
        self._K: Optional[np.ndarray] = None
        # id -> last-seen monotonic seconds
        self._active: Dict[str, float] = {}
        # id -> last-published (cx, cy, cz, dx, dy) tuple. Drives the
        # stability filter above. cz is the box CENTER z (= top_z - dz/2).
        # dz is constant per session so we don't need to store it.
        self._last_box: Dict[str, tuple] = {}

        self._co_pub = self.create_publisher(CollisionObject, self._co_topic, 10)
        self.create_subscription(Image, self._depth_topic, self._depth_cb, 10)
        self.create_subscription(CameraInfo, self._info_topic, self._info_cb, 10)
        self.create_subscription(String, self._detection_topic, self._detection_cb, 10)
        self.create_timer(0.2, self._cleanup_stale)

        self.get_logger().info(
            f"HazardCollisionInjector ready: detections={self._detection_topic}, "
            f"depth={self._depth_topic}, frame={self._frame}, "
            f"trigger={sorted(self._trigger)}, conf>={self._conf}, "
            f"extrinsics_key={self._ext_key}"
        )

    def _load_extrinsics(self, path: str, key: str):
        identity = (np.eye(3), np.zeros(3))
        if not path or not os.path.isfile(path):
            self.get_logger().error(
                f"Extrinsics file not found: '{path}'. Using identity — "
                f"collision objects will be in the camera frame (wrong)."
            )
            return identity
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            cam = data[key]
            R = np.array(cam["R"], dtype=np.float64).reshape(3, 3)
            t = np.array(cam["t"], dtype=np.float64).reshape(3)
            self.get_logger().info(f"Loaded '{key}' extrinsics from {path}")
            return R, t
        except (KeyError, ValueError, TypeError) as exc:
            self.get_logger().error(
                f"Failed to parse '{key}' from {path}: {exc}. Using identity."
            )
            return identity

    def _depth_cb(self, msg: Image) -> None:
        self._depth_msg = msg

    def _info_cb(self, msg: CameraInfo) -> None:
        self._K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def _decode_depth(self, msg: Image) -> Optional[np.ndarray]:
        if msg.encoding != "32FC1":
            self.get_logger().warn(
                f"Unexpected depth encoding '{msg.encoding}' (expected 32FC1)"
            )
            return None
        arr = np.frombuffer(bytes(msg.data), dtype=np.float32)
        arr = arr.reshape(msg.height, msg.step // 4)[:, : msg.width]
        return arr

    def _detection_cb(self, msg: String) -> None:
        if self._depth_msg is None or self._K is None:
            return
        try:
            payload = json.loads(msg.data)["hazard_detections"]
            detections = payload["detections"]
            det_w = int(payload["image_width"])
            det_h = int(payload["image_height"])
        except (ValueError, KeyError, TypeError):
            return

        depth = self._decode_depth(self._depth_msg)
        if depth is None:
            return
        dh, dw = depth.shape

        fx, fy = self._K[0, 0], self._K[1, 1]
        cx, cy = self._K[0, 2], self._K[1, 2]
        sx = dw / det_w if det_w else 1.0
        sy = dh / det_h if det_h else 1.0

        now = self._now_sec()
        idx_by_class: Dict[str, int] = {}
        for det in detections:
            try:
                cls_id = int(det["class_id"])
                conf = float(det["confidence"])
                polygon = det.get("polygon") or []
            except (KeyError, ValueError, TypeError):
                continue
            if cls_id not in self._trigger or conf < self._conf or len(polygon) < 3:
                continue

            poly = np.array(polygon, dtype=np.float64)
            poly[:, 0] *= sx
            poly[:, 1] *= sy
            pts_world = self._polygon_to_world(poly, depth, fx, fy, cx, cy)
            if pts_world is None:
                continue

            class_name = det.get("class_name", str(cls_id))
            i = idx_by_class.get(class_name, 0)
            idx_by_class[class_name] = i + 1
            obj_id = f"{self._id_prefix}{class_name}_{i}"
            self._publish_box(obj_id, pts_world)
            self._active[obj_id] = now

    def _polygon_to_world(self, poly, depth, fx, fy, cx, cy) -> Optional[np.ndarray]:
        dh, dw = depth.shape
        mask = np.zeros((dh, dw), dtype=np.uint8)
        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)

        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return None
        d = depth[ys, xs]
        valid = np.isfinite(d) & (d > self._min_depth) & (d < self._max_depth)
        if int(valid.sum()) < self._min_pts:
            return None
        xs, ys, d = xs[valid], ys[valid], d[valid]

        x_cam = (xs - cx) * d / fx
        y_cam = (ys - cy) * d / fy
        pts_cam = np.stack([x_cam, y_cam, d], axis=1)  # optical: +Z fwd, +X right, +Y down
        return pts_cam @ self._R.T + self._t

    def _publish_box(self, obj_id: str, pts_world: np.ndarray) -> None:
        p = self._extent_pct
        lo = np.percentile(pts_world, p, axis=0)
        hi = np.percentile(pts_world, 100.0 - p, axis=0)

        dx = max(float(hi[0] - lo[0]) + self._xy_margin, self._min_box)
        dy = max(float(hi[1] - lo[1]) + self._xy_margin, self._min_box)
        dz = self._obs_height
        # Box centred on the clipped AABB in XY; overhead view sees the top
        # surface (hi[2]) so the box hangs downward from there.
        cx = float(lo[0] + hi[0]) / 2.0
        cy = float(lo[1] + hi[1]) / 2.0
        top_z = float(hi[2])
        cz = top_z - dz / 2.0

        # Stability filter — skip republish when the new box is essentially
        # the same as the last one. Keeps the planning scene quiescent so
        # the local stop position and the global CheckStartStateCollision
        # adapter see the SAME box geometry, instead of one whose XYZ jitters
        # by a few mm between consecutive frames.
        prev = self._last_box.get(obj_id)
        if prev is not None:
            pcx, pcy, pcz, pdx, pdy = prev
            pos_drift = max(abs(cx - pcx), abs(cy - pcy), abs(cz - pcz))
            size_drift = max(abs(dx - pdx), abs(dy - pdy))
            if pos_drift < self._stable_thresh and size_drift < self._stable_thresh:
                # Refresh staleness so _cleanup_stale doesn't remove the
                # still-detected object just because we skipped a publish.
                self._active[obj_id] = self._now_sec()
                return
        self._last_box[obj_id] = (cx, cy, cz, dx, dy)

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [dx, dy, dz]

        pose = Pose()
        pose.position.x = cx
        pose.position.y = cy
        pose.position.z = cz
        pose.orientation.w = 1.0

        co = CollisionObject()
        co.header.frame_id = self._frame
        co.header.stamp = self.get_clock().now().to_msg()
        co.id = obj_id
        co.primitives.append(box)
        co.primitive_poses.append(pose)
        co.operation = CollisionObject.ADD
        self._co_pub.publish(co)

    def _cleanup_stale(self) -> None:
        now = self._now_sec()
        stale = [oid for oid, ts in self._active.items() if now - ts > self._clear_timeout]
        for oid in stale:
            co = CollisionObject()
            co.header.frame_id = self._frame
            co.header.stamp = self.get_clock().now().to_msg()
            co.id = oid
            co.operation = CollisionObject.REMOVE
            self._co_pub.publish(co)
            # Drop the cached box so a re-appearance gets a fresh ADD
            # (otherwise the stability filter would skip the new publish).
            self._last_box.pop(oid, None)
            del self._active[oid]
            self.get_logger().info(f"Hazard '{oid}' cleared — removed from planning scene")

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HazardCollisionInjectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
