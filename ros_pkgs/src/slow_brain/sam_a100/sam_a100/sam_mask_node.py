"""ROS 2 node: Qwen boxes + source frame -> mono8 label map for the projector.

Wiring only.  Transport lives in sam_client.py, the label-map contract in
label_map.py.

Subscribes
  /qwen/source_image           sensor_msgs/Image   **TRIGGER**
  /qwen/labeled_detections     std_msgs/String     cached
  <ee_camera_info_topic>       sensor_msgs/CameraInfo  cached — gives the depth
                                                   resolution the mask must match

Publishes
  /sam/mask_image              sensor_msgs/Image   mono8 label map, latched
  /sam/annotated_image         sensor_msgs/Image   bgr8 debug overlay, optional

Why the trigger is the IMAGE and not the detections: qwen_bridge publishes
detections, then grounding, then source_image LAST.  ROS 2 gives no ordering
guarantee across topics, so triggering on the first publish can fire before the
rest arrive.  Triggering on the last one is the same trick GSAM uses when it
publishes its mask last for the projector.

Only TARGET and DESTINATION are segmented.  Obstacles keep their slots in the
detections array (so the position-based join stays valid) but get no mask, and
therefore land in the projector as background/FREE.  That is intentional: the
top camera still contributes UNKNOWN geometry for obstacle avoidance.
"""
from __future__ import annotations

import json
import threading
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from .label_map import compose_label_map, mask_stats
from .sam_client import SamClient

LATCHED = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

SEGMENT_CATEGORIES = ("TARGET", "DESTINATION")


class SamMaskNode(Node):

    def __init__(self) -> None:
        super().__init__("sam_mask_node")

        self.declare_parameter("zmq_host", "127.0.0.1")
        self.declare_parameter("zmq_port", 5558)
        self.declare_parameter("zmq_timeout_ms", 30000)

        self.declare_parameter("source_image_topic", "/qwen/source_image")
        self.declare_parameter("detections_topic", "/qwen/labeled_detections")
        self.declare_parameter("ee_camera_info_topic", "/ee_rgbd_camera/camera_info")
        self.declare_parameter("mask_topic", "/sam/mask_image")
        self.declare_parameter("annotated_topic", "/sam/annotated_image")

        # Fallback when no CameraInfo has arrived. Empty means "use the source
        # image size", which is only correct if RGB and depth match.
        self.declare_parameter("mask_width", 0)
        self.declare_parameter("mask_height", 0)

        self.declare_parameter("multimask", False)
        self.declare_parameter("target_priority", True)
        self.declare_parameter("min_mask_pixels", 50)
        # A single mask covering more than this fraction of the frame is almost
        # never a real object. Usual cause: the server returned raw logits and
        # something downstream treated every pixel as inside the mask.
        self.declare_parameter("max_mask_fraction", 0.6)
        self.declare_parameter("publish_annotated", True)

        self._timeout_ms = int(self.get_parameter("zmq_timeout_ms").value)
        self._multimask = bool(self.get_parameter("multimask").value)
        self._target_priority = bool(self.get_parameter("target_priority").value)
        self._min_pixels = int(self.get_parameter("min_mask_pixels").value)
        self._max_fraction = float(self.get_parameter("max_mask_fraction").value)

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._busy = False
        self._detections: list[dict] | None = None
        self._depth_hw: tuple[int, int] | None = None

        self._client = SamClient(
            host=self.get_parameter("zmq_host").value,
            port=int(self.get_parameter("zmq_port").value),
            timeout_ms=self._timeout_ms,
        )

        self.create_subscription(
            String, self.get_parameter("detections_topic").value,
            self._detections_cb, 10)
        self.create_subscription(
            CameraInfo, self.get_parameter("ee_camera_info_topic").value,
            self._info_cb, 10)
        self.create_subscription(
            Image, self.get_parameter("source_image_topic").value,
            self._image_cb, 1)

        self._mask_pub = self.create_publisher(
            Image, self.get_parameter("mask_topic").value, LATCHED)
        self._annotated_pub = (
            self.create_publisher(
                Image, self.get_parameter("annotated_topic").value, 1)
            if self.get_parameter("publish_annotated").value else None)

        w = int(self.get_parameter("mask_width").value)
        h = int(self.get_parameter("mask_height").value)
        if w > 0 and h > 0:
            self._depth_hw = (h, w)
            self.get_logger().info(f"mask size pinned to {w}x{h} by parameter")

        self.get_logger().info(
            f"sam_mask_node ready — server "
            f"{self.get_parameter('zmq_host').value}:"
            f"{self.get_parameter('zmq_port').value}")

    # ── callbacks ────────────────────────────────────────────────────────────

    def _detections_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"detections parse error: {exc}")
            return
        with self._lock:
            self._detections = data

    def _info_cb(self, msg: CameraInfo) -> None:
        """Depth resolution — the mask MUST match this, not the RGB size."""
        with self._lock:
            self._depth_hw = (msg.height, msg.width)

    def _image_cb(self, msg: Image) -> None:
        with self._lock:
            if self._busy:
                self.get_logger().warn("segmentation in flight — frame dropped")
                return
            detections = self._detections
            if detections is None:
                self.get_logger().warn(
                    "source image arrived before detections — dropping. "
                    "qwen_bridge should publish detections first.")
                return
            depth_hw = self._depth_hw
            self._busy = True

        threading.Thread(target=self._run, args=(msg, detections, depth_hw),
                         daemon=True).start()

    # ── worker ───────────────────────────────────────────────────────────────

    def _run(self, img_msg: Image, detections: list[dict],
             depth_hw: tuple[int, int] | None) -> None:
        t0 = time.monotonic()
        try:
            image_bgr = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            src_h, src_w = image_bgr.shape[:2]

            # Which detections to segment, and their array positions — the
            # position IS the mask value (minus one), so it must come from the
            # original array, not from the filtered subset.
            picks = [(i, d) for i, d in enumerate(detections)
                     if d.get("category", "").upper() in SEGMENT_CATEGORIES]
            if not picks:
                self.get_logger().error(
                    "no TARGET/DESTINATION in detections — nothing to segment")
                return

            indices = [i for i, _ in picks]
            boxes = np.array([d["bbox_xyxy"] for _, d in picks], dtype=np.float32)

            masks, scores = self._client.segment(
                cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), boxes,
                multimask=self._multimask)

            if depth_hw is None:
                depth_hw = (src_h, src_w)
                self.get_logger().warn(
                    "no CameraInfo yet — sizing the mask to the RGB frame "
                    f"({src_w}x{src_h}). If depth differs, the projector will "
                    "mislabel or raise IndexError.")

            label_map = compose_label_map(
                masks, indices, depth_hw, target_priority=self._target_priority)

            stats = mask_stats(label_map)
            total_px = depth_hw[0] * depth_hw[1]
            for idx, det in picks:
                px = stats.get(idx + 1, 0)
                frac = px / total_px if total_px else 0.0
                if px < self._min_pixels:
                    self.get_logger().warn(
                        f"{det.get('category')} '{det.get('label')}' has only "
                        f"{px} px (mask={idx + 1}) — suspect a wrong "
                        "bbox_convention upstream")
                elif frac > self._max_fraction:
                    self.get_logger().warn(
                        f"{det.get('category')} '{det.get('label')}' covers "
                        f"{frac:.0%} of the frame (mask={idx + 1}) — that is not "
                        "an object. Check whether the server is returning raw "
                        "logits instead of thresholded masks, and whether the "
                        "box actually bounds the object.")

            mask_msg = self._bridge.cv2_to_imgmsg(label_map, encoding="mono8")
            # Carry the source frame's stamp so a mask can be traced back to the
            # exact image its boxes were computed on.
            mask_msg.header = img_msg.header
            self._mask_pub.publish(mask_msg)

            if self._annotated_pub is not None:
                vis_msg = self._bridge.cv2_to_imgmsg(
                    self._overlay(image_bgr, label_map, picks), encoding="bgr8")
                # Carry the source header. Without it the stamp is 0 and RViz's
                # Image display can drop the message as "too old", plus there is
                # no way to trace the overlay back to its source frame.
                vis_msg.header = img_msg.header
                self._annotated_pub.publish(vis_msg)

            summary = ", ".join(
                f"{d.get('category')}='{d.get('label')}'→{i + 1}({stats.get(i + 1, 0)}px)"
                for i, d in picks)
            self.get_logger().info(
                f"segmented in {time.monotonic() - t0:.2f}s — "
                f"{depth_hw[1]}x{depth_hw[0]} mask; {summary}; "
                f"scores={np.round(scores, 3).tolist()}")

        except Exception as exc:  # noqa: BLE001 — one bad frame must not kill the node
            self.get_logger().error(
                f"segmentation failed after {time.monotonic() - t0:.2f}s: {exc}")
        finally:
            with self._lock:
                self._busy = False

    def _overlay(self, image_bgr: np.ndarray, label_map: np.ndarray,
                 picks: list[tuple[int, dict]]) -> np.ndarray:
        """Debug view — masks tinted over the source frame."""
        colours = {1: (0, 200, 80), 2: (0, 220, 255)}
        vis = image_bgr.copy()
        h, w = vis.shape[:2]
        lm = (cv2.resize(label_map, (w, h), interpolation=cv2.INTER_NEAREST)
              if label_map.shape != (h, w) else label_map)
        for idx, det in picks:
            value = idx + 1
            sel = lm == value
            if not sel.any():
                continue
            tint = np.zeros_like(vis)
            tint[:] = colours.get(value, (200, 80, 200))
            vis[sel] = cv2.addWeighted(vis, 0.5, tint, 0.5, 0)[sel]
            ys, xs = np.nonzero(sel)
            cv2.putText(vis, f"{value}:{det.get('label', '')}",
                        (int(xs.min()), max(12, int(ys.min()) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        colours.get(value, (255, 255, 255)), 1, cv2.LINE_AA)
        return vis


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SamMaskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
