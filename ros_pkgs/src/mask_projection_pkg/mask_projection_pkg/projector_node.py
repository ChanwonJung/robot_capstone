"""
projector_node.py

ROS 2 node — wiring only.  No math, no message packing here.
All logic lives in the modules below:

  back_projection.depth_to_points()   depth image → 3D points
  label_mapper.apply_labels()         mask pixel → semantic category
  cloud_builder.build_pointcloud2()   CategoryPoints → PointCloud2 msg

Subscriptions  (all configurable via ROS 2 parameters)
-------------
  <depth_topic>       sensor_msgs/Image       (32FC1, meters)
  <camera_info_topic> sensor_msgs/CameraInfo
  <mask_topic>        sensor_msgs/Image       (mono8, 1-based index)
  <detections_topic>  std_msgs/String         (JSON, no header)

Publications  (all configurable via ROS 2 parameters)
------------
  <output_cloud_topic>   sensor_msgs/PointCloud2 (XYZRGB + category field)
  <output_result_topic>  std_msgs/String         (JSON centroid summary)

Parameters
----------
  depth_topic          string  /rgbd_camera/depth_image
  camera_info_topic    string  /rgbd_camera/camera_info
  mask_topic           string  /grounded_sam/mask_image
  detections_topic     string  /grounded_sam/detections_json
  output_cloud_topic   string  /labeled_points
  output_result_topic  string  /projection_result
  output_frame_id      string  ''   Override PointCloud2 frame_id.
                                    Empty string = use depth message header as-is.
  min_depth            float   0.05 Discard depth readings below this value (m)
  max_depth            float   15.0 Discard depth readings above this value (m)

Design note — why no ApproximateTimeSynchronizer
-------------------------------------------------
SAM runs on CPU and takes 30–40 s per frame.  By the time mask_image is
published, the depth queue has moved far ahead and timestamp matching fails.
Instead, depth/camera_info/detections are cached as "latest", and projection
fires the moment a new mask_image arrives.  This is correct because:
  - depth changes slowly (static tabletop scene)
  - mask timestamp already matches the depth that was captured (rgb and
    depth share the same Gazebo sim-time stamp from the same sensor)

When moving to Isaac Sim (real-time GPU inference) replace the cache pattern
with message_filters.ApproximateTimeSynchronizer across depth + mask.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import String

def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "sim").exists() and (parent / "ros_pkgs").exists():
            return parent
    return Path.home() / "robot_capstone"


_OUTPUT_DIR = _find_project_root() / "output"


from .back_projection import depth_to_points
from .cloud_builder import build_pointcloud2
from .label_mapper import apply_labels
from .ply_utils import build_result_json, save_ply_labeled, save_ply_xyz


class MaskProjectorNode(Node):

    def __init__(self) -> None:
        super().__init__('mask_projector_node')

        # ── parameters ───────────────────────────────────────────────────────
        # Topic names — override in launch file to swap simulator adapters.
        # Default values match the Gazebo bridge setup.
        self.declare_parameter('depth_topic',         '/rgbd_camera/depth_image')
        self.declare_parameter('camera_info_topic',   '/rgbd_camera/camera_info')
        self.declare_parameter('mask_topic',          '/grounded_sam/mask_image')
        self.declare_parameter('detections_topic',    '/grounded_sam/detections_json')
        self.declare_parameter('output_cloud_topic',  '/labeled_points')
        self.declare_parameter('output_result_topic', '/projection_result')
        # Empty string = inherit frame_id from incoming depth message header
        self.declare_parameter('output_frame_id',     '')
        # Prompt initials for output filenames, e.g. "tc" for "table, cup"
        self.declare_parameter('initials',            '')
        # Depth filter range
        self.declare_parameter('min_depth', 0.05)
        self.declare_parameter('max_depth', 15.0)
        self.declare_parameter('min_project_interval_sec', 2.0)
        self.declare_parameter('save_ply', False)
        self.declare_parameter('max_ply_saves', 3)
        self.declare_parameter('output_subdir', '')

        depth_topic         = self.get_parameter('depth_topic').value
        camera_info_topic   = self.get_parameter('camera_info_topic').value
        mask_topic          = self.get_parameter('mask_topic').value
        detections_topic    = self.get_parameter('detections_topic').value
        output_cloud_topic  = self.get_parameter('output_cloud_topic').value
        output_result_topic = self.get_parameter('output_result_topic').value
        self._output_frame_id = self.get_parameter('output_frame_id').value
        self._min_depth       = self.get_parameter('min_depth').value
        self._max_depth       = self.get_parameter('max_depth').value
        self._initials        = self.get_parameter('initials').value
        self._min_project_interval_sec = float(self.get_parameter('min_project_interval_sec').value)
        self._save_ply = bool(self.get_parameter('save_ply').value)
        self._max_ply_saves = int(self.get_parameter('max_ply_saves').value)
        output_subdir = str(self.get_parameter('output_subdir').value).strip().strip("/")
        self._output_dir = _OUTPUT_DIR / output_subdir if output_subdir else _OUTPUT_DIR
        self._last_project_time = 0.0
        self._ply_save_count = 0

        # ── cache — updated by individual subscribers ─────────────────────────
        self._bridge             = CvBridge()
        self._latest_depth:      Optional[Image]      = None
        self._latest_info:       Optional[CameraInfo] = None
        self._latest_detections: Optional[List[dict]] = None

        # ── subscribers ───────────────────────────────────────────────────────
        # depth and camera_info: just cache the latest frame
        self.create_subscription(Image,      depth_topic,      self._depth_cb, 10)
        self.create_subscription(CameraInfo, camera_info_topic, self._info_cb, 10)
        # detections_json: cache (no header, published together with mask)
        self.create_subscription(String,     detections_topic, self._json_cb, 10)
        # mask_image: TRIGGER — runs projection when a new mask arrives
        self.create_subscription(Image,      mask_topic,       self._mask_cb, 10)

        # ── publishers ───────────────────────────────────────────────────────
        self._pub_cloud  = self.create_publisher(PointCloud2, output_cloud_topic,  10)
        self._pub_result = self.create_publisher(String,      output_result_topic, 10)

        self.get_logger().info(
            f'MaskProjectorNode ready — '
            f'depth=[{self._min_depth}, {self._max_depth}]m  '
            f'trigger={mask_topic}  '
            f'min_project_interval_sec={self._min_project_interval_sec}  '
            f'save_ply={self._save_ply}  '
            f'max_ply_saves={self._max_ply_saves}  '
            f'output_dir={self._output_dir}'
        )

    # ── cache callbacks ───────────────────────────────────────────────────────

    def _depth_cb(self, msg: Image) -> None:
        self._latest_depth = msg

    def _info_cb(self, msg: CameraInfo) -> None:
        self._latest_info = msg

    def _json_cb(self, msg: String) -> None:
        try:
            self._latest_detections = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(f'detections_json parse error: {e}')

    # ── trigger callback ──────────────────────────────────────────────────────

    def _mask_cb(self, mask_msg: Image) -> None:
        now = time.monotonic()
        if now - self._last_project_time < self._min_project_interval_sec:
            return

        if self._latest_depth is None or self._latest_info is None:
            self.get_logger().warn('Waiting for depth/camera_info...')
            return
        if self._latest_detections is None:
            self.get_logger().warn('Waiting for detections_json...')
            return
        self._last_project_time = now

        # ── decode ────────────────────────────────────────────────────────────
        depth = self._bridge.imgmsg_to_cv2(self._latest_depth,
                                           desired_encoding='32FC1')
        mask  = self._bridge.imgmsg_to_cv2(mask_msg,
                                           desired_encoding='mono8')
        K     = np.array(self._latest_info.k, dtype=np.float64).reshape(3, 3)

        # ── back-project ──────────────────────────────────────────────────────
        points, pixel_coords = depth_to_points(
            depth, K,
            min_depth=self._min_depth,
            max_depth=self._max_depth,
        )
        if len(points) == 0:
            self.get_logger().warn('No valid depth points in this frame')
            return

        # ── label ─────────────────────────────────────────────────────────────
        category_points = apply_labels(
            points, pixel_coords, mask, self._latest_detections)

        self.get_logger().info(
            'Projected: ' +
            ', '.join(f'{cp.label}={len(cp.points)}pts' for cp in category_points)
        )

        # ── publish ───────────────────────────────────────────────────────────
        cloud_header = self._latest_depth.header
        if self._output_frame_id:
            # Allow overriding frame_id without changing depth source
            # (useful when Isaac Sim uses a different TF frame name)
            cloud_header = type(cloud_header)(
                stamp=cloud_header.stamp,
                frame_id=self._output_frame_id,
            )
        self._pub_cloud.publish(build_pointcloud2(cloud_header, category_points))
        self._pub_result.publish(String(data=build_result_json(category_points)))

        # ── save pointclouds to disk ──────────────────────────────────────────
        stamp  = self._latest_depth.header.stamp.sec
        prefix = f"{self._initials}_" if self._initials else ""
        orig_name    = f"cloud_original_{prefix}{stamp}.ply"
        labeled_name = f"cloud_labeled_{prefix}{stamp}.ply"
        save_ply_xyz(_OUTPUT_DIR / orig_name, points)
        save_ply_labeled(_OUTPUT_DIR / labeled_name, category_points)
        self.get_logger().info(f"Saved → {orig_name}  {labeled_name}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MaskProjectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
