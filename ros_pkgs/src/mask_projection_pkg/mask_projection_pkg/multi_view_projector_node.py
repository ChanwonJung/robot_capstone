"""
multi_view_projector_node.py

Two-camera world-frame PointCloud2 builder.

Pipeline
--------
  Top camera  : depth only  →  all points labeled FREE  (no GSAM on this view)
  EE camera   : depth + GSAM mask + detections  →  TARGET / WORKSPACE / OBSTACLE / FREE

Both views are transformed to a common world frame using fixed (R, t) matrices,
then merged into a single PointCloud2 published as /world_map.

Subscriptions (all configurable via ROS 2 parameters):
  <top_depth_topic>        sensor_msgs/Image      (32FC1, meters)
  <top_camera_info_topic>  sensor_msgs/CameraInfo
  <ee_depth_topic>         sensor_msgs/Image      (32FC1, meters)
  <ee_camera_info_topic>   sensor_msgs/CameraInfo
  <mask_topic>             sensor_msgs/Image      (mono8, 1-based)   ← TRIGGER
  <detections_topic>       std_msgs/String        (JSON array)

Publications:
  <output_cloud_topic>     sensor_msgs/PointCloud2  (frame_id="world")
  <output_result_topic>    std_msgs/String          (JSON centroid summary)

Camera → world transforms
-------------------------
R_TOP, t_TOP  top camera frame → world frame  (camera is fixed to ceiling/stand)
R_EE,  t_EE   EE camera frame  → world frame  (snapshot at robot home pose)

# TODO (teammate): Replace the identity placeholders below with actual values.
#   Measurement procedure:
#     1. Place a calibration target at a known world-frame coordinate.
#     2. Measure R (3×3 rotation matrix, camera-to-world) and
#        t (3-vector, camera origin in world frame, meters).
#     3. Paste the values into _R_TOP/_t_TOP and _R_EE/_t_EE below.
#   For Isaac Sim: use the camera pose from the stage (USD transform).

Isaac Sim topic override
------------------------
All topic names are ROS 2 parameters.  Override them at launch time:
  ros2 launch mask_projection_pkg multi_view_projector.launch.py \
    top_depth_topic:=/isaac/top/depth \
    top_camera_info_topic:=/isaac/top/camera_info \
    ee_depth_topic:=/isaac/ee/depth \
    ee_camera_info_topic:=/isaac/ee/camera_info
(see multi_view_projector.launch.py for the full override example)

GSAM node must subscribe to the EE camera RGB topic:
  ros2 launch grounded_sam_pkg grounded_sam.launch.py \
    image_topic:=/ee_camera/image \
    prompt:="cup, table, object"

Design note — cache pattern (same as projector_node.py)
---------------------------------------------------------
GSAM runs on CPU (~30–40 s/frame).  By the time mask_image arrives, the depth
queue has moved far ahead, so timestamp-based sync fails.  Instead, depth /
camera_info / detections are cached as "latest", and projection fires the
moment a new mask_image arrives.

When moving to Isaac Sim with GPU inference (real-time), replace the cache
pattern with message_filters.ApproximateTimeSynchronizer across ee_depth + mask.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Header, String

from .back_projection import depth_to_points
from .cloud_builder import build_pointcloud2
from .label_mapper import (
    CATEGORY_COLOR,
    CATEGORY_FREE,
    CATEGORY_TARGET,
    CATEGORY_WORKSPACE,
    CategoryPoints,
    apply_labels,
)

_OUTPUT_DIR = Path.home() / "gsam_ws" / "output"

# ── Camera → world transforms ────────────────────────────────────────────────
#
# TODO (teammate): Replace np.eye(3) / np.zeros(3) with measured values.
#
# Convention:
#   p_world = R @ p_cam + t
#
# _R_TOP : (3,3) rotation  — top camera axes expressed in world frame
# _t_TOP : (3,)  translation — top camera optical centre in world frame (m)
#
_R_TOP: np.ndarray = np.eye(3, dtype=np.float64)    # TODO: fill in
_t_TOP: np.ndarray = np.zeros(3, dtype=np.float64)  # TODO: fill in

# _R_EE  : (3,3) rotation  — EE camera axes in world frame at home pose
# _t_EE  : (3,)  translation — EE camera optical centre in world frame (m)
#
_R_EE: np.ndarray = np.eye(3, dtype=np.float64)     # TODO: fill in
_t_EE: np.ndarray = np.zeros(3, dtype=np.float64)   # TODO: fill in


class MultiViewProjectorNode(Node):

    def __init__(self) -> None:
        super().__init__('multi_view_projector_node')

        # ── parameters ───────────────────────────────────────────────────────
        # All topic names are parameters so Isaac Sim can override them without
        # touching this file.  Default values match the Gazebo bridge setup.
        self.declare_parameter('top_depth_topic',        '/top_camera/depth_image')
        self.declare_parameter('top_camera_info_topic',  '/top_camera/camera_info')
        self.declare_parameter('ee_depth_topic',         '/ee_camera/depth_image')
        self.declare_parameter('ee_camera_info_topic',   '/ee_camera/camera_info')
        self.declare_parameter('mask_topic',             '/grounded_sam/mask_image')
        self.declare_parameter('detections_topic',       '/grounded_sam/detections_json')
        self.declare_parameter('output_cloud_topic',     '/world_map')
        self.declare_parameter('output_result_topic',    '/world_map_result')
        self.declare_parameter('initials',               '')
        self.declare_parameter('min_depth', 0.05)
        self.declare_parameter('max_depth', 15.0)

        top_depth_topic        = self.get_parameter('top_depth_topic').value
        top_camera_info_topic  = self.get_parameter('top_camera_info_topic').value
        ee_depth_topic         = self.get_parameter('ee_depth_topic').value
        ee_camera_info_topic   = self.get_parameter('ee_camera_info_topic').value
        mask_topic             = self.get_parameter('mask_topic').value
        detections_topic       = self.get_parameter('detections_topic').value
        output_cloud_topic     = self.get_parameter('output_cloud_topic').value
        output_result_topic    = self.get_parameter('output_result_topic').value
        self._min_depth        = self.get_parameter('min_depth').value
        self._max_depth        = self.get_parameter('max_depth').value
        self._initials         = self.get_parameter('initials').value

        # ── cache ─────────────────────────────────────────────────────────────
        self._bridge = CvBridge()

        # Top camera (optional — node warns and falls back to EE-only if absent)
        self._top_depth:  Optional[Image]      = None
        self._top_info:   Optional[CameraInfo] = None

        # EE camera
        self._ee_depth:   Optional[Image]      = None
        self._ee_info:    Optional[CameraInfo] = None

        # GSAM outputs
        self._latest_detections: Optional[List[Dict]] = None

        # ── subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Image,      top_depth_topic,       self._top_depth_cb,  10)
        self.create_subscription(CameraInfo, top_camera_info_topic, self._top_info_cb,   10)
        self.create_subscription(Image,      ee_depth_topic,        self._ee_depth_cb,   10)
        self.create_subscription(CameraInfo, ee_camera_info_topic,  self._ee_info_cb,    10)
        self.create_subscription(String,     detections_topic,      self._json_cb,       10)
        # mask_image arrival = TRIGGER for projection
        self.create_subscription(Image,      mask_topic,            self._mask_cb,       10)

        # ── publishers ────────────────────────────────────────────────────────
        self._pub_cloud  = self.create_publisher(PointCloud2, output_cloud_topic,  10)
        self._pub_result = self.create_publisher(String,      output_result_topic, 10)

        self.get_logger().info(
            f'MultiViewProjectorNode ready — '
            f'depth=[{self._min_depth}, {self._max_depth}]m  '
            f'trigger={mask_topic}'
        )
        self.get_logger().warn(
            'R/t transforms are identity placeholders. '
            'Fill in _R_TOP/_t_TOP/_R_EE/_t_EE in multi_view_projector_node.py '
            'before real deployment.'
        )

    # ── cache callbacks ───────────────────────────────────────────────────────

    def _top_depth_cb(self, msg: Image) -> None:
        self._top_depth = msg

    def _top_info_cb(self, msg: CameraInfo) -> None:
        self._top_info = msg

    def _ee_depth_cb(self, msg: Image) -> None:
        self._ee_depth = msg

    def _ee_info_cb(self, msg: CameraInfo) -> None:
        self._ee_info = msg

    def _json_cb(self, msg: String) -> None:
        try:
            self._latest_detections = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(f'detections_json parse error: {e}')

    # ── trigger callback ──────────────────────────────────────────────────────

    def _mask_cb(self, mask_msg: Image) -> None:
        # EE camera is required — top camera is optional (degrades gracefully)
        if self._ee_depth is None or self._ee_info is None:
            self.get_logger().warn('Waiting for EE camera depth/camera_info...')
            return
        if self._latest_detections is None:
            self.get_logger().warn('Waiting for detections_json...')
            return

        all_category_points: List[CategoryPoints] = []

        # ── Top camera view (depth only → all FREE) ───────────────────────────
        if self._top_depth is not None and self._top_info is not None:
            top_pts = self._project_free(self._top_depth, self._top_info,
                                         _R_TOP, _t_TOP)
            if top_pts is not None:
                all_category_points.append(top_pts)
        else:
            self.get_logger().warn(
                'Top camera not available — publishing EE view only. '
                'Check top_depth_topic / top_camera_info_topic parameters.'
            )

        # ── EE camera view (GSAM mask → semantic labels) ──────────────────────
        ee_pts = self._project_labeled(
            self._ee_depth, self._ee_info, mask_msg,
            self._latest_detections, _R_EE, _t_EE,
        )
        if ee_pts:
            all_category_points.extend(ee_pts)

        if not all_category_points:
            self.get_logger().warn('No valid points from either camera.')
            return

        self.get_logger().info(
            'World map: ' +
            ', '.join(f'{cp.label}={len(cp.points)}pts' for cp in all_category_points)
        )

        # ── publish ───────────────────────────────────────────────────────────
        header = Header()
        header.stamp    = self.get_clock().now().to_msg()
        header.frame_id = 'world'

        self._pub_cloud.publish(build_pointcloud2(header, all_category_points))
        self._pub_result.publish(String(data=_build_result_json(all_category_points)))

        # ── save PLY to disk ──────────────────────────────────────────────────
        stamp  = self._ee_depth.header.stamp.sec
        prefix = f"{self._initials}_" if self._initials else ""
        _save_ply_labeled(
            _OUTPUT_DIR / f"world_map_{prefix}{stamp}.ply",
            all_category_points,
        )

    # ── projection helpers ────────────────────────────────────────────────────

    def _project_free(
        self,
        depth_msg: Image,
        info_msg:  CameraInfo,
        R: np.ndarray,
        t: np.ndarray,
    ) -> Optional[CategoryPoints]:
        """Back-project depth image and label all points FREE (no GSAM)."""
        depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        K     = np.array(info_msg.k, dtype=np.float64).reshape(3, 3)

        pts_cam, _ = depth_to_points(depth, K,
                                     min_depth=self._min_depth,
                                     max_depth=self._max_depth)
        if len(pts_cam) == 0:
            return None

        pts_world = (R @ pts_cam.T).T + t
        return _make_free_points(pts_world)

    def _project_labeled(
        self,
        depth_msg:  Image,
        info_msg:   CameraInfo,
        mask_msg:   Image,
        detections: List[Dict],
        R: np.ndarray,
        t: np.ndarray,
    ) -> List[CategoryPoints]:
        """Back-project EE depth, transform to world, apply GSAM labels."""
        depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        mask  = self._bridge.imgmsg_to_cv2(mask_msg,  desired_encoding='mono8')
        K     = np.array(info_msg.k, dtype=np.float64).reshape(3, 3)

        # NOTE: pixel_coords are at depth image resolution.
        # apply_labels indexes mask_image (built at RGB resolution by GSAM) using
        # those coords.  If depth and RGB resolutions differ, add a resize step here.
        # Standard RGBD sensors produce equal resolutions → no resize needed.
        pts_cam, pixel_coords = depth_to_points(depth, K,
                                                min_depth=self._min_depth,
                                                max_depth=self._max_depth)
        if len(pts_cam) == 0:
            return []

        pts_world = (R @ pts_cam.T).T + t
        return apply_labels(pts_world, pixel_coords, mask, detections)


# ── module-level helpers ──────────────────────────────────────────────────────

def _make_free_points(points: np.ndarray) -> CategoryPoints:
    """Wrap an (N, 3) world-frame array as a FREE CategoryPoints."""
    n     = len(points)
    color = CATEGORY_COLOR[CATEGORY_FREE]
    return CategoryPoints(
        label      = 'free',
        category   = CATEGORY_FREE,
        points     = points.astype(np.float32),
        colors     = np.tile(np.array(color, dtype=np.uint8), (n, 1)),
        categories = np.full(n, CATEGORY_FREE, dtype=np.uint8),
    )


def _build_result_json(category_points: List[CategoryPoints]) -> str:
    """
    JSON summary: label, centroid (world frame), point_count per category.

    Shape (extensible for Qwen target_coordinate protocol):
    {
      "target":    {"label": "cup",   "centroid": [x, y, z], "point_count": N},
      "workspace": {"label": "table", "centroid": [x, y, z], "point_count": N},
      "free":      {"label": "free",  "centroid": [x, y, z], "point_count": N}
    }

    # TODO (Qwen integration): Replace prompt-order-based category assignment with
    #   Qwen VLM output.  Qwen will determine which mask ID is TARGET vs WORKSPACE.
    #   Expected change: build a custom MASK_VALUE_TO_CATEGORY dict from Qwen's JSON
    #   and pass it into apply_labels() before calling this function.
    """
    _CATEGORY_KEY = {
        CATEGORY_TARGET:    'target',
        CATEGORY_WORKSPACE: 'workspace',
    }
    out: Dict = {}
    for cp in category_points:
        key      = _CATEGORY_KEY.get(cp.category, cp.label)
        centroid = cp.points.mean(axis=0).tolist()
        out[key] = {
            'label':       cp.label,
            'centroid':    [round(v, 4) for v in centroid],
            'point_count': len(cp.points),
        }
    return json.dumps(out)


def _save_ply_labeled(path: Path, category_points: List[CategoryPoints]) -> None:
    """Save world-frame labeled points (XYZ + RGB + category) as PLY."""
    all_pts  = np.concatenate([cp.points     for cp in category_points], axis=0)
    all_col  = np.concatenate([cp.colors     for cp in category_points], axis=0)
    all_cats = np.concatenate([cp.categories for cp in category_points], axis=0)
    N = len(all_pts)
    path.parent.mkdir(parents=True, exist_ok=True)
    dt = np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('red', np.uint8), ('green', np.uint8), ('blue', np.uint8),
        ('category', np.uint8),
    ])
    arr             = np.zeros(N, dtype=dt)
    arr['x']        = all_pts[:, 0]
    arr['y']        = all_pts[:, 1]
    arr['z']        = all_pts[:, 2]
    arr['red']      = all_col[:, 0]
    arr['green']    = all_col[:, 1]
    arr['blue']     = all_col[:, 2]
    arr['category'] = all_cats
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar category\n"
        "end_header\n"
    )
    with open(path, 'wb') as f:
        f.write(header.encode())
        f.write(arr.tobytes())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MultiViewProjectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
