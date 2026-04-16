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
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import String

_OUTPUT_DIR = Path.home() / "gsam_ws" / "output"


from .back_projection import depth_to_points
from .cloud_builder import build_pointcloud2
from .label_mapper import CategoryPoints, apply_labels, CATEGORY_TARGET, CATEGORY_WORKSPACE


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

        # ── cache — updated by individual subscribers ─────────────────────────
        self._bridge             = CvBridge()
        self._latest_depth:      Optional[Image]      = None
        self._latest_info:       Optional[CameraInfo] = None
        self._latest_detections: Optional[List[Dict]] = None

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
            f'trigger={mask_topic}'
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
        if self._latest_depth is None or self._latest_info is None:
            self.get_logger().warn('Waiting for depth/camera_info...')
            return
        if self._latest_detections is None:
            self.get_logger().warn('Waiting for detections_json...')
            return

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
        self._pub_result.publish(String(data=_build_result_json(category_points)))

        # ── save pointclouds to disk ──────────────────────────────────────────
        stamp  = self._latest_depth.header.stamp.sec
        prefix = f"{self._initials}_" if self._initials else ""
        orig_name    = f"cloud_original_{prefix}{stamp}.ply"
        labeled_name = f"cloud_labeled_{prefix}{stamp}.ply"
        _save_ply_xyz(_OUTPUT_DIR / orig_name, points)
        _save_ply_labeled(_OUTPUT_DIR / labeled_name, category_points)
        self.get_logger().info(f"Saved → {orig_name}  {labeled_name}")


# ── helpers (module-level, easy to move/extend) ───────────────────────────────

def _save_ply_xyz(path: Path, points: np.ndarray) -> None:
    """Save (N, 3) float32 XYZ points as binary-little-endian PLY."""
    N = len(points)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    )
    with open(path, 'wb') as f:
        f.write(header.encode())
        f.write(points.astype(np.float32).tobytes())


def _save_ply_labeled(path: Path, category_points: List[CategoryPoints]) -> None:
    """Save labeled points (XYZ + RGB + category) as binary-little-endian PLY."""
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


def _build_result_json(category_points: List[CategoryPoints]) -> str:
    """
    Build a JSON summary with centroid + point count per category.

    Output shape (extensible toward target_coordinate protocol):
    {
      "target":    {"label": "cup",   "centroid": [x, y, z], "point_count": N},
      "workspace": {"label": "table", "centroid": [x, y, z], "point_count": N}
    }
    """
    category_to_key = {
        CATEGORY_TARGET:    'target',
        CATEGORY_WORKSPACE: 'workspace',
    }
    out: Dict = {}
    for cp in category_points:
        key      = category_to_key.get(cp.category, f'category_{cp.category}')
        centroid = cp.points.mean(axis=0).tolist()
        out[key] = {
            'label':       cp.label,
            'centroid':    [round(v, 4) for v in centroid],
            'point_count': len(cp.points),
        }
    return json.dumps(out)


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
