"""Two-camera world-frame PointCloud2 builder."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header, String

# /world_map_result 는 Slow Brain 의 1회성 결과 — 한 번 발행 후 BT 가 늦게
# 구독해도 마지막 메시지 받도록 TRANSIENT_LOCAL. depth=1 (최신 1개만 보관).
_LATCHED_QOS = QoSProfile(
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
)

from .cloud_builder import build_pointcloud2
from .label_mapper import CategoryPoints
from .ply_utils import build_result_json, save_ply_labeled
from .projection_engine import (
    collect_seg_points,
    filter_free_by_unknown,
    load_extrinsics,
    project_labeled,
    project_unknown,
)

def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "sim").exists() and (parent / "ros_pkgs").exists():
            return parent
    return Path.home() / "robot_capstone"


_OUTPUT_DIR = _find_project_root() / "output"


class MultiViewProjectorNode(Node):

    def __init__(self) -> None:
        super().__init__('multi_view_projector_node')

        # ── parameters ───────────────────────────────────────────────────────
        self.declare_parameter('top_depth_topic',        '/top_camera/depth_image')
        self.declare_parameter('top_camera_info_topic',  '/top_camera/camera_info')
        self.declare_parameter('ee_depth_topic',         '/ee_camera/depth_image')
        self.declare_parameter('ee_camera_info_topic',   '/ee_camera/camera_info')
        self.declare_parameter('mask_topic',             '/grounded_sam/mask_image')
        self.declare_parameter('detections_topic',       '/grounded_sam/detections_json')
        self.declare_parameter('output_cloud_topic',     '/world_map')
        self.declare_parameter('output_result_topic',    '/world_map_result')
        self.declare_parameter('output_raw_cloud_topic', '/world_cloud_raw')
        self.declare_parameter('initials',               '')
        self.declare_parameter('min_depth', 0.05)
        self.declare_parameter('max_depth', 15.0)
        self.declare_parameter('ee_seg_filter_radius',    0.015)  # metres — XY footprint radius
        self.declare_parameter('ee_seg_z_margin',         0.10)   # metres — Z gate for seg filter
        self.declare_parameter('free_unknown_xy_radius',  0.05)   # metres — Pass 2 XY radius
        self.declare_parameter('free_unknown_z_margin',   0.10)   # metres — Pass 2 Z gate
        # 한 번 publish 후 새 mask 입력 무시 (true) — robot 이 움직여 EE 카메라 시점이
        # 바뀌면 정적 extrinsics 가 깨져 재투영이 망가짐 ([[project-slow-brain-static-extrinsics]]).
        # 첫 발행 후 freeze 하면 BT 가 일관된 데이터로 끝까지 작업 가능.
        self.declare_parameter('freeze_after_first_publish', False)

        _default_extrinsics = os.path.join(
            get_package_share_directory('mask_projection_pkg'),
            'config', 'camera_extrinsics.yaml',
        )
        self.declare_parameter('extrinsics_config', _default_extrinsics)

        top_depth_topic        = self.get_parameter('top_depth_topic').value
        top_camera_info_topic  = self.get_parameter('top_camera_info_topic').value
        ee_depth_topic         = self.get_parameter('ee_depth_topic').value
        ee_camera_info_topic   = self.get_parameter('ee_camera_info_topic').value
        mask_topic             = self.get_parameter('mask_topic').value
        detections_topic       = self.get_parameter('detections_topic').value
        output_cloud_topic     = self.get_parameter('output_cloud_topic').value
        output_result_topic    = self.get_parameter('output_result_topic').value
        output_raw_cloud_topic = self.get_parameter('output_raw_cloud_topic').value
        self._min_depth        = self.get_parameter('min_depth').value
        self._max_depth        = self.get_parameter('max_depth').value
        self._ee_seg_filter_radius    = self.get_parameter('ee_seg_filter_radius').value
        self._ee_seg_z_margin         = self.get_parameter('ee_seg_z_margin').value
        self._free_unknown_xy_radius  = self.get_parameter('free_unknown_xy_radius').value
        self._free_unknown_z_margin   = self.get_parameter('free_unknown_z_margin').value
        self._initials                = self.get_parameter('initials').value
        self._freeze_after_first      = bool(self.get_parameter('freeze_after_first_publish').value)
        self._frozen                  = False
        extrinsics_path        = self.get_parameter('extrinsics_config').value
        # launch file may pass '' (empty string) → fall back to package default
        if not extrinsics_path:
            extrinsics_path = _default_extrinsics

        # ── camera extrinsics (loaded from YAML) ──────────────────────────────
        (self._R_TOP, self._t_TOP,
         self._R_EE,  self._t_EE, _ext_warn) = load_extrinsics(extrinsics_path)
        if _ext_warn:
            self.get_logger().warn(
                f'Failed to load extrinsics from "{extrinsics_path}": {_ext_warn}. '
                'Using identity transforms — point cloud will be in camera frame.'
            )
        else:
            self.get_logger().info(f'Loaded camera extrinsics from {extrinsics_path}')

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
        self.create_subscription(Image,      mask_topic,            self._mask_cb,       10)

        # ── publishers ────────────────────────────────────────────────────────
        self._pub_cloud     = self.create_publisher(PointCloud2, output_cloud_topic,     10)
        # /world_map_result 만 latched — Slow Brain 결과를 BT 가 늦게 받아도 됨.
        # PointCloud2 들은 RViz 표시용이라 VOLATILE 유지 (latching 시 RViz 가
        # 오래된 cloud 누적 표시할 수 있어 의도와 다름).
        self._pub_result    = self.create_publisher(String,      output_result_topic,    _LATCHED_QOS)
        self._pub_raw_cloud = self.create_publisher(PointCloud2, output_raw_cloud_topic, 10)

        self.get_logger().info(
            f'MultiViewProjectorNode ready — '
            f'depth=[{self._min_depth}, {self._max_depth}]m  '
            f'trigger={mask_topic}  '
            f'extrinsics={extrinsics_path}'
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
        # Freeze mode: 한 번 publish 한 뒤로 신규 mask 전부 drop. latched QoS 가
        # 마지막 발행을 BT 에 영원히 노출하므로 추가 갱신은 오히려 해로움.
        if self._frozen:
            return
        # EE camera is required — top camera is optional (degrades gracefully)
        if self._ee_depth is None or self._ee_info is None:
            self.get_logger().warn('Waiting for EE camera depth/camera_info...')
            return
        if self._latest_detections is None:
            self.get_logger().warn('Waiting for detections_json...')
            return

        all_category_points: List[CategoryPoints] = []

        # ── EE camera view ────────────────────────────────────────────────────
        ee_pts = self._project_labeled(
            self._ee_depth, self._ee_info, mask_msg,
            self._latest_detections, self._R_EE, self._t_EE,
        )

        # ── Top camera view — Pass 1: remove UNKNOWN near EE seg (XY + Z gate) ──
        top_pts: Optional[CategoryPoints] = None
        if self._top_depth is not None and self._top_info is not None:
            ee_seg_pts = collect_seg_points(ee_pts)
            top_pts = self._project_unknown(
                self._top_depth, self._top_info,
                self._R_TOP, self._t_TOP,
                ee_seg_pts=ee_seg_pts,
                ee_seg_filter_radius=self._ee_seg_filter_radius,
                ee_seg_z_margin=self._ee_seg_z_margin,
            )
        else:
            self.get_logger().warn(
                'Top camera not available — publishing EE view only. '
                'Check top_depth_topic / top_camera_info_topic parameters.'
            )

        # ── Pass 2: UNKNOWN > FREE — remove EE FREE near top UNKNOWN ─────────
        if top_pts is not None:
            ee_pts = filter_free_by_unknown(
                ee_pts, top_pts,
                self._free_unknown_xy_radius,
                self._free_unknown_z_margin,
            )

        if ee_pts:
            all_category_points.extend(ee_pts)
        if top_pts is not None:
            all_category_points.append(top_pts)

        if not all_category_points:
            self.get_logger().warn('No valid points from either camera.')
            return

        self.get_logger().info(
            'World map: ' +
            ', '.join(f'{cp.label}={len(cp.points)}pts' for cp in all_category_points)
        )

        header = Header()
        header.stamp    = self.get_clock().now().to_msg()
        header.frame_id = 'world'

        # ── publish ───────────────────────────────────────────────────────────
        self._pub_cloud.publish(build_pointcloud2(header, all_category_points))
        self._pub_result.publish(String(data=build_result_json(all_category_points)))
        self._pub_raw_cloud.publish(self._build_raw_cloud(header, all_category_points))

        # 첫 발행 후 freeze — 이후 mask 입력은 모두 무시.
        if self._freeze_after_first and not self._frozen:
            self._frozen = True
            self.get_logger().info('FROZEN — 이후 mask 입력 무시. latched /world_map_result 그대로 유지.')

        # ── save PLY to disk ──────────────────────────────────────────────────
        stamp  = self._ee_depth.header.stamp.sec
        prefix = f"{self._initials}_" if self._initials else ""
        save_ply_labeled(
            _OUTPUT_DIR / f"world_map_{prefix}{stamp}.ply",
            all_category_points,
        )

    # ── raw cloud (geometry-only for MoveIt2 OctoMap) ────────────────────────

    def _build_raw_cloud(self, header, category_points: List[CategoryPoints]) -> PointCloud2:
        all_pts = np.concatenate([cp.points for cp in category_points], axis=0).astype(np.float32)
        N = len(all_pts)
        dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
        arr = np.zeros(N, dtype=dt)
        arr['x'] = all_pts[:, 0]
        arr['y'] = all_pts[:, 1]
        arr['z'] = all_pts[:, 2]
        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg = PointCloud2()
        msg.header     = header
        msg.height     = 1
        msg.width      = N
        msg.fields     = fields
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step   = 12 * N
        msg.is_dense   = True
        msg.data       = arr.tobytes()
        return msg

    # ── projection helpers (ROS decode → engine call) ─────────────────────────

    def _project_unknown(
        self,
        depth_msg:             Image,
        info_msg:              CameraInfo,
        R:                     np.ndarray,
        t:                     np.ndarray,
        ee_seg_pts:            Optional[np.ndarray] = None,
        ee_seg_filter_radius:  float = 0.015,
        ee_seg_z_margin:       float = 0.10,
    ) -> Optional[CategoryPoints]:
        depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        K     = np.array(info_msg.k, dtype=np.float64).reshape(3, 3)
        return project_unknown(
            depth, K, R, t,
            self._min_depth, self._max_depth,
            ee_seg_pts, ee_seg_filter_radius, ee_seg_z_margin,
        )

    def _project_labeled(
        self,
        depth_msg:  Image,
        info_msg:   CameraInfo,
        mask_msg:   Image,
        detections: List[Dict],
        R:          np.ndarray,
        t:          np.ndarray,
    ) -> List[CategoryPoints]:
        depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        mask  = self._bridge.imgmsg_to_cv2(mask_msg,  desired_encoding='mono8')
        K     = np.array(info_msg.k, dtype=np.float64).reshape(3, 3)
        return project_labeled(depth, K, mask, detections, R, t,
                               self._min_depth, self._max_depth)


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
