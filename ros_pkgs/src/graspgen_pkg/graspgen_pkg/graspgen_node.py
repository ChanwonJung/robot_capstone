"""
graspgen_node.py — ROS 2 node: EE depth + GSAM mask → TARGET cloud → ZMQ GraspGen → /grasp_candidates

Drop-in replacement for vgn_grasp_node. Identical /grasp_candidates JSON and
/grasp_markers so bt_pkg needs no changes.

Data flow:
  /ee_camera/depth_image   ─┐
  /ee_camera/camera_info   ─┼─ cache ──→ _result_cb (trigger: /world_map_result)
  /qwen/mask_image         ─┤            ├─ cloud_extractor → ZMQ → candidates
  /qwen/labeled_detections ─┘            └─ marker_publisher → /grasp_markers

Multi-camera: see cloud_extractor.extract_target_cloud() docstring.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as Rot
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
import tf2_ros

from .zmq_client import GraspGenClient, check_deps
from .depth_utils import (decode_depth, decode_mask, extract_K,
                           load_ee_extrinsics, apply_world_to_robot_tf)
from .cloud_extractor import find_target_mask_val, extract_target_cloud
from .marker_publisher import build_grasp_markers, build_target_cloud_msg

_WS = Path(os.environ.get('GSAM_WS', str(Path.home() / 'gsam_ws')))
_DEFAULT_EXTRINSICS = str(_WS / 'src/mask_projection_pkg/config/camera_extrinsics.yaml')


class GraspGenNode(Node):

    def __init__(self) -> None:
        super().__init__('graspgen_node')

        ok, err = check_deps()
        if not ok:
            self.get_logger().error(f'Missing deps: {err}\nRun: pip install pyzmq msgpack')
            raise ImportError(err)

        self._declare_params()
        self._load_params()
        self._init_extrinsics()
        self._init_zmq()
        self._init_tf()
        self._init_cache()
        self._init_pubsub()

        self.get_logger().info(
            f'graspgen_node ready  topk={self._topk}  min_pts={self._min_pts}')

    # ── initialisation helpers ────────────────────────────────────────────────

    def _declare_params(self) -> None:
        p = self.declare_parameter
        p('zmq_host',                 '127.0.0.1')
        p('zmq_port',                 5556)
        p('zmq_timeout_ms',           5000)
        p('num_grasps',               50)
        p('topk_num_grasps',          5)
        p('min_point_count',          50)
        p('max_points',               4096)
        p('min_depth',                0.05)
        p('max_depth',                15.0)
        p('gripper_width',            0.08)
        p('ee_depth_topic',           '/ee_camera/depth_image')
        p('ee_camera_info_topic',     '/ee_camera/camera_info')
        p('mask_topic',               '/qwen/mask_image')
        p('labeled_detections_topic', '/qwen/labeled_detections')
        p('world_map_result_topic',   '/world_map_result')
        p('grasp_candidates_topic',   '/grasp_candidates')
        p('extrinsics_config',        '')
        p('world_frame',              'world')
        p('robot_frame',              'panda_link0')

    def _load_params(self) -> None:
        g = self.get_parameter
        self._num_grasps    = g('num_grasps').value
        self._topk          = g('topk_num_grasps').value
        self._min_pts       = g('min_point_count').value
        self._max_pts       = g('max_points').value
        self._min_depth     = g('min_depth').value
        self._max_depth     = g('max_depth').value
        self._gripper_width = g('gripper_width').value
        self._world_frame   = g('world_frame').value
        self._robot_frame   = g('robot_frame').value

    def _init_extrinsics(self) -> None:
        path = self.get_parameter('extrinsics_config').value or _DEFAULT_EXTRINSICS
        self._R_ee, self._t_ee = load_ee_extrinsics(path)
        self.get_logger().info(f'Extrinsics: {path}')

    def _init_zmq(self) -> None:
        g = self.get_parameter
        host, port, timeout = g('zmq_host').value, g('zmq_port').value, g('zmq_timeout_ms').value
        self._client = GraspGenClient(host, port, timeout)
        self.get_logger().info(f'GraspGen ZMQ → tcp://{host}:{port}  timeout={timeout}ms')

    def _init_tf(self) -> None:
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

    def _init_cache(self) -> None:
        self._ee_depth:       Optional[np.ndarray] = None
        self._ee_K:           Optional[np.ndarray] = None
        self._mask:           Optional[np.ndarray] = None
        self._labeled_dets:   Optional[list]       = None
        self._pending_result: Optional[str]        = None

    def _init_pubsub(self) -> None:
        g = self.get_parameter
        self.create_subscription(Image,      g('ee_depth_topic').value,           self._ee_depth_cb, 10)
        self.create_subscription(CameraInfo, g('ee_camera_info_topic').value,     self._ee_info_cb,  10)
        self.create_subscription(Image,      g('mask_topic').value,               self._mask_cb,     10)
        self.create_subscription(String,     g('labeled_detections_topic').value, self._dets_cb,     10)
        self.create_subscription(String,     g('world_map_result_topic').value,   self._result_cb,   10)

        self._grasp_pub  = self.create_publisher(String,       g('grasp_candidates_topic').value, 10)
        self._marker_pub = self.create_publisher(MarkerArray,  '/grasp_markers',                  10)
        self._cloud_pub  = self.create_publisher(PointCloud2,  '/graspgen/target_cloud',          10)

    # ── cache callbacks ───────────────────────────────────────────────────────

    def _ee_depth_cb(self, msg: Image) -> None:
        try:
            self._ee_depth = decode_depth(msg)
        except ValueError as e:
            self.get_logger().warn(f'depth decode: {e}')
            return
        self._try_flush()

    def _ee_info_cb(self, msg: CameraInfo) -> None:
        self._ee_K = extract_K(msg)
        self._try_flush()

    def _mask_cb(self, msg: Image) -> None:
        self._mask = decode_mask(msg)
        self._try_flush()

    def _dets_cb(self, msg: String) -> None:
        try:
            self._labeled_dets = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(f'labeled_detections parse: {e}')

    def _try_flush(self) -> None:
        if (self._pending_result is None
                or self._ee_depth is None
                or self._ee_K is None
                or self._mask is None):
            return
        self.get_logger().info('캐시 완료 — 대기 중인 world_map_result 처리')
        msg      = String()
        msg.data = self._pending_result
        self._pending_result = None
        self._result_cb(msg)

    # ── main trigger ─────────────────────────────────────────────────────────

    def _result_cb(self, msg: String) -> None:
        if self._ee_depth is None or self._ee_K is None or self._mask is None:
            self._pending_result = msg.data
            self.get_logger().warn('EE 캐시 미도착 — 대기')
            return

        try:
            result      = json.loads(msg.data)
            target_info = result.get('target', {})
            centroid    = np.array(target_info['centroid'], dtype=np.float32)
            point_count = int(target_info.get('point_count', 0))
        except (KeyError, json.JSONDecodeError, ValueError) as e:
            self.get_logger().warn(f'world_map_result parse: {e}')
            return

        if point_count < self._min_pts:
            self.get_logger().info(f'TARGET count={point_count} < min={self._min_pts} — skip')
            return

        t0         = time.monotonic()
        target_val = find_target_mask_val(self._labeled_dets)
        pts_world  = extract_target_cloud(
            self._ee_depth, self._ee_K, self._mask, target_val,
            self._R_ee, self._t_ee,
            self._min_depth, self._max_depth, self._max_pts,
        )

        if pts_world is None or len(pts_world) < self._min_pts:
            n = len(pts_world) if pts_world is not None else 0
            self.get_logger().info(f'TARGET 포인트 부족 ({n}) — skip')
            return

        self.get_logger().info(f'TARGET {len(pts_world)} pts → GraspGen')
        stamp = self.get_clock().now().to_msg()
        self._cloud_pub.publish(build_target_cloud_msg(pts_world, self._world_frame, stamp))

        try:
            grasps, confs = self._client.request(pts_world, self._num_grasps, self._topk)
        except (RuntimeError, ValueError) as e:
            self.get_logger().error(f'GraspGen: {e}')
            return

        if len(grasps) == 0:
            self.get_logger().warn('GraspGen: 결과 없음')
            return

        order  = np.argsort(confs)[::-1]
        grasps = grasps[order[:self._topk]]
        confs  = confs[order[:self._topk]]

        tf_stamped, output_frame = self._lookup_tf()
        candidates = self._build_candidates(grasps, confs, tf_stamped, output_frame)

        out      = String()
        out.data = json.dumps({
            'candidates':      candidates,
            'target_centroid': centroid.tolist(),
            'stamp':           self.get_clock().now().nanoseconds * 1e-9,
        })
        self._grasp_pub.publish(out)

        clear_ma, markers_ma = build_grasp_markers(candidates, output_frame, stamp, self._gripper_width)
        self._marker_pub.publish(clear_ma)
        self._marker_pub.publish(markers_ma)

        self.get_logger().info(
            f'Published {len(candidates)} grasp(s)  '
            f'best={confs[0]:.3f}  elapsed={time.monotonic()-t0:.2f}s')

    # ── helpers ───────────────────────────────────────────────────────────────

    def _lookup_tf(self) -> tuple:
        """Return (tf_stamped | None, output_frame_id)."""
        try:
            tf_stamped = self._tf_buffer.lookup_transform(
                self._robot_frame, self._world_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
            return tf_stamped, self._robot_frame
        except Exception as e:
            self.get_logger().warn(
                f'TF ({self._world_frame}→{self._robot_frame}) failed: {e}'
                ' — world frame으로 발행')
            return None, self._world_frame

    def _build_candidates(
        self,
        grasps: np.ndarray,
        confs: np.ndarray,
        tf_stamped,
        output_frame: str,
    ) -> list[dict]:
        candidates: list[dict] = []
        for mat, conf in zip(grasps, confs):
            pos  = mat[:3, 3].astype(np.float32)
            quat = Rot.from_matrix(mat[:3, :3].astype(np.float64)).as_quat().astype(np.float32)
            if tf_stamped is not None:
                pos, quat = apply_world_to_robot_tf(tf_stamped, pos, quat)
            candidates.append({
                'position':   pos.tolist(),
                'quaternion': quat.tolist(),
                'width':      self._gripper_width,
                'quality':    float(conf),
                'frame':      output_frame,
            })
        return candidates

    def destroy_node(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
        super().destroy_node()


# ── entry point ───────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)
    node = GraspGenNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
