"""
vgn_grasp_node.py — ROS 2 node: raw depth → signed TSDF → VGN grasp detection.

Subscriptions:
  /ee_camera/depth_image   (Image)       — cached; EE depth for TSDF
  /ee_camera/camera_info   (CameraInfo)  — cached; EE intrinsics
  /top_camera/depth_image  (Image)       — cached; Top depth (use_top_depth)
  /top_camera/camera_info  (CameraInfo)  — cached; Top intrinsics
  /world_map_result        (String JSON) — trigger; target centroid + bbox_3d

Publications:
  /grasp_candidates  (String JSON)  — Top-K grasp poses
  /grasp_markers     (MarkerArray)  — RViz2 arrows/spheres
  /tsdf_debug        (PointCloud2)  — TSDF voxels (green=surface, red=inside, blue=near-outside)

Extrinsics: src/mask_projection_pkg/config/camera_extrinsics.yaml
  p_world = R @ p_cam + t  (same convention as mask_projection_pkg)
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
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import String, ColorRGBA
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros

from .depth_utils import decode_depth, extract_K, load_extrinsics
from .tsdf_builder import build_tsdf_raycasting, tsdf_to_pointcloud
from .vgn_inference import (
    VGN_OK, VGN_IMPORT_ERROR, load_network,
    vgn_predict, vgn_process, vgn_select, from_voxel_coordinates,
)

_WS = Path(os.environ.get('GSAM_WS', str(Path.home() / 'gsam_ws')))


# ── TF helper ─────────────────────────────────────────────────────────────────

def _apply_tf(tf_stamped, p_world: np.ndarray, quat_world: np.ndarray):
    from scipy.spatial.transform import Rotation as R
    t       = tf_stamped.transform.translation
    r       = tf_stamped.transform.rotation
    tf_rot  = R.from_quat([r.x, r.y, r.z, r.w])
    tf_trans = np.array([t.x, t.y, t.z], dtype=np.float64)
    p_robot  = tf_rot.apply(p_world.astype(np.float64)) + tf_trans
    q_robot  = tf_rot * R.from_quat(quat_world)
    return p_robot.astype(np.float32), q_robot.as_quat().astype(np.float32)


# ── Node ──────────────────────────────────────────────────────────────────────

class VgnGraspNode(Node):

    def __init__(self) -> None:
        super().__init__('vgn_grasp_node')

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('roi_size_m',             0.30)
        self.declare_parameter('tsdf_resolution',        40)
        self.declare_parameter('vgn_model_path',         'models/vgn_conv.pth')
        self.declare_parameter('min_quality',            0.5)
        self.declare_parameter('max_candidates',         5)
        self.declare_parameter('min_point_count',        50)
        self.declare_parameter('ee_depth_topic',         '/ee_camera/depth_image')
        self.declare_parameter('ee_camera_info_topic',   '/ee_camera/camera_info')
        self.declare_parameter('top_depth_topic',        '/top_camera/depth_image')
        self.declare_parameter('top_camera_info_topic',  '/top_camera/camera_info')
        self.declare_parameter('world_map_result_topic', '/world_map_result')
        self.declare_parameter('grasp_candidates_topic', '/grasp_candidates')
        self.declare_parameter('extrinsics_config',      '')
        self.declare_parameter('use_top_depth',          True)
        self.declare_parameter('world_frame',            'world')
        self.declare_parameter('robot_frame',            'panda_link0')

        self._roi_size_m  = self.get_parameter('roi_size_m').value
        self._reso        = self.get_parameter('tsdf_resolution').value
        self._min_quality = self.get_parameter('min_quality').value
        self._max_k       = self.get_parameter('max_candidates').value
        self._min_pts     = self.get_parameter('min_point_count').value
        self._use_top     = self.get_parameter('use_top_depth').value
        self._world_frame = self.get_parameter('world_frame').value
        self._robot_frame = self.get_parameter('robot_frame').value

        # ── Extrinsics ────────────────────────────────────────────────────────
        ext_param = self.get_parameter('extrinsics_config').value or str(
            _WS / 'src/mask_projection_pkg/config/camera_extrinsics.yaml')
        try:
            self._R_ee, self._t_ee, self._R_top, self._t_top = load_extrinsics(ext_param)
            self.get_logger().info(f'Extrinsics loaded: {ext_param}')
        except Exception as e:
            self.get_logger().error(f'Failed to load extrinsics ({ext_param}): {e}')
            raise

        # ── VGN network ───────────────────────────────────────────────────────
        if not VGN_OK:
            self.get_logger().error(
                f'VGN import failed: {VGN_IMPORT_ERROR}\n'
                'Run: git submodule add https://github.com/ethz-asl/vgn external/vgn\n'
                'Then: source launch_env.bash'
            )
            raise RuntimeError('VGN library not available')

        import torch
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'VGN device: {self._device}')

        model_path = Path(self.get_parameter('vgn_model_path').value)
        if not model_path.is_absolute():
            model_path = _WS / model_path
        if not model_path.exists():
            self.get_logger().error(f'VGN model not found: {model_path}')
            raise FileNotFoundError(str(model_path))
        self._net = load_network(model_path, self._device)
        self._net.eval()
        self.get_logger().info(f'VGN model loaded: {model_path}')

        # ── TF2 ───────────────────────────────────────────────────────────────
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── Cache ─────────────────────────────────────────────────────────────
        self._ee_depth:       Optional[np.ndarray] = None
        self._ee_K:           Optional[np.ndarray] = None
        self._top_depth:      Optional[np.ndarray] = None
        self._top_K:          Optional[np.ndarray] = None
        self._pending_result: Optional[str]        = None

        # ── Subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(Image,      self.get_parameter('ee_depth_topic').value,        self._ee_depth_cb,  10)
        self.create_subscription(CameraInfo, self.get_parameter('ee_camera_info_topic').value,  self._ee_info_cb,   10)
        if self._use_top:
            self.create_subscription(Image,      self.get_parameter('top_depth_topic').value,       self._top_depth_cb, 10)
            self.create_subscription(CameraInfo, self.get_parameter('top_camera_info_topic').value, self._top_info_cb,  10)
        self.create_subscription(String, self.get_parameter('world_map_result_topic').value,
                                 self._result_cb, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        grasp_topic      = self.get_parameter('grasp_candidates_topic').value
        self._grasp_pub  = self.create_publisher(String,      grasp_topic,       10)
        self._marker_pub = self.create_publisher(MarkerArray, '/grasp_markers',  10)
        self._tsdf_pub   = self.create_publisher(PointCloud2, '/tsdf_debug',     10)

        self.get_logger().info(
            f'vgn_grasp_node ready  '
            f'roi={self._roi_size_m}m  reso={self._reso}  '
            f'min_quality={self._min_quality}  use_top={self._use_top}'
        )

    # ── Depth / CameraInfo callbacks ──────────────────────────────────────────

    def _ee_depth_cb(self, msg: Image) -> None:
        try:
            self._ee_depth = decode_depth(msg)
        except ValueError as e:
            self.get_logger().warn(f'EE depth decode error: {e}')
            return
        self._try_flush_pending()

    def _ee_info_cb(self, msg: CameraInfo) -> None:
        self._ee_K = extract_K(msg)
        self._try_flush_pending()

    def _top_depth_cb(self, msg: Image) -> None:
        try:
            self._top_depth = decode_depth(msg)
        except ValueError as e:
            self.get_logger().warn(f'Top depth decode error: {e}')

    def _top_info_cb(self, msg: CameraInfo) -> None:
        self._top_K = extract_K(msg)

    def _try_flush_pending(self) -> None:
        if self._pending_result is None or self._ee_depth is None or self._ee_K is None:
            return
        self.get_logger().info('EE depth+info 도착 — 대기 중인 world_map_result 처리')
        pending = String()
        pending.data = self._pending_result
        self._pending_result = None
        self._result_cb(pending)

    # ── Main trigger callback ─────────────────────────────────────────────────

    def _result_cb(self, msg: String) -> None:
        if self._ee_depth is None or self._ee_K is None:
            self._pending_result = msg.data
            self.get_logger().warn('EE depth/info 미도착 — world_map_result 캐시 후 대기')
            return

        try:
            result      = json.loads(msg.data)
            target_info = result.get('target', {})
            centroid    = np.array(target_info['centroid'],   dtype=np.float32)
            point_count = int(target_info['point_count'])
            bbox_raw    = target_info.get('bbox_3d_world', None)
        except (KeyError, json.JSONDecodeError, ValueError) as e:
            self.get_logger().warn(f'world_map_result parse error: {e}')
            return

        if point_count < self._min_pts:
            self.get_logger().info(f'TARGET point_count={point_count} < min={self._min_pts} — skip')
            return

        t0 = time.monotonic()

        half    = self._roi_size_m * 0.5
        roi_min = (centroid - half).astype(np.float64)

        top_args: dict = {}
        if self._use_top and self._top_depth is not None and self._top_K is not None:
            top_args = dict(top_depth=self._top_depth, top_K=self._top_K,
                            R_top=self._R_top, t_top=self._t_top)

        grid = build_tsdf_raycasting(
            roi_min, self._roi_size_m, self._reso,
            self._ee_depth, self._ee_K, self._R_ee, self._t_ee,
            **top_args,
        )

        self._publish_tsdf_debug(grid, roi_min)

        qual_vol, rot_vol, width_vol = vgn_predict(grid, self._net, self._device)
        qual_vol, rot_vol, width_vol = vgn_process(grid, qual_vol, rot_vol, width_vol)

        voxel_size        = self._roi_size_m / self._reso
        grasps_voxel, scores = vgn_select(qual_vol, rot_vol, width_vol,
                                          threshold=self._min_quality)
        if not grasps_voxel:
            self.get_logger().info('No grasp candidates after NMS')
            # DEBUG: quality 무관 best 1개 시각화 (git push 시 주석 처리)
            debug_all, debug_scores = vgn_select(qual_vol, rot_vol, width_vol, threshold=0.0)
            if debug_all:
                best_idx = int(np.argmax(debug_scores))
                best_g   = from_voxel_coordinates(debug_all[best_idx], voxel_size)
                p_w      = (roi_min + best_g.pose.translation).astype(np.float32)
                self._publish_markers([{
                    'position':   p_w.tolist(),
                    'quaternion': best_g.pose.rotation.as_quat().tolist(),
                    'width':      float(best_g.width),
                    'quality':    float(debug_scores[best_idx]),
                    'frame':      self._world_frame,
                }], self._world_frame)
            return

        grasps = [from_voxel_coordinates(g, voxel_size) for g in grasps_voxel]

        if bbox_raw is not None:
            bbox_min = np.array(bbox_raw['min'], dtype=np.float32)
            bbox_max = np.array(bbox_raw['max'], dtype=np.float32)
            kept_grasps, kept_scores = [], []
            for grasp, score in zip(grasps, scores):
                p_w = (roi_min + grasp.pose.translation).astype(np.float32)
                if np.all(p_w >= bbox_min) and np.all(p_w <= bbox_max):
                    kept_grasps.append(grasp)
                    kept_scores.append(float(score))
            if not kept_grasps:
                self.get_logger().info('All candidates rejected by bbox semantic filter')
                # DEBUG: bbox 필터 거른 경우에도 best 1개 시각화 (git push 시 주석 처리)
                best_idx = int(np.argmax(scores))
                best_g   = grasps[best_idx]
                p_w      = (roi_min + best_g.pose.translation).astype(np.float32)
                debug_candidate = [{
                    'position':   p_w.tolist(),
                    'quaternion': best_g.pose.rotation.as_quat().tolist(),
                    'width':      float(best_g.width),
                    'quality':    float(scores[best_idx]),
                    'frame':      self._world_frame,
                }]
                self._publish_markers(debug_candidate, self._world_frame)
                return
        else:
            self.get_logger().warn('No bbox_3d_world — skipping semantic filter')
            kept_grasps = list(grasps)
            kept_scores = [float(s) for s in scores]

        order      = np.argsort(kept_scores)[::-1]
        top_grasps = [kept_grasps[i] for i in order[:self._max_k]]
        top_scores = [kept_scores[i] for i in order[:self._max_k]]

        try:
            tf_stamped = self._tf_buffer.lookup_transform(
                self._robot_frame, self._world_frame,
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1))
            tf_ok = True
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed ({e}) — publishing in world frame')
            tf_ok = False

        output_frame = self._robot_frame if tf_ok else self._world_frame
        candidates   = []
        for grasp, score in zip(top_grasps, top_scores):
            p_w  = (roi_min + grasp.pose.translation).astype(np.float32)
            quat = grasp.pose.rotation.as_quat()
            if tf_ok:
                p_w, quat = _apply_tf(tf_stamped, p_w, quat)
            candidates.append({
                'position':   p_w.tolist(),
                'quaternion': quat.tolist(),
                'width':      float(grasp.width),
                'quality':    score,
                'frame':      output_frame,
            })

        out_msg      = String()
        out_msg.data = json.dumps({
            'candidates':      candidates,
            'target_centroid': centroid.tolist(),
            'stamp':           self.get_clock().now().nanoseconds * 1e-9,
        })
        self._grasp_pub.publish(out_msg)
        self._publish_markers(candidates, output_frame)

        self.get_logger().info(
            f'Published {len(candidates)} grasp(s)  '
            f'best_quality={top_scores[0]:.3f}  elapsed={time.monotonic()-t0:.2f}s'
        )

    # ── TSDF debug publisher ──────────────────────────────────────────────────

    def _publish_tsdf_debug(self, grid: np.ndarray, roi_min: np.ndarray) -> None:
        """Publish TSDF as PointCloud2 on /tsdf_debug for RViz2 inspection."""
        pts, colors = tsdf_to_pointcloud(grid, roi_min, self._roi_size_m)
        if len(pts) == 0:
            return

        # Pack xyz + RGB into float32 columns (PointCloud2 rgb convention)
        rgb_packed = (colors[:, 2].astype(np.uint32)
                      | (colors[:, 1].astype(np.uint32) << 8)
                      | (colors[:, 0].astype(np.uint32) << 16))
        rgb_f = rgb_packed.view(np.float32)
        data  = np.column_stack([pts, rgb_f.reshape(-1, 1)]).astype(np.float32).tobytes()

        msg                 = PointCloud2()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self._world_frame
        msg.height          = 1
        msg.width           = len(pts)
        msg.is_dense        = True
        msg.is_bigendian    = False
        msg.point_step      = 16
        msg.row_step        = 16 * len(pts)
        msg.fields          = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.data = data
        self._tsdf_pub.publish(msg)

    # ── RViz markers ─────────────────────────────────────────────────────────

    def _publish_markers(self, candidates: list, frame: str) -> None:
        from scipy.spatial.transform import Rotation as R

        now = self.get_clock().now().to_msg()

        clear_ma            = MarkerArray()
        clear_m             = Marker()
        clear_m.header.frame_id = frame
        clear_m.header.stamp    = now
        clear_m.ns              = 'vgn_grasps'
        clear_m.action          = Marker.DELETEALL
        clear_ma.markers        = [clear_m]
        self._marker_pub.publish(clear_ma)

        markers: list[Marker] = []
        for i, c in enumerate(candidates):
            q      = float(c['quality'])
            color  = ColorRGBA(r=0.0, g=0.6 * q, b=1.0, a=1.0)
            pos    = c['position']
            quat   = c['quaternion']
            rot    = R.from_quat(quat)
            approach  = rot.apply([0.0, 0.0, -1.0])
            arrow_len = 0.20

            arrow = Marker()
            arrow.header.frame_id = frame
            arrow.header.stamp    = now
            arrow.ns   = 'vgn_grasps'
            arrow.id   = i * 3 + 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.points = [
                Point(x=pos[0] - approach[0] * arrow_len,
                      y=pos[1] - approach[1] * arrow_len,
                      z=pos[2] - approach[2] * arrow_len),
                Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
            ]
            arrow.scale = Vector3(x=0.02, y=0.04, z=0.0)
            arrow.color = color
            markers.append(arrow)

            sphere = Marker()
            sphere.header.frame_id = frame
            sphere.header.stamp    = now
            sphere.ns     = 'vgn_grasps'
            sphere.id     = i * 3 + 2
            sphere.type   = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            sphere.scale  = Vector3(x=0.03, y=0.03, z=0.03)
            sphere.color  = color
            markers.append(sphere)

            width = float(c.get('width', 0.05))
            ring  = Marker()
            ring.header.frame_id = frame
            ring.header.stamp    = now
            ring.ns     = 'vgn_grasps'
            ring.id     = i * 3 + 3
            ring.type   = Marker.CYLINDER
            ring.action = Marker.ADD
            ring.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            gripper_x  = rot.apply([1.0, 0.0, 0.0])
            z_axis     = np.array([0.0, 0.0, 1.0])
            cross      = np.cross(z_axis, gripper_x)
            cross_norm = np.linalg.norm(cross)
            if cross_norm > 1e-6:
                from scipy.spatial.transform import Rotation as _R2
                axis   = cross / cross_norm
                angle  = float(np.arccos(np.clip(np.dot(z_axis, gripper_x), -1, 1)))
                q_ring = _R2.from_rotvec(axis * angle).as_quat()
            else:
                q_ring = np.array([0.0, 0.0, 0.0, 1.0])
            ring.pose.orientation.x = float(q_ring[0])
            ring.pose.orientation.y = float(q_ring[1])
            ring.pose.orientation.z = float(q_ring[2])
            ring.pose.orientation.w = float(q_ring[3])
            ring.scale = Vector3(x=width, y=width, z=0.005)
            ring.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            markers.append(ring)

        ma          = MarkerArray()
        ma.markers  = markers
        self._marker_pub.publish(ma)


# ── entry point ───────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)
    node = VgnGraspNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
