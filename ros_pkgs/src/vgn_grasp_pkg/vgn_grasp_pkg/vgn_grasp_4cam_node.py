"""
vgn_grasp_4cam_node.py — 4-camera variant of vgn_grasp_node.

Cameras: ee (front) + top (overhead) + right (+Y side) + left (-Y side).
Uses build_tsdf_ncam() and load_extrinsics_4cam().
Extrinsics default: camera_extrinsics_4cam.yaml (mask_projection_pkg/config/).

Subscriptions:
  /ee_camera/depth_image    (Image)      — cached
  /ee_camera/camera_info    (CameraInfo) — cached
  /top_camera/depth_image   (Image)      — cached
  /top_camera/camera_info   (CameraInfo) — cached
  /right_camera/depth_image (Image)      — cached
  /right_camera/camera_info (CameraInfo) — cached
  /left_camera/depth_image  (Image)      — cached
  /left_camera/camera_info  (CameraInfo) — cached
  /world_map_result         (String JSON) — trigger

Publications:
  /grasp_candidates  (String JSON)
  /grasp_markers     (MarkerArray)
  /tsdf_debug        (PointCloud2)
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

from .depth_utils import decode_depth, extract_K, load_extrinsics_4cam
from .tsdf_builder import build_tsdf_ncam, tsdf_to_pointcloud
from .vgn_inference import (
    VGN_OK, VGN_IMPORT_ERROR, load_network,
    vgn_predict, vgn_process, vgn_select, from_voxel_coordinates,
)

_ROOT = Path(os.environ.get('ROBOT_CAPSTONE_ROOT', str(Path.home() / 'robot_capstone')))
_DEFAULT_EXTRINSICS = str(
    _ROOT / 'ros_pkgs/src/mask_projection_pkg/config/camera_extrinsics_4cam.yaml')


def _apply_tf(tf_stamped, p_world: np.ndarray, quat_world: np.ndarray):
    from scipy.spatial.transform import Rotation as R
    t        = tf_stamped.transform.translation
    r        = tf_stamped.transform.rotation
    tf_rot   = R.from_quat([r.x, r.y, r.z, r.w])
    tf_trans = np.array([t.x, t.y, t.z], dtype=np.float64)
    p_robot  = tf_rot.apply(p_world.astype(np.float64)) + tf_trans
    q_robot  = tf_rot * R.from_quat(quat_world)
    return p_robot.astype(np.float32), q_robot.as_quat().astype(np.float32)


class VgnGrasp4CamNode(Node):

    def __init__(self) -> None:
        super().__init__('vgn_grasp_4cam_node')

        self.declare_parameter('roi_size_m',              0.30)
        self.declare_parameter('tsdf_resolution',         40)
        self.declare_parameter('vgn_model_path',          'models/vgn_conv.pth')
        self.declare_parameter('min_quality',             0.5)
        self.declare_parameter('max_grasp_candidates',    5)
        self.declare_parameter('min_point_count',         50)
        self.declare_parameter('ee_depth_topic',          '/ee_camera/depth_image')
        self.declare_parameter('ee_camera_info_topic',    '/ee_camera/camera_info')
        self.declare_parameter('top_depth_topic',         '/top_camera/depth_image')
        self.declare_parameter('top_camera_info_topic',   '/top_camera/camera_info')
        self.declare_parameter('right_depth_topic',       '/right_camera/depth_image')
        self.declare_parameter('right_camera_info_topic', '/right_camera/camera_info')
        self.declare_parameter('left_depth_topic',        '/left_camera/depth_image')
        self.declare_parameter('left_camera_info_topic',  '/left_camera/camera_info')
        self.declare_parameter('world_map_result_topic',  '/world_map_result')
        self.declare_parameter('grasp_candidates_topic',  '/grasp_candidates')
        self.declare_parameter('extrinsics_config',       '')
        self.declare_parameter('use_top_depth',           True)
        self.declare_parameter('use_side_depth',          True)
        self.declare_parameter('top_occlude_filter',      True)
        self.declare_parameter('trunc_factor',            4.0)
        self.declare_parameter('ee_weight',               4.0)
        self.declare_parameter('top_weight',              4.0)
        self.declare_parameter('side_weight',             4.0)
        self.declare_parameter('table_top_z',             -999.0)
        self.declare_parameter('world_frame',             'world')
        self.declare_parameter('robot_frame',             'panda_link0')

        self._roi_size_m         = self.get_parameter('roi_size_m').value
        self._reso               = self.get_parameter('tsdf_resolution').value
        self._min_quality        = self.get_parameter('min_quality').value
        self._max_k              = self.get_parameter('max_grasp_candidates').value
        self._min_pts            = self.get_parameter('min_point_count').value
        self._use_top            = self.get_parameter('use_top_depth').value
        self._use_side           = self.get_parameter('use_side_depth').value
        self._top_occlude_filter = self.get_parameter('top_occlude_filter').value
        self._trunc_factor       = self.get_parameter('trunc_factor').value
        self._ee_weight          = self.get_parameter('ee_weight').value
        self._top_weight         = self.get_parameter('top_weight').value
        self._side_weight        = self.get_parameter('side_weight').value
        self._table_top_z_param  = self.get_parameter('table_top_z').value
        self._world_frame        = self.get_parameter('world_frame').value
        self._robot_frame        = self.get_parameter('robot_frame').value

        ext_param = self.get_parameter('extrinsics_config').value or _DEFAULT_EXTRINSICS
        try:
            self._extrinsics = load_extrinsics_4cam(ext_param)
            self.get_logger().info(f'Extrinsics loaded: {ext_param}')
        except Exception as e:
            self.get_logger().error(f'Failed to load extrinsics ({ext_param}): {e}')
            raise

        if not VGN_OK:
            self.get_logger().error(f'VGN import failed: {VGN_IMPORT_ERROR}')
            raise RuntimeError('VGN library not available')

        import torch
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'VGN device: {self._device}')

        model_path = Path(self.get_parameter('vgn_model_path').value)
        if not model_path.is_absolute():
            model_path = _ROOT / model_path
        if not model_path.exists():
            self.get_logger().error(f'VGN model not found: {model_path}')
            raise FileNotFoundError(str(model_path))
        self._net = load_network(model_path, self._device)
        self._net.eval()
        self.get_logger().info(f'VGN model loaded: {model_path}')

        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._cache: dict = {
            'ee':    {'depth': None, 'K': None},
            'top':   {'depth': None, 'K': None},
            'right': {'depth': None, 'K': None},
            'left':  {'depth': None, 'K': None},
        }
        self._pending_result: Optional[str] = None

        qos = 10
        self.create_subscription(
            Image, self.get_parameter('ee_depth_topic').value,
            lambda m: self._depth_cb(m, 'ee'), qos)
        self.create_subscription(
            CameraInfo, self.get_parameter('ee_camera_info_topic').value,
            lambda m: self._info_cb(m, 'ee'), qos)

        if self._use_top:
            self.create_subscription(
                Image, self.get_parameter('top_depth_topic').value,
                lambda m: self._depth_cb(m, 'top'), qos)
            self.create_subscription(
                CameraInfo, self.get_parameter('top_camera_info_topic').value,
                lambda m: self._info_cb(m, 'top'), qos)

        if self._use_side:
            self.create_subscription(
                Image, self.get_parameter('right_depth_topic').value,
                lambda m: self._depth_cb(m, 'right'), qos)
            self.create_subscription(
                CameraInfo, self.get_parameter('right_camera_info_topic').value,
                lambda m: self._info_cb(m, 'right'), qos)
            self.create_subscription(
                Image, self.get_parameter('left_depth_topic').value,
                lambda m: self._depth_cb(m, 'left'), qos)
            self.create_subscription(
                CameraInfo, self.get_parameter('left_camera_info_topic').value,
                lambda m: self._info_cb(m, 'left'), qos)

        self.create_subscription(
            String, self.get_parameter('world_map_result_topic').value,
            self._result_cb, qos)

        grasp_topic      = self.get_parameter('grasp_candidates_topic').value
        self._grasp_pub  = self.create_publisher(String,      grasp_topic,      qos)
        self._marker_pub = self.create_publisher(MarkerArray, '/grasp_markers', qos)
        self._tsdf_pub   = self.create_publisher(PointCloud2, '/tsdf_debug',    qos)

        self.get_logger().info(
            f'vgn_grasp_4cam_node ready  '
            f'roi={self._roi_size_m}m  reso={self._reso}  '
            f'min_quality={self._min_quality}  '
            f'use_top={self._use_top}  use_side={self._use_side}'
        )

    # ── Cache callbacks ───────────────────────────────────────────────────────

    def _depth_cb(self, msg: Image, name: str) -> None:
        try:
            self._cache[name]['depth'] = decode_depth(msg)
        except ValueError as e:
            self.get_logger().warn(f'{name} depth decode error: {e}')
            return
        if name == 'ee':
            self._try_flush_pending()

    def _info_cb(self, msg: CameraInfo, name: str) -> None:
        self._cache[name]['K'] = extract_K(msg)
        if name == 'ee':
            self._try_flush_pending()

    def _try_flush_pending(self) -> None:
        ee = self._cache['ee']
        if self._pending_result is None or ee['depth'] is None or ee['K'] is None:
            return
        self.get_logger().info('EE depth+info 도착 — 대기 중인 world_map_result 처리')
        pending = String()
        pending.data = self._pending_result
        self._pending_result = None
        self._result_cb(pending)

    # ── Main trigger ──────────────────────────────────────────────────────────

    def _result_cb(self, msg: String) -> None:
        ee = self._cache['ee']
        if ee['depth'] is None or ee['K'] is None:
            self._pending_result = msg.data
            self.get_logger().warn('EE depth/info 미도착 — world_map_result 캐시 후 대기')
            return

        try:
            result      = json.loads(msg.data)
            target_info = result.get('target', {})
            centroid    = np.array(target_info['centroid'],  dtype=np.float32)
            point_count = int(target_info['point_count'])
            bbox_raw    = target_info.get('bbox_3d_world', None)
        except (KeyError, json.JSONDecodeError, ValueError) as e:
            self.get_logger().warn(f'world_map_result parse error: {e}')
            return

        if point_count < self._min_pts:
            self.get_logger().info(
                f'TARGET point_count={point_count} < min={self._min_pts} — skip')
            return

        t0 = time.monotonic()

        half    = self._roi_size_m * 0.5
        roi_min = (centroid - half).astype(np.float64)

        # table_top_z: ROS param 우선, -999이면 workspace JSON centroid Z 사용
        if self._table_top_z_param > -999.0:
            table_top_z = float(self._table_top_z_param)
        else:
            ws_info = result.get('workspace', {})
            if ws_info:
                table_top_z = float(ws_info['centroid'][2])
            else:
                table_top_z = -np.inf
        roi_max_z = float(roi_min[2]) + self._roi_size_m
        if table_top_z >= roi_max_z:
            self.get_logger().warn(
                f'table_top_z={table_top_z:.4f} >= roi_max_z={roi_max_z:.4f} — outlier, disabling clip')
            table_top_z = -np.inf
        self.get_logger().info(
            f'table_top_z={table_top_z:.4f}m  roi_min_z={roi_min[2]:.4f}m  roi_max_z={roi_max_z:.4f}m')

        cameras = [
            {
                'name':           'ee',
                'depth':          ee['depth'],
                'K':              ee['K'],
                'R':              self._extrinsics['ee_camera']['R'],
                't':              self._extrinsics['ee_camera']['t'],
                'weight':         self._ee_weight,
                'occlude_filter': True,
            }
        ]

        top = self._cache['top']
        if self._use_top and top['depth'] is not None and top['K'] is not None:
            cameras.append({
                'name':           'top',
                'depth':          top['depth'],
                'K':              top['K'],
                'R':              self._extrinsics['top_camera']['R'],
                't':              self._extrinsics['top_camera']['t'],
                'weight':         self._top_weight,
                'occlude_filter': self._top_occlude_filter,
            })

        if self._use_side:
            for side in ('right', 'left'):
                cam_name = f'{side}_camera'
                c = self._cache[side]
                if c['depth'] is not None and c['K'] is not None:
                    cameras.append({
                        'name':           side,
                        'depth':          c['depth'],
                        'K':              c['K'],
                        'R':              self._extrinsics[cam_name]['R'],
                        't':              self._extrinsics[cam_name]['t'],
                        'weight':         self._side_weight,
                        'occlude_filter': True,
                    })

        self.get_logger().info(
            'Building TSDF — cameras: '
            + ', '.join(f"{c['name']}(w={c['weight']:.1f})" for c in cameras)
        )

        grid = build_tsdf_ncam(
            roi_min, self._roi_size_m, self._reso,
            cameras, trunc_factor=self._trunc_factor,
            z_min=table_top_z,
        )

        self._publish_tsdf_debug(grid, roi_min)

        qual_vol, rot_vol, width_vol = vgn_predict(grid, self._net, self._device)

        tsdf_flat = grid.squeeze()
        n_surface = int(np.sum((tsdf_flat >= 0.4) & (tsdf_flat <= 0.6)))
        n_inside  = int(np.sum(tsdf_flat < 0.4))
        qual_max  = float(qual_vol.max())
        self.get_logger().info(
            f'TSDF stats — surface voxels: {n_surface}  inside: {n_inside}  '
            f'VGN qual_max(pre-process): {qual_max:.4f}'
        )

        qual_vol, rot_vol, width_vol = vgn_process(grid, qual_vol, rot_vol, width_vol)

        voxel_size = self._roi_size_m / self._reso
        grasps_voxel, scores = vgn_select(qual_vol, rot_vol, width_vol,
                                          threshold=self._min_quality)
        if not grasps_voxel:
            debug_all, debug_scores = vgn_select(qual_vol, rot_vol, width_vol, threshold=0.0)
            if not debug_all:
                self.get_logger().warn(
                    f'VGN quality all-zero after process (surface voxels={n_surface}) — '
                    'TSDF 품질 불량. /tsdf_debug 확인 필요. '
                    'min_quality:=0.1 또는 trunc_factor 조정 시도'
                )
                return
            best_idx = int(np.argmax(debug_scores))
            best_g   = from_voxel_coordinates(debug_all[best_idx], voxel_size)
            p_w      = (roi_min + best_g.pose.translation).astype(np.float32)
            self.get_logger().info(
                f'No grasp above min_quality={self._min_quality:.2f} — '
                f'DEBUG: best={debug_scores[best_idx]:.4f}  pos={p_w.tolist()}'
            )
            self._publish_markers([{
                'position':   p_w.tolist(),
                'quaternion': best_g.pose.rotation.as_quat().tolist(),
                'width':      float(best_g.width),
                'quality':    float(debug_scores[best_idx]),
                'frame':      self._world_frame,
            }], self._world_frame)
            return

        grasps = [from_voxel_coordinates(g, voxel_size) for g in grasps_voxel]

        # grasp z 안전장치: 테이블 면 아래 후보 제거
        _z_floor = table_top_z + 0.005
        grasps_zf, scores_zf = [], []
        for g, s in zip(grasps, scores):
            gz = float(roi_min[2] + g.pose.translation[2])
            if gz >= _z_floor:
                grasps_zf.append(g)
                scores_zf.append(s)
        if not grasps_zf:
            self.get_logger().info(f'All candidates below table floor z={_z_floor:.4f}')
            return
        grasps, scores = grasps_zf, scores_zf

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
                best_idx = int(np.argmax(scores))
                best_g   = grasps[best_idx]
                p_w      = (roi_min + best_g.pose.translation).astype(np.float32)
                self._publish_markers([{
                    'position':   p_w.tolist(),
                    'quaternion': best_g.pose.rotation.as_quat().tolist(),
                    'width':      float(best_g.width),
                    'quality':    float(scores[best_idx]),
                    'frame':      self._world_frame,
                }], self._world_frame)
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

    # ── TSDF debug ────────────────────────────────────────────────────────────

    def _publish_tsdf_debug(self, grid: np.ndarray, roi_min: np.ndarray) -> None:
        pts, colors = tsdf_to_pointcloud(grid, roi_min, self._roi_size_m)
        if len(pts) == 0:
            return
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

    # ── RViz markers ──────────────────────────────────────────────────────────

    def _publish_markers(self, candidates: list, frame: str) -> None:
        from scipy.spatial.transform import Rotation as R

        now = self.get_clock().now().to_msg()

        clear_ma                = MarkerArray()
        clear_m                 = Marker()
        clear_m.header.frame_id = frame
        clear_m.header.stamp    = now
        clear_m.ns              = 'vgn_grasps'
        clear_m.action          = Marker.DELETEALL
        clear_ma.markers        = [clear_m]
        self._marker_pub.publish(clear_ma)

        markers: list[Marker] = []
        for i, c in enumerate(candidates):
            q         = float(c['quality'])
            color     = ColorRGBA(r=0.0, g=0.6 * q, b=1.0, a=1.0)
            pos       = c['position']
            quat      = c['quaternion']
            rot       = R.from_quat(quat)
            approach  = rot.apply([0.0, 0.0, -1.0])
            arrow_len = 0.20

            arrow               = Marker()
            arrow.header.frame_id = frame
            arrow.header.stamp  = now
            arrow.ns            = 'vgn_grasps'
            arrow.id            = i * 3 + 1
            arrow.type          = Marker.ARROW
            arrow.action        = Marker.ADD
            arrow.points        = [
                Point(x=pos[0] - approach[0] * arrow_len,
                      y=pos[1] - approach[1] * arrow_len,
                      z=pos[2] - approach[2] * arrow_len),
                Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
            ]
            arrow.scale         = Vector3(x=0.02, y=0.04, z=0.0)
            arrow.color         = color
            markers.append(arrow)

            sphere               = Marker()
            sphere.header.frame_id = frame
            sphere.header.stamp  = now
            sphere.ns            = 'vgn_grasps'
            sphere.id            = i * 3 + 2
            sphere.type          = Marker.SPHERE
            sphere.action        = Marker.ADD
            sphere.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            sphere.scale         = Vector3(x=0.03, y=0.03, z=0.03)
            sphere.color         = color
            markers.append(sphere)

            width = float(c.get('width', 0.05))
            ring  = Marker()
            ring.header.frame_id = frame
            ring.header.stamp    = now
            ring.ns              = 'vgn_grasps'
            ring.id              = i * 3 + 3
            ring.type            = Marker.CYLINDER
            ring.action          = Marker.ADD
            ring.pose.position   = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
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
            ring.scale              = Vector3(x=width, y=width, z=0.005)
            ring.color              = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            markers.append(ring)

        ma         = MarkerArray()
        ma.markers = markers
        self._marker_pub.publish(ma)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VgnGrasp4CamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
