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
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from scipy.spatial.transform import Rotation as Rot
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
import tf2_ros

# /grasp_candidates 는 한 번 추론 후 결과 고정 — BT 가 늦게 구독해도
# 마지막 grasp pool 받도록 TRANSIENT_LOCAL.
_LATCHED_QOS = QoSProfile(
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
)

from .zmq_client import GraspGenClient, check_deps
from .depth_utils import (decode_depth, decode_mask, extract_K,
                           load_ee_extrinsics, apply_world_to_robot_tf)
from .cloud_extractor import find_target_mask_val, extract_target_cloud
from .marker_publisher import build_grasp_markers, build_target_cloud_msg
from .grasp_filter import (
    top_down_filter, confidence_top_n, IKFeasibilityChecker)

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
        self._init_filters()

        self.get_logger().info(
            f'graspgen_node ready  topk={self._topk}  '
            f'num_grasps={self._num_grasps}  min_quality={self._min_quality:.2f}  '
            f'min_pts={self._min_pts}  '
            f'top_down={self._td_enabled}@{self._td_angle_deg:.0f}°  '
            f'ik={self._ik_enabled}  max_pub={self._max_published}')

    # ── initialisation helpers ────────────────────────────────────────────────

    def _declare_params(self) -> None:
        p = self.declare_parameter
        p('zmq_host',                 '127.0.0.1')
        p('zmq_port',                 5556)
        p('zmq_timeout_ms',           5000)
        # Paper-aligned defaults (Sundaralingam et al., 2025):
        #   num_grasps = diffusion batch (Fig. 14); topk = 100 (§5.1);
        #   min_quality = confidence threshold ≥ 0.5 (§6.10).
        # The paper recommends publishing the full top-K as a goal set to
        # the downstream planner *without* NMS — NMS toggle is server-side
        # (see zmq_client.py). BT's retry budget is decoupled via
        # robot_defaults.yaml::bt_pick_retries.
        p('num_grasps',               200)
        p('topk_num_grasps',          100)
        p('min_quality',              0.5)
        p('min_point_count',          50)
        p('max_points',               4096)
        p('min_depth',                0.05)
        p('max_depth',                15.0)
        p('gripper_width',            0.08)
        # Workspace floor margin (m): drop server-returned grasps whose
        # translation Z is more than this far below the TARGET bbox bottom.
        # Set to 0.01 so a top-down approach can land right at the table
        # surface, but rejects "from below" grasps that try to spear through
        # the supporting plane. Set negative to disable.
        p('z_floor_margin',           0.01)
        # ── Gripper TCP offset ────────────────────────────────────────────
        # GraspGen returns the grasp pose at the *grasp center* (between
        # the open fingertips at fingertip depth). bt_pkg's MoveAction
        # targets `panda_link8` (wrist plate). The Franka standard
        # panda_link8 → fingertip distance is ~0.103 m along the link's +Z.
        # Without compensation, sending the grasp center as the
        # panda_link8 goal drives the fingertips 0.103 m past the object.
        # The offset shifts the published pose *back* along the gripper's
        # approach direction (-Z under the convention below) so the
        # fingertips land on the original grasp center. Magnitude only —
        # sign is applied in _build_candidates. Set to 0.0 to disable.
        p('panda_link8_offset',       0.103)
        # 180° X-flip of the orientation.
        # Empirical finding (R[2,2] diagnosis 2026-06-04): GraspGen's
        # output ALREADY uses +Z = approach (toward fingertips/object),
        # which matches panda_link8 directly. Applying the flip inverted
        # both orientation and TCP-offset direction, sending grasps below
        # the table (graspgen_5 regression). Default is now OFF — only
        # enable if a server-side convention change re-introduces a sign
        # flip and TCP offset needs to follow.
        p('flip_orientation_x',       False)
        # Optional X-Y override: replace each grasp's (x, y) with the
        # TARGET bbox horizontal center. Helps when GraspGen's raw output
        # is X-Y-offset from the object centroid (observed for flat
        # objects like a book on a table). Z and orientation are kept.
        p('override_xy_with_bbox_center', False)
        # GraspGen 의 yaw 가 직사각형 평판 (책 등) 의 긴 축과 정렬되어 finger 가
        # 객체 *옆면* 이 아니라 *위* 를 누르는 경우 — close 시 finger 가 객체를
        # 옆으로 밀어 쓰러뜨림. true 면 TARGET bbox 의 XY 짧은 축 방향으로
        # gripper finger plane (+X) 을 강제 정렬. approach (+Z) 는 유지.
        # 모든 candidate 가 같은 yaw 가 되므로 후보 다양성은 줄지만 정렬 성공률↑.
        # 주의: world AABB 기반이라 객체가 yaw 회전돼 있으면 부정확.
        p('align_yaw_with_bbox_short_axis', False)
        # GraspGen 의 noisy/tilted orientation 을 완전히 버리고 깨끗한 수직
        # top-down grasp 을 합성. true 면:
        #   - approach (+Z) = 월드 수직 아래 (panda_link0 -Z)
        #   - finger spread (+X) = TARGET 포인트클라우드 XY PCA 의 *짧은* 축
        #     (= 객체의 가장 얇은 수평 방향, 회전돼 있어도 정확)
        # 모든 candidate 가 동일한 깨끗한 수직 grasp 이 됨. tilted 접근으로
        # finger 가 객체에 걸리는 문제를 근본 해결. 평판/책에 권장.
        # align_yaw_with_bbox_short_axis 보다 우선 적용됨.
        p('force_top_down_orientation', False)
        # force_top_down 일 때 fingertip 의 목표 z 를 TARGET centroid 기준
        # 으로 결정론적으로 설정 (GraspGen 의 scatter 된 z 무시).
        # fingertip_z = centroid_z + 이 값. 0 이면 정확히 centroid 높이에서
        # 잡음 (책 중간). 음수면 더 깊이, 양수면 더 위. wrist 는 자동으로
        # fingertip + 0.103 (panda_link8→fingertip) 위로 설정됨.
        p('force_top_down_grasp_z_offset', 0.0)
        # ── Client-side grasp filters ────────────────────────────────────
        # The paper recommends publishing ~100 grasps as a goal set, but
        # our MoveIt + BT pipeline is single-goal sequential. We pre-filter
        # client-side to ship a small, BT-friendly pool. See grasp_filter.py.
        p('top_down_filter_enabled',  True)
        p('top_down_angle_deg',       45.0)   # ≤45° from straight down
        p('ik_filter_enabled',        True)
        p('ik_service_name',          '/compute_ik')
        p('ik_planning_group',        'panda_arm')
        p('ik_ee_link',               'panda_link8')
        p('ik_per_call_timeout_sec',  0.10)   # per-grasp IK budget
        p('ik_service_wait_sec',      5.0)    # startup wait for MoveIt
        # Final cap on the published pool. BT only retries `bt_pick_retries`
        # of these, but keep a safety margin (extra reachable candidates
        # behind the active retry budget in case future BT logic uses them).
        p('max_published_grasps',     10)
        p('ee_depth_topic',           '/ee_camera/depth_image')
        p('ee_camera_info_topic',     '/ee_camera/camera_info')
        p('mask_topic',               '/qwen/mask_image')
        p('labeled_detections_topic', '/qwen/labeled_detections')
        p('world_map_result_topic',   '/world_map_result')
        p('grasp_candidates_topic',   '/grasp_candidates')
        p('extrinsics_config',        '')
        p('world_frame',              'world')
        p('robot_frame',              'panda_link0')
        # 한 번 publish 후 새 /world_map_result 입력 무시 (true) — projector freeze
        # 와 함께 쓰면 이중 방어. ZMQ 추론 비용도 아낌.
        p('freeze_after_first_publish', False)

    def _load_params(self) -> None:
        g = self.get_parameter
        self._num_grasps    = g('num_grasps').value
        self._topk          = g('topk_num_grasps').value
        self._min_quality   = float(g('min_quality').value)
        self._min_pts       = g('min_point_count').value
        self._max_pts       = g('max_points').value
        self._min_depth     = g('min_depth').value
        self._max_depth     = g('max_depth').value
        self._gripper_width = g('gripper_width').value
        self._z_floor_margin= float(g('z_floor_margin').value)
        self._tcp_offset    = float(g('panda_link8_offset').value)
        self._flip_x        = bool(g('flip_orientation_x').value)
        self._world_frame   = g('world_frame').value
        self._robot_frame   = g('robot_frame').value
        self._override_xy   = bool(g('override_xy_with_bbox_center').value)
        self._align_yaw     = bool(g('align_yaw_with_bbox_short_axis').value)
        self._force_top_down = bool(g('force_top_down_orientation').value)
        self._ftd_grasp_z_off = float(g('force_top_down_grasp_z_offset').value)
        self._td_enabled    = bool(g('top_down_filter_enabled').value)
        self._td_angle_deg  = float(g('top_down_angle_deg').value)
        self._ik_enabled    = bool(g('ik_filter_enabled').value)
        self._max_published = int(g('max_published_grasps').value)
        self._freeze_after_first = bool(g('freeze_after_first_publish').value)
        self._frozen = False

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

    def _init_filters(self) -> None:
        """Set up the optional IK feasibility checker.

        The IK client uses a ReentrantCallbackGroup so its response
        callback can dispatch on a *different* thread than the one
        polling `future.done()`. This requires main() to spin the node on
        a MultiThreadedExecutor — see __main__.
        """
        self._ik_checker: Optional[IKFeasibilityChecker] = None
        if not self._ik_enabled:
            return

        g = self.get_parameter
        self._ik_checker = IKFeasibilityChecker(
            node                 = self,
            service_name         = g('ik_service_name').value,
            planning_group       = g('ik_planning_group').value,
            ee_link              = g('ik_ee_link').value,
            frame_id             = self._robot_frame,
            per_call_timeout_sec = float(g('ik_per_call_timeout_sec').value),
        )
        wait = float(g('ik_service_wait_sec').value)
        ok   = self._ik_checker.wait_for_service(wait)
        if ok:
            self.get_logger().info(
                f'IK service ready: {g("ik_service_name").value} '
                f'(group={g("ik_planning_group").value}, link={g("ik_ee_link").value})')
        else:
            self.get_logger().warn(
                f'IK service {g("ik_service_name").value} not available after '
                f'{wait:.1f}s — IK filter will pass-through until MoveIt comes up')

    def _init_pubsub(self) -> None:
        g = self.get_parameter
        self.create_subscription(Image,      g('ee_depth_topic').value,           self._ee_depth_cb, 10)
        self.create_subscription(CameraInfo, g('ee_camera_info_topic').value,     self._ee_info_cb,  10)
        self.create_subscription(Image,      g('mask_topic').value,               self._mask_cb,     10)
        self.create_subscription(String,     g('labeled_detections_topic').value, self._dets_cb,     10)
        self.create_subscription(String,     g('world_map_result_topic').value,   self._result_cb,   10)

        # grasp_candidates 만 latched. markers/target_cloud 는 RViz 디버그용이라
        # VOLATILE 유지 (latching 시 stale 마커가 누적될 수 있음).
        self._grasp_pub  = self.create_publisher(String,       g('grasp_candidates_topic').value, _LATCHED_QOS)
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
        # Freeze mode: 한 번 성공 publish 후 신규 world_map_result 입력 무시.
        # latched /grasp_candidates 가 BT 에 영원히 노출되어 추가 추론은 불필요.
        if self._frozen:
            return
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

        # PCA 로 TARGET 의 수평(XY) 주축 계산 — force_top_down / yaw 정렬에 사용.
        # 짧은 축 = 객체의 가장 얇은 수평 방향 = finger 가 span 해야 할 방향.
        # world AABB 와 달리 객체가 yaw 회전돼 있어도 정확. (static TF 가
        # panda_link0=world identity 라 robot frame 과 동일.)
        self._target_short_axis = None
        try:
            xy = np.asarray(pts_world, dtype=np.float64)[:, :2]
            xy_c = xy - xy.mean(axis=0)
            cov = xy_c.T @ xy_c
            evals, evecs = np.linalg.eigh(cov)   # 오름차순 — evecs[:,0] = 최소 분산
            short2d = evecs[:, 0]                 # 짧은 축 (얇은 방향)
            self._target_short_axis = np.array([short2d[0], short2d[1], 0.0])
            self._target_short_axis /= (np.linalg.norm(self._target_short_axis) + 1e-12)
            ratio = float(evals[0] / (evals[1] + 1e-12))
            self.get_logger().info(
                f'TARGET PCA: short_axis=({short2d[0]:+.2f}, {short2d[1]:+.2f}) '
                f'분산비={ratio:.2f}')
        except (ValueError, np.linalg.LinAlgError) as e:
            self.get_logger().warn(f'PCA short-axis 계산 실패: {e}')

        try:
            grasps, confs = self._client.request(pts_world, self._num_grasps, self._topk)
        except (RuntimeError, ValueError) as e:
            self.get_logger().error(f'GraspGen: {e}')
            return

        if len(grasps) == 0:
            self.get_logger().warn('GraspGen: 결과 없음')
            return

        # Workspace floor filter — drop grasps that pierce the table.
        # The GraspGen server is trained object-centric and does not know
        # where the supporting surface is; without this filter top-down
        # grasps coming from below the object are returned as valid.
        if self._z_floor_margin >= 0.0:
            try:
                bbox_min = target_info['bbox_3d_world']['min']
                z_floor  = float(bbox_min[2]) - self._z_floor_margin
                z_vals   = np.asarray(grasps)[:, 2, 3]
                keep     = z_vals >= z_floor
                n_drop   = int((~keep).sum())
                if n_drop:
                    self.get_logger().info(
                        f'Workspace filter: dropped {n_drop}/{len(grasps)} '
                        f'grasp(s) with z < {z_floor:.3f}m')
                grasps = grasps[keep]
                confs  = confs[keep]
            except (KeyError, IndexError, TypeError, ValueError) as e:
                self.get_logger().warn(f'workspace filter skipped: {e}')

        if len(grasps) == 0:
            self.get_logger().warn('GraspGen: 필터 후 후보 없음')
            return

        # Confidence threshold (paper §6.10 recommends ≥ 0.5).
        if self._min_quality > 0.0:
            keep   = confs >= self._min_quality
            n_drop = int((~keep).sum())
            if n_drop:
                self.get_logger().info(
                    f'min_quality filter: dropped {n_drop}/{len(grasps)} '
                    f'grasp(s) with conf < {self._min_quality:.2f}')
            grasps = grasps[keep]
            confs  = confs[keep]

        if len(grasps) == 0:
            self.get_logger().warn(
                f'GraspGen: min_quality≥{self._min_quality:.2f} 통과 후보 없음 '
                f'(num_grasps batch={self._num_grasps} 증가 또는 threshold↓ 검토)')
            return

        order  = np.argsort(confs)[::-1]
        grasps = grasps[order[:self._topk]]
        confs  = confs[order[:self._topk]]

        tf_stamped, output_frame = self._lookup_tf()
        candidates = self._build_candidates(grasps, confs, tf_stamped, output_frame)

        # Optional X-Y override → align with TARGET bbox horizontal center.
        if self._override_xy and candidates:
            try:
                bbox_min = np.asarray(target_info['bbox_3d_world']['min'])
                bbox_max = np.asarray(target_info['bbox_3d_world']['max'])
                cx       = float((bbox_min[0] + bbox_max[0]) * 0.5)
                cy       = float((bbox_min[1] + bbox_max[1]) * 0.5)
                for c in candidates:
                    c['position'][0] = cx
                    c['position'][1] = cy
                self.get_logger().info(
                    f'XY override: ({cx:+.3f}, {cy:+.3f}) applied to '
                    f'{len(candidates)} candidate(s)')
            except (KeyError, IndexError, TypeError, ValueError) as e:
                self.get_logger().warn(f'XY override skipped: {e}')

        # Force top-down: orientation 전체를 깨끗한 수직 grasp 으로 교체.
        # approach (+Z) = 아래 (panda_link0 -Z), finger spread (+X) = PCA 짧은 축.
        # GraspGen 의 tilted 출력을 버려 finger 걸림 문제를 근본 해결.
        if self._force_top_down and candidates:
            if self._target_short_axis is None:
                self.get_logger().warn(
                    'force_top_down: PCA short-axis 없음 — bbox 로 폴백 시도')
            try:
                down = np.array([0.0, 0.0, -1.0])     # +Z = approach (아래)
                if self._target_short_axis is not None:
                    x_axis = self._target_short_axis.copy()
                else:
                    # 폴백: bbox 짧은 축
                    bbox_min = np.asarray(target_info['bbox_3d_world']['min'])
                    bbox_max = np.asarray(target_info['bbox_3d_world']['max'])
                    dx = float(bbox_max[0] - bbox_min[0])
                    dy = float(bbox_max[1] - bbox_min[1])
                    x_axis = (np.array([1.0, 0.0, 0.0]) if dx < dy
                              else np.array([0.0, 1.0, 0.0]))
                # x_axis 를 down 에 직교화 (이미 수평이라 거의 그대로)
                x_axis = x_axis - np.dot(x_axis, down) * down
                x_axis /= (np.linalg.norm(x_axis) + 1e-12)
                y_axis = np.cross(down, x_axis)
                R_td   = np.column_stack([x_axis, y_axis, down])
                quat_td = Rot.from_matrix(R_td).as_quat().tolist()
                # 결정론적 Z: fingertip 을 centroid_z + offset 에 두고, wrist(panda_link8)
                # 는 그보다 0.103m 위 (approach=down 이므로 +Z 방향). GraspGen 의
                # scatter 된 z 를 무시해 "가장 낮은 후보가 1등 → 테이블 뚫음" 방지.
                fingertip_z = float(centroid[2]) + self._ftd_grasp_z_off
                wrist_z = fingertip_z + 0.103
                for c in candidates:
                    c['quaternion'] = quat_td
                    c['position'][2] = wrist_z
                self.get_logger().info(
                    f'force_top_down: 수직 grasp (+X=({x_axis[0]:+.2f},'
                    f'{x_axis[1]:+.2f}), fingertip_z={fingertip_z:.3f}, '
                    f'wrist_z={wrist_z:.3f}) → {len(candidates)} candidate(s)')
            except (KeyError, IndexError, TypeError, ValueError) as e:
                self.get_logger().warn(f'force_top_down skipped: {e}')

        # Yaw 정렬: finger plane (+X) 을 bbox 짧은 axis 방향으로 강제.
        # approach (+Z) 은 candidate 별로 유지 (top-down filter 통과한 값).
        # force_top_down 활성 시 이미 orientation 교체됐으므로 건너뜀.
        if self._align_yaw and not self._force_top_down and candidates:
            try:
                bbox_min = np.asarray(target_info['bbox_3d_world']['min'])
                bbox_max = np.asarray(target_info['bbox_3d_world']['max'])
                dx = float(bbox_max[0] - bbox_min[0])
                dy = float(bbox_max[1] - bbox_min[1])
                # 짧은 axis 가 finger plane 이 향해야 할 방향 (close 시 객체
                # 좁은 너비를 잡아 옆으로 밀지 않게).
                short_axis_world = (np.array([1.0, 0.0, 0.0]) if dx < dy
                                    else np.array([0.0, 1.0, 0.0]))
                axis_name = 'X' if dx < dy else 'Y'
                n_done = 0
                for c in candidates:
                    quat = np.asarray(c['quaternion'], dtype=np.float64)
                    R    = Rot.from_quat(quat).as_matrix()
                    z_w  = R[:, 2]
                    # x_new = short_axis 의 z_w 직교 성분 (정사영 제거)
                    x_new = short_axis_world - np.dot(short_axis_world, z_w) * z_w
                    n     = np.linalg.norm(x_new)
                    if n < 1e-6:
                        # approach 가 short axis 와 평행: 회전 정의 불가, 건너뜀
                        continue
                    x_new /= n
                    y_new = np.cross(z_w, x_new)
                    R_new = np.column_stack([x_new, y_new, z_w])
                    c['quaternion'] = Rot.from_matrix(R_new).as_quat().tolist()
                    n_done += 1
                self.get_logger().info(
                    f'Yaw align: short_axis=world-{axis_name} '
                    f'(dx={dx:.3f}, dy={dy:.3f}) applied to {n_done}/{len(candidates)}')
            except (KeyError, IndexError, TypeError, ValueError) as e:
                self.get_logger().warn(f'Yaw align skipped: {e}')

        # ── Client-side filters (Option A — shrink the goal set for the
        #    sequential motion planner). Order: cheap → expensive.
        n_after_build = len(candidates)

        if self._td_enabled:
            candidates = top_down_filter(
                candidates, self._td_angle_deg, logger=self.get_logger())
            self.get_logger().info(
                f'top_down(≤{self._td_angle_deg:.0f}°): '
                f'{n_after_build} → {len(candidates)}')
            if not candidates:
                self.get_logger().warn(
                    'top_down filter dropped all candidates — disable filter '
                    'or relax angle if scene has no top-down approach')
                return

        if self._ik_enabled and self._ik_checker is not None:
            n_before_ik = len(candidates)
            t_ik = time.monotonic()
            candidates, ik_stats = self._ik_checker.filter(candidates)
            ik_dt = time.monotonic() - t_ik
            if ik_stats['service_down']:
                self.get_logger().warn(
                    f'IK service unavailable — pass-through ({n_before_ik} kept)')
            else:
                self.get_logger().info(
                    f'IK feasibility: {ik_stats["kept"]}/{ik_stats["checked"]} '
                    f'reachable  elapsed={ik_dt:.2f}s')
            if not candidates:
                self.get_logger().warn(
                    'IK filter dropped all candidates — robot cannot reach any '
                    'top-down approach for this target')
                return

        # Final confidence cap. Already confidence-sorted upstream, but
        # filters may have removed leading entries — resort then truncate.
        candidates = confidence_top_n(candidates, self._max_published)

        out      = String()
        out.data = json.dumps({
            'candidates':      candidates,
            'target_centroid': centroid.tolist(),
            'stamp':           self.get_clock().now().nanoseconds * 1e-9,
        })
        self._grasp_pub.publish(out)

        # 마커는 실제 Franka panda_link8 → fingertip 기하 (0.103m) 로 렌더링.
        # self._tcp_offset 은 Z 보정 hack 값 (음수일 수 있음) 이라 마커에 쓰면
        # fingertip 이 손목 위로 그려져 깨짐. 시각화는 진짜 기하를 써야 함.
        clear_ma, markers_ma = build_grasp_markers(
            candidates, output_frame, stamp, self._gripper_width,
            tcp_offset=0.103)
        self._marker_pub.publish(clear_ma)
        self._marker_pub.publish(markers_ma)

        best_q = candidates[0]['quality'] if candidates else 0.0
        self.get_logger().info(
            f'Published {len(candidates)} grasp(s)  '
            f'best={best_q:.3f}  elapsed={time.monotonic()-t0:.2f}s')

        # 첫 성공 publish 후 freeze — 이후 world_map_result 입력 무시.
        if self._freeze_after_first and not self._frozen and candidates:
            self._frozen = True
            self.get_logger().info('FROZEN — 이후 /world_map_result 입력 무시. latched /grasp_candidates 그대로.')

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
        # Static 180°-X rotation in gripper frame, applied to convert from
        # GraspGen convention (-Z = approach) to panda_link8 (+Z = approach).
        _R_FLIP_X = Rot.from_euler('x', 180, degrees=True)

        candidates: list[dict] = []
        for mat, conf in zip(grasps, confs):
            R_world = mat[:3, :3].astype(np.float64)
            pos     = mat[:3, 3].astype(np.float64)

            # TCP offset: shift pose origin BACK along the gripper's -Z axis
            # (away from the object) so panda_link8 lands behind the fingertips
            # by self._tcp_offset metres. The convention is +Z = approach
            # (toward the object), so moving *back* means subtracting along
            # R_world[:, 2]. mat[:3, 2] is the gripper +Z in world.
            if self._tcp_offset != 0.0:
                pos = pos - self._tcp_offset * R_world[:, 2]

            # Orientation flip — multiply on the right (intrinsic rotation) so
            # the offset axis we just used (R_world[:, 2]) is the pre-flip Z.
            R_final = (Rot.from_matrix(R_world) * _R_FLIP_X) if self._flip_x \
                      else Rot.from_matrix(R_world)
            quat = R_final.as_quat().astype(np.float32)
            pos  = pos.astype(np.float32)

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
    # MultiThreadedExecutor is required for the IK feasibility filter:
    # the per-grasp `future.done()` poll loop runs on the result callback
    # thread, and the service response callback must dispatch on a
    # different thread to unblock that loop. ReentrantCallbackGroup on
    # the IK client (see grasp_filter.IKFeasibilityChecker) provides that
    # parallelism. SingleThreadedExecutor would deadlock here.
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
