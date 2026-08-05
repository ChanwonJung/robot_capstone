"""
graspgen_node.py — ROS 2 node: EE depth + GSAM mask → TARGET cloud → ZMQ GraspGen → /grasp_candidates

Publishes /grasp_candidates in the schema bt_pkg expects. Identical JSON and
/grasp_markers so bt_pkg needs no changes.

Data flow:
  /ee_camera/depth_image   ─┐
  /ee_camera/camera_info   ─┼─ cache ──→ _result_cb (trigger: /world_map_result)
  /sam/mask_image          ─┤            ├─ cloud_extractor → ZMQ → candidates
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
import yaml
from ament_index_python.packages import get_package_share_directory
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

# 같은 프로파일을 Slow Brain 입력 구독에도 쓴다. 이게 없으면 발행자가
# TRANSIENT_LOCAL 이어도 VOLATILE 구독자에게는 이력이 전달되지 않는다 —
# QoS 는 호환되므로 경고 한 줄 없이, 이 노드보다 먼저 나간 스캔 결과를
# 영원히 못 받는다. Slow Brain 은 명령당 1회만 도는데 graspgen 은 보통
# 나중에 뜨므로, 이 비대칭이 곧 "GraspGen 이 조용히 아무것도 안 함" 이었다.
_LATCHED_SUB_QOS = _LATCHED_QOS

from .zmq_client import GraspGenClient, check_deps
from .depth_utils import (decode_depth, decode_mask, extract_K,
                           load_ee_extrinsics, apply_world_to_robot_tf)
from .cloud_extractor import find_target_mask_val, extract_target_cloud
from .swindrnet_client import SwinDRNetClient
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
        self._init_swindrnet()
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
        # force_top_down 일 때 fingertip 의 목표 z 를 "복원된 TARGET 포인트
        # 클라우드의 실제 높이 분포" 기준으로 결정 (GraspGen 의 scatter 된 z,
        # 그리고 투명물체에서 see-through 라 테이블 높이(≈0)로 찍히는 projector
        # centroid[2] 를 둘 다 무시한다).
        #   z_bot=10퍼센타일, z_top=90퍼센타일
        #   fingertip_z = z_bot + z_frac*(z_top - z_bot) + z_offset
        # z_frac: 0=바닥(테이블), 1=상단(rim). 0.65 = 몸통 상단부(벽 확실히
        #   잡고 테이블 회피). z_offset: 그 위에 얹는 미세 보정(±m).
        # wrist(panda_link8) 는 자동으로 fingertip + 0.103 위.
        p('force_top_down_grasp_z_frac',   0.65)
        p('force_top_down_grasp_z_offset', 0.0)
        # force_top_down 일 때 XY 를 복원 cloud 의 어느 통계로 잡을지.
        #   'median' — 점들의 중앙값. 점 밀도가 높은 쪽으로 끌린다.
        #   'extent' — 실루엣 폭의 중점 (p2+p98)/2. 밀도와 무관.
        # TARGET 포인트는 EE 카메라 한 시점에서만 나오므로 카메라를 향한 면에
        # 점이 몰린다. 컵/책은 이 편향이 작지만 구에서는 median 이 눈에 띄게
        # 밀려 손가락이 하강 중에 공을 쳐버렸다 (여유가 한쪽 5.6mm 뿐 —
        # 그리퍼 개폐 80mm vs 공 지름 68.8mm). 한쪽 면만 찍혀도 실루엣의
        # 좌우 끝은 양쪽 다 잡히므로 그 중점은 밀도에 안 흔들린다.
        # min/max 가 아니라 2/98 퍼센타일인 건 아웃라이어 한두 점 때문.
        # 기본값은 컵/책에서 검증된 median 을 유지 — 바꾸려면 명시적으로 켤 것.
        p('grasp_xy_anchor', 'median')
        # force_top_down 접근축을 타깃의 기울기에 맞출지. 기본은 정확히 수직.
        # 기울어 선 평판(책)에서만 켜면 된다 — 회전 대칭 물체는 기울일 이유가
        # 없고, 평판이 아닌 클라우드에서 최소분산축을 법선으로 믿으면 엉뚱한
        # 자세가 나온다. 그래서 프로파일에서 물체별로 켠다.
        p('force_top_down_align_tilt', False)
        # 평판 판정: 최소축 std < ratio × 중간축 std. 책 실측은 6.3 vs 32.4mm
        # (비 0.19) 라 여유롭게 통과하고, 구/컵은 통과하지 못한다.
        p('force_top_down_tilt_planar_ratio', 0.5)
        # 접근축이 수직에서 이 각을 넘으면 기울이지 않는다. top_down_filter 의
        # 45° 보다 작게 잡아 필터와 싸우지 않도록 한다.
        p('force_top_down_tilt_max_deg', 30.0)
        # ── Per-object grasp profiles ────────────────────────────────────
        # z_frac / z_offset / xy_anchor 는 물체 모양에 따라 값이 다르다 (구는
        # 적도 아래, 컵은 림 근처). 여태 launch 인자로 매번 넣던 것을 TARGET
        # 라벨 기반으로 자동 선택한다. 표는 config/grasp_profiles.yaml.
        p('use_grasp_profiles', True)
        p('grasp_profiles_config', '')   # 빈 값 = 패키지 기본 config
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
        p('ee_camera_rgb_topic',      '/ee_camera/image_raw')
        p('mask_topic',               '/sam/mask_image')
        p('labeled_detections_topic', '/qwen/labeled_detections')
        p('world_map_result_topic',   '/world_map_result')
        p('grasp_candidates_topic',   '/grasp_candidates')
        p('extrinsics_config',        '')
        p('world_frame',              'world')
        p('robot_frame',              'panda_link0')
        # 한 번 publish 후 새 /world_map_result 입력 무시 (true) — projector freeze
        # 와 함께 쓰면 이중 방어. ZMQ 추론 비용도 아낌.
        p('freeze_after_first_publish', False)
        # ── 투명물체(유리컵) 깊이 복원 ────────────────────────────────────
        # see-through 로 깨진 투명 TARGET depth 를 복원. 2단계 구현:
        # Stage 1: analytic 원통 (기존)
        # Stage 2: SwinDRNet 학습 모델 (새로움, A100 ZMQ 서버)
        # use_swindrnet=true 면 SwinDRNet 시도, 실패하면 analytic fallback.
        p('transparent_reconstruct_enabled', False)
        p('transparent_force',               False)
        p('transparent_labels',              ['glass', 'cup', 'bottle', 'transparent', 'wine'])
        # ──── Stage 2: SwinDRNet (학습 모델) ──────────────────────────────
        p('swindrnet_enabled',               False)
        p('swindrnet_host',                  '127.0.0.1')
        p('swindrnet_port',                  5557)
        p('swindrnet_timeout_ms',            30000)
        # ──── Stage 1: Analytic 원통 (fallback) ──────────────────────────
        p('transparent_cylinder_height',     -1.0)    # <0 → 데이터 추정, 아니면 고정(m)
        p('transparent_radius_min',          0.015)
        p('transparent_radius_max',          0.10)

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
        self._ftd_grasp_z_frac = float(g('force_top_down_grasp_z_frac').value)
        self._ftd_grasp_z_off = float(g('force_top_down_grasp_z_offset').value)
        self._xy_anchor     = str(g('grasp_xy_anchor').value).strip().lower()
        if self._xy_anchor not in ('median', 'extent'):
            self.get_logger().warn(
                f"grasp_xy_anchor='{self._xy_anchor}' 는 알 수 없는 값 — "
                "'median' 으로 되돌린다 (허용: median | extent)")
            self._xy_anchor = 'median'
        # launch 가 준 값을 그대로 보존한다. _apply_grasp_profile 이 매 스캔
        # 여기서 다시 출발해야 대상이 바뀔 때 프로파일이 누적되지 않는다.
        self._align_tilt       = bool(g('force_top_down_align_tilt').value)
        self._tilt_planar_ratio = float(
            g('force_top_down_tilt_planar_ratio').value)
        self._tilt_max_deg     = float(g('force_top_down_tilt_max_deg').value)
        self._base_z_frac      = self._ftd_grasp_z_frac
        self._base_z_off       = self._ftd_grasp_z_off
        self._base_xy_anchor   = self._xy_anchor
        self._base_align_tilt  = self._align_tilt
        self._target_axes      = None
        self._target_axis_std  = None
        self._use_profiles     = bool(g('use_grasp_profiles').value)
        self._grasp_profiles   = self._load_grasp_profiles(
            str(g('grasp_profiles_config').value))
        self._td_enabled    = bool(g('top_down_filter_enabled').value)
        self._td_angle_deg  = float(g('top_down_angle_deg').value)
        self._ik_enabled    = bool(g('ik_filter_enabled').value)
        self._max_published = int(g('max_published_grasps').value)
        self._freeze_after_first = bool(g('freeze_after_first_publish').value)
        self._frozen = False
        self._tr_enabled = bool(g('transparent_reconstruct_enabled').value)
        self._tr_force   = bool(g('transparent_force').value)
        self._tr_labels  = [str(s).lower() for s in g('transparent_labels').value]
        self._swindrnet_enabled = bool(g('swindrnet_enabled').value)
        self._tr_height  = float(g('transparent_cylinder_height').value)
        self._tr_rmin    = float(g('transparent_radius_min').value)
        self._tr_rmax    = float(g('transparent_radius_max').value)

    def _init_extrinsics(self) -> None:
        path = self.get_parameter('extrinsics_config').value or _DEFAULT_EXTRINSICS
        self._R_ee, self._t_ee = load_ee_extrinsics(path)
        self.get_logger().info(f'Extrinsics: {path}')
        self.get_logger().info(f'  t_ee: {self._t_ee}')
        self.get_logger().info(f'  R_ee[2,:] (world Z row): {self._R_ee[2, :]}')

    def _init_zmq(self) -> None:
        g = self.get_parameter
        host, port, timeout = g('zmq_host').value, g('zmq_port').value, g('zmq_timeout_ms').value
        self._client = GraspGenClient(host, port, timeout)
        self.get_logger().info(f'GraspGen ZMQ → tcp://{host}:{port}  timeout={timeout}ms')

    def _init_swindrnet(self) -> None:
        """Initialize SwinDRNet client (optional). Lazy connect on first use."""
        self._swindrnet_client: Optional[SwinDRNetClient] = None
        if not self._swindrnet_enabled:
            return
        g = self.get_parameter
        host = g('swindrnet_host').value
        port = g('swindrnet_port').value
        timeout = g('swindrnet_timeout_ms').value
        try:
            self._swindrnet_client = SwinDRNetClient(host, port, timeout)
            self.get_logger().info(f'SwinDRNet ZMQ → tcp://{host}:{port}  timeout={timeout}ms')
        except Exception as e:
            self.get_logger().error(f'SwinDRNet connection failed: {e} — will fall back to analytic')
            self._swindrnet_client = None

    def _init_tf(self) -> None:
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

    def _init_cache(self) -> None:
        self._ee_depth:       Optional[np.ndarray] = None
        self._ee_K:           Optional[np.ndarray] = None
        self._ee_rgb:         Optional[np.ndarray] = None
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
        self.create_subscription(Image,      g('ee_camera_rgb_topic').value,      self._ee_rgb_cb,   10)
        # Slow Brain 출력 3개는 latched — 구독도 TRANSIENT_LOCAL 이어야 이력이 온다.
        self.create_subscription(Image,      g('mask_topic').value,               self._mask_cb,     _LATCHED_SUB_QOS)
        self.create_subscription(String,     g('labeled_detections_topic').value, self._dets_cb,     _LATCHED_SUB_QOS)
        self.create_subscription(String,     g('world_map_result_topic').value,   self._result_cb,   _LATCHED_SUB_QOS)

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

    def _ee_rgb_cb(self, msg: Image) -> None:
        try:
            h, w = msg.height, msg.width
            data = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding == 'rgb8':
                self._ee_rgb = data.reshape((h, w, 3))
            elif msg.encoding == 'bgr8':
                self._ee_rgb = data.reshape((h, w, 3))
            else:
                self.get_logger().warn(f'RGB encoding not rgb8/bgr8: {msg.encoding}')
                return
        except ValueError as e:
            self.get_logger().warn(f'RGB decode: {e}')
            return
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
        self.get_logger().debug('캐시 완료 — 대기 중인 world_map_result 처리')
        msg      = String()
        msg.data = self._pending_result
        self._pending_result = None
        self._result_cb(msg)

    def _load_grasp_profiles(self, path: str) -> list:
        """config/grasp_profiles.yaml 을 읽어 프로파일 리스트를 돌려준다.

        읽기 실패는 치명적이지 않다 — launch 값으로 계속 돈다. 다만 조용히
        넘어가면 "왜 프로파일이 안 먹지"를 몇 시간 헤매게 되므로 크게 찍는다.
        """
        if not self._use_profiles:
            self.get_logger().info('use_grasp_profiles=false — launch 값만 사용')
            return []
        if not path:
            path = os.path.join(
                get_package_share_directory('graspgen_pkg'),
                'config', 'grasp_profiles.yaml')
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
            profiles = cfg.get('profiles') or []
            for prof in profiles:
                prof['match'] = [str(k).lower() for k in prof.get('match', [])]
            names = ', '.join(str(p.get('name', '?')) for p in profiles)
            self.get_logger().info(
                f'grasp profiles ({len(profiles)}) from {path}: {names}')
            return profiles
        except Exception as e:  # noqa: BLE001 — 파지는 계속 가능해야 한다
            self.get_logger().error(
                f'grasp_profiles 로드 실패 ({path}): {e} — launch 값으로 진행. '
                '물체별 z_frac/z_offset 은 인자로 직접 넣어야 한다.')
            return []

    def _target_text(self) -> str:
        """TARGET detection 의 모든 값을 합친 소문자 문자열 (키워드 매칭용).

        VLM 라벨은 자유 문자열이라 ('red sphere', 'white cylinder') 정확 일치가
        아니라 키워드 부분일치로 본다. TARGET 이 없으면 빈 문자열.
        """
        for det in (self._labeled_dets or []):
            if str(det.get('category', '')).upper() == 'TARGET':
                return ' '.join(str(v) for v in det.values()).lower()
        return ''

    def _is_transparent_target(self) -> bool:
        """TARGET 이 투명물체인지 판정. force=true 면 항상 True(테스트용),
        아니면 TARGET detection 의 값 문자열에 transparent_labels 키워드 매칭."""
        if self._tr_force:
            return True
        text = self._target_text()
        return bool(text) and any(k in text for k in self._tr_labels)

    def _apply_grasp_profile(self) -> None:
        """TARGET 라벨에 맞는 force_top_down 튜닝을 이번 스캔에 적용한다.

        구는 적도 아래를 물어야 하고 컵은 림 근처를 물어야 해서, launch 기본값
        하나로는 둘 다 못 맞춘다. 여태 매번 손으로 인자를 넣었고, 하나 빠뜨리면
        인지 버그처럼 보이는 파지 실패로 재현됐다 — 그걸 라벨로 자동화한다.

        _base_* 를 매번 기준으로 삼는다. 직전 스캔이 남긴 값에서 출발하면
        대상이 바뀔 때 프로파일이 누적돼 엉뚱한 높이가 된다.
        """
        self._ftd_grasp_z_frac = self._base_z_frac
        self._ftd_grasp_z_off = self._base_z_off
        self._xy_anchor = self._base_xy_anchor
        self._align_tilt = self._base_align_tilt

        if not self._use_profiles or not self._grasp_profiles:
            return

        text = self._target_text()
        if not text:
            return

        for prof in self._grasp_profiles:
            if not any(k in text for k in prof.get('match', [])):
                continue
            self._ftd_grasp_z_frac = float(
                prof.get('z_frac', self._base_z_frac))
            self._ftd_grasp_z_off = float(
                prof.get('z_offset', self._base_z_off))
            self._xy_anchor = str(
                prof.get('xy_anchor', self._base_xy_anchor)).strip().lower()
            self._align_tilt = bool(
                prof.get('align_tilt', self._base_align_tilt))
            self.get_logger().info(
                f"grasp profile '{prof.get('name', '?')}' matched TARGET "
                f"→ z_frac={self._ftd_grasp_z_frac:.2f} "
                f"z_offset={self._ftd_grasp_z_off:+.3f} "
                f"xy_anchor={self._xy_anchor} "
                f"align_tilt={self._align_tilt} "
                f"(launch 값 {self._base_z_frac:.2f}/{self._base_z_off:+.3f}/"
                f"{self._base_xy_anchor} 대체 — 끄려면 use_grasp_profiles:=false)")
            return

        self.get_logger().info(
            f'매칭되는 grasp profile 없음 → launch 값 유지 '
            f'z_frac={self._base_z_frac:.2f} z_offset={self._base_z_off:+.3f} '
            f'xy_anchor={self._base_xy_anchor}')

    # ── main trigger ─────────────────────────────────────────────────────────

    def _result_cb(self, msg: String) -> None:
        # Freeze mode: 한 번 성공 publish 후 신규 world_map_result 입력 무시.
        # latched /grasp_candidates 가 BT 에 영원히 노출되어 추가 추론은 불필요.
        if self._frozen:
            return
        if self._ee_depth is None or self._ee_K is None or self._mask is None or self._ee_rgb is None:
            self._pending_result = msg.data
            self.get_logger().warn('EE 캐시 미도착 — 대기')
            return

        # 대상이 바뀌면 파지 높이도 바뀐다. detections 가 캐시된 뒤, 실제 추론에
        # 쓰이기 전에 이번 스캔의 TARGET 라벨로 프로파일을 고른다.
        self._apply_grasp_profile()

        # [디버그] 원본 EE depth 범위 확인
        self.get_logger().debug(
            f'[디버그] 원본 ee_depth: shape={self._ee_depth.shape} '
            f'dtype={self._ee_depth.dtype} '
            f'min={self._ee_depth.min():.6f} max={self._ee_depth.max():.6f} '
            f'mean={self._ee_depth.mean():.6f}')

        # inf = 창밖 하늘 등 센서 레인지 밖. 유리컵과 무관하다 (유리컵 영역은
        # see-through 로 뒤 테이블 depth 가 찍힌다 — 그게 SwinDRNet 이 고칠 입력).
        # 표준 RGB-D 규약대로 0 = invalid 로 표기해서 모델에 넘긴다.
        ee_depth_clean = self._ee_depth.copy()
        inf_mask = ~np.isfinite(ee_depth_clean)
        if inf_mask.any():
            ee_depth_clean[inf_mask] = 0.0
            self.get_logger().debug(
                f'inf/NaN {inf_mask.sum()} px (센서 레인지 밖) → 0 = invalid')
        self._ee_depth = ee_depth_clean

        try:
            result      = json.loads(msg.data)
            # Projection JSON 구조: {"target": {...}, "glass cup": {...}, "obstacle": {...}, ...}
            # 투명물체는 "glass cup", "cup" 등의 라벨로 들어올 수 있음
            target_info = None
            target_label = None

            # 1. "target" 라벨 먼저 시도
            if 'target' in result:
                target_info = result['target']
                target_label = 'target'

            # 2. transparent_labels 중 매칭되는 라벨 찾기
            if target_info is None:
                for label in self._tr_labels:
                    if label in result:
                        target_info = result[label]
                        target_label = label
                        break

            # 3. 일반적인 라벨들 시도
            if target_info is None:
                for label in ['glass cup', 'cup', 'mug', 'bottle']:
                    if label in result:
                        target_info = result[label]
                        target_label = label
                        break

            if target_info is None:
                available = list(result.keys())
                self.get_logger().info(f'TARGET 라벨 없음. 사용 가능한 키: {available}')
                return

            centroid    = np.array(target_info['centroid'], dtype=np.float32)
            point_count = int(target_info.get('point_count', 0))
            self.get_logger().debug(f'TARGET 찾음: {target_label}, points={point_count}')
        except (KeyError, json.JSONDecodeError, ValueError) as e:
            self.get_logger().warn(f'world_map_result parse: {e}')
            return

        if point_count < self._min_pts:
            self.get_logger().info(f'TARGET count={point_count} < min={self._min_pts} — skip')
            return

        t0         = time.monotonic()
        t_extract  = time.monotonic()
        target_val = find_target_mask_val(self._labeled_dets)

        # [진단] Glass mask 영역의 깊이값 확인
        glass_mask = (self._mask == target_val)
        glass_region = self._ee_depth[glass_mask]

        if glass_region.size > 0:
            self.get_logger().debug(
                f'[진단] Glass mask 영역 깊이: '
                f'size={glass_region.size}, '
                f'min={np.nanmin(glass_region):.6f}, '
                f'max={np.nanmax(glass_region):.6f}, '
                f'mean={np.nanmean(glass_region):.6f}, '
                f'inf_count={np.isinf(glass_region).sum()}, '
                f'nan_count={np.isnan(glass_region).sum()}, '
                f'zero_count={(glass_region == 0.0).sum()}, '
                f'valid_count={np.isfinite(glass_region).sum()}')

        # 투명 TARGET → SwinDRNet 으로 depth 복원.
        # 기하학적 prior(inpaint/원통/경계보간) 는 쓰지 않는다. 모델이 복원하지
        # 못하면 복원 실패로 보고하고 raw depth 로 진행한다 — 실패를 감추지 말 것.
        pts_world = None
        if self._tr_enabled and self._is_transparent_target():
            self.get_logger().debug('[투명복원] 투명 TARGET 감지')

            if self._swindrnet_enabled and self._swindrnet_client:
                try:
                    import cv2
                    t_swin = time.monotonic()

                    # ── fine-tune 전처리와 1:1 일치 (필수) ──────────────────
                    # 모델은 컵중심 252 크롭 → 224, 컵=구멍(0), off-center pp 로
                    # 학습됨(finetune_swindrnet.py CupDS). 풀프레임을 그대로 넣으면
                    # 스케일/FOV 가 out-of-distribution 이 되어 fine-tune 효과가
                    # 사라진다. 여기서 학습과 동일한 크롭 입력을 만들어 보낸다.
                    # (대조실험: 크롭+use_pp 경로가 학습경로와 컵 L1 ~1mm 일치 확인.)
                    CROP = 252
                    rgb_uint8 = self._ee_rgb.astype(np.uint8)
                    Hf, Wf = self._ee_depth.shape[:2]

                    m_cup = (self._mask == target_val)
                    ys, xs = np.where(m_cup)
                    if len(xs) < 30:
                        raise RuntimeError(
                            f'유리 마스크 픽셀 부족 ({len(xs)}) — 252 크롭 불가')
                    ccx, ccy = float(xs.mean()), float(ys.mean())
                    x0 = int(np.clip(round(ccx - CROP // 2), 0, max(0, Wf - CROP)))
                    y0 = int(np.clip(round(ccy - CROP // 2), 0, max(0, Hf - CROP)))

                    # 컵 영역을 구멍(0=invalid)으로 만든 뒤 크롭 (see-through
                    # 값을 모델이 채우게). broken depth 는 미터, 정규화 금지.
                    depth_hole = self._ee_depth.copy()
                    depth_hole[m_cup] = 0.0
                    depth_crop = depth_hole[y0:y0 + CROP, x0:x0 + CROP]
                    rgb_crop = rgb_uint8[y0:y0 + CROP, x0:x0 + CROP]

                    # 크롭은 초점거리 불변, principal point 만 (x0,y0) 이동.
                    # use_pp=True 로 서버가 이 off-center pp 를 쓰게 한다.
                    K_crop = self._ee_K.astype(np.float64).copy()
                    K_crop[0, 2] -= x0
                    K_crop[1, 2] -= y0

                    rest_crop = self._swindrnet_client.restore(
                        rgb_crop, depth_crop, K_crop, use_pp=True)  # (CROP, CROP)
                    dt_swin = time.monotonic() - t_swin

                    # 복원 패치를 풀프레임 depth 의 크롭 위치에 되붙임 (컵 밖은
                    # raw depth 유지 — 배경은 원래 정상). 이후 extract_target_cloud
                    # 은 풀프레임 K 로 그대로 back-project.
                    restored_depth = self._ee_depth.copy()
                    restored_depth[y0:y0 + CROP, x0:x0 + CROP] = rest_crop
                    self.get_logger().debug(
                        f'[투명복원] 크롭 x0={x0} y0={y0} '
                        f'centroid=({ccx:.0f},{ccy:.0f}) pp→'
                        f'({K_crop[0,2]:.0f},{K_crop[1,2]:.0f}) use_pp=True')

                    # 복원 품질 평가: TARGET 마스크 안에서 모델이 raw 대비
                    # 얼마나 depth 를 당겼는가 (유리 표면은 테이블보다 카메라에
                    # 가까우므로 raw 보다 작아야 정상).
                    m_t       = (self._mask == target_val)
                    raw_med   = float(np.median(self._ee_depth[m_t])) if m_t.any() else float('nan')
                    rest_med  = float(np.median(restored_depth[m_t])) if m_t.any() else float('nan')
                    delta_mm  = (raw_med - rest_med) * 1000.0
                    self.get_logger().info(
                        f'[투명복원] SwinDRNet {dt_swin*1000:.0f}ms | '
                        f'TARGET median raw={raw_med:.4f}m → restored={rest_med:.4f}m '
                        f'(Δ={delta_mm:+.1f}mm, 양수여야 유리 표면 복원)')

                    self._save_swindrnet_debug(self._ee_depth, restored_depth, rgb_uint8)

                    pts_world = extract_target_cloud(
                        restored_depth, self._ee_K, self._mask, target_val,
                        self._R_ee, self._t_ee,
                        self._min_depth, self._max_depth, self._max_pts,
                    )
                    n_pts = len(pts_world) if pts_world is not None else 0
                    if n_pts == 0:
                        self.get_logger().warn('[투명복원] SwinDRNet 복원 후 TARGET 포인트 0')
                        pts_world = None
                except Exception as e:
                    self.get_logger().error(f'[투명복원] SwinDRNet 실패: {e}')
                    pts_world = None

            if pts_world is None:
                self.get_logger().warn(
                    '[투명복원] 복원 실패 — raw depth 로 진행 (유리컵은 테이블이 '
                    'see-through 로 비친 평면으로 나옴. 이건 복원이 아니다)')

        if pts_world is None:
            pts_world = extract_target_cloud(
                self._ee_depth, self._ee_K, self._mask, target_val,
                self._R_ee, self._t_ee,
                self._min_depth, self._max_depth, self._max_pts,
            )
        dt_extract = time.monotonic() - t_extract

        if pts_world is None or len(pts_world) < self._min_pts:
            n = len(pts_world) if pts_world is not None else 0
            self.get_logger().info(f'TARGET 포인트 부족 ({n}) — skip')
            return

        # Point cloud 타입 및 형태 확인
        pts_world = np.asarray(pts_world, dtype=np.float32)
        self.get_logger().debug(f'TARGET {len(pts_world)} pts → GraspGen (shape={pts_world.shape}, dtype={pts_world.dtype})')

        # [디버그] pts_world 좌표 범위 확인
        self.get_logger().debug(
            f'[디버그] pts_world 좌표범위: '
            f'X [{pts_world[:, 0].min():.6f}, {pts_world[:, 0].max():.6f}] '
            f'Y [{pts_world[:, 1].min():.6f}, {pts_world[:, 1].max():.6f}] '
            f'Z [{pts_world[:, 2].min():.6f}, {pts_world[:, 2].max():.6f}] '
            f'Z_mean={pts_world[:, 2].mean():.6f}')

        stamp = self.get_clock().now().to_msg()
        self._cloud_pub.publish(build_target_cloud_msg(pts_world, self._world_frame, stamp))

        t_pca = time.monotonic()
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
            self.get_logger().debug(
                f'TARGET PCA: short_axis=({short2d[0]:+.2f}, {short2d[1]:+.2f}) '
                f'분산비={ratio:.2f}')
        except (ValueError, np.linalg.LinAlgError) as e:
            self.get_logger().warn(f'PCA short-axis 계산 실패: {e}')

        # 3D PCA — 기울어진 평판(책)의 접근축을 세우는 데 쓴다. 위의 2D PCA 는
        # z 성분을 0 으로 박아 기울기를 통째로 버리므로 별도로 계산한다.
        # 축은 분산 내림차순: axes[:,0]=최대 … axes[:,2]=최소.
        # 책 표지처럼 평면인 클라우드에서는 최소축이 곧 표지 법선(=두께 방향)
        # 이고, 그 std 가 나머지보다 확연히 작다는 점으로 "평판인지"를 판정한다.
        self._target_axes = None
        self._target_axis_std = None
        try:
            P3 = np.asarray(pts_world, dtype=np.float64)
            Q3 = P3 - P3.mean(axis=0)
            ev3, evec3 = np.linalg.eigh(Q3.T @ Q3)
            order = np.argsort(ev3)[::-1]
            self._target_axes = evec3[:, order]
            self._target_axis_std = np.sqrt(
                np.maximum(ev3[order], 0.0) / max(len(P3), 1))
            self.get_logger().debug(
                'TARGET PCA3D std(mm)=' + ' '.join(
                    f'{s * 1000:.1f}' for s in self._target_axis_std))
        except (ValueError, np.linalg.LinAlgError) as e:
            self.get_logger().warn(f'PCA 3D 계산 실패: {e}')
        dt_pca = time.monotonic() - t_pca

        try:
            t_zmq = time.monotonic()
            grasps, confs = self._client.request(pts_world, self._num_grasps, self._topk)
            dt_zmq = time.monotonic() - t_zmq
            self.get_logger().debug(
                f'[PROFILE][graspgen][zmq] round_trip={dt_zmq:.3f}s '
                f'input_points={len(pts_world)} num_grasps={self._num_grasps} topk={self._topk}'
            )
        except (RuntimeError, ValueError) as e:
            self.get_logger().error(f'GraspGen: {e}')
            return

        if len(grasps) == 0:
            self.get_logger().warn('GraspGen: 결과 없음')
            return

        t_workspace = time.monotonic()
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
                    self.get_logger().debug(
                        f'Workspace filter: dropped {n_drop}/{len(grasps)} '
                        f'grasp(s) with z < {z_floor:.3f}m')
                grasps = grasps[keep]
                confs  = confs[keep]
            except (KeyError, IndexError, TypeError, ValueError) as e:
                self.get_logger().warn(f'workspace filter skipped: {e}')

        dt_workspace = time.monotonic() - t_workspace

        if len(grasps) == 0:
            self.get_logger().warn('GraspGen: 필터 후 후보 없음')
            return

        t_quality = time.monotonic()
        # Confidence threshold (paper §6.10 recommends ≥ 0.5).
        if self._min_quality > 0.0:
            keep   = confs >= self._min_quality
            n_drop = int((~keep).sum())
            if n_drop:
                self.get_logger().debug(
                    f'min_quality filter: dropped {n_drop}/{len(grasps)} '
                    f'grasp(s) with conf < {self._min_quality:.2f}')
            grasps = grasps[keep]
            confs  = confs[keep]

        dt_quality = time.monotonic() - t_quality

        if len(grasps) == 0:
            self.get_logger().warn(
                f'GraspGen: min_quality≥{self._min_quality:.2f} 통과 후보 없음 '
                f'(num_grasps batch={self._num_grasps} 증가 또는 threshold↓ 검토)')
            return

        t_build = time.monotonic()
        order  = np.argsort(confs)[::-1]
        grasps = grasps[order[:self._topk]]
        confs  = confs[order[:self._topk]]

        tf_stamped, output_frame = self._lookup_tf()
        candidates = self._build_candidates(grasps, confs, tf_stamped, output_frame)
        dt_build = time.monotonic() - t_build

        t_pose_adjust = time.monotonic()
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
                self.get_logger().debug(
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
                # 부호 정규화 — 반드시 필요하다.
                # 평행 그리퍼는 접근축(+Z) 기준 180° 회전에 대해 물리적으로
                # 동일한 파지다. 그런데 x_axis 는 PCA 고유벡터라 부호가 임의로
                # 정해진다. 뒤집힌 쪽이 걸리면 손목 yaw 만 π 다른 "같은 파지"를
                # 목표로 삼게 되고, MoveIt 은 그걸 j1/j3 null-space 자전으로
                # 풀어낸다: 책에서 실측한 궤적이 j1 +134° / j3 -146° / j7 -167°
                # (합계 7.8rad) 인데 EE 는 제자리였고, gripper yaw(≈j1+j3+j7)만
                # 정확히 -π 바뀌었다. 관절공간 이동이 너무 길어 local planner 가
                # 1.5Hz 로 다 소화하지 못하고 stuck abort 로 죽었다.
                # 회전 대칭 타깃(사과/공/컵)은 어느 부호든 무관해서 이 버그가
                # 책에서만 드러났다.
                if (x_axis[0] < 0.0) or (abs(x_axis[0]) < 1e-9 and x_axis[1] < 0.0):
                    x_axis = -x_axis
                approach = down.copy()
                # ── 기울기 정렬 (align_tilt 프로파일) ──────────────────────
                # 기본 경로는 접근축을 정확히 수직으로 고정하고 폐쇄축을 수평면에
                # 투영한다. 회전 대칭 물체엔 맞지만, 기울어 선 책에서는 패드가
                # 표지와 어긋나 모서리로만 닿는다. 실측(world_map_396.ply, 책
                # 6040점)에서 표지 법선이 수평에서 7.5°, 높이축이 수직에서 8.2°
                # 기울어 있었다.
                #   폐쇄축(+X) = 최소분산축 = 표지 법선(두께 방향)
                #   접근축(+Z) = 나머지 두 축 중 더 수직인 쪽을 아래로
                # 평판이 아닌 클라우드에서 엉뚱한 축을 잡지 않도록 두 가지를
                # 검사한다: (1) 최소축 std 가 중간축 std 보다 확실히 작을 것
                # (평판성), (2) 얻어진 접근축이 수직에서 max_tilt 이내일 것.
                # 하나라도 어긋나면 조용히 기존 top-down 으로 되돌아간다.
                tilt_applied = False
                if self._align_tilt and self._target_axes is not None:
                    axes = self._target_axes
                    std = self._target_axis_std
                    planar = std[2] < self._tilt_planar_ratio * std[1]
                    n_axis = axes[:, 2]                      # 최소분산 = 법선
                    # 접근축 후보: 최대/중간축 중 더 수직인 쪽, 아래로 향하게
                    cand = max((axes[:, 0], axes[:, 1]),
                               key=lambda v: abs(float(v[2])))
                    if cand[2] > 0.0:
                        cand = -cand
                    tilt_deg = float(np.degrees(
                        np.arccos(min(1.0, abs(float(cand[2]))))))
                    if planar and tilt_deg <= self._tilt_max_deg:
                        x_axis = n_axis - np.dot(n_axis, cand) * cand
                        x_axis /= (np.linalg.norm(x_axis) + 1e-12)
                        if (x_axis[0] < 0.0) or (abs(x_axis[0]) < 1e-9
                                                 and x_axis[1] < 0.0):
                            x_axis = -x_axis
                        approach = cand
                        tilt_applied = True
                        self.get_logger().info(
                            f'align_tilt: 접근축을 수직에서 {tilt_deg:.1f}° 기울임 '
                            f'| 법선 std={std[2] * 1000:.1f}mm vs 중간축 '
                            f'{std[1] * 1000:.1f}mm | approach=('
                            f'{approach[0]:+.3f},{approach[1]:+.3f},{approach[2]:+.3f})')
                    else:
                        self.get_logger().info(
                            f'align_tilt: 조건 미충족 → 수직 top-down 유지 '
                            f'(평판성={"O" if planar else "X"} '
                            f'std {std[2] * 1000:.1f}/{std[1] * 1000:.1f}mm, '
                            f'기울기 {tilt_deg:.1f}° > {self._tilt_max_deg:.0f}°)')
                y_axis = np.cross(approach, x_axis)
                y_axis /= (np.linalg.norm(y_axis) + 1e-12)
                x_axis = np.cross(y_axis, approach)          # 재직교화
                x_axis /= (np.linalg.norm(x_axis) + 1e-12)
                R_td   = np.column_stack([x_axis, y_axis, approach])
                # ★ panda_link8 프레임 보정 — 45° 를 빼먹으면 안 된다.
                # 위 행렬은 "폐쇄축 = 프레임의 +X" 를 가정하지만, Panda 는
                # 그렇지 않다. URDF 에서 panda_hand 는 panda_link8 에 Rz(-45°)
                # 로 붙고 손가락은 panda_hand 의 +Y 를 따라 미끄러진다. 따라서
                # 실제 폐쇄 방향은 link8 프레임의 Rz(-45°)·(0,1,0) =
                # (0.7071, 0.7071, 0) — +X 에서 45° 돌아간 대각선이다.
                # R_td·Rz(-45°) 로 후곱하면 그 대각선이 x_axis 를 향하게 되고
                # 접근축(+Z)은 그대로 유지된다.
                #
                # 이걸 빠뜨려 모든 파지가 yaw 로 45° 틀어져 있었다. 사과/공/컵은
                # 회전 대칭이라 아무 차이가 없어 여태 안 드러났고, 책에서 처음
                # 터졌다: 조가 두께 26.6mm 대신 26.6·cos45 + 171.5·sin45 =
                # 140mm 를 물어야 해서 63mm 에서 끼고, 자유물체인 책이 패드에
                # 맞춰 회전하며 대롱대롱 매달렸다.
                # Isaac 실측으로 확인: 명령 +X 방위 -43.0° vs 실제 조 방향
                # (panda_hand +Y) -89.5° → 46.5°, 그리고 명령 프레임의
                # (0.707,0.707,0) 을 월드로 옮기면 실측과 1.2° 일치.
                c45 = np.sqrt(0.5)
                R_link8_fix = np.array([[c45,  c45, 0.0],
                                        [-c45, c45, 0.0],
                                        [0.0,  0.0, 1.0]])
                R_td = R_td @ R_link8_fix
                quat_td = Rot.from_matrix(R_td).as_quat().tolist()
                # 결정론적 Z: fingertip 을 "복원된 pts_world 의 실제 높이"
                # 기준으로 둔다. projector 의 centroid[2] 는 투명물체에서
                # see-through(테이블 높이 ≈0) 라 못 쓴다 — 그걸 쓰면 손끝이
                # 테이블로 내려가거나(=0) GraspGen scatter z 로 컵 위에서 닫힌다.
                # z_frac 로 몸통 어디를 잡을지 튜닝. wrist(panda_link8) 는
                # 손끝에서 접근축 반대로 0.103m 뒤 — 수직 접근이면 정확히 +Z
                # 0.103m 이고, align_tilt 로 기울면 XY 도 같이 밀린다(8° 기울기
                # 에서 약 14mm). 이걸 +Z 로만 두면 기울인 만큼 손끝이 목표에서
                # 빗나간다.
                pw    = np.asarray(pts_world, dtype=np.float64)
                zc    = pw[:, 2]
                z_bot = float(np.percentile(zc, 10))
                z_top = float(np.percentile(zc, 90))
                fingertip_z = (z_bot + self._ftd_grasp_z_frac * (z_top - z_bot)
                               + self._ftd_grasp_z_off)
                # XY 도 복원 cloud 중심(robust median)으로 센터링. GraspGen 원본
                # XY 는 컵에서 최대 ~8cm 벗어나 있어 수직 grasp 이 옆으로 빗나감.
                # projector bbox 대신 복원 cloud 를 쓰는 이유는 Z 앵커와 동일.
                # 'extent' 는 실루엣 폭의 중점이라 점 밀도 편향에 안 끌린다
                # (grasp_xy_anchor 선언부 주석 참조).
                if self._xy_anchor == 'extent':
                    xlo, xhi = np.percentile(pw[:, 0], (2.0, 98.0))
                    ylo, yhi = np.percentile(pw[:, 1], (2.0, 98.0))
                    cx_r = float((xlo + xhi) * 0.5)
                    cy_r = float((ylo + yhi) * 0.5)
                else:
                    cx_r = float(np.median(pw[:, 0]))
                    cy_r = float(np.median(pw[:, 1]))
                fingertip = np.array([cx_r, cy_r, fingertip_z])
                wrist     = fingertip - 0.103 * approach
                for c in candidates:
                    c['quaternion']  = quat_td
                    c['position'][0] = float(wrist[0])
                    c['position'][1] = float(wrist[1])
                    c['position'][2] = float(wrist[2])
                # 두 앵커를 항상 같이 찍는다 — 한 번 돌리면 편향(Δ)이 바로
                # 측정된다. 여유가 한쪽 5.6mm 라 이 값이 성패를 가른다.
                mx = float(np.median(pw[:, 0]))
                my = float(np.median(pw[:, 1]))
                ex = float(np.percentile(pw[:, 0], 2.0)
                           + np.percentile(pw[:, 0], 98.0)) * 0.5
                ey = float(np.percentile(pw[:, 1], 2.0)
                           + np.percentile(pw[:, 1], 98.0)) * 0.5
                self.get_logger().info(
                    f'force_top_down: {"기울인" if tilt_applied else "수직"} grasp | '
                    f'cloud z=[{z_bot:.3f},{z_top:.3f}] '
                    f'frac={self._ftd_grasp_z_frac:.2f} off={self._ftd_grasp_z_off:+.3f} '
                    f'→ fingertip=({cx_r:.3f},{cy_r:.3f},{fingertip_z:.3f}) '
                    f'wrist=({wrist[0]:.3f},{wrist[1]:.3f},{wrist[2]:.3f}) '
                    f'(+X=({x_axis[0]:+.2f},{x_axis[1]:+.2f},{x_axis[2]:+.2f})) '
                    f'× {len(candidates)}')
                self.get_logger().info(
                    f'  xy_anchor={self._xy_anchor} | '
                    f'median=({mx:.3f},{my:.3f}) extent=({ex:.3f},{ey:.3f}) '
                    f'Δ=({(ex-mx)*1000:+.1f},{(ey-my)*1000:+.1f})mm | '
                    f'cloud n={len(pw)} '
                    f'xy_span=({(np.percentile(pw[:,0],98)-np.percentile(pw[:,0],2))*1000:.1f},'
                    f'{(np.percentile(pw[:,1],98)-np.percentile(pw[:,1],2))*1000:.1f})mm')
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
                self.get_logger().debug(
                    f'Yaw align: short_axis=world-{axis_name} '
                    f'(dx={dx:.3f}, dy={dy:.3f}) applied to {n_done}/{len(candidates)}')
            except (KeyError, IndexError, TypeError, ValueError) as e:
                self.get_logger().warn(f'Yaw align skipped: {e}')

        dt_pose_adjust = time.monotonic() - t_pose_adjust

        # ── Client-side filters (Option A — shrink the goal set for the
        #    sequential motion planner). Order: cheap → expensive.
        n_after_build = len(candidates)
        dt_top_down = 0.0
        dt_ik = 0.0

        if self._td_enabled:
            t_top_down = time.monotonic()
            candidates_before = len(candidates)
            candidates = top_down_filter(
                candidates, self._td_angle_deg, logger=self.get_logger())
            dt_top_down = time.monotonic() - t_top_down
            self.get_logger().debug(
                f'top_down(≤{self._td_angle_deg:.0f}°): '
                f'{candidates_before} → {len(candidates)}')
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
            dt_ik = ik_dt
            if ik_stats['service_down']:
                self.get_logger().warn(
                    f'IK service unavailable — pass-through ({n_before_ik} kept)')
            else:
                self.get_logger().debug(
                    f'IK feasibility: {ik_stats["kept"]}/{ik_stats["checked"]} '
                    f'reachable  elapsed={ik_dt:.2f}s')
            if not candidates:
                self.get_logger().warn(
                    'IK filter dropped all candidates — robot cannot reach any '
                    'top-down approach for this target')
                return

        t_final_cap = time.monotonic()
        # Final confidence cap. Already confidence-sorted upstream, but
        # filters may have removed leading entries — resort then truncate.
        candidates = confidence_top_n(candidates, self._max_published)
        dt_final_cap = time.monotonic() - t_final_cap

        t_publish = time.monotonic()
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
        dt_publish = time.monotonic() - t_publish

        best_q = candidates[0]['quality'] if candidates else 0.0
        total_dt = time.monotonic() - t0
        self.get_logger().debug(
            f'[PROFILE][graspgen] total={total_dt:.3f}s extract_cloud={dt_extract:.3f}s '
            f'pca={dt_pca:.3f}s zmq={dt_zmq:.3f}s workspace={dt_workspace:.3f}s '
            f'quality={dt_quality:.3f}s build_candidates={dt_build:.3f}s '
            f'pose_adjust={dt_pose_adjust:.3f}s top_down={dt_top_down:.3f}s '
            f'ik={dt_ik:.3f}s final_cap={dt_final_cap:.3f}s publish={dt_publish:.3f}s '
            f'published={len(candidates)} best={best_q:.3f}'
        )
        self.get_logger().info(
            f'Published {len(candidates)} grasp(s)  '
            f'best={best_q:.3f}  elapsed={total_dt:.2f}s')

        # 첫 성공 publish 후 freeze — 이후 world_map_result 입력 무시.
        if self._freeze_after_first and not self._frozen and candidates:
            self._frozen = True
            self.get_logger().info('FROZEN — 이후 /world_map_result 입력 무시. latched /grasp_candidates 그대로.')

    # ── helpers ───────────────────────────────────────────────────────────────

    def _save_swindrnet_debug(self, raw: np.ndarray, restored: np.ndarray,
                              rgb: np.ndarray) -> None:
        """Dump raw/restored depth + RGB to /tmp for visual inspection."""
        import cv2

        def _colorize(d):
            v = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
            v = (v / (v.max() + 1e-6) * 255).astype(np.uint8)
            return cv2.applyColorMap(v, cv2.COLORMAP_TURBO)

        try:
            cv2.imwrite('/tmp/swindrnet_00_raw_depth.png', _colorize(raw))
            cv2.imwrite('/tmp/swindrnet_01_restored_depth.png', _colorize(restored))
            cv2.imwrite('/tmp/swindrnet_02_mask.png', (self._mask * 255).astype(np.uint8))
            cv2.imwrite('/tmp/swindrnet_03_rgb.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        except Exception as e:
            self.get_logger().warn(f'디버그 이미지 저장 실패: {e}')

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
