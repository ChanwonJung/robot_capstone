"""
vgn_grasp_node.py

Subscribes:
  /ee_camera/depth_image   (Image)      — cached; EE depth for TSDF integration
  /ee_camera/camera_info   (CameraInfo) — cached; EE intrinsics K
  /top_camera/depth_image  (Image)      — cached; Top depth (optional, use_top_depth)
  /top_camera/camera_info  (CameraInfo) — cached; Top intrinsics K (optional)
  /world_map_result        (String JSON) — trigger; target centroid + bbox_3d + point_count

Processing (on each /world_map_result):
  1. Parse target centroid, bbox_3d_world, point_count  → skip if < min_point_count
  2. Define 30 cm ROI cube centred on centroid
  3. Ray-cast signed TSDF from EE depth (+ Top depth) → 40×40×40 float32 grid
  4. VGN inference → NMS → grasp candidates
  5. Semantic filter: keep grasps whose centre falls inside target bbox_3d_world
  6. TF world → panda_link0
  7. Publish /grasp_candidates (JSON String)
  8. Publish /grasp_markers (MarkerArray) — RViz visualization

/world_map is NOT subscribed here — it is used only as RViz2 debug visualization.
Semantic filtering uses bbox_3d_world from /world_map_result (no /world_map dependency).

Extrinsics convention: p_world = R @ p_cam + t  (same as camera_extrinsics.yaml)
K matrix comes from /camera_info topics at runtime (not stored in YAML).

External dependency: external/vgn  (ethz-asl/vgn git submodule)
  git submodule add https://github.com/ethz-asl/vgn external/vgn
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, ColorRGBA
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros

# ── VGN library path (external/vgn submodule) ────────────────────────────────
_WS = Path(os.environ.get('GSAM_WS', str(Path.home() / 'gsam_ws')))
_VGN_SRC = _WS / 'external' / 'vgn' / 'src'
if _VGN_SRC.exists() and str(_VGN_SRC) not in sys.path:
    sys.path.insert(0, str(_VGN_SRC))

try:
    import torch
    from scipy import ndimage as _ndimage
    from vgn.networks import load_network
    from vgn.utils.transform import Transform as _VgnTransform, Rotation as _VgnRotation

    class _Grasp:
        def __init__(self, pose, width):
            self.pose = pose
            self.width = width

    def _vgn_predict(tsdf_vol, net, device):
        assert tsdf_vol.shape == (1, 40, 40, 40)
        x = torch.from_numpy(tsdf_vol).unsqueeze(0).to(device)
        with torch.no_grad():
            qual_vol, rot_vol, width_vol = net(x)
        return (qual_vol.cpu().squeeze().numpy(),
                rot_vol.cpu().squeeze().numpy(),
                width_vol.cpu().squeeze().numpy())

    def _vgn_process(tsdf_vol, qual_vol, rot_vol, width_vol,
                     gaussian_filter_sigma=1.0, min_width=1.33, max_width=9.33):
        tsdf_vol  = tsdf_vol.squeeze()
        qual_vol  = _ndimage.gaussian_filter(qual_vol, sigma=gaussian_filter_sigma,
                                             mode='nearest')
        outside   = tsdf_vol > 0.5
        inside    = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < 0.5)
        valid     = _ndimage.morphology.binary_dilation(
                        outside, iterations=2, mask=np.logical_not(inside))
        qual_vol[valid == False] = 0.0  # noqa: E712
        qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0
        return qual_vol, rot_vol, width_vol

    def _vgn_select(qual_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4):
        qual_vol = qual_vol.copy()
        qual_vol[qual_vol < threshold] = 0.0
        max_vol  = _ndimage.maximum_filter(qual_vol, size=max_filter_size)
        qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
        grasps, scores = [], []
        for idx in np.argwhere(np.where(qual_vol, 1.0, 0.0)):
            i, j, k = idx
            score = qual_vol[i, j, k]
            ori   = _VgnRotation.from_quat(rot_vol[:, i, j, k])
            pos   = np.array([i, j, k], dtype=np.float64)
            grasps.append(_Grasp(_VgnTransform(ori, pos), width_vol[i, j, k]))
            scores.append(score)
        return grasps, scores

    def from_voxel_coordinates(grasp, voxel_size):
        grasp.pose.translation = grasp.pose.translation * voxel_size
        grasp.width = grasp.width * voxel_size
        return grasp

    _VGN_OK = True
except ImportError as _e:
    _VGN_OK = False
    _VGN_IMPORT_ERROR = str(_e)


# ── depth image decoding ──────────────────────────────────────────────────────

def _decode_depth(msg: Image) -> np.ndarray:
    """Decode ROS Image to (H, W) float32 depth in metres."""
    if msg.encoding == '32FC1':
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
    if msg.encoding == '16UC1':
        raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        return raw.astype(np.float32) * 0.001
    raise ValueError(f'Unsupported depth encoding: {msg.encoding}')


def _extract_K(msg: CameraInfo) -> np.ndarray:
    """Extract 3×3 intrinsics matrix from CameraInfo."""
    return np.array(msg.k, dtype=np.float64).reshape(3, 3)


# ── extrinsics loading ────────────────────────────────────────────────────────

def _load_extrinsics(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load R_ee, t_ee, R_top, t_top from YAML.

    Convention: p_world = R @ p_cam + t
    Returns (R_ee, t_ee, R_top, t_top) as float64 arrays.
    Raises on missing keys or wrong shapes.
    """
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    R_ee  = np.array(cfg['ee_camera']['R'],  dtype=np.float64)
    t_ee  = np.array(cfg['ee_camera']['t'],  dtype=np.float64)
    R_top = np.array(cfg['top_camera']['R'], dtype=np.float64)
    t_top = np.array(cfg['top_camera']['t'], dtype=np.float64)
    assert R_ee.shape == (3, 3) and t_ee.shape == (3,)
    assert R_top.shape == (3, 3) and t_top.shape == (3,)
    return R_ee, t_ee, R_top, t_top


# ── signed TSDF via ray-casting ───────────────────────────────────────────────

def _project_voxels(
    p_world: np.ndarray,   # (N, 3) voxel centres in world frame
    depth:   np.ndarray,   # (H, W) float32 metres
    K:       np.ndarray,   # (3, 3) camera intrinsics
    R:       np.ndarray,   # (3, 3) cam→world rotation  (p_world = R @ p_cam + t)
    t:       np.ndarray,   # (3,)   camera origin in world frame
    trunc:   float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project voxel centres onto depth image and compute signed TSDF values.

    SDF sign: positive = voxel in front of observed surface (free space),
              negative = voxel behind observed surface (inside object).

    Normalisation: tsdf = clip(sdf / trunc, -1, 1) * 0.5 + 0.5
      → 1.0 = far outside,  0.5 = on surface,  0.0 = deep inside

    VGN _vgn_process() uses tsdf > 0.5 as outside and 1e-3 < tsdf < 0.5 as
    inside, so this [0,1] mapping is required.

    Returns:
      tsdf_vals (N,) float32 in [0, 1] — unobserved voxels default to 1.0
      valid     (N,) bool  — True for voxels with a valid depth reading
    """
    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # world → camera:  p_cam = R^T @ (p_world - t)
    p_cam = (R.T @ (p_world - t).T).T          # (N, 3)
    pz    = p_cam[:, 2]

    in_front = pz > 0.01

    # avoid divide-by-zero for behind-camera voxels
    safe_pz = np.where(in_front, pz, 1.0)
    u = fx * p_cam[:, 0] / safe_pz + cx
    v = fy * p_cam[:, 1] / safe_pz + cy

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    in_image = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    proj_ok  = in_front & in_image

    # default: 1.0 (unobserved = outside)
    tsdf_vals = np.ones(len(p_world), dtype=np.float32)
    valid     = np.zeros(len(p_world), dtype=bool)

    if proj_ok.any():
        idx   = np.where(proj_ok)[0]
        d_obs = depth[vi[idx], ui[idx]].astype(np.float64)

        depth_ok   = (d_obs > 0.01) & np.isfinite(d_obs)
        idx_valid  = idx[depth_ok]

        sdf = d_obs[depth_ok] - pz[idx_valid]
        tsdf_vals[idx_valid] = (np.clip(sdf / trunc, -1.0, 1.0) * 0.5 + 0.5).astype(np.float32)
        valid[idx_valid] = True

    return tsdf_vals, valid


def _build_tsdf_raycasting(
    roi_min:   np.ndarray,           # (3,) world-frame ROI origin
    roi_size_m: float,
    reso:      int,
    ee_depth:  np.ndarray,           # (H, W) float32
    ee_K:      np.ndarray,           # (3, 3)
    R_ee:      np.ndarray,           # (3, 3)
    t_ee:      np.ndarray,           # (3,)
    top_depth: Optional[np.ndarray] = None,
    top_K:     Optional[np.ndarray] = None,
    R_top:     Optional[np.ndarray] = None,
    t_top:     Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build signed TSDF grid (1, reso, reso, reso) float32 via depth ray-casting.

    Each voxel accumulates signed TSDF values from all cameras with valid
    depth readings.  Unobserved voxels default to 1.0 (outside/free).
    """
    voxel_size = roi_size_m / reso
    trunc      = 4.0 * voxel_size

    # voxel centres in world frame  (N=reso^3, 3)
    idx_arr = np.arange(reso, dtype=np.float32)
    gi, gj, gk = np.meshgrid(idx_arr, idx_arr, idx_arr, indexing='ij')
    p_world = ((np.stack([gi, gj, gk], axis=-1) + 0.5) * voxel_size
               + roi_min).reshape(-1, 3)                                # (N, 3)

    tsdf_sum = np.zeros(len(p_world), dtype=np.float64)
    weight   = np.zeros(len(p_world), dtype=np.float64)

    def _integrate(depth, K, R, t):
        vals, valid = _project_voxels(p_world, depth, K, R, t, trunc)
        tsdf_sum[valid] += vals[valid]
        weight[valid]   += 1.0

    _integrate(ee_depth, ee_K, R_ee, t_ee)
    if top_depth is not None and top_K is not None and R_top is not None and t_top is not None:
        _integrate(top_depth, top_K, R_top, t_top)

    # weighted average; unobserved → 1.0 (outside/free)
    result = np.ones(len(p_world), dtype=np.float32)
    obs = weight > 0
    result[obs] = (tsdf_sum[obs] / weight[obs]).astype(np.float32)

    return result.reshape(1, reso, reso, reso)


# ── node ──────────────────────────────────────────────────────────────────────

class VgnGraspNode(Node):

    def __init__(self) -> None:
        super().__init__('vgn_grasp_node')

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('roi_size_m',             0.30)
        self.declare_parameter('tsdf_resolution',        40)
        self.declare_parameter('vgn_model_path',         'models/vgn_conv.pth')
        self.declare_parameter('min_quality',            0.5)
        self.declare_parameter('max_grasp_candidates',   5)
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

        self._roi_size_m   = self.get_parameter('roi_size_m').value
        self._reso         = self.get_parameter('tsdf_resolution').value
        self._min_quality  = self.get_parameter('min_quality').value
        self._max_k        = self.get_parameter('max_grasp_candidates').value
        self._min_pts      = self.get_parameter('min_point_count').value
        self._use_top      = self.get_parameter('use_top_depth').value
        self._world_frame  = self.get_parameter('world_frame').value
        self._robot_frame  = self.get_parameter('robot_frame').value

        # ── Extrinsics ────────────────────────────────────────────────────────
        ext_param = self.get_parameter('extrinsics_config').value
        if not ext_param:
            ext_param = str(_WS / 'src/mask_projection_pkg/config/camera_extrinsics.yaml')
        try:
            self._R_ee, self._t_ee, self._R_top, self._t_top = _load_extrinsics(ext_param)
            self.get_logger().info(f'Extrinsics loaded: {ext_param}')
        except Exception as e:
            self.get_logger().error(f'Failed to load extrinsics ({ext_param}): {e}')
            raise

        # ── VGN network ───────────────────────────────────────────────────────
        if not _VGN_OK:
            self.get_logger().error(
                f'VGN import failed: {_VGN_IMPORT_ERROR}\n'
                f'Run: git submodule add https://github.com/ethz-asl/vgn external/vgn\n'
                f'Then: source launch_env.bash'
            )
            raise RuntimeError('VGN library not available')

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
        self._ee_depth:  Optional[np.ndarray] = None   # (H, W) float32
        self._ee_K:      Optional[np.ndarray] = None   # (3, 3)
        self._top_depth: Optional[np.ndarray] = None   # (H, W) float32
        self._top_K:     Optional[np.ndarray] = None   # (3, 3)
        self._pending_result: Optional[str]   = None   # raw JSON string

        # ── Subscriptions ─────────────────────────────────────────────────────
        ee_depth_topic  = self.get_parameter('ee_depth_topic').value
        ee_info_topic   = self.get_parameter('ee_camera_info_topic').value
        top_depth_topic = self.get_parameter('top_depth_topic').value
        top_info_topic  = self.get_parameter('top_camera_info_topic').value
        result_topic    = self.get_parameter('world_map_result_topic').value

        self.create_subscription(Image,      ee_depth_topic,  self._ee_depth_cb,  10)
        self.create_subscription(CameraInfo, ee_info_topic,   self._ee_info_cb,   10)
        if self._use_top:
            self.create_subscription(Image,      top_depth_topic, self._top_depth_cb, 10)
            self.create_subscription(CameraInfo, top_info_topic,  self._top_info_cb,  10)
        self.create_subscription(String, result_topic, self._result_cb, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        grasp_topic      = self.get_parameter('grasp_candidates_topic').value
        self._grasp_pub  = self.create_publisher(String,      grasp_topic,      10)
        self._marker_pub = self.create_publisher(MarkerArray, '/grasp_markers',  10)

        self.get_logger().info(
            f'vgn_grasp_node ready  '
            f'roi={self._roi_size_m}m  reso={self._reso}  '
            f'min_quality={self._min_quality}  use_top={self._use_top}'
        )

    # ── Depth / CameraInfo callbacks ──────────────────────────────────────────

    def _ee_depth_cb(self, msg: Image) -> None:
        try:
            self._ee_depth = _decode_depth(msg)
        except ValueError as e:
            self.get_logger().warn(f'EE depth decode error: {e}')
            return
        self._try_flush_pending()

    def _ee_info_cb(self, msg: CameraInfo) -> None:
        self._ee_K = _extract_K(msg)
        self._try_flush_pending()

    def _top_depth_cb(self, msg: Image) -> None:
        try:
            self._top_depth = _decode_depth(msg)
        except ValueError as e:
            self.get_logger().warn(f'Top depth decode error: {e}')

    def _top_info_cb(self, msg: CameraInfo) -> None:
        self._top_K = _extract_K(msg)

    def _try_flush_pending(self) -> None:
        if self._pending_result is None:
            return
        if self._ee_depth is None or self._ee_K is None:
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

        # ── 1. Parse trigger ──────────────────────────────────────────────────
        try:
            result      = json.loads(msg.data)
            target_info = result.get('target', {})
            centroid    = np.array(target_info['centroid'],      dtype=np.float32)
            point_count = int(target_info['point_count'])
            bbox_raw    = target_info.get('bbox_3d_world', None)
        except (KeyError, json.JSONDecodeError, ValueError) as e:
            self.get_logger().warn(f'world_map_result parse error: {e}')
            return

        if point_count < self._min_pts:
            self.get_logger().info(
                f'TARGET point_count={point_count} < min={self._min_pts} — skip'
            )
            return

        t0 = time.monotonic()

        # ── 2. Define ROI ─────────────────────────────────────────────────────
        half    = self._roi_size_m * 0.5
        roi_min = (centroid - half).astype(np.float64)

        # ── 3. Build signed TSDF via ray-casting ──────────────────────────────
        top_args: dict = {}
        if self._use_top and self._top_depth is not None and self._top_K is not None:
            top_args = dict(
                top_depth=self._top_depth,
                top_K=self._top_K,
                R_top=self._R_top,
                t_top=self._t_top,
            )

        grid = _build_tsdf_raycasting(
            roi_min, self._roi_size_m, self._reso,
            self._ee_depth, self._ee_K, self._R_ee, self._t_ee,
            **top_args,
        )

        # ── 4. VGN inference ──────────────────────────────────────────────────
        qual_vol, rot_vol, width_vol = _vgn_predict(grid, self._net, self._device)
        qual_vol, rot_vol, width_vol = _vgn_process(grid, qual_vol, rot_vol, width_vol)

        # ── 5. NMS + voxel → ROI frame ────────────────────────────────────────
        voxel_size   = self._roi_size_m / self._reso
        grasps_voxel, scores = _vgn_select(
            qual_vol, rot_vol, width_vol, threshold=self._min_quality,
        )

        if not grasps_voxel:
            self.get_logger().info('No grasp candidates after NMS')
            return

        grasps = [from_voxel_coordinates(g, voxel_size) for g in grasps_voxel]

        # ── 6. Semantic filter: grasp centre inside target bbox_3d_world ──────
        if bbox_raw is not None:
            bbox_min = np.array(bbox_raw['min'], dtype=np.float32)
            bbox_max = np.array(bbox_raw['max'], dtype=np.float32)
            kept_grasps, kept_scores = [], []
            for grasp, score in zip(grasps, scores):
                p_world = (roi_min + grasp.pose.translation).astype(np.float32)
                if np.all(p_world >= bbox_min) and np.all(p_world <= bbox_max):
                    kept_grasps.append(grasp)
                    kept_scores.append(float(score))
            if not kept_grasps:
                self.get_logger().info('All candidates rejected by bbox semantic filter')
                return
        else:
            self.get_logger().warn('No bbox_3d_world in world_map_result — skipping semantic filter')
            kept_grasps = list(grasps)
            kept_scores = [float(s) for s in scores]

        # ── 7. Sort, Top-K ────────────────────────────────────────────────────
        order      = np.argsort(kept_scores)[::-1]
        top_grasps = [kept_grasps[i] for i in order[:self._max_k]]
        top_scores = [kept_scores[i] for i in order[:self._max_k]]

        # ── 8. TF world → panda_link0 ─────────────────────────────────────────
        try:
            tf_stamped = self._tf_buffer.lookup_transform(
                self._robot_frame, self._world_frame,
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1),
            )
            tf_ok = True
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed ({e}) — publishing in world frame')
            tf_ok = False

        output_frame = self._robot_frame if tf_ok else self._world_frame

        candidates = []
        for grasp, score in zip(top_grasps, top_scores):
            p_world = (roi_min + grasp.pose.translation).astype(np.float32)
            quat    = grasp.pose.rotation.as_quat()  # [qx, qy, qz, qw]
            if tf_ok:
                p_world, quat = _apply_tf(tf_stamped, p_world, quat)
            candidates.append({
                'position':   p_world.tolist(),
                'quaternion': quat.tolist(),
                'width':      float(grasp.width),
                'quality':    score,
                'frame':      output_frame,
            })

        payload = {
            'candidates':      candidates,
            'target_centroid': centroid.tolist(),
            'stamp':           self.get_clock().now().nanoseconds * 1e-9,
        }
        out_msg = String()
        out_msg.data = json.dumps(payload)
        self._grasp_pub.publish(out_msg)
        self._publish_markers(candidates, output_frame)

        elapsed = time.monotonic() - t0
        self.get_logger().info(
            f'Published {len(candidates)} grasp(s)  '
            f'best_quality={top_scores[0]:.3f}  elapsed={elapsed:.2f}s'
        )

    # ── RViz markers ─────────────────────────────────────────────────────────

    def _publish_markers(self, candidates: list, frame: str) -> None:
        """
        Publish MarkerArray for RViz2 visualization.

        Per candidate:
          - ARROW  (id=i*3+1): approach direction, color = quality (blue tint)
          - SPHERE (id=i*3+2): grasp position
          - CYLINDER (id=i*3+3): gripper width indicator
        DELETEALL is sent in a separate prior message.
        """
        from scipy.spatial.transform import Rotation as R

        now = self.get_clock().now().to_msg()

        clear_ma = MarkerArray()
        clear_m  = Marker()
        clear_m.header.frame_id = frame
        clear_m.header.stamp    = now
        clear_m.ns              = 'vgn_grasps'
        clear_m.action          = Marker.DELETEALL
        clear_ma.markers        = [clear_m]
        self._marker_pub.publish(clear_ma)

        markers: list[Marker] = []
        for i, c in enumerate(candidates):
            q     = float(c['quality'])
            color = ColorRGBA(r=0.0, g=0.6 * q, b=1.0, a=1.0)
            color_ring = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)

            pos  = c['position']
            quat = c['quaternion']
            rot  = R.from_quat(quat)

            approach  = rot.apply([0.0, 0.0, -1.0])
            arrow_len = 0.20

            arrow = Marker()
            arrow.header.frame_id = frame
            arrow.header.stamp    = now
            arrow.ns              = 'vgn_grasps'
            arrow.id              = i * 3 + 1
            arrow.type            = Marker.ARROW
            arrow.action          = Marker.ADD
            tail = Point(x=pos[0] - approach[0] * arrow_len,
                         y=pos[1] - approach[1] * arrow_len,
                         z=pos[2] - approach[2] * arrow_len)
            head = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            arrow.points = [tail, head]
            arrow.scale  = Vector3(x=0.02, y=0.04, z=0.0)
            arrow.color  = color
            markers.append(arrow)

            sphere = Marker()
            sphere.header.frame_id = frame
            sphere.header.stamp    = now
            sphere.ns              = 'vgn_grasps'
            sphere.id              = i * 3 + 2
            sphere.type            = Marker.SPHERE
            sphere.action          = Marker.ADD
            sphere.pose.position   = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            sphere.scale           = Vector3(x=0.03, y=0.03, z=0.03)
            sphere.color           = color
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
                axis   = cross / cross_norm
                angle  = float(np.arccos(np.clip(np.dot(z_axis, gripper_x), -1, 1)))
                from scipy.spatial.transform import Rotation as _R2
                q_ring = _R2.from_rotvec(axis * angle).as_quat()
            else:
                q_ring = np.array([0.0, 0.0, 0.0, 1.0])
            ring.pose.orientation.x = float(q_ring[0])
            ring.pose.orientation.y = float(q_ring[1])
            ring.pose.orientation.z = float(q_ring[2])
            ring.pose.orientation.w = float(q_ring[3])
            ring.scale = Vector3(x=width, y=width, z=0.005)
            ring.color = color_ring
            markers.append(ring)

        ma = MarkerArray()
        ma.markers = markers
        self._marker_pub.publish(ma)


# ── TF helper ─────────────────────────────────────────────────────────────────

def _apply_tf(
    tf_stamped,
    p_world:    np.ndarray,
    quat_world: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation as R
    t        = tf_stamped.transform.translation
    r        = tf_stamped.transform.rotation
    tf_rot   = R.from_quat([r.x, r.y, r.z, r.w])
    tf_trans = np.array([t.x, t.y, t.z], dtype=np.float64)
    p_robot  = tf_rot.apply(p_world.astype(np.float64)) + tf_trans
    q_robot  = tf_rot * R.from_quat(quat_world)
    return p_robot.astype(np.float32), q_robot.as_quat().astype(np.float32)


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
