"""
depth_utils.py — depth decode, intrinsics, extrinsics, backprojection.

Intentionally self-contained: no cross-package imports so graspgen_pkg
remains independent of vgn_grasp_pkg even though the APIs are similar.
"""
from __future__ import annotations

import numpy as np
from sensor_msgs.msg import CameraInfo, Image


def decode_depth(msg: Image) -> np.ndarray:
    """32FC1 or 16UC1 → (H, W) float32 metres."""
    if msg.encoding == '32FC1':
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
    if msg.encoding == '16UC1':
        raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        return raw.astype(np.float32) * 0.001
    raise ValueError(f'Unsupported depth encoding: {msg.encoding}')


def decode_mask(msg: Image) -> np.ndarray:
    """mono8 mask → (H, W) uint8; pixel = 1-based detection index (0 = FREE)."""
    return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width).copy()


def extract_K(msg: CameraInfo) -> np.ndarray:
    """CameraInfo → (3, 3) float64 intrinsic matrix."""
    return np.array(msg.k, dtype=np.float64).reshape(3, 3)


def load_ee_extrinsics(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load ee_camera R (3,3) and t (3,) from camera_extrinsics.yaml.

    Convention: p_world = R @ p_cam + t
    """
    import yaml
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    R = np.array(cfg['ee_camera']['R'], dtype=np.float64)
    t = np.array(cfg['ee_camera']['t'], dtype=np.float64)
    assert R.shape == (3, 3) and t.shape == (3,)
    return R, t


def depth_to_points(
    depth: np.ndarray,
    K: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Backproject depth image to camera-frame points.

    Returns
    -------
    pts : (N, 3) float32 — 3D points in camera optical frame (+Z forward)
    pix : (N, 2) int32   — corresponding (row, col) pixel coordinates
    """
    H, W   = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    Z      = depth.astype(np.float32)
    valid  = np.isfinite(Z) & (Z > min_depth) & (Z < max_depth)
    Z_v, u_v, v_v = Z[valid], uu[valid], vv[valid]
    pts = np.stack([(u_v - cx) * Z_v / fx,
                    (v_v - cy) * Z_v / fy,
                    Z_v], axis=1)
    pix = np.stack([v_v.astype(np.int32), u_v.astype(np.int32)], axis=1)
    return pts, pix


def apply_world_to_robot_tf(
    tf_stamped,
    pos: np.ndarray,
    quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform (pos, quat) from world frame to robot frame via a TF2 stamped transform."""
    from scipy.spatial.transform import Rotation as Rot
    tr  = tf_stamped.transform.translation
    ro  = tf_stamped.transform.rotation
    rot = Rot.from_quat([ro.x, ro.y, ro.z, ro.w])
    t   = np.array([tr.x, tr.y, tr.z], dtype=np.float64)
    p_r = rot.apply(pos.astype(np.float64)) + t
    q_r = rot * Rot.from_quat(quat)
    return p_r.astype(np.float32), q_r.as_quat().astype(np.float32)
