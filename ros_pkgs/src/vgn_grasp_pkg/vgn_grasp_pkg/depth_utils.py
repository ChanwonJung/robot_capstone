"""depth_utils.py — Depth image decoding and camera extrinsics loading."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import yaml


def decode_depth(msg) -> np.ndarray:
    """Decode ROS Image to (H, W) float32 depth in metres."""
    if msg.encoding == '32FC1':
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
    if msg.encoding == '16UC1':
        raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        return raw.astype(np.float32) * 0.001
    raise ValueError(f'Unsupported depth encoding: {msg.encoding}')


def extract_K(msg) -> np.ndarray:
    """Extract 3×3 intrinsics matrix from CameraInfo."""
    return np.array(msg.k, dtype=np.float64).reshape(3, 3)


def load_extrinsics(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load R_ee, t_ee, R_top, t_top from YAML.

    Convention: p_world = R @ p_cam + t
    Returns (R_ee, t_ee, R_top, t_top) as float64 arrays.
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
