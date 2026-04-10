"""
back_projection.py

Depth image → 3D points in camera frame.
Pure numpy, no ROS dependency — easy to unit-test or swap out.

Math
----
  X = (u - cx) * Z / fx
  Y = (v - cy) * Z / fy
  Z = depth_image[v, u]        (meters, float32)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

# depth image의 각 픽셀 카메라 3차원 좌표로 변환
def depth_to_points(
    depth_image: np.ndarray,   # (H, W) float32, meters
    K: np.ndarray,             # (3, 3) camera intrinsic matrix
    min_depth: float = 0.05,
    max_depth: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Back-project every valid pixel to a 3D point in the camera frame.

    Returns
    -------
    points       : (N, 3) float32 — XYZ in camera frame (meters)
    pixel_coords : (N, 2) int32   — (row, col) of each point in the image
    """
    H, W = depth_image.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # build pixel grid
    cols = np.arange(W, dtype=np.float32)
    rows = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(cols, rows)          # (H, W) each

    Z = depth_image.astype(np.float32)
    valid = np.isfinite(Z) & (Z > min_depth) & (Z < max_depth)

    Z_v = Z[valid]
    u_v = uu[valid]
    v_v = vv[valid]

    X = (u_v - cx) * Z_v / fx
    Y = (v_v - cy) * Z_v / fy

    points       = np.stack([X, Y, Z_v], axis=1)                      # (N, 3)
    pixel_coords = np.stack([v_v.astype(np.int32),
                              u_v.astype(np.int32)], axis=1)           # (N, 2)
    return points, pixel_coords
