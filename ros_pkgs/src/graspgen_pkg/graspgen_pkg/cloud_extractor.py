"""
cloud_extractor.py — Extract TARGET point cloud from EE depth + GSAM mask.

Multi-camera extensibility:
    Call extract_target_cloud() for each camera with its own (depth, K, mask, R, t)
    and np.vstack() the results before sending to GraspGen. The node only needs to
    cache additional depth/info topics and add calls here — core logic is unchanged.
"""
from __future__ import annotations

import numpy as np

from .depth_utils import depth_to_points

_FALLBACK_TARGET_MASK_VAL = 1  # first detection when labeled_detections unavailable


def find_target_mask_val(labeled_dets: list | None) -> int:
    """Return the 1-based mask pixel value for the TARGET category detection.

    Falls back to 1 (first detection) if labeled_detections is None or has no TARGET.
    """
    if labeled_dets:
        for det in labeled_dets:
            if det.get('category', '').upper() == 'TARGET':
                return int(det['idx']) + 1  # 1-based mask encoding
    return _FALLBACK_TARGET_MASK_VAL


def extract_target_cloud(
    depth: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray,
    target_val: int,
    R_cam: np.ndarray,
    t_cam: np.ndarray,
    min_depth: float,
    max_depth: float,
    max_points: int,
) -> np.ndarray | None:
    """Backproject TARGET-masked pixels to world-frame (N, 3) float32.

    Returns None if fewer than 1 TARGET point is found after filtering.

    Multi-camera usage example:
        clouds = [extract_target_cloud(d, K, m, val, R, t, ...) for d, K, m, R, t in cameras]
        clouds = [c for c in clouds if c is not None]
        pts_world = np.vstack(clouds) if clouds else None
    """
    pts_cam, pix = depth_to_points(depth, K, min_depth, max_depth)
    if len(pts_cam) == 0:
        return None

    rows, cols  = pix[:, 0], pix[:, 1]
    sel         = mask[rows, cols] == target_val
    pts_cam_sel = pts_cam[sel]

    if len(pts_cam_sel) == 0:
        return None

    pts_world = (R_cam @ pts_cam_sel.T).T + t_cam
    valid     = np.all(np.isfinite(pts_world), axis=1)
    pts_world = pts_world[valid].astype(np.float32)

    if len(pts_world) == 0:
        return None

    if len(pts_world) > max_points:
        idx       = np.random.choice(len(pts_world), max_points, replace=False)
        pts_world = pts_world[idx]

    return pts_world
