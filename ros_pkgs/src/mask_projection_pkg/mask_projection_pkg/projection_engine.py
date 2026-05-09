"""Pure-numpy projection and filtering logic — no ROS dependencies.

All functions operate on plain numpy arrays so they can be unit-tested
without a running ROS node.  The node (multi_view_projector_node.py) is
responsible only for ROS I/O: decoding messages, publishing results.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from scipy.spatial import KDTree

from .back_projection import depth_to_points
from .label_mapper import (
    CATEGORY_COLOR,
    CATEGORY_FREE,
    CATEGORY_UNKNOWN,
    CategoryPoints,
    apply_labels,
)


# ── extrinsics ────────────────────────────────────────────────────────────────

def load_extrinsics(
    path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[str]]:
    """Load camera extrinsics from YAML.

    Returns (R_top, t_top, R_ee, t_ee, warn_msg).
    warn_msg is None on success; an error string on fallback to identity.
    p_world = R @ p_cam + t
    """
    try:
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        R_top = np.array(cfg['top_camera']['R'], dtype=np.float64)
        t_top = np.array(cfg['top_camera']['t'], dtype=np.float64)
        R_ee  = np.array(cfg['ee_camera']['R'],  dtype=np.float64)
        t_ee  = np.array(cfg['ee_camera']['t'],  dtype=np.float64)
        assert R_top.shape == (3, 3) and t_top.shape == (3,)
        assert R_ee.shape  == (3, 3) and t_ee.shape  == (3,)
        return R_top, t_top, R_ee, t_ee, None
    except Exception as exc:
        return np.eye(3), np.zeros(3), np.eye(3), np.zeros(3), str(exc)


# ── projection ────────────────────────────────────────────────────────────────

def project_labeled(
    depth:      np.ndarray,
    K:          np.ndarray,
    mask:       np.ndarray,
    detections: List[Dict],
    R:          np.ndarray,
    t:          np.ndarray,
    min_depth:  float,
    max_depth:  float,
) -> List[CategoryPoints]:
    """Back-project EE depth, transform to world frame, apply GSAM labels.

    depth: (H, W) float32 metres
    K:     (3, 3) intrinsic matrix
    mask:  (H, W) uint8 — pixel value = 1-based detection index (0 = FREE)
    """
    pts_cam, pixel_coords = depth_to_points(depth, K,
                                            min_depth=min_depth,
                                            max_depth=max_depth)
    if len(pts_cam) == 0:
        return []
    pts_world = (R @ pts_cam.T).T + t
    return apply_labels(pts_world, pixel_coords, mask, detections)


def project_unknown(
    depth:                np.ndarray,
    K:                    np.ndarray,
    R:                    np.ndarray,
    t:                    np.ndarray,
    min_depth:            float,
    max_depth:            float,
    ee_seg_pts:           Optional[np.ndarray] = None,
    ee_seg_filter_radius: float = 0.015,
    ee_seg_z_margin:      float = 0.10,
) -> Optional[CategoryPoints]:
    """Back-project top-view depth → UNKNOWN points.

    Pass 1 filter: removes points within ee_seg_filter_radius (XY) AND
    ee_seg_z_margin (Z) of any EE-segmented point.  The Z gate prevents floor
    points from being removed by the XY footprint of elevated objects (e.g.
    table at Z=1 m whose footprint would otherwise cover the floor at Z=0 m).
    """
    pts_cam, _ = depth_to_points(depth, K, min_depth=min_depth, max_depth=max_depth)
    if len(pts_cam) == 0:
        return None

    pts_world = (R @ pts_cam.T).T + t

    if ee_seg_pts is not None and len(ee_seg_pts) > 0:
        tree = KDTree(ee_seg_pts[:, :2])
        dists, idx = tree.query(pts_world[:, :2], workers=-1)
        z_diff = np.abs(pts_world[:, 2] - ee_seg_pts[idx, 2])
        remove = (dists <= ee_seg_filter_radius) & (z_diff <= ee_seg_z_margin)
        pts_world = pts_world[~remove]

    if len(pts_world) == 0:
        return None

    return _make_unknown_points(pts_world)


# ── filtering helpers ─────────────────────────────────────────────────────────

def collect_seg_points(category_points: List[CategoryPoints]) -> Optional[np.ndarray]:
    """Return (N, 3) array of non-FREE EE segmentation points, or None if empty."""
    seg = [cp.points for cp in category_points if cp.category != CATEGORY_FREE]
    if not seg:
        return None
    return np.concatenate(seg, axis=0)


def filter_free_by_unknown(
    ee_pts:      List[CategoryPoints],
    top_unknown: CategoryPoints,
    xy_radius:   float,
    z_margin:    float,
) -> List[CategoryPoints]:
    """Pass 2: remove EE FREE points that overlap top UNKNOWN (XY + Z gate).

    Implements UNKNOWN > FREE priority.  Must be called after Pass 1 so that
    already-removed UNKNOWN locations do not incorrectly suppress FREE points.
    """
    if len(top_unknown.points) == 0:
        return ee_pts

    tree = KDTree(top_unknown.points[:, :2])
    result: List[CategoryPoints] = []
    for cp in ee_pts:
        if cp.category != CATEGORY_FREE:
            result.append(cp)
            continue
        dists, idx = tree.query(cp.points[:, :2], workers=-1)
        z_diff = np.abs(cp.points[:, 2] - top_unknown.points[idx, 2])
        keep = ~((dists <= xy_radius) & (z_diff <= z_margin))
        if keep.any():
            result.append(CategoryPoints(
                label=cp.label,
                category=cp.category,
                points=cp.points[keep],
                colors=cp.colors[keep],
                categories=cp.categories[keep],
            ))
    return result


# ── internal ──────────────────────────────────────────────────────────────────

def _make_unknown_points(points: np.ndarray) -> CategoryPoints:
    n     = len(points)
    color = CATEGORY_COLOR[CATEGORY_UNKNOWN]
    return CategoryPoints(
        label      = 'unknown',
        category   = CATEGORY_UNKNOWN,
        points     = points.astype(np.float32),
        colors     = np.tile(np.array(color, dtype=np.uint8), (n, 1)),
        categories = np.full(n, CATEGORY_UNKNOWN, dtype=np.uint8),
    )
