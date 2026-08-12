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
    CATEGORY_DESTINATION,
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


# ── tabletop occupancy ────────────────────────────────────────────────────────

def extract_tabletop_obstacles(
    category_points: List[CategoryPoints],
    surface_z:       float,
    *,
    min_height:      float = 0.020,
    max_height:      float = 0.400,
    cell:            float = 0.020,
    min_points:      int   = 15,
    workspace_x:     Tuple[float, float] = (0.15, 0.95),
    workspace_y:     Tuple[float, float] = (-0.75, 0.75),
    arm_top_z:       float = 0.250,
) -> List[Dict]:
    """Things standing on the surface, as XY footprints to place around.

    Height, not category, is what separates them. A "table" mask covers the
    objects sitting on it, so they inherit DESTINATION rather than OBSTACLE:
    one measured scan put 197 k points under DESTINATION spanning z to 0.516,
    with only 172 left in UNKNOWN. Categories cannot be trusted here; the 2 cm
    of clear air above the tabletop can.

    Clustering is 8-connected on a `cell` grid rather than KDTree/DBSCAN —
    footprints are what matter, the grid IS the output resolution, and it costs
    one pass over a few thousand points.

    `arm_top_z` drops the robot itself, which the overhead view sees as a tall
    narrow column (one scan: 84 points at (0.469, 0.007) reaching z = 0.400).
    Objects on this table top out around 0.13.

    Returns dicts of {centroid, xy_radius, top_z, point_count} — xy_radius is
    the circumscribed radius of the cluster's footprint, so a caller can keep
    its own footprint clear of it without reasoning about shape.
    """
    pts = [cp.points for cp in category_points
           if cp.category in (CATEGORY_DESTINATION, CATEGORY_UNKNOWN)
           and len(cp.points)]
    if not pts:
        return []
    p = np.vstack(pts)

    keep = ((p[:, 0] > workspace_x[0]) & (p[:, 0] < workspace_x[1]) &
            (p[:, 1] > workspace_y[0]) & (p[:, 1] < workspace_y[1]) &
            (p[:, 2] > surface_z + min_height) &
            (p[:, 2] < surface_z + max_height))
    p = p[keep]
    if len(p) < min_points:
        return []

    gx = np.floor(p[:, 0] / cell).astype(np.int64)
    gy = np.floor(p[:, 1] / cell).astype(np.int64)
    cells: Dict[Tuple[int, int], List[int]] = {}
    for i in range(len(p)):
        cells.setdefault((int(gx[i]), int(gy[i])), []).append(i)

    out: List[Dict] = []
    seen: set = set()
    for start in cells:
        if start in seen:
            continue
        seen.add(start)
        stack, comp = [start], []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    nb = (cur[0] + da, cur[1] + db)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        idx = [i for c in comp for i in cells[c]]
        if len(idx) < min_points:
            continue
        q = p[idx]
        top_z = float(q[:, 2].max())
        if top_z > surface_z + arm_top_z:
            continue                      # the arm, not a tabletop object
        c = q[:, :2].mean(axis=0)
        radius = float(np.max(np.linalg.norm(q[:, :2] - c, axis=1)))
        out.append({
            'centroid':    [round(float(c[0]), 4), round(float(c[1]), 4),
                            round(float(q[:, 2].mean()), 4)],
            'xy_radius':   round(radius, 4),
            'top_z':       round(top_z, 4),
            'point_count': len(idx),
        })
    out.sort(key=lambda o: -o['point_count'])
    return out


def obstacles_from_boxes(
    boxes:      List[Dict],
    K:          np.ndarray,
    R:          np.ndarray,
    t:          np.ndarray,
    surface_z:  float,
    *,
    min_radius: float = 0.020,
) -> List[Dict]:
    """XY footprints for objects the depth stream cannot see, from 2D boxes.

    A glass returns no depth, so it is absent from the cloud entirely — the one
    obstacle that most needs avoiding is the one extract_tabletop_obstacles()
    cannot find. The VLM does see it (`glass` is in every detection list), so
    intersect the box's corner rays with the support plane instead of trusting
    depth. An overhead camera looks nearly straight down, which is what keeps
    the XY error small; the same trick under the wrist camera's oblique view
    would smear the footprint badly.

    Height is NOT recovered — top_z is None. Callers must treat these as
    "occupied XY, unknown height".

    boxes: [{'label': str, 'bbox_xyxy': [x1, y1, x2, y2]}] in the image the
    K/R/t describe. R, t map camera frame → world (panda_link0).
    """
    if not len(boxes):
        return []
    Kinv = np.linalg.inv(np.asarray(K, dtype=np.float64).reshape(3, 3))
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    out: List[Dict] = []
    for b in boxes:
        x1, y1, x2, y2 = (float(v) for v in b['bbox_xyxy'])
        corners = np.array([[x1, y1, 1.0], [x2, y1, 1.0],
                            [x2, y2, 1.0], [x1, y2, 1.0]]).T
        dirs = R @ (Kinv @ corners)          # (3, 4) ray directions in world
        hits = []
        for i in range(dirs.shape[1]):
            d = dirs[:, i]
            # Parallel to the plane, or pointing away from it: no intersection.
            if abs(d[2]) < 1e-6:
                continue
            s = (surface_z - t[2]) / d[2]
            if s <= 0:
                continue
            hits.append(t + s * d)
        if len(hits) < 3:
            continue
        h = np.asarray(hits)
        c = h[:, :2].mean(axis=0)
        radius = max(float(np.max(np.linalg.norm(h[:, :2] - c, axis=1))),
                     min_radius)
        out.append({
            'label':       b.get('label', ''),
            'centroid':    [round(float(c[0]), 4), round(float(c[1]), 4),
                            round(float(surface_z), 4)],
            'xy_radius':   round(radius, 4),
            'top_z':       None,            # depth-less: height unknown
            'point_count': 0,
            'source':      'box',
        })
    return out


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
