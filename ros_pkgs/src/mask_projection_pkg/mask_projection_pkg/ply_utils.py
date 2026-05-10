"""Shared PLY I/O and result-JSON utilities for mask_projection_pkg nodes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .label_mapper import CATEGORY_TARGET, CATEGORY_WORKSPACE, CategoryPoints


def save_ply_xyz(path: Path, points: np.ndarray) -> None:
    """Save (N, 3) float32 XYZ points as binary-little-endian PLY."""
    N = len(points)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    )
    with open(path, 'wb') as f:
        f.write(header.encode())
        f.write(points.astype(np.float32).tobytes())


def save_ply_labeled(path: Path, category_points: List[CategoryPoints]) -> None:
    """Save world-frame labeled points (XYZ + RGB + category) as binary PLY."""
    all_pts  = np.concatenate([cp.points     for cp in category_points], axis=0)
    all_col  = np.concatenate([cp.colors     for cp in category_points], axis=0)
    all_cats = np.concatenate([cp.categories for cp in category_points], axis=0)
    N = len(all_pts)
    path.parent.mkdir(parents=True, exist_ok=True)
    dt = np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('red', np.uint8), ('green', np.uint8), ('blue', np.uint8),
        ('category', np.uint8),
    ])
    arr             = np.zeros(N, dtype=dt)
    arr['x']        = all_pts[:, 0]
    arr['y']        = all_pts[:, 1]
    arr['z']        = all_pts[:, 2]
    arr['red']      = all_col[:, 0]
    arr['green']    = all_col[:, 1]
    arr['blue']     = all_col[:, 2]
    arr['category'] = all_cats
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar category\n"
        "end_header\n"
    )
    with open(path, 'wb') as f:
        f.write(header.encode())
        f.write(arr.tobytes())


def build_result_json(category_points: List[CategoryPoints]) -> str:
    """
    JSON summary per category: label, centroid, bbox_3d_world, point_count.

    {
      "target":    {"label": "cup",   "centroid": [x,y,z],
                    "bbox_3d_world": {"min": [x,y,z], "max": [x,y,z]},
                    "point_count": N},
      "workspace": { ... },
      ...
    }
    """
    _CATEGORY_KEY = {
        CATEGORY_TARGET:    'target',
        CATEGORY_WORKSPACE: 'workspace',
    }
    out: Dict = {}
    for cp in category_points:
        key      = _CATEGORY_KEY.get(cp.category, cp.label)
        centroid = cp.points.mean(axis=0).tolist()
        pts_min  = cp.points.min(axis=0).tolist()
        pts_max  = cp.points.max(axis=0).tolist()
        out[key] = {
            'label':         cp.label,
            'centroid':      [round(v, 4) for v in centroid],
            'bbox_3d_world': {
                'min': [round(v, 4) for v in pts_min],
                'max': [round(v, 4) for v in pts_max],
            },
            'point_count':   len(cp.points),
        }
    return json.dumps(out)
