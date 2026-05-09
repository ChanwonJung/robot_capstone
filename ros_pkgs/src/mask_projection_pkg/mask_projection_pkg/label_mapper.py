"""
Maps mask pixel values → semantic categories → colors.

Category assignment (order-based, tied to GSAM prompt order):
  mask pixel 0  → CATEGORY_FREE      (background)
  mask pixel 1  → CATEGORY_TARGET    (detections_json[0])
  mask pixel 2  → CATEGORY_WORKSPACE (detections_json[1])
  mask pixel 3+ → CATEGORY_OBSTACLE  (detections_json[2+])

All depth pixels are kept — background → FREE, detected → labeled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


# ── Category IDs ─────────────────────────────────────────────────────────────
CATEGORY_FREE:      int = 0   # background / unoccupied space (EE view)
CATEGORY_TARGET:    int = 1   # object to be moved
CATEGORY_WORKSPACE: int = 2   # surface / operational area
CATEGORY_OBSTACLE:  int = 3   # anything else detected → treat as obstacle
CATEGORY_UNKNOWN:   int = 4   # top-view geometry (unclassified by GSAM)

# ── Per-category color (R, G, B) ─────────────────────────────────────────────
CATEGORY_COLOR: Dict[int, tuple] = {
    CATEGORY_FREE:      ( 80,  80,  80),  # grey
    CATEGORY_TARGET:    (  0, 200,  80),  # green
    CATEGORY_WORKSPACE: (255, 220,   0),  # yellow
    CATEGORY_OBSTACLE:  (220,  40,  40),  # red
    CATEGORY_UNKNOWN:   (150,  80, 200),  # purple
}

# ── Mask pixel value → category ──────────────────────────────────────────────
# mask pixel value is 1-based index into detections_json.
# pixel=0 (background) → CATEGORY_FREE (handled separately in apply_labels).
# Only this dict needs to change when the upstream protocol changes.
MASK_VALUE_TO_CATEGORY: Dict[int, int] = {
    1: CATEGORY_TARGET,     # detections_json[0]
    2: CATEGORY_WORKSPACE,  # detections_json[1]
    # 3+ → OBSTACLE (handled by fallback in apply_labels)
}

# ── "category" string → category ID (Qwen / stub path) ───────────────────────
# Used when detections carry an explicit "category" field (e.g. from qwen_stub).
# Fallback to MASK_VALUE_TO_CATEGORY when field is absent (legacy path).
_CATEGORY_STR_TO_ID: Dict[str, int] = {
    "TARGET":    CATEGORY_TARGET,
    "WORKSPACE": CATEGORY_WORKSPACE,
    "OBSTACLE":  CATEGORY_OBSTACLE,
}


@dataclass
class CategoryPoints:
    """All 3D points belonging to one semantic category."""
    label:      str           # e.g. "cup", "table", "free", "obstacle"
    category:   int           # CATEGORY_* constant
    points:     np.ndarray    # (N, 3) float32  XYZ in camera frame
    colors:     np.ndarray    # (N, 3) uint8    RGB per point
    categories: np.ndarray    # (N,)   uint8    category ID per point


def apply_labels(
    points:       np.ndarray,   # (N, 3)  float32  — ALL depth points
    pixel_coords: np.ndarray,   # (N, 2)  int32    (row, col)
    mask_image:   np.ndarray,   # (H, W)  uint8
    detections:   List[Dict],   # detections_json list (may be empty)
) -> List[CategoryPoints]:
    """
    Assign a semantic category to every 3D point using its source pixel.

    All points are kept (background → FREE, detected → labeled).
    Returns one CategoryPoints per active category, sorted by category ID.
    """
    rows, cols = pixel_coords[:, 0], pixel_coords[:, 1]
    mask_vals  = mask_image[rows, cols].astype(np.int32)   # 1-based, 0=bg

    result: List[CategoryPoints] = []

    # collect all unique mask values present in this frame
    unique_vals = np.unique(mask_vals)

    for mv in sorted(unique_vals):
        sel = (mask_vals == mv)
        n   = int(sel.sum())

        if mv == 0:
            # background → FREE
            result.append(CategoryPoints(
                label      = 'free',
                category   = CATEGORY_FREE,
                points     = points[sel],
                colors     = np.tile(
                    np.array(CATEGORY_COLOR[CATEGORY_FREE], dtype=np.uint8), (n, 1)),
                categories = np.full(n, CATEGORY_FREE, dtype=np.uint8),
            ))
        else:
            det_idx = mv - 1
            det     = detections[det_idx] if det_idx < len(detections) else None

            # Qwen/stub path: use explicit "category" field when present
            # Legacy path: fall back to position-based MASK_VALUE_TO_CATEGORY
            if det and "category" in det:
                category_id = _CATEGORY_STR_TO_ID.get(det["category"], CATEGORY_OBSTACLE)
            else:
                category_id = MASK_VALUE_TO_CATEGORY.get(mv, CATEGORY_OBSTACLE)

            color = CATEGORY_COLOR[category_id]
            label = det["label"] if det else f"obstacle_{mv}"

            result.append(CategoryPoints(
                label      = label,
                category   = category_id,
                points     = points[sel],
                colors     = np.tile(
                    np.array(color, dtype=np.uint8), (n, 1)),
                categories = np.full(n, category_id, dtype=np.uint8),
            ))

    return result
