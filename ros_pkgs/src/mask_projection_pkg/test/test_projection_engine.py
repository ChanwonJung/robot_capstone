"""Tabletop occupancy extraction — pure numpy, no ROS.

These guard the assumption the placement logic rests on: that objects standing
on a surface can be separated by HEIGHT, because they cannot be separated by
category. A "table" mask covers everything sitting on it, so a measured scan put
197 k points under DESTINATION (spanning z to 0.516) and left only 172 in
UNKNOWN.
"""
import numpy as np

from mask_projection_pkg.label_mapper import (
    CATEGORY_DESTINATION,
    CATEGORY_TARGET,
    CATEGORY_UNKNOWN,
    CategoryPoints,
)
from mask_projection_pkg.projection_engine import (
    extract_tabletop_obstacles,
    obstacles_from_boxes,
)


def _cp(points, category=CATEGORY_DESTINATION, label='table'):
    p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return CategoryPoints(
        label=label, category=category, points=p,
        colors=np.zeros((len(p), 3), np.uint8),
        categories=np.full(len(p), category, np.uint8))


def _blob(cx, cy, top_z, n=60, r=0.03, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 2 * np.pi, n)
    d = r * np.sqrt(rng.uniform(0, 1, n))
    return np.stack([cx + d * np.cos(a),
                     cy + d * np.sin(a),
                     rng.uniform(top_z * 0.5, top_z, n)], axis=1)


def _plane(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(0.2, 0.9, n),
                     rng.uniform(-0.6, 0.6, n),
                     rng.normal(0.0, 0.001, n)], axis=1)


def test_finds_objects_by_height_not_category():
    # Both blobs wear DESTINATION, exactly as a table mask makes them.
    pts = np.vstack([_plane(), _blob(0.50, 0.20, 0.08, seed=2),
                     _blob(0.65, -0.15, 0.10, seed=3)])
    obs = extract_tabletop_obstacles([_cp(pts)], surface_z=0.0)
    assert len(obs) == 2
    found = sorted((o['centroid'][0], o['centroid'][1]) for o in obs)
    assert abs(found[0][0] - 0.50) < 0.02 and abs(found[0][1] - 0.20) < 0.02
    assert abs(found[1][0] - 0.65) < 0.02 and abs(found[1][1] + 0.15) < 0.02


def test_ignores_the_surface_itself():
    assert extract_tabletop_obstacles([_cp(_plane())], surface_z=0.0) == []


def test_drops_the_robot_arm():
    # The overhead view sees the arm as a tall narrow column: one scan had 84
    # points at (0.469, 0.007) reaching z = 0.400, while table objects top out
    # near 0.13.
    pts = np.vstack([_plane(), _blob(0.469, 0.007, 0.40, n=90, r=0.02, seed=4),
                     _blob(0.60, 0.25, 0.07, seed=5)])
    obs = extract_tabletop_obstacles([_cp(pts)], surface_z=0.0)
    assert len(obs) == 1
    assert abs(obs[0]['centroid'][0] - 0.60) < 0.02


def test_surface_z_offsets_the_whole_threshold():
    # A table 10 cm up must behave exactly like one at zero, not report its own
    # surface as an obstacle.
    pts = np.vstack([_plane() + [0, 0, 0.10], _blob(0.55, 0.10, 0.18, seed=6)])
    obs = extract_tabletop_obstacles([_cp(pts)], surface_z=0.10)
    assert len(obs) == 1
    assert abs(obs[0]['centroid'][0] - 0.55) < 0.02


def test_radius_covers_the_footprint():
    pts = np.vstack([_plane(), _blob(0.55, 0.10, 0.08, n=200, r=0.06, seed=7)])
    obs = extract_tabletop_obstacles([_cp(pts)], surface_z=0.0)
    assert len(obs) == 1
    assert obs[0]['xy_radius'] >= 0.05


def test_ignores_points_outside_the_workspace():
    pts = np.vstack([_plane(), _blob(1.40, 0.10, 0.08, seed=8)])
    assert extract_tabletop_obstacles([_cp(pts)], surface_z=0.0) == []


def test_reads_unknown_as_well_as_destination():
    obs = extract_tabletop_obstacles(
        [_cp(_plane()),
         _cp(_blob(0.55, 0.10, 0.08, seed=9), CATEGORY_UNKNOWN, 'unknown')],
        surface_z=0.0)
    assert len(obs) == 1


def test_target_points_are_not_treated_as_clutter():
    # TARGET is excluded by category: the caller measures the support plane from
    # it, and by place time it is in the gripper.
    obs = extract_tabletop_obstacles(
        [_cp(_blob(0.55, 0.10, 0.08, seed=10), CATEGORY_TARGET, 'book')],
        surface_z=0.0)
    assert obs == []


# ── depth-less footprints from 2D boxes ──────────────────────────────────────

def _overhead(height=2.0):
    """Camera at (0, 0, height) looking straight down, +x right, +y down."""
    K = np.array([[400.0, 0.0, 320.0],
                  [0.0, 400.0, 240.0],
                  [0.0, 0.0, 1.0]])
    R = np.array([[1.0, 0.0, 0.0],     # cam +x → world +x
                  [0.0, -1.0, 0.0],    # cam +y → world -y
                  [0.0, 0.0, -1.0]])   # cam +z (forward) → world -z
    t = np.array([0.0, 0.0, height])
    return K, R, t


def test_box_projects_onto_the_support_plane():
    # A glass returns no depth, so its footprint has to come from the box. The
    # image centre must land directly under the camera.
    K, R, t = _overhead(2.0)
    out = obstacles_from_boxes(
        [{'label': 'glass', 'bbox_xyxy': [300, 220, 340, 260]}],
        K, R, t, surface_z=0.0)
    assert len(out) == 1
    assert abs(out[0]['centroid'][0]) < 1e-6
    assert abs(out[0]['centroid'][1]) < 1e-6
    # 40 px at 400 px focal over 2 m → 0.20 m across, so radius ~0.14 (corners).
    assert 0.10 < out[0]['xy_radius'] < 0.18
    # Height is genuinely unknown — callers must not read it as a number.
    assert out[0]['top_z'] is None


def test_box_offset_maps_to_the_right_direction():
    K, R, t = _overhead(2.0)
    out = obstacles_from_boxes(
        [{'label': 'glass', 'bbox_xyxy': [420, 220, 460, 260]}],
        K, R, t, surface_z=0.0)
    # +x in the image is +x in the world under this R.
    assert out[0]['centroid'][0] > 0.4
    assert abs(out[0]['centroid'][1]) < 1e-6


def test_rays_pointing_away_from_the_plane_are_dropped():
    # Camera below the surface: nothing it sees can intersect the plane ahead.
    K, R, t = _overhead(-1.0)
    assert obstacles_from_boxes(
        [{'label': 'glass', 'bbox_xyxy': [300, 220, 340, 260]}],
        K, R, t, surface_z=0.0) == []


def test_no_boxes_is_empty():
    K, R, t = _overhead()
    assert obstacles_from_boxes([], K, R, t, surface_z=0.0) == []
