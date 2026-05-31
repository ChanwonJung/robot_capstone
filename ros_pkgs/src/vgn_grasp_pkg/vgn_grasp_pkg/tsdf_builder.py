"""tsdf_builder.py — Signed TSDF construction via depth ray-casting (ROS-free)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def project_voxels(
    p_world:   np.ndarray,
    depth:     np.ndarray,
    K:         np.ndarray,
    R:         np.ndarray,
    t:         np.ndarray,
    trunc:     float,
    occlude_filter: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project voxel centres onto a depth image and compute signed TSDF values.

    SDF sign: positive = voxel in front of observed surface (free space),
              negative = voxel behind observed surface (inside object).

    Normalisation: tsdf = clip(sdf / trunc, -1, 1) * 0.5 + 0.5
      → 1.0 = far outside,  0.5 = on surface,  0.0 = deep inside

    occlude_filter: if True, voxels with sdf < -trunc are treated as unknown
      (not updated). Use for top-down cameras to avoid misclassifying cup side
      walls as inside — such voxels are occluded by the rim, not inside the object.

    Returns:
      tsdf_vals (N,) float32 in [0, 1] — unobserved voxels default to 1.0
      valid     (N,) bool  — True for voxels with a valid depth reading
    """
    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    p_cam    = (R.T @ (p_world - t).T).T          # (N, 3)
    pz       = p_cam[:, 2]
    in_front = pz > 0.01
    safe_pz  = np.where(in_front, pz, 1.0)

    u  = fx * p_cam[:, 0] / safe_pz + cx
    v  = fy * p_cam[:, 1] / safe_pz + cy
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    in_image = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    proj_ok  = in_front & in_image

    tsdf_vals = np.ones(len(p_world), dtype=np.float32)
    valid     = np.zeros(len(p_world), dtype=bool)

    if proj_ok.any():
        idx       = np.where(proj_ok)[0]
        d_obs     = depth[vi[idx], ui[idx]].astype(np.float64)
        depth_ok  = (d_obs > 0.01) & np.isfinite(d_obs)
        idx_valid = idx[depth_ok]
        sdf       = d_obs[depth_ok] - pz[idx_valid]
        if occlude_filter:
            not_occluded = sdf >= -trunc
            idx_valid = idx_valid[not_occluded]
            sdf       = sdf[not_occluded]
        tsdf_vals[idx_valid] = (np.clip(sdf / trunc, -1.0, 1.0) * 0.5 + 0.5).astype(np.float32)
        valid[idx_valid] = True

    return tsdf_vals, valid


def build_tsdf_raycasting(
    roi_min:    np.ndarray,
    roi_size_m: float,
    reso:       int,
    ee_depth:   np.ndarray,
    ee_K:       np.ndarray,
    R_ee:       np.ndarray,
    t_ee:       np.ndarray,
    top_depth:  Optional[np.ndarray] = None,
    top_K:      Optional[np.ndarray] = None,
    R_top:      Optional[np.ndarray] = None,
    t_top:      Optional[np.ndarray] = None,
    top_occlude_filter: bool = True,
    trunc_factor: float = 4.0,
) -> np.ndarray:
    """Build signed TSDF grid (1, reso, reso, reso) float32 via depth ray-casting.

    Each voxel accumulates signed TSDF values from all cameras with valid depth.
    Unobserved voxels default to 1.0 (outside/free).
    EE camera is weighted 2×, Top camera 4× for better surface reconstruction.
    """
    voxel_size = roi_size_m / reso
    trunc      = trunc_factor * voxel_size

    idx_arr = np.arange(reso, dtype=np.float32)
    gi, gj, gk = np.meshgrid(idx_arr, idx_arr, idx_arr, indexing='ij')
    p_world = ((np.stack([gi, gj, gk], axis=-1) + 0.5) * voxel_size
               + roi_min).reshape(-1, 3)

    tsdf_sum = np.zeros(len(p_world), dtype=np.float64)
    weight   = np.zeros(len(p_world), dtype=np.float64)

    def _integrate(depth, K, R, t, occlude_filter=False, w=1.0):
        vals, valid = project_voxels(p_world, depth, K, R, t, trunc, occlude_filter)
        tsdf_sum[valid] += vals[valid] * w
        weight[valid]   += w

    _integrate(ee_depth, ee_K, R_ee, t_ee, occlude_filter=True, w=2.0)
    if top_depth is not None and top_K is not None and R_top is not None and t_top is not None:
        _integrate(top_depth, top_K, R_top, t_top, occlude_filter=top_occlude_filter, w=4.0)

    result = np.ones(len(p_world), dtype=np.float32)
    obs = weight > 0
    result[obs] = (tsdf_sum[obs] / weight[obs]).astype(np.float32)
    return result.reshape(1, reso, reso, reso)


def build_tsdf_ncam(
    roi_min:    np.ndarray,
    roi_size_m: float,
    reso:       int,
    cameras:    list,
    trunc_factor: float = 4.0,
    z_min:      float = -np.inf,
) -> np.ndarray:
    """Build signed TSDF grid from N cameras.

    cameras: list of dicts, each with keys:
      depth  (np.ndarray H×W float32)
      K      (np.ndarray 3×3 float64)
      R      (np.ndarray 3×3 float64)
      t      (np.ndarray (3,) float64)
      weight (float, default 1.0)
      occlude_filter (bool, default False) — set True for top-down cameras

    z_min: world-frame Z floor (table top z). Voxels below this height are
      skipped during integration and remain at default 1.0 (free/outside).
    """
    voxel_size = roi_size_m / reso
    trunc      = trunc_factor * voxel_size

    idx_arr = np.arange(reso, dtype=np.float32)
    gi, gj, gk = np.meshgrid(idx_arr, idx_arr, idx_arr, indexing='ij')
    p_world = ((np.stack([gi, gj, gk], axis=-1) + 0.5) * voxel_size
               + roi_min).reshape(-1, 3)

    above    = p_world[:, 2] >= z_min
    tsdf_sum = np.zeros(len(p_world), dtype=np.float64)
    weight   = np.zeros(len(p_world), dtype=np.float64)

    for cam in cameras:
        w   = float(cam.get('weight', 1.0))
        occ = bool(cam.get('occlude_filter', False))
        vals, valid = project_voxels(
            p_world, cam['depth'], cam['K'], cam['R'], cam['t'], trunc, occ)
        valid_above = valid & above
        tsdf_sum[valid_above] += vals[valid_above] * w
        weight[valid_above]   += w

    result = np.ones(len(p_world), dtype=np.float32)
    obs = weight > 0
    result[obs] = (tsdf_sum[obs] / weight[obs]).astype(np.float32)
    return result.reshape(1, reso, reso, reso)


def tsdf_to_pointcloud(
    grid:       np.ndarray,
    roi_min:    np.ndarray,
    roi_size_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert TSDF grid to colored point arrays for RViz2 debug.

    Excludes voxels with tsdf >= 0.8 (clearly free space).

    Color coding:
      red   (255,   0, 0) : inside object  tsdf < 0.40
      green (  0, 255, 0) : surface        tsdf 0.40–0.60
      blue  (  0, 100, 255): near-outside  tsdf 0.60–0.80

    Returns:
      pts    (N, 3) float32 — world-frame voxel centres
      colors (N, 3) uint8  — RGB
    """
    vol        = grid.squeeze()
    reso       = vol.shape[0]
    voxel_size = roi_size_m / reso

    idx_arr = np.arange(reso, dtype=np.float32)
    gi, gj, gk = np.meshgrid(idx_arr, idx_arr, idx_arr, indexing='ij')
    p_world   = ((np.stack([gi, gj, gk], axis=-1) + 0.5) * voxel_size
                 + roi_min).reshape(-1, 3)
    tsdf_flat = vol.reshape(-1)

    mask = tsdf_flat < 0.8
    pts  = p_world[mask].astype(np.float32)
    vals = tsdf_flat[mask]

    colors = np.zeros((len(pts), 3), dtype=np.uint8)
    colors[vals < 0.40]                          = [255, 0,   0  ]   # red   — inside
    colors[(vals >= 0.40) & (vals <= 0.60)]      = [0,   255, 0  ]   # green — surface
    colors[vals > 0.60]                          = [0,   100, 255]   # blue  — near-outside

    return pts, colors
