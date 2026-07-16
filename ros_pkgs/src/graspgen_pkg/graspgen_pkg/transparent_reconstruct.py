"""transparent_reconstruct.py — 투명물체(유리컵) 원통 복원 (ROS-free numpy).

see-through 로 깨진 투명물체 depth 를, GSAM 마스크 + "원통" 형상 prior 로
기하학적으로 복원한다. 딥러닝 없음 — 책 analytic top-down 과 같은 계열.

핵심 아이디어
-------------
투명 유리는 벽이 see-through 라 depth 가 뒤 테이블을 찍는다(=틀린 값). 그래서
back-projection 하면 포인트가 시선방향으로 늘어진다. 하지만:
  1) 유리 마스크는 정확 → "어느 픽셀이 유리인지" 안다.
  2) 컵은 테이블에 수직으로 선 원통 → 강한 형상 prior.
이 둘로 컵 표면을 계산으로 합성한다:

  ① 유리 마스크 픽셀의 카메라 레이를 테이블 평면(z=table_z)에 투영 → 바닥 그림자 G
  ② G 는 사선 뷰에서 시선방향으로 늘어진 타원 → PCA 로 분해
        긴 축  = 시선방향 스미어  → 무시
        짧은 축 = 진짜 컵 지름     → 반지름 r
  ③ 중심 = (짧은축) 평균 + (긴축) 카메라쪽 끝 - r   ← 스미어 편향 보정
  ④ 높이 = 부분표면 z 신호 + prior 클램프 (또는 고정값)
  ⑤ 원통 옆면+윗면 샘플 → dense TARGET 클라우드

좌표 규약: p_world = R @ p_cam + t  (extract_target_cloud 과 동일). t = 카메라 광심(world).
"""
from __future__ import annotations

import numpy as np


def _pixel_ray_dirs(u: np.ndarray, v: np.ndarray, K: np.ndarray,
                    R: np.ndarray) -> np.ndarray:
    """픽셀 (u,v) → world 프레임 레이 방향 (N,3), 정규화 안 함 (Z 성분 유지)."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    d_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=1)
    return (R @ d_cam.T).T


def reconstruct_cylinder_cloud(
    depth: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray,
    target_val: int,
    R: np.ndarray,
    t: np.ndarray,
    min_depth: float,
    max_depth: float,
    *,
    table_percentile: float = 50.0,
    radius_min: float = 0.015,
    radius_max: float = 0.10,
    height: float = -1.0,          # <0 → 데이터에서 추정, 아니면 고정(m)
    height_clip: tuple = (0.03, 0.22),
    n_theta: int = 48,
    n_z: int = 14,
) -> tuple[np.ndarray | None, dict]:
    """투명 TARGET 마스크 → 복원된 원통 point cloud (world, (M,3) float32).

    Returns (cloud|None, info). info 는 로깅용 dict(table_z/center/radius/height/n).
    마스크 픽셀이 너무 적으면 (None, {...}).
    """
    info: dict = {}
    ys, xs = np.where(mask == target_val)
    if len(xs) < 30:
        return None, {'reason': f'유리 마스크 픽셀 부족 ({len(xs)})'}

    u = xs.astype(np.float64)
    v = ys.astype(np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    cam_xy = t[:2]

    # ── 1) 유리 픽셀의 (깨진) world 포인트 — table_z / height 신호용 ──────────
    Z = depth[ys, xs].astype(np.float64)
    ok = np.isfinite(Z) & (Z > min_depth) & (Z < max_depth)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    pc = np.stack([(u[ok] - cx) * Z[ok] / fx,
                   (v[ok] - cy) * Z[ok] / fy,
                   Z[ok]], axis=1)
    pw = (R @ pc.T).T + t                       # (Nv,3) world (see-through)
    if len(pw) < 5:
        return None, {'reason': '유효 depth 포인트 부족'}

    # WORKAROUND: coordinate transform may have sign inversion.
    # Clamp table_z to [0, camera_z] — table is always below camera.
    # This is a temporary fix pending extrinsics validation.
    table_z_raw = float(np.percentile(pw[:, 2], table_percentile))
    camera_z = float(t[2])
    table_z = float(np.clip(table_z_raw, 0.0, camera_z - 0.05))

    if table_z_raw < 0:
        print(f"[WARN] table_z before clamp: {table_z_raw:.4f} (negative!), "
              f"camera_z: {camera_z:.4f}, clamped to: {table_z:.4f}")

    # ── 2) 바닥 그림자 G: 모든 유리 픽셀 레이를 테이블 평면에 투영 ───────────
    d_world = _pixel_ray_dirs(u, v, K, R)
    denom = d_world[:, 2]
    valid = np.abs(denom) > 1e-6
    s = np.full_like(denom, np.nan)
    s[valid] = (table_z - t[2]) / denom[valid]
    front = valid & (s > 0) & np.isfinite(s)
    if front.sum() < 20:
        return None, {'reason': f'테이블 평면 교차 포인트 부족 ({int(front.sum())})'}
    G = t[:2][None, :] + s[front, None] * d_world[front, :2]      # (Ng,2)

    # ── 3) PCA: 짧은 축 = 지름, 긴 축 = 시선 스미어 ─────────────────────────
    c0 = G.mean(axis=0)
    Gc = G - c0
    evals, evecs = np.linalg.eigh(Gc.T @ Gc)     # 오름차순
    e_short = evecs[:, 0]
    e_long = evecs[:, 1]
    # 긴 축을 카메라 쪽으로 향하게 정렬
    if np.dot(cam_xy - c0, e_long) < 0:
        e_long = -e_long
    proj_short = Gc @ e_short
    proj_long = Gc @ e_long

    # 반지름: 짧은축 5~95% 폭의 절반 (스미어 무관)
    short_ext = np.percentile(proj_short, 95) - np.percentile(proj_short, 5)
    r = float(np.clip(short_ext / 2.0, radius_min, radius_max))

    # 중심: 짧은축=평균(0), 긴축=카메라쪽 끝(90%) - r  (근접 rim 에서 반지름만큼 후퇴)
    near = float(np.percentile(proj_long, 90))
    center = c0 + e_long * (near - r)

    # ── 4) 높이 ─────────────────────────────────────────────────────────────
    if height >= 0.0:
        h = float(height)
    else:
        h = float(np.percentile(pw[:, 2], 90) - table_z)
        h = float(np.clip(h, height_clip[0], height_clip[1]))
        if not np.isfinite(h) or h < height_clip[0]:
            h = height_clip[0]

    # ── 5) 원통 샘플 (옆면 + 윗면 rim + 윗면 disk 몇 개) ─────────────────────
    th = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    ring_x = center[0] + r * np.cos(th)
    ring_y = center[1] + r * np.sin(th)
    zs = np.linspace(table_z, table_z + h, n_z)
    pts = []
    for z in zs:                                  # 옆면
        pts.append(np.stack([ring_x, ring_y, np.full(n_theta, z)], axis=1))
    # 윗면 disk (top-down grasp 가 표면을 보게)
    for rr in np.linspace(0.0, r, 4)[1:]:
        dx = center[0] + rr * np.cos(th)
        dy = center[1] + rr * np.sin(th)
        pts.append(np.stack([dx, dy, np.full(n_theta, table_z + h)], axis=1))
    cloud = np.vstack(pts).astype(np.float32)

    info = {'table_z': round(table_z, 4),
            'center': [round(float(center[0]), 4), round(float(center[1]), 4)],
            'radius': round(r, 4), 'height': round(h, 4),
            'n_shadow': int(front.sum()), 'n_out': len(cloud)}
    return cloud, info
