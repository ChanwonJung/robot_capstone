#!/usr/bin/env python3
"""gen_transparent_dataset.py — SwinDRNet 투명물체 fine-tune 데이터 생성 (headless).

각 프레임마다 컵 포즈를 랜덤화하고, 같은 포즈에서 두 번 렌더한다:
  · 투명 상태 DIP   → 입력 (컵 = see-through, 뒤 테이블 depth)
  · opaque 토글 DIP → GT   (컵 = 실제 표면, 미터 Z-depth)  ← DSD 안 씀, DIP 만
컵 마스크는 두 DIP 의 차이로 자동 추출(컵만 두 상태 사이에서 변한다).

노이즈/구멍은 여기서 굽지 않는다 — 데이터셋은 깨끗하게 두고, 학습 시
augmentation 으로 (input_dip 을 mask 로 뚫고 노이즈 합성). 그래야 재현·유연.

usage:
  ./isaacsim/python.sh sim/gen_transparent_dataset.py --frames 1            # 1프레임 검증
  ./isaacsim/python.sh sim/gen_transparent_dataset.py --frames 200 --jitter 0.15
  (GUI Isaac 이 떠 있으면 먼저 닫을 것 — SimulationApp 은 하나만.)
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SIM_DIR.parent

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--frames', type=int, default=1)
parser.add_argument('--out', default=str(SIM_DIR / 'dataset_out'))
parser.add_argument('--warmup', type=int, default=16, help='머티리얼 토글 후 RTX 안정화 프레임')
parser.add_argument('--jitter', type=float, default=0.0, help='컵 XY 로컬 지터(±). frame0 은 항상 원포즈')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--preview', type=int, default=4, help='PNG 미리보기 저장할 프레임 수')
parser.add_argument('--mask-thresh', type=float, default=0.01, help='컵 마스크 depth 차이 임계(m)')
parser.add_argument('--gui', action='store_true')
parser.add_argument('--stage', default=None)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402
simulation_app = SimulationApp({'headless': not args.gui})
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.utils.stage import open_stage, is_stage_loading  # noqa: E402
from isaacsim.sensors.camera import SingleViewDepthSensor  # noqa: E402
from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, Sdf, Gf  # noqa: E402
import math  # noqa: E402

DOWNLOADS_DIR = Path(os.environ.get('ROBOT_CAPSTONE_DOWNLOADS_DIR', Path.home() / 'Downloads')).expanduser()
XR_CONTENT_ROOT = Path(os.environ.get('ROBOT_CAPSTONE_XR_CONTENT_ROOT', DOWNLOADS_DIR / 'XR_Content_NVD@10010')).expanduser()
# exported scene(robot_capstone_scene.usd)는 팀원 머신 절대경로가 박혀 깨질 수 있어
# 신뢰하지 않는다. GUI 와 동일하게 로컬 SOURCE 를 열고 apply_scene() 으로 직접 빌드.
SOURCE_STAGE = XR_CONTENT_ROOT / 'Assets' / 'XR' / 'Stages' / 'robot_capstone.usd'

EE_CAMERA_PATH = '/Franka/panda_hand/EEViewCameraMount/CameraRig/CameraFrame/EEViewCamera'
GLASS_PRIM_PATH = '/World/CapstoneAdditions/TabletopItems/Glass'
GLASS_VISUAL_PATH = GLASS_PRIM_PATH + '/Visual'
OPAQUE_MAT_PATH = '/World/DataGen/CupOpaqueMat'
RESOLUTION = (640, 480)
DIP = 'distance_to_image_plane'
MIN_D, MAX_D = 0.05, 15.0

# Panda "ready" 포즈 (config/robot_defaults.yaml) — MoveIt/BT 가 로봇을 이 자세로 홈시킨다.
# 이 포즈여야 EE 카메라가 테이블을 ~0.4m 에서 내려다본다(배포 분포와 일치).
HOME_JOINTS = {'panda_joint1': 0.0, 'panda_joint2': -0.785, 'panda_joint3': 0.0,
               'panda_joint4': -2.356, 'panda_joint5': 0.0, 'panda_joint6': 1.571,
               'panda_joint7': 0.785}


def log(m):
    print(m, flush=True)


class Capture:
    """EE 카메라 render product 에 DIP/rgb/camera_params annotator 부착."""
    def __init__(self, camera_path, resolution, sim=None):
        self.sim = sim  # SimulationContext 존재 시 rep.orchestrator 대신 sim.step 으로 렌더
        self.sensor = SingleViewDepthSensor(prim_path=camera_path, resolution=resolution)
        self.sensor.initialize(attach_rgb_annotator=False)
        self.sensor.set_enabled(False)  # DSD AOV 안 만들게 — DIP 만 쓴다
        self.rp = self.sensor.get_render_product_path()
        self.ann = {}
        for name in (DIP, 'rgb', 'camera_params'):
            a = rep.AnnotatorRegistry.get_annotator(name)
            a.attach([self.rp])
            self.ann[name] = a

    def step(self, n):
        # pause_timeline=False 여야 material 토글이 render 에 반영된다(True 면 render 가 얼어
        # 투명/opaque 가 동일 프레임이 됨). arm 은 홈 드라이브 타겟이 잡아주므로 타임라인이
        # 흘러도 카메라는 정지 상태를 유지한다.
        for _ in range(n):
            if self.sim is not None:
                self.sim.step(render=True)
                continue
            try:
                rep.orchestrator.step(rt_subframes=1, pause_timeline=False)
            except Exception:  # noqa: BLE001
                simulation_app.update()

    def wait_ready(self, max_frames=200):
        for i in range(max_frames):
            self.step(1)
            try:
                a = np.asarray(self.ann[DIP].get_data())
            except Exception:  # noqa: BLE001
                continue
            if a.ndim >= 2 and a.size > 0:
                return i + 1
        return -1

    def grab_dip(self):
        return np.asarray(self.ann[DIP].get_data(), dtype=np.float32)

    def grab_rgb(self):
        a = np.asarray(self.ann['rgb'].get_data())
        if a.ndim == 3 and a.shape[2] >= 3:
            return np.ascontiguousarray(a[:, :, :3].astype(np.uint8))
        return None

    def K(self):
        cp = self.ann['camera_params'].get_data()
        w, h = RESOLUTION
        f = float(np.asarray(cp['cameraFocalLength']).reshape(-1)[0])
        ap = np.asarray(cp['cameraAperture']).reshape(-1)
        fx, fy = f / float(ap[0]) * w, f / float(ap[1]) * h
        return np.array([[fx, 0, w / 2.0], [0, fy, h / 2.0], [0, 0, 1]], dtype=np.float64)

    def cup_uv(self, stage):
        prim = stage.GetPrimAtPath(GLASS_PRIM_PATH)
        t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        w = np.array([[float(t[0]), float(t[1]), float(t[2])]], dtype=np.float32)
        c = self.sensor.get_image_coords_from_world_points(w)
        if c is None or len(c) == 0:
            return None
        return int(c[0][0]), int(c[0][1])


def make_opaque_material(stage):
    mat = UsdShade.Material.Define(stage, OPAQUE_MAT_PATH)
    sh = UsdShade.Shader.Define(stage, OPAQUE_MAT_PATH + '/Shader')
    sh.CreateIdAttr('UsdPreviewSurface')
    sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.55, 0.6, 0.7))
    sh.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(1.0)
    sh.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
    sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.8)
    sh.CreateInput('ior', Sdf.ValueTypeNames.Float).Set(1.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
    return mat


def cup_bind_targets(stage):
    """컵 Visual 서브트리의 메시/Gprim prim 들 — 여기에 opaque 를 바인딩/언바인딩.
    ★ 참조가 instanceable 이면 PrimRange 가 내부 메시로 못 들어가고 인스턴스 프록시엔
    머티리얼 바인딩이 무시된다 → 먼저 instanceable 해제해서 실제 메시를 노출시킨다."""
    root = stage.GetPrimAtPath(GLASS_VISUAL_PATH)
    root.SetInstanceable(False)
    for p in Usd.PrimRange(root):
        if p.IsInstanceable():
            p.SetInstanceable(False)
    targets = [root]
    for p in Usd.PrimRange(root):
        if p.IsA(UsdGeom.Mesh) or p.IsA(UsdGeom.Gprim):
            targets.append(p)
    return targets


def set_opaque(targets, mat, on):
    for p in targets:
        b = UsdShade.MaterialBindingAPI.Apply(p)
        if on:
            b.Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        else:
            b.UnbindDirectBinding()


def set_cup_xy(stage, base_xyz, dx, dy):
    prim = stage.GetPrimAtPath(GLASS_PRIM_PATH)
    attr = prim.GetAttribute('xformOp:translate')
    if attr and attr.IsValid():
        attr.Set(Gf.Vec3d(base_xyz[0] + dx, base_xyz[1] + dy, base_xyz[2]))
        return True
    ok = UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(base_xyz[0] + dx, base_xyz[1] + dy, base_xyz[2]))
    return bool(ok)


def set_franka_home_drive(stage):
    """USD 드라이브 타겟을 홈 포즈로 설정. SimulationContext 는 쓰지 않는다(그게 있으면
    replicator annotator 가 얼어 material 토글이 render 에 안 잡힌다). 타임라인을 play 하면
    드라이브가 arm 을 여기로 몰아 붙잡는다."""
    franka = stage.GetPrimAtPath('/Franka')
    n = 0
    for p in Usd.PrimRange(franka):
        name = p.GetName()
        if name not in HOME_JOINTS:
            continue
        drive = UsdPhysics.DriveAPI.Get(p, 'angular')
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(p, 'angular')
        deg = math.degrees(HOME_JOINTS[name])  # USD angular drive 타겟은 도(degree)
        drive.CreateTargetPositionAttr().Set(deg)
        n += 1
    log(f'드라이브 홈타겟 설정 {n}개 관절')
    return n


def valid(d):
    return np.isfinite(d) & (d > MIN_D) & (d < MAX_D)


def save_preview(path, rgb, input_dip, gt_dip, mask):
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001
        return
    def cmap(d):
        m = valid(d); v = d.copy()
        lo, hi = np.percentile(v[m], [2, 98]) if m.any() else (0, 1)
        n = np.clip((v - lo) / (hi - lo + 1e-9), 0, 1); n[~m] = 0
        r = np.clip(1.5 - abs(4 * n - 3), 0, 1); g = np.clip(1.5 - abs(4 * n - 2), 0, 1); b = np.clip(1.5 - abs(4 * n - 1), 0, 1)
        img = (np.stack([r, g, b], -1) * 255).astype(np.uint8); img[~m] = [40, 40, 40]
        return img
    mk = np.zeros((*mask.shape, 3), np.uint8); mk[mask] = [255, 60, 60]
    row = [rgb if rgb is not None else np.zeros((*input_dip.shape, 3), np.uint8),
           cmap(input_dip), cmap(gt_dip), mk]
    Image.fromarray(np.concatenate(row, axis=1)).save(path)


def main():
    stage_path = Path(args.stage) if args.stage else SOURCE_STAGE
    if not stage_path.exists():
        log(f'stage 없음: {stage_path} — run_capstone_scene.sh 로 씬 생성 필요'); return 1
    log(f'SOURCE stage: {stage_path}')
    open_stage(str(stage_path))
    stage = omni.usd.get_context().get_stage()
    n = 0
    while is_stage_loading() and n < 3000:
        simulation_app.update(); n += 1
    log(f'SOURCE 로딩 완료 ({n} 프레임)')

    # GUI 와 동일하게 씬 빌드 — 카메라/컵/물체 생성.
    # ⚠️ setup_initial_scene.py 는 마지막 줄에 `asyncio.ensure_future(main())` 가 있어
    # import 만 해도 setup 의 async main() 이 스케줄돼 스테이지를 다시 열고 apply_scene 을
    # 또 돌린다(그래프 중복·스테이지 재오픈으로 캡처 오염). import 동안 ensure_future 를
    # 무력화해 그 자동 실행을 막는다.
    sys.path.insert(0, str(SIM_DIR))
    import asyncio
    _orig_ef = asyncio.ensure_future
    asyncio.ensure_future = lambda *a, **k: None
    try:
        import setup_initial_scene as sis  # noqa: E402
    finally:
        asyncio.ensure_future = _orig_ef
    # ROS 브리지/조인트 그래프는 헤드리스에서 노드타입이 없어 실패한다 — no-op 로 대체해
    # apply_scene 이 끝까지 깨끗이 돌게 한다(카메라/물체 생성엔 영향 없음).
    sis.build_ee_view_bridge = lambda *a, **k: None
    sis.build_top_view_bridge = lambda *a, **k: None
    sis.create_ros2_joint_graph = lambda *a, **k: None
    try:
        sis.apply_scene()
    except Exception as e:  # noqa: BLE001
        log(f'apply_scene 일부 실패(무시): {type(e).__name__}: {e}')
    for _ in range(30):
        simulation_app.update()
    log('apply_scene 완료 — 씬 빌드됨')

    # 로봇을 ready 포즈로 — 안 하면 EE 카메라가 방을 가로질러(테이블 2.5m) 봐서
    # 컵이 작고 마스크가 먼 배경 노이즈를 잡는다. 홈 포즈면 테이블 ~0.4m 정면.
    set_franka_home_drive(stage)
    # 타임라인 play → 물리로 드라이브가 arm 을 홈으로 이동/유지. 정착까지 스텝.
    omni.timeline.get_timeline_interface().play()
    for _ in range(150):
        simulation_app.update()
    log('arm 홈 정착 완료 (timeline playing)')

    if not stage.GetPrimAtPath(EE_CAMERA_PATH).IsValid():
        log(f'EE 카메라 없음: {EE_CAMERA_PATH}'); return 1
    cup_prim = stage.GetPrimAtPath(GLASS_PRIM_PATH)
    if not cup_prim.IsValid():
        log(f'컵 없음: {GLASS_PRIM_PATH}'); return 1

    cap = Capture(EE_CAMERA_PATH, RESOLUTION)  # sim=None → rep.orchestrator 캡처(live 갱신)
    if cap.wait_ready() < 0:
        log('annotator 데이터 안 나옴 — 렌더 파이프라인 문제'); return 1
    K = cap.K()
    log(f'K: fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}')

    mat = make_opaque_material(stage)
    targets = cup_bind_targets(stage)
    log(f'opaque 바인딩 대상 prim {len(targets)}개')

    base = cup_prim.GetAttribute('xformOp:translate').Get()
    base_xyz = (float(base[0]), float(base[1]), float(base[2])) if base else (0, 0, 0)
    log(f'컵 base translate: {base_xyz}')

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    saved = 0
    for i in range(args.frames):
        dx = dy = 0.0
        if i > 0 and args.jitter > 0:
            dx = rng.uniform(-args.jitter, args.jitter)
            dy = rng.uniform(-args.jitter, args.jitter)
        set_cup_xy(stage, base_xyz, dx, dy)

        # 1) 투명 입력 (컵 손 안 댐)
        set_opaque(targets, mat, False)
        cap.step(args.warmup)
        input_dip = cap.grab_dip()
        rgb = cap.grab_rgb()
        uv = cap.cup_uv(stage)
        if uv is None or not (0 <= uv[0] < RESOLUTION[0] and 0 <= uv[1] < RESOLUTION[1]):
            log(f'[{i}] 컵 시야 밖 (uv={uv}) — 스킵'); continue

        # 2) opaque GT
        set_opaque(targets, mat, True)
        cap.step(args.warmup)
        gt_dip = cap.grab_dip()

        # 3) 투명 복원(다음 프레임 대비)
        set_opaque(targets, mat, False)

        # 4) 마스크 = 컵이 솟은(=가까워진) 픽셀
        m = valid(input_dip) & valid(gt_dip)
        rise = np.zeros_like(gt_dip); rise[m] = input_dip[m] - gt_dip[m]
        # 컵은 최대 ~150mm 솟는다. 상한 250mm 로 먼 배경 depth 노이즈(수백mm~수m) 제거.
        mask = (rise > args.mask_thresh) & (rise < 0.25)
        cup_px = int(mask.sum())
        if cup_px < 50:
            r = rise[m]
            uz_in = float(input_dip[uv[1], uv[0]]); uz_gt = float(gt_dip[uv[1], uv[0]])
            log(f'[{i}] 마스크부족({cup_px}) rise: max={float(r.max())*1000:.1f}mm '
                f'>1mm={int((r > 0.001).sum())} >5mm={int((r > 0.005).sum())} | '
                f'cup uv={uv} depth in={uz_in:.3f} gt={uz_gt:.3f} Δ={(uz_in - uz_gt) * 1000:.1f}mm')
            save_preview(out / f'skip_{i:05d}.png', rgb, input_dip, gt_dip, mask)
            continue
        rise_mm = float(rise[mask].max() * 1000)
        table = float(np.median(input_dip[valid(input_dip)]))

        np.savez_compressed(out / f'frame_{i:05d}.npz',
                            input_dip=input_dip, gt_dip=gt_dip, mask=mask, rgb=rgb,
                            K=K, cup_uv=np.array(uv), cup_dxy=np.array([dx, dy]))
        if saved < args.preview:
            save_preview(out / f'preview_{i:05d}.png', rgb, input_dip, gt_dip, mask)
        saved += 1
        log(f'[{i}] uv={uv} 컵마스크={cup_px}px 최대솟음={rise_mm:.1f}mm 테이블={table:.3f}m → 저장')

    log(f'\n완료: {saved}/{args.frames} 프레임 → {out}')
    return 0


if __name__ == '__main__':
    code = 1
    try:
        code = main()
    except Exception:  # noqa: BLE001
        import traceback; traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
    sys.stdout.flush(); sys.stderr.flush()
    simulation_app.close()
    sys.exit(code)
