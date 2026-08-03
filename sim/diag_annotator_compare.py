#!/usr/bin/env python3
"""Phase 1 진단 — distance_to_image_plane vs DepthSensorDistance 대조.

SwinDRNet 이 우리 프레임에서 depth 를 1~2mm 밖에 안 바꾸는 이유가
"모델이 우리 도메인을 모른다"가 아니라 "고칠 게 없는 입력을 줬다"일 수 있다.
DREDS 는 시뮬 스테레오 센서 depth(노이즈·홀 포함) 로 학습됐는데, 우리가 넣는
distance_to_image_plane 은 레이트레이싱된 무결점 Z-depth 라 사실상 GT 다.

이 스크립트가 답하는 것:
  1. 규약     — DepthSensorDistance 가 Z-depth 인가 radial 인가?
                radial 이면 compute_xyz 핀홀 백프로젝션과 안 맞아 데이터셋이 오염된다.
  2. 커버리지 — minDistance 기본 0.5m 가 우리 작업거리(테이블 0.33~0.54m)를 죽이는가?
  3. 유리컵   — 컵 영역이 see-through 인가 hole 인가?
  4. 노이즈   — 평평한 테이블에서 (DSD - DIP) 의 표준편차.
                ★ 가설의 사활이 걸린 지표. 0 에 가까우면 DIP 와 다를 게 없다는
                뜻이고, 그러면 입력 도메인 가설은 기각이다.
  + intrinsics 대조 — depthSensor:focalLengthPixel(기본 897) vs 우리 실제 fx(≈183)
  + 간섭 확인       — 센서를 켜면 기존 DIP 출력이 오염되는가?
                      (오염되면 Phase 3 ROS 배선이 위험해진다)

결과는 npz 로 저장 → Phase 2 에서 SwinDRNet 에 그대로 투입해 Δ 재측정.

usage:
  ./isaacsim/python.sh sim/diag_annotator_compare.py
  ./isaacsim/python.sh sim/diag_annotator_compare.py --gui
  ./isaacsim/python.sh sim/diag_annotator_compare.py --out /tmp/cmp.npz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SIM_DIR.parent

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--gui', action='store_true', help='GUI 로 실행 (기본 headless)')
parser.add_argument('--out', default=str(SIM_DIR / 'diag_out' / 'annotator_compare.npz'),
                    help='npz 저장 경로')
parser.add_argument('--warmup', type=int, default=20, help='설정 변경 후 렌더 안정화 프레임 수')
parser.add_argument('--stage', default=None, help='USD stage 경로 (기본: robot_capstone_scene.usd)')
args = parser.parse_args()

# SimulationApp 은 다른 omni import 보다 반드시 먼저 생성해야 한다.
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({'headless': not args.gui})

# simulation_app.close() 는 프로세스를 하드 종료시켜 버퍼에 남은 stdout 을 통째로
# 날린다. 파일로 리다이렉트하면 stdout 이 block-buffered 라 진단 출력이 전부 유실됐다.
# 줄 단위 flush 로 강제해 그때그때 나가게 한다.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from isaacsim.core.utils.stage import open_stage  # noqa: E402
from isaacsim.sensors.camera import SingleViewDepthSensor  # noqa: E402
from pxr import UsdGeom  # noqa: E402
import omni.usd  # noqa: E402

sys.path.insert(0, str(SIM_DIR))

# setup_initial_scene 과 상수를 공유해 경로가 어긋나지 않게 한다.
DOWNLOADS_DIR = Path(os.environ.get('ROBOT_CAPSTONE_DOWNLOADS_DIR', Path.home() / 'Downloads')).expanduser()
XR_CONTENT_ROOT = Path(
    os.environ.get('ROBOT_CAPSTONE_XR_CONTENT_ROOT', DOWNLOADS_DIR / 'XR_Content_NVD@10010')
).expanduser()
DEFAULT_STAGE = XR_CONTENT_ROOT / 'Assets' / 'XR' / 'Stages' / 'robot_capstone_scene.usd'

EE_CAMERA_PATH = '/Franka/panda_hand/EEViewCameraMount/CameraRig/CameraFrame/EEViewCamera'
GLASS_PRIM_PATH = '/World/CapstoneAdditions/TabletopItems/Glass'
RESOLUTION = (640, 480)

DIP = 'distance_to_image_plane'
DSD = 'DepthSensorDistance'

# 우리 파이프라인(graspgen/projector)이 쓰는 유효 depth 범위.
MIN_D, MAX_D = 0.05, 15.0


def hr(title: str) -> None:
    print('\n' + '=' * 78)
    print(f'  {title}')
    print('=' * 78)


def valid_mask(d: np.ndarray) -> np.ndarray:
    return np.isfinite(d) & (d >= MIN_D) & (d <= MAX_D)


def squeeze2d(a, name: str = ''):
    """(H,W,1) / (H,W) → (H,W) float32.

    annotator 가 아직 안 찼으면 빈 1-D 배열이 온다. 그걸 그대로 흘려보내면
    한참 뒤 엉뚱한 곳에서 터지므로 여기서 None 으로 걸러 보고한다.
    """
    if a is None:
        return None
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[:, :, 0]
    if a.ndim != 2 or a.size == 0:
        print(f'  [경고] {name or "annotator"} 데이터 형상 이상: shape={a.shape} — 미수신 취급')
        return None
    return a.astype(np.float32)


def extract_rgb(a):
    """rgb annotator → (H,W,3) uint8. 미수신/형상 이상이면 None."""
    if a is None:
        return None
    arr = np.asarray(a)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        print(f'  [경고] rgb 형상 이상: shape={arr.shape} — 미수신 취급')
        return None
    return arr[:, :, :3]


class Capture:
    """EE 카메라 render product 에 두 annotator 를 동시에 붙여 같은 프레임을 뜬다."""

    def __init__(self, camera_path: str, resolution):
        self.sensor = SingleViewDepthSensor(prim_path=camera_path, resolution=resolution)
        # initialize() 가 OmniSensorDepthSensorSingleViewAPI 를 render product 에
        # 적용하고 set_enabled(True) 까지 한다. 지금 setup_initial_scene.py 는
        # add_template_render_product() 만 부르고 이걸 안 불러서 센서가 죽어 있다.
        self.sensor.initialize(attach_rgb_annotator=False)
        self.rp = self.sensor.get_render_product_path()

        # DSD 는 센서가 켜진 뒤에 붙인다. 꺼진 상태에서 붙여두면 렌더러가 그 AOV 를
        # 안 만들어서 매 프레임 "Failed to export AOV" 가 쏟아진다.
        self.ann = {}
        for name in (DIP, 'rgb', 'camera_params'):
            self._attach(name)

    def _attach(self, name: str) -> None:
        a = rep.AnnotatorRegistry.get_annotator(name)
        a.attach([self.rp])
        self.ann[name] = a

    def attach_dsd(self) -> None:
        if DSD not in self.ann:
            self._attach(DSD)

    def step(self, n: int) -> None:
        # rep.orchestrator.step() 이 replicator 파이프라인을 제대로 돌린다.
        # simulation_app.update() 만으로는 annotator 버퍼가 안 차는 경우가 있다.
        for _ in range(n):
            try:
                rep.orchestrator.step(rt_subframes=1, pause_timeline=False)
            except Exception:  # noqa: BLE001
                simulation_app.update()

    def wait_ready(self, max_frames: int = 200) -> int:
        """annotator 가 실제 (H,W) 데이터를 뱉을 때까지 스텝.

        replicator annotator 는 데이터가 차기 전까지 빈 1-D 배열을 반환한다.
        고정 프레임수만 돌리고 get_data() 하면 그 빈 배열을 잡게 된다.
        """
        for i in range(max_frames):
            self.step(1)
            try:
                a = np.asarray(self.ann[DIP].get_data())
            except Exception:  # noqa: BLE001
                continue
            if a.ndim >= 2 and a.size > 0:
                return i + 1
        return -1

    def grab(self) -> dict:
        out = {}
        for name, a in self.ann.items():
            try:
                out[name] = a.get_data()
            except Exception as e:  # noqa: BLE001
                print(f'  [경고] annotator {name} get_data 실패: {e}')
                out[name] = None
        return out


def intrinsics_from_usd(stage, camera_path: str, resolution):
    """USD 속성에서 손으로 계산한 K. camera_params 와 대조용."""
    cam = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
    if not cam:
        return None
    f = float(cam.GetFocalLengthAttr().Get())
    ha = float(cam.GetHorizontalApertureAttr().Get())
    va = float(cam.GetVerticalApertureAttr().Get())
    w, h = resolution
    return {
        'focal_mm': f, 'h_aperture_mm': ha, 'v_aperture_mm': va,
        'fx': f / ha * w, 'fy': f / va * h, 'cx': w / 2.0, 'cy': h / 2.0,
    }


def intrinsics_from_annotator(cam_params, resolution):
    """camera_params annotator 가 주는 값 = 렌더러가 실제로 쓴 값 (권위값)."""
    if not cam_params:
        return None
    try:
        w, h = resolution
        f = float(np.asarray(cam_params['cameraFocalLength']).reshape(-1)[0])
        ap = np.asarray(cam_params['cameraAperture']).reshape(-1)
        ha, va = float(ap[0]), float(ap[1])
        return {'focal_mm': f, 'h_aperture_mm': ha, 'v_aperture_mm': va,
                'fx': f / ha * w, 'fy': f / va * h, 'cx': w / 2.0, 'cy': h / 2.0}
    except Exception as e:  # noqa: BLE001
        print(f'  [경고] camera_params 해석 실패: {e}  keys={list(cam_params)}')
        return None


def radial_factor(K, resolution):
    """각 픽셀의 (radial 거리 / Z-depth) 비율.

    radial = Z * sqrt(1 + ((u-cx)/fx)^2 + ((v-cy)/fy)^2)
    DepthSensorDistance 가 radial 이면 DSD/DIP 가 이 값과 일치한다.
    """
    w, h = resolution
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x = (uu - K['cx']) / K['fx']
    y = (vv - K['cy']) / K['fy']
    return np.sqrt(1.0 + x ** 2 + y ** 2)


def roi_median(a: np.ndarray, m: np.ndarray, cy: int, cx: int, r: int = 20):
    """(cy,cx) 중심 2r x 2r 창에서 유효 픽셀 중앙값."""
    h, w = a.shape
    y0, y1 = max(0, cy - r), min(h, cy + r)
    x0, x1 = max(0, cx - r), min(w, cx + r)
    patch, pm = a[y0:y1, x0:x1], m[y0:y1, x0:x1]
    if not pm.any():
        return None
    return float(np.median(patch[pm]))


# ─────────────────────────────────────────────────────────────────────────────
# 진단 본체
# ─────────────────────────────────────────────────────────────────────────────

def analyze_convention(dip, dsd, K, resolution):
    hr('1. 규약 — DepthSensorDistance 는 Z-depth 인가 radial 인가?')
    m = valid_mask(dip) & valid_mask(dsd)
    if m.sum() < 100:
        print('  유효 픽셀 부족 — 판정 불가')
        return None

    ratio = np.where(m, dsd / np.maximum(dip, 1e-6), np.nan)
    rf = radial_factor(K, resolution)
    h, w = dip.shape

    print(f'  유효 픽셀 {m.sum()} / {m.size}')
    print(f'  {"위치":<12}{"DSD/DIP 실측":>14}{"radial 예측":>14}{"Z 예측":>10}')
    spots = [
        ('중심',      h // 2, w // 2),
        ('좌상 코너', int(h * 0.12), int(w * 0.12)),
        ('우상 코너', int(h * 0.12), int(w * 0.88)),
        ('좌하 코너', int(h * 0.88), int(w * 0.12)),
        ('우하 코너', int(h * 0.88), int(w * 0.88)),
    ]
    rows = []
    for name, y, x in spots:
        meas = roi_median(ratio, m & np.isfinite(ratio), y, x)
        pred = float(rf[y, x])
        rows.append((name, meas, pred))
        ms = f'{meas:.4f}' if meas is not None else '  --  '
        print(f'  {name:<12}{ms:>14}{pred:>14.4f}{1.0:>10.4f}')

    # 코너에서 두 가설의 예측이 가장 크게 갈린다 → 거기서 판정.
    corners = [(m_, p_) for n_, m_, p_ in rows[1:] if m_ is not None]
    if not corners:
        print('\n  판정: 코너 유효 픽셀 없음 — 불가')
        return None
    err_z = float(np.mean([abs(mm - 1.0) for mm, _ in corners]))
    err_r = float(np.mean([abs(mm - pp) for mm, pp in corners]))
    print(f'\n  코너 평균오차 — Z 가설: {err_z:.4f}   radial 가설: {err_r:.4f}')
    if err_z < err_r:
        verdict = 'Z-depth'
        print('  ✅ 판정: Z-depth — distance_to_image_plane 과 같은 규약.')
        print('     compute_xyz 핀홀 백프로젝션에 그대로 넣어도 된다.')
    else:
        verdict = 'radial'
        print('  ⚠️  판정: radial — distance_to_camera 계열!')
        print('     그대로 쓰면 코너에서 체계적 오차. 반드시 Z 로 변환 후 사용:')
        print('       Z = DSD / sqrt(1 + ((u-cx)/fx)^2 + ((v-cy)/fy)^2)')
    return verdict


def analyze_coverage(dip, dsd, label: str):
    hr(f'2. 커버리지 — {label}')
    for name, d in ((DIP, dip), (DSD, dsd)):
        if d is None:
            print(f'  {name:<26} 데이터 없음')
            continue
        v = valid_mask(d)
        pct = 100.0 * v.sum() / v.size
        zero = 100.0 * float((d == 0).sum()) / d.size
        nonfin = 100.0 * float((~np.isfinite(d)).sum()) / d.size
        rng = f'{d[v].min():.3f}~{d[v].max():.3f}m' if v.any() else 'n/a'
        print(f'  {name:<26} valid {pct:5.1f}%   zero {zero:5.1f}%   '
              f'inf/nan {nonfin:5.1f}%   범위 {rng}')
    if dsd is not None:
        v = valid_mask(dsd)
        if v.sum() / v.size < 0.2:
            print('\n  ⚠️  DSD 커버리지가 20% 미만 — minDistance/maxDisparityPixel 이')
            print('      우리 작업거리(0.33~0.54m)를 잘라내고 있을 가능성이 크다.')


def analyze_noise(dip, dsd):
    hr('4. 노이즈 프로파일 ★ 가설의 사활이 걸린 지표')
    m = valid_mask(dip) & valid_mask(dsd)
    if m.sum() < 500:
        print('  유효 픽셀 부족 — 판정 불가')
        return None

    # 테이블 평면 = DIP 중앙값 근처 픽셀. 물체/배경을 배제해 평면만 남긴다.
    table = float(np.median(dip[m]))
    flat = m & (np.abs(dip - table) < 0.01)
    print(f'  테이블 평면 ≈ {table:.3f}m,  평면 픽셀 {flat.sum()}개')
    if flat.sum() < 200:
        print('  평면 픽셀 부족 — 판정 불가')
        return None

    # 평면의 기하 기울기를 제거하려고 std(DIP) 가 아니라 std(DSD - DIP) 를 본다.
    # 이게 순수 센서 노이즈다.
    diff = (dsd - dip)[flat]
    bias, sigma = float(np.mean(diff)), float(np.std(diff))
    print(f'  DIP 자체 표준편차     : {float(np.std(dip[flat])) * 1000:8.3f} mm  (평면 기울기 포함)')
    print(f'  (DSD - DIP) 평균(bias): {bias * 1000:8.3f} mm')
    print(f'  (DSD - DIP) 표준편차  : {sigma * 1000:8.3f} mm  ← 센서 노이즈')

    print()
    if sigma * 1000 < 0.5:
        print('  ❌ 노이즈가 0.5mm 미만 — DSD 가 DIP 와 사실상 동일하다.')
        print('     입력 도메인 가설 기각. 센서 시뮬이 안 켜졌거나 노이즈 파라미터가')
        print('     죽어 있는지 먼저 확인하고, 그래도 같으면 fine-tuning 으로 간다.')
    else:
        print(f'  ✅ 센서 노이즈 {sigma * 1000:.2f}mm 확인 — DIP 와 명확히 다른 도메인이다.')
        print('     DREDS 입력(시뮬 스테레오)에 근접했을 가능성. Phase 2 로 진행해')
        print('     SwinDRNet Δ 가 1~2mm 를 벗어나는지 확인할 것.')
    return sigma


def analyze_glass(cap: Capture, dip, dsd, stage):
    hr('3. 유리컵 — see-through 인가 hole 인가?')
    prim = stage.GetPrimAtPath(GLASS_PRIM_PATH)
    if not prim or not prim.IsValid():
        print(f'  유리컵 prim 없음: {GLASS_PRIM_PATH} — 건너뜀')
        return
    from pxr import Usd
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = xf.ExtractTranslation()
    world = np.array([[float(t[0]), float(t[1]), float(t[2])]], dtype=np.float32)
    print(f'  유리컵 world 위치: {world[0]}')

    try:
        coords = cap.sensor.get_image_coords_from_world_points(world)
    except Exception as e:  # noqa: BLE001
        print(f'  이미지 좌표 투영 실패: {e} — 건너뜀')
        return
    if coords is None or len(coords) == 0:
        print('  투영 결과 없음 — 카메라 시야 밖일 수 있음')
        return

    u, v = int(coords[0][0]), int(coords[0][1])
    h, w = dip.shape
    print(f'  이미지 좌표: (u={u}, v={v})   해상도 {w}x{h}')
    if not (0 <= u < w and 0 <= v < h):
        print('  ⚠️  시야 밖 — EE 카메라가 유리컵을 안 보고 있다. 씬/포즈 확인 필요.')
        return

    mdip, mdsd = valid_mask(dip), valid_mask(dsd)
    table = float(np.median(dip[mdip])) if mdip.any() else float('nan')
    d_dip = roi_median(dip, mdip, v, u, r=10)
    d_dsd = roi_median(dsd, mdsd, v, u, r=10)
    print(f'  테이블 평면 depth      : {table:.3f} m')
    print(f'  컵 중심 {DIP:<24}: {d_dip if d_dip is None else f"{d_dip:.3f} m"}')
    print(f'  컵 중심 {DSD:<24}: {d_dsd if d_dsd is None else f"{d_dsd:.3f} m"}')
    print()
    if d_dsd is None:
        print('  → DSD 가 컵 영역에서 무효(hole). 실제 depth 센서의 투명물체 실패와 유사.')
    elif d_dip is not None and abs(d_dsd - d_dip) < 0.005:
        print('  → DSD 도 DIP 와 같은 see-through. 스테레오 시뮬이 레이트레이싱 depth 를')
        print('     그대로 쓰는 듯 — 유리 실패 모드는 추가 안 됨(노이즈만 붙음).')
    else:
        print('  → DSD 가 DIP 와 다른 값. 스테레오 매칭이 컵 표면을 잡았을 수 있다.')


def analyze_intrinsics(cap: Capture, stage, cam_params, resolution):
    hr('+ intrinsics 대조 — depthSensor 가 우리 카메라와 같은 렌즈를 가정하는가?')
    k_usd = intrinsics_from_usd(stage, EE_CAMERA_PATH, resolution)
    k_ann = intrinsics_from_annotator(cam_params, resolution)
    for name, K in (('USD 속성 계산', k_usd), ('camera_params(권위)', k_ann)):
        if K is None:
            print(f'  {name:<22} 해석 실패')
            continue
        print(f'  {name:<22} focal={K["focal_mm"]:.3f}mm  aperture={K["h_aperture_mm"]:.2f}'
              f'x{K["v_aperture_mm"]:.2f}mm  fx={K["fx"]:.1f}  fy={K["fy"]:.1f}')

    fl_px = cap.sensor.get_focal_length_pixel()
    ss_px = cap.sensor.get_sensor_size_pixel()
    print(f'\n  depthSensor:focalLengthPixel = {fl_px}')
    print(f'  depthSensor:sensorSizePixel  = {ss_px}   (우리 렌더 폭 {resolution[0]})')

    K = k_ann or k_usd
    if K and fl_px:
        ratio = float(fl_px) / K['fx']
        print(f'\n  focalLengthPixel / 실제 fx = {ratio:.2f}x')
        if abs(ratio - 1.0) > 0.05:
            print('  ⚠️  불일치. 센서 시뮬이 가정하는 렌즈와 실제 렌더 렌즈가 다르다.')
            print(f'     → set_focal_length_pixel({K["fx"]:.1f}) 로 맞춰야 Z 스케일이 산다.')
    if ss_px and int(ss_px) != int(resolution[0]):
        print(f'  ⚠️  sensorSizePixel({ss_px}) != 렌더 폭({resolution[0]}) → set_sensor_size_pixel({resolution[0]})')
    return K


def analyze_interference(dip_off, dip_on):
    hr('+ 간섭 — 센서를 켜면 기존 distance_to_image_plane 이 오염되는가?')
    if dip_off is None or dip_on is None:
        print('  비교 데이터 없음')
        return
    m = valid_mask(dip_off) & valid_mask(dip_on)
    if not m.any():
        print('  공통 유효 픽셀 없음')
        return
    d = np.abs(dip_on - dip_off)[m]
    print(f'  |DIP(on) - DIP(off)|  최대 {d.max() * 1000:.4f} mm   평균 {d.mean() * 1000:.4f} mm')
    if d.max() * 1000 < 0.1:
        print('  ✅ 영향 없음 — 센서를 켜도 기존 ROS depth 경로는 그대로다.')
        print('     Phase 3 에서 두 토픽을 병행 발행해도 안전.')
    else:
        print('  ⚠️  DIP 출력이 바뀐다! 센서를 켜면 현재 파이프라인이 영향을 받는다.')
        print('     rgbDepthOutputMode 등을 확인하고, Phase 3 배선을 신중히 할 것.')


def main() -> int:
    stage_path = Path(args.stage) if args.stage else DEFAULT_STAGE
    hr('Phase 1 — annotator 대조 진단')
    print(f'  stage : {stage_path}')
    print(f'  camera: {EE_CAMERA_PATH}')
    if not stage_path.exists():
        print(f'\n  stage 파일 없음: {stage_path}')
        print('  ./run_capstone_scene.sh 로 씬을 한 번 생성했는지 확인할 것.')
        return 1

    open_stage(str(stage_path))
    stage = omni.usd.get_context().get_stage()

    # 에셋 스트리밍이 끝나기 전에 렌더하면 annotator 가 빈 채로 돌아온다.
    from isaacsim.core.utils.stage import is_stage_loading
    n = 0
    while is_stage_loading() and n < 2000:
        simulation_app.update()
        n += 1
    print(f'  스테이지 로딩 완료 ({n} 프레임 대기)')

    if not stage.GetPrimAtPath(EE_CAMERA_PATH).IsValid():
        print(f'\n  EE 카메라 prim 없음: {EE_CAMERA_PATH}')
        return 1

    cap = Capture(EE_CAMERA_PATH, RESOLUTION)

    # ── A. 센서 OFF — 기준 DIP ────────────────────────────────────────────
    cap.sensor.set_enabled(False)
    ready = cap.wait_ready()
    if ready < 0:
        print('\n  annotator 가 끝내 데이터를 안 뱉음 — 렌더 파이프라인 문제. 중단.')
        return 1
    print(f'  annotator 데이터 수신 ({ready} 프레임 소요)')
    cap.step(args.warmup)
    a = cap.grab()
    dip_off = squeeze2d(a.get(DIP), 'DIP(off)')
    rgb = extract_rgb(a.get('rgb'))
    K = analyze_intrinsics(cap, stage, a.get('camera_params'), RESOLUTION)

    # ── B. 센서 ON, 기본 파라미터 ─────────────────────────────────────────
    cap.sensor.set_enabled(True)
    cap.attach_dsd()
    cap.step(args.warmup)
    b = cap.grab()
    dip_on, dsd_default = squeeze2d(b.get(DIP), 'DIP(on)'), squeeze2d(b.get(DSD), 'DSD(기본)')
    analyze_interference(dip_off, dip_on)
    analyze_coverage(dip_on, dsd_default, '센서 ON, 기본 파라미터 (minDistance=0.5)')

    # ── C. 센서 ON, 우리 작업거리에 맞게 튜닝 ─────────────────────────────
    # minDistance 기본 0.5m 는 테이블(0.33~0.54m)을 잘라먹는다. focalLengthPixel
    # 도 우리 실제 fx 로 맞춰야 Z 스케일이 산다.
    cap.sensor.set_min_distance(0.05)
    cap.sensor.set_max_distance(10.0)
    if K:
        cap.sensor.set_focal_length_pixel(float(K['fx']))
    cap.sensor.set_sensor_size_pixel(int(RESOLUTION[0]))
    cap.step(args.warmup)
    c = cap.grab()
    dip_t, dsd_tuned = squeeze2d(c.get(DIP), 'DIP(튜닝)'), squeeze2d(c.get(DSD), 'DSD(튜닝)')
    analyze_coverage(dip_t, dsd_tuned, '센서 ON, 튜닝 (minDistance=0.05, fx 정합)')

    # 튜닝본으로 본 진단 — 커버리지가 살아난 쪽을 쓴다.
    use_dsd = dsd_tuned if dsd_tuned is not None else dsd_default
    use_dip = dip_t if dip_t is not None else dip_on
    verdict = sigma = None
    if use_dsd is not None and use_dip is not None and K:
        verdict = analyze_convention(use_dip, use_dsd, K, RESOLUTION)
        analyze_glass(cap, use_dip, use_dsd, stage)
        sigma = analyze_noise(use_dip, use_dsd)

    # ── 저장 — Phase 2 에서 SwinDRNet 에 그대로 투입 ──────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for k, v in (('rgb', rgb), ('dip_off', dip_off), ('dip_on', dip_on),
                 ('dsd_default', dsd_default), ('dip_tuned', dip_t), ('dsd_tuned', dsd_tuned)):
        if v is not None:
            payload[k] = v
    if K:
        payload['K'] = np.array([[K['fx'], 0, K['cx']], [0, K['fy'], K['cy']], [0, 0, 1]], dtype=np.float64)
    np.savez_compressed(out, **payload)

    hr('요약')
    print(f'  규약        : {verdict or "판정 불가"}')
    print(f'  센서 노이즈 : {f"{sigma * 1000:.2f} mm" if sigma is not None else "판정 불가"}')
    print(f'  저장        : {out}   (keys: {", ".join(sorted(payload))})')
    print('\n  다음: Phase 2 — 이 npz 의 dsd_tuned 를 SwinDRNet 에 넣어 Δ 재측정.')
    return 0


if __name__ == '__main__':
    # simulation_app.close() 가 하드 종료라, 예외를 여기서 직접 찍고 flush 하지 않으면
    # traceback 이 통째로 삼켜지고 exit 0 으로 나온다 (실패가 성공처럼 보인다).
    code = 1
    try:
        code = main()
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    simulation_app.close()
    sys.exit(code)
