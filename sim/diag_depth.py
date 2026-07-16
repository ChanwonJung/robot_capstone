#!/usr/bin/env python3
"""depth 진단 — 투명물체(유리컵) depth 구멍 여부 확인용 (throwaway).

depth 이미지 한 프레임을 받아서:
  1) 전체 유효/무효(구멍) 픽셀 통계 (valid% / zero% / nan% / min·max·mean)
  2) 터미널에 ASCII "구멍 지도" — '#'=유효 depth, ' '(공백)=무효(구멍)
     → 유리컵이 투명 투과로 depth를 못 잡으면 그 자리가 빈 공간으로 보임.

usage:
  source launch_env.bash
  python3 sim/diag_depth.py                         # 기본 EE depth
  python3 sim/diag_depth.py /rgbd_camera/depth_image   # TOP depth
  python3 sim/diag_depth.py /ee_rgbd_camera/depth_image --frames 3
"""
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

MIN_D, MAX_D = 0.05, 15.0          # 유효 depth 범위 (graspgen/projector 기준)
ASCII_COLS, ASCII_ROWS = 90, 30    # 구멍 지도 해상도


def decode_depth(msg: Image) -> np.ndarray:
    """Image -> (H,W) float32 미터. 32FC1 / 16UC1(mm) 지원."""
    h, w, enc = msg.height, msg.width, msg.encoding.lower()
    buf = bytes(msg.data)
    if enc in ('32fc1', '32f', 'f32'):
        d = np.frombuffer(buf, dtype='<f4').reshape(h, w).astype(np.float32)
    elif enc in ('16uc1', 'mono16', '16u'):
        d = np.frombuffer(buf, dtype='<u2').reshape(h, w).astype(np.float32) / 1000.0
    else:
        raise ValueError(f'지원 안 하는 encoding: {msg.encoding}')
    return d


def detect_objects(d: np.ndarray, margin: float = 0.015) -> None:
    """테이블 평면보다 margin(m) 이상 가까운 픽셀 = 물체. 자동 탐지 후 보고.

    RGB 없이 물체 위치/개수/표면depth 를 확인 — 유리컵이 near-blob 으로 잡히면
    표면 depth 있음(정상), 안 잡히면 see-through/구멍.
    """
    h, w = d.shape
    valid = np.isfinite(d) & (d >= MIN_D) & (d <= MAX_D)
    if not valid.any():
        print('유효 depth 없음'); return
    table = float(np.median(d[valid]))
    near = valid & (d < table - margin)
    print(f'\n[물체 자동탐지]  테이블 평면 ≈ {table:.3f}m,  '
          f'그보다 {margin*100:.0f}cm↑ 가까운 픽셀 = 물체 후보 ({near.sum()}px)')

    try:
        from scipy import ndimage
        lbl, n = ndimage.label(near)
        print(f'  연결 컴포넌트 {n}개 (면적 200px↑만 표시):')
        rows_out = 0
        for i in range(1, n + 1):
            ys, xs = np.where(lbl == i)
            if ys.size < 200:
                continue
            vv = d[ys, xs]
            print(f'   • blob{i:<2} px=({xs.min()},{ys.min()})-({xs.max()},{ys.max()}) '
                  f'면적={ys.size:>5}  depth median={np.median(vv):.3f} '
                  f'(테이블보다 {100*(table-np.median(vv)):.1f}cm 가까움)')
            rows_out += 1
        if rows_out == 0:
            print('  → 200px↑ 물체 blob 없음. 물체가 안 잡히거나 see-through 의심.')
    except ImportError:
        print(f'  (scipy 없음 — 클러스터 생략) near depth: '
              f'min={d[near].min():.3f} max={d[near].max():.3f}')

    # 물체 강조 ASCII
    ys = np.linspace(0, h - 1, ASCII_ROWS).astype(int)
    xs = np.linspace(0, w - 1, ASCII_COLS).astype(int)
    print('  "O"=물체(가까움) "."=테이블 " "=무효:')
    print('  +' + '-' * ASCII_COLS + '+')
    for y in ys:
        row = ''.join('O' if near[y, x] else ('.' if valid[y, x] else ' ') for x in xs)
        print('  |' + row + '|')
    print('  +' + '-' * ASCII_COLS + '+')


def report(d: np.ndarray, bbox=None, objects=False) -> None:
    h, w = d.shape
    if objects:
        detect_objects(d)
        return
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        sub = d[y0:y1, x0:x1]
        m = np.isfinite(sub) & (sub >= MIN_D) & (sub <= MAX_D)
        print(f'\n[영역 프로브] bbox=({x0},{y0})-({x1},{y1})  {sub.size}px')
        if m.any():
            v = sub[m]
            print(f'  유효 {100*m.sum()/sub.size:.1f}%  '
                  f'min={v.min():.3f} max={v.max():.3f} mean={v.mean():.3f} '
                  f'median={np.median(v):.3f} (m)')
        else:
            print('  유효 depth 없음 (전부 구멍/무효) → (A) 투과실패')
        return
    finite = np.isfinite(d)
    nan_inf = ~finite
    zero = finite & (d <= 0.0)
    valid = finite & (d >= MIN_D) & (d <= MAX_D)
    tot = h * w

    print(f'\n해상도 {w}x{h}  ({tot} px)')
    print(f'  유효(valid) : {valid.sum():>8} ({100*valid.sum()/tot:5.1f}%)')
    print(f'  구멍(0 이하) : {zero.sum():>8} ({100*zero.sum()/tot:5.1f}%)')
    print(f'  nan/inf     : {nan_inf.sum():>8} ({100*nan_inf.sum()/tot:5.1f}%)')
    if valid.any():
        v = d[valid]
        print(f'  유효 depth  : min={v.min():.3f}  max={v.max():.3f}  '
              f'mean={v.mean():.3f}  median={np.median(v):.3f} (m)')

    ys = np.linspace(0, h - 1, ASCII_ROWS).astype(int)
    xs = np.linspace(0, w - 1, ASCII_COLS).astype(int)

    # ── ASCII 구멍 지도 ────────────────────────────────────────────────
    print('\n[구멍 지도]  "#"=depth 있음, " "=구멍/무효 — 유리컵 자리가 빈칸이면 (A)투과실패:')
    print('  +' + '-' * ASCII_COLS + '+')
    for y in ys:
        row = ''.join('#' if (np.isfinite(d[y, x]) and MIN_D <= d[y, x] <= MAX_D)
                      else ' ' for x in xs)
        print('  |' + row + '|')
    print('  +' + '-' * ASCII_COLS + '+')

    # ── ASCII depth 그라디언트 (가까움→멈: . : - = + * o O # @) ─────────
    ramp = '.:-=+*oO#@'
    if valid.any():
        lo, hi = np.percentile(d[valid], [2, 98])
        span = max(hi - lo, 1e-6)
        print(f'\n[depth 그라디언트]  가까움({lo:.2f}m) . : - = + * o O # @ 멈({hi:.2f}m),'
              f'  " "=무효 — 유리컵이 주변 테이블과 "다른 밝기"면 표면 잡힌 것(정상),'
              f' "같은 밝기"면 (B)see-through:')
        print('  +' + '-' * ASCII_COLS + '+')
        for y in ys:
            row = ''
            for x in xs:
                v = d[y, x]
                if np.isfinite(v) and MIN_D <= v <= MAX_D:
                    idx = int(np.clip((v - lo) / span, 0, 1) * (len(ramp) - 1))
                    row += ramp[idx]
                else:
                    row += ' '
            print('  |' + row + '|')
        print('  +' + '-' * ASCII_COLS + '+')


class Diag(Node):
    def __init__(self, topic: str, frames: int, bbox=None, objects=False):
        super().__init__('diag_depth')
        self._need = frames
        self._got = 0
        self._bbox = bbox
        self._objects = objects
        self.create_subscription(Image, topic, self._cb, 10)
        self.get_logger().info(f'구독: {topic} — {frames} 프레임 대기...')

    def _cb(self, msg: Image):
        self._got += 1
        try:
            d = decode_depth(msg)
        except ValueError as e:
            self.get_logger().error(str(e))
            rclpy.shutdown(); return
        print(f'\n===== frame {self._got}/{self._need}  encoding={msg.encoding} =====')
        report(d, self._bbox, self._objects)
        if self._got >= self._need:
            rclpy.shutdown()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    topic = args[0] if args else '/ee_rgbd_camera/depth_image'
    frames = 1
    if '--frames' in sys.argv:
        frames = int(sys.argv[sys.argv.index('--frames') + 1])
    bbox = None
    if '--bbox' in sys.argv:
        i = sys.argv.index('--bbox')
        bbox = tuple(int(v) for v in sys.argv[i + 1:i + 5])
    objects = '--objects' in sys.argv
    rclpy.init()
    node = Diag(topic, frames, bbox, objects)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
