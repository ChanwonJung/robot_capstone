import asyncio
import os
import sys
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.timeline
import omni.ui as ui
import omni.usd
from isaacsim.sensors.camera import Camera
from isaacsim.sensors.camera import SingleViewDepthSensorAsset
from isaacsim.core.utils.stage import open_stage
from omni.kit.viewport.utility import get_active_viewport
from omni.kit.viewport.window import ViewportWindow, get_viewport_window_instances
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


SIM_DIR = Path(__file__).resolve().parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from isaac_ros_camera_bridge import build_ee_view_bridge, build_top_view_bridge
from isaac_ros_joint_bridge import create_ros2_joint_graph

PROJECT_ROOT = SIM_DIR.parent
DOWNLOADS_DIR = Path(os.environ.get("ROBOT_CAPSTONE_DOWNLOADS_DIR", Path.home() / "Downloads")).expanduser()
XR_CONTENT_ROOT = Path(
    os.environ.get("ROBOT_CAPSTONE_XR_CONTENT_ROOT", DOWNLOADS_DIR / "XR_Content_NVD@10010")
).expanduser()
IMPORTED_ASSETS_DIR = SIM_DIR / "assets" / "imported"
ISAACSIM_ROOT = PROJECT_ROOT / "isaacsim"

SOURCE_STAGE = XR_CONTENT_ROOT / "Assets" / "XR" / "Stages" / "robot_capstone.usd"
OUTPUT_STAGE = XR_CONTENT_ROOT / "Assets" / "XR" / "Stages" / "robot_capstone_scene.usd"

APPLE_ASSET = IMPORTED_ASSETS_DIR / "Apple.usd"
USE_APPLE_MESH = True
USE_GLASS_MESH = True
RED_BALL_ASSET = IMPORTED_ASSETS_DIR / "Red_Ball.usd"
BOOK_ASSET = IMPORTED_ASSETS_DIR / "Book_brown.usd"
BASKET_ASSET = IMPORTED_ASSETS_DIR / "Basket.usd"
GLASS_ASSET = XR_CONTENT_ROOT / "Assets" / "XR" / "Stages" / "Indoor" / "Modern_House" / "SubUSDs" / "P_Glassware_Short.usd"
BEDSIDE_TABLE_POSITION = np.array([3.3, -1.79, -0.73])
BEDSIDE_TABLE_ROTATION_DEG = np.array([0.0, 0.0, 25.0])
# xy: panda_link0 기준 0.660m 에 오도록 옮겼다. 원래 자리는 0.747m 로, pick 이
#   검증된 책(0.624m)·유리컵(0.654m) 대역 밖이었다. Panda 최대 도달은 0.855m 지만
#   그건 팔을 수평으로 뻗었을 때고, top-down 파지는 손목이 물체 위 + pre-grasp
#   가 거기서 또 120mm 위라 실제 작업영역은 훨씬 좁다. 구 파지 자체를 시험하는데
#   도달 한계까지 겹치면 실패 원인이 안 갈린다.
# z: 콜라이더 구(반지름 77.5*scale)의 바닥이 상판 면(-0.00137)에 닿는 값.
#   이전 0.682 는 콜라이더가 z=[-101, +8]mm 로 거의 전부 테이블 아래에 있었다.
APPLE_TRANSLATE = np.array([-2.4938, 3.0971, 0.761911])
APPLE_ROTATION_DEG = np.array([90.0, 0.0, 0.0])
# ★ 비주얼 정렬 — 이전 값 [-4.691566, -83.639191, 65.429489] 은 에셋 bbox 중심의
# 단순 음수라 회전(90°X)을 반영하지 않았고, 그 결과 사과 비주얼이 콜라이더보다
# 54.8mm 위에 떠 보였다(육안 확인 + 오프라인 bbox 측정 일치). 축 순서가
# set_xform 의 op 순서([translate, orient, scale, rotateXYZ] — 참조 에셋이 이미
# 가진 op 를 재사용해서 이렇게 된다)에 얽혀 있어 손계산이 아니라 야코비안으로
# 역산했다. 조건: 비주얼 bbox 중심 XY == 콜라이더 XY, 비주얼 바닥 == 콜라이더 바닥.
#   z=-75.65 = -(77.5 - 1.85) = -(콜라이더 반지름 - 콜라이더 오프셋)
APPLE_VISUAL_TRANSLATE = np.array([-4.691567, -65.429491, -75.650001])
APPLE_COLLIDER_TRANSLATE = np.array([0.0, 0.0, 1.85])
# z 는 컵 바닥이 테이블 상판 면에 정확히 닿도록 측정으로 맞춘 값 (0.71 → 0.733 → 0.72863).
# 부모 TabletopItems 가 z=-0.730 이므로 월드 z = 0.72863-0.730 = -0.00137.
# 유리컵은 콜라이더 바닥 == 비주얼 바닥 == root 원점이다 — GLASS_COLLIDER_TRANSLATE(4.5)
# 가 원통 반높이(9.0/2)를 정확히 상쇄하므로, root z 를 상판 면에 맞추면 그대로 안착한다.
#
# 상판 면 = -0.00137. table_low 메쉬에서 컵 XY 를 덮는 삼각형을 직접 찾아 잰 값이다
# (윗면 -0.00137 / 아랫면 -0.07122, 두께 70mm). 상판이 큰 면 몇 장짜리 성긴 메쉬라
# 컵 주변 35cm 안에 정점이 1개뿐 — 정점 최댓값(+0.01041)이나 bbox 로는 못 잰다.
# 주의: PhysX 레이캐스트는 여기서 +0.01526 을 준다(비주얼보다 16.6mm 위). 충돌면과
# 비주얼면이 어긋나 있으니, 이 값을 상판으로 쓰면 컵이 눈에 띄게 뜬다.
GLASS_TRANSLATE = np.array([-2.23, 3.03, 0.72863])
GLASS_ROTATION_DEG = np.array([0.0, 0.0, 0.0])
GLASS_COLLIDER_TRANSLATE = np.array([0.0, 0.0, 4.5])
# xy: panda_link0 기준 0.650m 로 옮겼다 — 원래 자리는 0.844m 로 최대 도달
#   0.855m 의 경계였다 (APPLE_TRANSLATE 주석 참조).
# z: 콜라이더 구(반지름 1.93*scale)의 바닥이 상판 면(-0.00137)에 닿는 값.
#   이전 0.84 는 콜라이더 바닥이 +43mm 라 공이 눈에 띄게 떠 있었다.
RED_BALL_TRANSLATE = np.array([-2.3367, 3.0647, 0.762686])
RED_BALL_ROTATION_DEG = np.array([-90.0, 0.0, 0.0])
# ★ 비주얼 정렬 — APPLE_VISUAL_TRANSLATE 와 같은 회전 미반영 버그. 이전 값
# [2.981419, -1.258126, -0.150547] 로는 공 비주얼이 콜라이더보다 22.4mm 아래로
# 내려가 테이블에 파묻혀 보였다. Y 도 17.4mm 어긋나 있었다 — 카메라는 비주얼을
# 보고 그리퍼는 콜라이더에 닿으므로 파지 정확도에 직접 영향이다.
RED_BALL_VISUAL_TRANSLATE = np.array([2.981419, -0.150547, 1.274956])
RED_BALL_COLLIDER_TRANSLATE = np.array([0.0, 0.0, 0.02])
BOOK_TRANSLATE = np.array([-2.11, 2.92, 0.787])   # 0.7배 축소 후 바닥 보정 (원래 0.80)
BOOK_ROTATION_DEG = np.array([0.0, -90.0, -68.0])
BOOK_COLLIDER_TRANSLATE = np.array([0.0, 0.0, 0.0])
# 이전 위치 [-2.05, 2.30, 0.72] 는 panda_link0 기준 (0.068, -0.470) 으로, EE
# 카메라 광축과의 내적이 음수 — 즉 **카메라 뒤쪽**이라 EE view 에 아예 안 잡혔다.
# destination 을 EE 시야 가장자리에 걸치도록 로봇 앞으로 495mm 당겼다.
#   link0 (0.480, -0.420), 베이스에서 0.638m — 검증된 도달 대역(책 0.624 /
#   컵 0.654) 한복판이고, 책과 171mm 떨어져 있다 (바구니 모서리 기준).
#
# 역할 분담 (destination grounding):
#   top view — 8/8 꼭짓점이 화면 안, 111x113px 로 잡힌다. 실제 추론은 여기서.
#   EE view  — 먼 쪽 윗 모서리 2개만 v=370 부근에 걸린다 = "조금만 보이는" 상태.
#              물체들이 v=377~415 라 같은 대역이다.
# EE 근거리 한계: 카메라가 x=0.442 에서 앞아래를 보므로 테이블 바닥면이 화면에
#   들어오는 최근접은 x≈0.52 다. 바구니는 높이가 131mm 라 x=0.48 에서도 윗
#   모서리가 살아남는다 — 점이 아니라 상자로 계산해야 나오는 값이다.
# z(local 0.72) 는 그대로 — link0 z = 0.72-0.73 = -0.010 으로 이전과 동일.
BASKET_TRANSLATE = np.array([-1.9213, 2.6944, 0.72])
# yaw 는 원래 값(-25) 로 되돌렸다. 부모 테이블이 +25 회전이라 -25 가 이를 상쇄해
# 바구니가 월드 축에 정렬된다. **yaw 로는 가로/세로를 못 바꾼다** — 에셋 발자국이
# 235x213mm 로 사실상 정사각이라 90도 돌려도 눈에 띄는 차이가 없다 (실측 확인).
# 가로를 길게 하려면 아래 SCALE 을 비균일로 줘야 한다.
BASKET_ROTATION_DEG = np.array([90.0, 0.0, -25.0])
# ── Phone — stacking destination for "put X on top of the phone" ──────────────
# poly.pizza/m/1L9oJAw6nY2, "Phone" by Alex Safayan, CC-BY 3.0 (credit required).
#
# 회전에 X+90 을 주면 안 된다. 사과·바구니는 에셋이 Y-up 이라 X+90 으로 세우지만,
# 이 에셋은 stage upAxis=Y 인데도 납작한 면의 법선이 **Z** 다 (실측 bbox
# 0.760 x 1.504 x 0.176, 얇은 축 = Z). 즉 무회전이 이미 "눕힌" 자세이고,
# X+90 을 주면 폰이 세로로 선다. yaw -25 는 부모 테이블의 +25 를 상쇄해
# (바구니와 같은 관용구) 장축을 panda_link0 +X 에 정렬시킨다.
PHONE_ASSET = IMPORTED_ASSETS_DIR / "Phone.usd"
# link0 (0.520, 0.420), 베이스에서 0.668m — 검증된 도달 대역(컵 0.654) 안이다.
# 기존 5개 물체는 link0 y = +0.269(사과) ~ -0.420(바구니) 로 거의 일직선이라
# 폰(198x100mm)이 들어갈 틈이 없었다. 줄 바깥 +Y 로 빼서:
#   책과 0.57m — top view 에서 유일한 다른 직사각형이 책이라(세워둔 윗면
#   172x27mm 가 어두운 막대로 보인다) 붙여두면 DESTINATION 오선택 위험이 있다.
#   사이에 둥근 사과가 끼어 형태 혼동도 없다. 가장 가까운 이웃(사과)과 66mm.
# top camera FOV 는 2m 에서 x[0.12,1.56] y[-0.96,0.96] — 여유 있게 안쪽이다.
# z: 에셋 바닥이 원점 아래 0.07861 이므로 0.72863(상판) + 0.07861*0.1316.
PHONE_TRANSLATE = np.array([-2.6657, 3.0857, 0.73898])
PHONE_ROTATION_DEG = np.array([0.0, 0.0, -25.0])
# 콜라이더는 본체 박스만 잡는다. bbox 전체(Z 0.0975)를 쓰면 한쪽 모서리 돌출부
# 높이에 평면이 생겨 공이 비주얼 표면보다 6mm 떠서 놓인다. 본체 상단은 Z 0.049.
# convexHull/Decomposition 을 피하는 이유는 테이블 상판과 같다 — 두께 자체가
# 23mm 라 십수 mm 부풀면 치명적이다 (fix_table_collision 주석 참조).
PHONE_COLLIDER_SIZE = np.array([0.740, 1.500, 0.120])
PHONE_COLLIDER_TRANSLATE = np.array([-0.0048, 0.0739, -0.0110])
# 0.17 → 0.12. 에셋 원본 bbox = X 1.960 / Y 1.093(위) / Z 1.771 이므로
#   scale 0.12 -> 가로 235 x 세로 213 x 높이 131 mm.
# 피벗은 바닥이다 (에셋 Y_min=0.037 ≈ 0) — 축소해도 상판 접지가 안 틀어진다.
BASKET_SCALE = np.array([0.12, 0.12, 0.12])
# 테이블 위 파지 대상(사과·유리컵·빨간공·책) 전체 축소 배율. 콜라이더·오프셋이
# 모두 *_SCALE 에서 파생되므로 이 배율 하나로 비주얼+물리가 함께 축소됨.
# 1.0 = 원래 크기. 바구니(BASKET)는 목적지라 제외.
_TABLETOP_SCALE = 0.7
# 원본 에셋 154.57 x 167.28 x 151.29 units (콜라이더 반지름 77.5 는 X 반경 77.28 과 일치).
# 0.001 에서는 지름 108.5mm 로 그리퍼 개폐 80mm 를 넘어 감싸 쥘 수 없었다.
# 유리컵과 같은 68mm 로 맞춘다 — 실제 사과 크기이고 여유 11.8mm 로 검증된 값이다.
#   가로 68.0mm (콜라이더 68.2mm) / 세로 73.6mm — 세로가 긴 건 줄기 때문.
APPLE_SCALE = np.array([0.00062848, 0.00062848, 0.00062848]) * _TABLETOP_SCALE
# 원본 에셋은 150.2mm 지름 x 133.4mm 높이 — 컵이 아니라 넓적한 사발 비율이고,
# panda 최대 개폐(80mm)보다 커서 바깥에서 감싸 쥘 수 없었다. (림 파지만 남는데
# 그건 복원 결과에 '빈 속'이 필요하고, SwinDRNet 은 꽉 찬 덩어리를 내놓는다.)
#
# xy 와 z 를 다르게 준다 — 균등 축소로는 사발 비율이 그대로라 '작은 사발'이 된다.
#   xy 0.00906 → 지름  68mm  (실제 유리컵 6~7cm, 그리퍼 80mm 안쪽)
#   z  0.01499 → 높이 100mm  (실제 유리컵 9.5~11cm)
GLASS_SCALE = np.array([0.00906, 0.00906, 0.01499]) * _TABLETOP_SCALE
# 원본 에셋 3.8137 x 3.8537 x 3.8137 units (콜라이더 반지름 1.93 은 비주얼 1.907 과 일치).
# 0.05 에서는 지름 133.5mm(콜라이더 135.1mm)로 그리퍼 개폐 80mm 를 훨씬 넘었다.
# 유리컵과 같은 68mm — 테니스공(67mm) 크기, 여유 11.2mm.
RED_BALL_SCALE = np.array([0.02547213, 0.02547213, 0.02547213]) * _TABLETOP_SCALE
BOOK_SCALE = np.array([0.10, 0.10, 0.10]) * _TABLETOP_SCALE
# 에셋 원본 0.760 x 1.504 x 0.176 -> 0.1316 배로 100.0 x 198.0 x 23.2 mm.
# 폭 100mm 를 먼저 정하고 거기서 역산한 값이다: 빨간공 지름이 68mm 이고 place
# 정확도가 ~10mm 라, 폭이 90mm 밑으로 내려가면 공이 얹힐 자리가 안 나온다.
# 실제 폰(75~80mm)보다 넓어 Qwen 이 "tablet" 이라 부를 수는 있는데, 그래도
# 동작한다 — destination_label 은 parse_scene 로그에서만 쓰이고 어떤 분기도
# 이 문자열을 보지 않는다. 좁히려면 이 배율 하나만 바꾸면 콜라이더까지 따라온다.
PHONE_SCALE = np.array([0.188, 0.188, 0.188]) * _TABLETOP_SCALE
# 책 무게 — 평행 그리퍼 grasp 유지를 쉽게 하려고 실제(~0.35kg)보다 가볍게.
# 마찰력 F = μ·N 이라 무게가 가벼우면 적은 grip force 로도 안 미끄러짐.
BOOK_MASS = 0.15

# ── 투명물체 depth-restoration 실험용 유리 3종 (2026-09) ─────────────────────
# Sketchfab CC-BY, GLB→USD 변환은 sim/import_downloaded_assets.py 로 함.
# ★ 아래 배치는 diag_asset_bbox.py 로 잰 원본 bbox 로만 역산한 1차 추정치다.
# 사과/공/폰 때처럼 회전 부호(90 vs -90)나 비주얼-콜라이더 XY 정렬이 실측 전엔
# 틀릴 수 있다 — Isaac Sim 에서 직접 보고 필요하면 위 두 항목(ROTATION_DEG 부호,
# 필요시 VISUAL_TRANSLATE 추가)부터 고칠 것. 콜라이더는 그 불확실성을 피하려고
# 이상화한 도형이 아니라 비주얼 메쉬 자체에서 convexHull 로 뽑는다(바구니와 동일
# 전략) — 비주얼이 어디 있든 콜라이더가 항상 따라가므로 피벗 오차가 안 남는다.

GLASS_BOTTLE_PACK_ASSET = IMPORTED_ASSETS_DIR / "Glass_Bottle_Pack.usd"
# 팩 안 55개 leaf mesh 중 height/diameter 비율(3.22, 가장 병처럼 길쭉함)로 고른 것.
# 나머지 후보는 diag_asset_bbox.py 출력 참고 — 컵/잔 계열은 이 비율이 훨씬 낮다.
GLASS_BOTTLE_PRIM_PATH = (
    "/World/node_93c05a82ce5435ea57e7f1b8c8ee88d_fbx/RootNode/Cylinder_042"
    "/Cylinder_042_Material_0/Cylinder_042_Material_0"
)
GLASS_BOWL_ASSET = IMPORTED_ASSETS_DIR / "Glass_Bowl.usd"
GLASS_JAR_ASSET = IMPORTED_ASSETS_DIR / "Glass_Jar.usd"
GLASS_MUG_ASSET = IMPORTED_ASSETS_DIR / "Glass_Mug.usd"

# 원본(Y-up, 미터 아닌 임의 unit) bbox 실측:
#   Bottle(Cylinder_042): size(1.4599, 4.6950, 1.4599) y_min≈0 (원점=바닥, 컵과 같은 편한 케이스)
#   Bowl  (전체):         size(1.4884, 0.9743, 1.4951) y_min=-0.6937 (원점이 바닥보다 위)
#   Jar   (전체):         size(0.7088, 0.9601, 0.7102) y_min= 0.2392 (원점이 바닥보다 아래)
# Y-up → 씬의 Z-up 으로 옮기는 회전은 사과와 같은 X+90 을 1차로 쓴다(Rx(+90): y→z).
# 그 변환에서 로컬 y_min 이 root 기준 z 오프셋이 되므로, 상판 접지 z 는
#   root_z = TABLE_SURFACE_Z(0.72863) - y_min * scale
# 로 역산했다 (y_min>0 이면 바닥이 원점보다 위이므로 root 를 낮춤, 음수면 올림).
# ★ Bottle 만 회전이 다르다 — prim_path 로 leaf mesh 하나만 참조하면(Bowl/Jar 와
# 달리 파일 전체 defaultPrim 을 참조하는 게 아니라서) 조상 Xform 들의 회전이 전혀
# 안 딸려온다. 이 메쉬는 raw extent 로 봤을 때 이미 로컬 Z 축이 높이축(Z: -3.19~
# 1.50, span 4.695 = height/diameter 비율 3.22 그대로)이라, 씬의 Z-up 과 그대로
# 맞는다 — 무회전이 정답이었다(90 회전을 주면 옆으로 눕는다, 실측 확인).
GLASS_BOTTLE_ROTATION_DEG = np.array([0.0, 0.0, 0.0])
GLASS_BOWL_ROTATION_DEG = np.array([90.0, 0.0, 0.0])
GLASS_JAR_ROTATION_DEG = np.array([90.0, 0.0, 0.0])

# 지름 56mm(그리퍼 80mm 안쪽, 몸통 파지 가능) 목표. 높이(Z)는 무회전이라 직접
# 스케일 반영돼 180mm 로 실측 정확히 맞았지만, X/Y(지름)는 실측하니 74.4mm 로
# 나와서(변환기가 raw glTF accessor 에 없던 unit 보정을 추가로 먹인 것으로 보임 —
# 원인 확정은 못 했고 diag_verify_glass_props.py 실측으로 역산해 고쳤다) X/Y 만
# 0.03834*(56/74.4) 로 축소. Z 는 이미 정확해서 그대로 둔다.
GLASS_BOTTLE_SCALE = np.array([0.02885, 0.02885, 0.03834])
# 무회전이라 raw local z_min(-3.194) 이 그대로 world 오프셋이 된다.
GLASS_BOTTLE_TRANSLATE = np.array([-2.6657, 3.0857, 0.72863 + 3.194 * 0.03834])

# 지름 150mm(그리퍼 80mm 초과 — 유리컵과 같은 림 파지 전제, "빈 속" 위상 필요).
# ★ 균등 스케일 0.10033 으로 1차 계산했더니 실측 지름 264mm, 높이 130mm 로 목표
# (150mm/97.7mm) 보다 한참 컸다 — Bottle 과 마찬가지로 변환기의 unit 보정으로
# 추정되는 배율이 축마다 다르게 껴 있어서(지름 비율 1.76 vs 높이 비율 1.328,
# 안 같음 — 균등 스케일로는 못 고친다) X/Z(지름)와 Y(높이)를 실측 기반으로 따로
# 축소했다. 회전이 X+90 이라 world 매핑은 X→X, Y→world Z(높이), Z→world Y(지름).
GLASS_BOWL_SCALE = np.array([0.0570, 0.0755, 0.0570])
# world Z(높이)를 좌우하는 성분(scale[1]=Y)이 0.10033→0.0755 로 바뀌었으므로,
# 이전 실측 보정 오프셋(0.6937*0.10033+0.0247=0.09430)도 같은 비율로 스케일했다.
GLASS_BOWL_TRANSLATE = np.array([-1.9213, 2.6944, 0.72863 + 0.09430 * (0.0755 / 0.10033)])

# 지름 80mm(그리퍼 한계선 — 몸통 파지 안 되면 림/목 파지로 넘어갈 후보).
# 실측 지름 106mm(목표比 1.328, Bottle 과 같은 배율 — 이쪽은 높이는 이미 정확했다)
# 라 X/Z(지름) 만 축소. Y(높이, world Z 를 좌우) 는 그대로 둬서 Z-translate 도 불변.
GLASS_JAR_SCALE = np.array([0.0848, 0.11265, 0.0848])
GLASS_JAR_TRANSLATE = np.array([-2.11, 2.92, 0.72863 - 0.2392 * 0.11265])

# 손잡이 컵(Glass Mug) — 2026-09 추가. 원본 bbox: X 20.36 / Y 26.07(=높이, y_min=2.30)
# / Z 29.37. h/d = 26.07/29.37 = 0.887 (Bowl/Jar 와 같은 Y-up → X+90 회전 가정).
# 지름 80mm 목표 — 기존 유리컵(68mm)과 달리 몸통을 그리퍼로 못 감싸도 된다(손잡이
# 파지가 목적이라 애초에 몸통 그립을 전제하지 않음). Bottle/Bowl/Jar 모두 균등
# 스케일 예측이 실측과 어긋났으므로(변환기 unit 보정) 이 값도 diag_verify_glass_props.py
# 로 확인 후 X/Z(지름)와 Y(높이)를 따로 보정해야 할 가능성이 높다 — 1차 추정치.
GLASS_MUG_SCALE = np.array([0.002724, 0.002724, 0.002724])
GLASS_MUG_ROTATION_DEG = np.array([90.0, 0.0, 0.0])
GLASS_MUG_TRANSLATE = np.array([-2.4938, 3.0971, 0.72863 - 2.3017 * 0.002724])

# 유리이므로 다른 유리컵(0.18kg)과 비슷한 대역의 1차 추정치. 파지 시 미끄러지면
# 유리컵처럼 gripper force 를 낮추는 쪽으로 튜닝할 것(무게보다 grip force 가 결정적이었음).
GLASS_BOTTLE_MASS = 0.15
GLASS_BOWL_MASS = 0.20
GLASS_JAR_MASS = 0.20
GLASS_MUG_MASS = 0.20

# Hazard placeholders for Fast Brain testing. Procedural for now; positions
# spawn outside the top-view FOV and have initial velocity so the hazards "fly
# in" toward the workspace when the simulation plays. Tune in viewport, then
# Ctrl+S to persist.
CARDBOX_ASSET = (
    XR_CONTENT_ROOT
    / "Assets" / "XR" / "Stages" / "Indoor" / "Warehouse" / "Containers" / "Cardboard"
    / "Cardbox_C2.usd"
)
# Top-view camera FOV at z=0.5 is roughly x∈[-0.7,0.7], y∈[-0.2,0.6]
# (focal=4mm, 3.84mm aperture, camera at z=2.0). Spawning hazards at
# |x|≈2.5 or |y|≈2.5 keeps them clearly outside the frame at t=0 and gives
# ~4-5s of fly-in time at v=0.4 m/s. Gravity disabled so they cruise straight
# at constant altitude through the robot's reaching path (z≈0.5).
HAZARD_BOX_TRANSLATE = np.array([2.50, -0.09, 0.35])
HAZARD_BOX_ROTATION_DEG = np.array([0.0, 0.0, 30.0])
# Cardbox_C2 USD is authored in centimeters (metersPerUnit=0.01) while the
# parent stage is in meters. USD does NOT auto-rescale references, so the
# visual scale bakes in the 0.01 unit factor.
# native ≈ 0.51 m, so scale 0.003 → ~0.15 m box, scale 0.002 → ~0.10 m box.
# Uniform scale override via ROBOT_CAPSTONE_BOX_SCALE — useful for the
# stop+resume demo where a smaller box has a smaller intersection with the
# arm body and the planning-scene halt fires with less actual contact.
_BOX_SCALE_UNIFORM = float(os.environ.get("ROBOT_CAPSTONE_BOX_SCALE", "0.003"))
HAZARD_BOX_SCALE = np.array([_BOX_SCALE_UNIFORM] * 3)
# Fallback box (when the cardbox USD isn't available) keeps a fixed scale-
# relative size so it's roughly visually equivalent to the loaded USD.
_BOX_FALLBACK_RATIO = 0.14 / 0.003  # 0.14 m fallback at default scale 0.003
HAZARD_BOX_FALLBACK_SIZE = np.full(3, _BOX_SCALE_UNIFORM * _BOX_FALLBACK_RATIO)
# Box flight velocity at /hazard/launch_bottle trigger. All three axes are
# env-controllable so the stop+resume demo can fly the box laterally across
# the arm's reach path (VY) rather than along it (VX) — the lateral cross
# is what makes the halt visible without the box physically slamming the
# arm body, since the box never sits in the same lane as the arm.
HAZARD_BOX_LINEAR_VELOCITY = np.array([
    float(os.environ.get("ROBOT_CAPSTONE_BOX_VX", "-0.4")),
    float(os.environ.get("ROBOT_CAPSTONE_BOX_VY", "0.0")),
    float(os.environ.get("ROBOT_CAPSTONE_BOX_VZ", "0.0")),
])

# pet_bottle hazard. Uses real PET USD captured into the v3 dataset so train
# and inference share the same render. Spawn / velocity mirror HazardBox so
# the top-view sees an off-frame approach then a base/glass crossing.
HAZARD_BOTTLE_ASSET = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/DigitalTwin/Assets/Warehouse/Storage/Bottles/Plastic/"
    "NaturalBostonRound_A/NaturalBostonRoundBottle_A02_PR_NVD_01.usd"
)
# x=0.85: just outside the top-view FOV edge (~0.7) so it's off-frame at rest but
# enters the workspace ~0.5 s after the launch trigger (fired when the arm starts).
# Appears-in-FOV delay ≈ (x - 0.7) / |velocity| s. Pull toward 0.78 to appear even
# sooner, push out for later. (Stay under the stage wall that blocked 4.5+.)
# X / Y / Z are env-controllable so the spawn can be repositioned for both
# the replan demo (parked above the book, ~constant Y) and the stop+resume
# demo (off to one side, flies laterally across the arm's reach path).
HAZARD_BOTTLE_TRANSLATE = np.array([
    float(os.environ.get("ROBOT_CAPSTONE_BOTTLE_SPAWN_X", "0.85")),
    float(os.environ.get("ROBOT_CAPSTONE_BOTTLE_SPAWN_Y", "-0.09")),
    float(os.environ.get("ROBOT_CAPSTONE_BOTTLE_SPAWN_Z", "0.35")),
])
HAZARD_BOTTLE_ROTATION_DEG = np.array([0.0, 0.0, 0.0])
# Warehouse Bottles authored in centimeters (metersPerUnit=0.01) like Cardbox.
# Bake the unit factor + final-size scale into one factor. Tune in viewport
# if the bottle reads too small/large after a fresh import.
HAZARD_BOTTLE_SCALE = np.array([0.01, 0.01, 0.01])
HAZARD_BOTTLE_RADIUS = 0.035    # collider approx (real PET ~6cm diameter)
HAZARD_BOTTLE_HEIGHT = 0.22     # collider approx
# Velocity applied to the bottle when /hazard/launch_bottle fires (it sits still
# until then). From x=1.0 at -0.3 m/s it reaches the arm's goal-x (~0.71) in ~1 s
# and crosses the workspace over the next ~2 s — i.e. during the grasp motion.
# Tune magnitude to match how fast you want the hazard to cross.
# Slower (-0.3) than the original -0.9 so "park" mode stops it precisely: at high
# speed the per-frame park check overshoots PARK_X by 10-20 cm. -0.3 keeps the
# per-frame travel small so the bottle halts near PARK_X. Still flies in well
# within the (very slow) arm motion.
#
# Override via ROBOT_CAPSTONE_BOTTLE_VX. For the EE-detected replan demo the
# bottle should be SLOW (e.g. -0.15) so the EE camera has time to see it,
# the injector to publish /collision_object, and the global planner to issue
# a re-plan before the bottle reaches PARK_X.
HAZARD_BOTTLE_LINEAR_VELOCITY = np.array([
    float(os.environ.get("ROBOT_CAPSTONE_BOTTLE_VX", "-0.3")),
    0.0, 0.0,
])

# Hazard scenario mode (env ROBOT_CAPSTONE_HAZARD_MODE):
#   "flythrough" (default) — bottle flies straight through the workspace and
#                            exits: a TRANSIENT hazard for the stop+resume demo.
#   "park"                 — bottle flies in, then HALTS in front of the arm and
#                            stays put: a PERSISTENT hazard for the avoidance
#                            (replan) demo. Top YOLO keeps detecting it, so the
#                            injected collision object persists and the global
#                            re-plan routes the arm around it.
HAZARD_BOTTLE_MODE = os.environ.get("ROBOT_CAPSTONE_HAZARD_MODE", "flythrough").strip().lower()
# Which hazard asset gets spawned + launched (env ROBOT_CAPSTONE_HAZARD_OBJECT):
#   "bottle" (default) — pet_bottle USD, used for v3 dataset / runtime hazard.
#   "box"              — cardbox USD, spawned at the bottle's position with the
#                        bottle's flight params so capture parity holds.
HAZARD_OBJECT = os.environ.get("ROBOT_CAPSTONE_HAZARD_OBJECT", "bottle").strip().lower()
# In "park" mode, zero the bottle's velocity once its world x drops to this value.
# It flies from x=0.85 toward -x, so stopping near the arm's goal-x leaves it
# sitting in the reaching path. Tune via env ROBOT_CAPSTONE_HAZARD_PARK_X so it
# actually blocks the path the arm takes to the goal.
HAZARD_BOTTLE_PARK_X = float(os.environ.get("ROBOT_CAPSTONE_HAZARD_PARK_X", "0.45"))
# Auto-trigger the hazard launch the first time the arm starts moving instead
# of waiting for a manual `ros2 topic pub /hazard/launch_bottle`. The launcher
# subscribes to /joint_states and, AFTER an arming delay (so MoveIt hybrid
# startup and any home-pose settling don't trip the trigger), fires once when
# any joint exceeds the threshold from the latched reference pose.
#
# Tuning notes:
#  - AUTO_TRIGGER_RAD: 0.10 rad ≈ 5.7° per joint. Real goal-directed motion
#    blows past this; Isaac physics jitter and hybrid startup wiggle do not.
#  - AUTO_ARM_SEC: 5 s comfortably swallows the hybrid container init + the
#    first MoveAction "settle to current state" wiggle.
HAZARD_AUTO_LAUNCH = os.environ.get("ROBOT_CAPSTONE_HAZARD_AUTO_LAUNCH", "0").strip() == "1"
HAZARD_AUTO_TRIGGER_RAD = float(os.environ.get("ROBOT_CAPSTONE_AUTO_TRIGGER_RAD", "0.10"))
HAZARD_AUTO_ARM_SEC = float(os.environ.get("ROBOT_CAPSTONE_AUTO_ARM_SEC", "5.0"))

# Capture-only static humans for hand/forearm dataset (NOT hazards — no rigid
# body, no motion). NVIDIA 4.2 People catalog has ~8 characters total. Pick
# whichever reads least uncanny + has exposed forearms; swap the URL below.
PEOPLE_ROOT = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/4.2/Isaac/People/Characters"
)
CAPTURE_HUMAN_ASSET_BUSINESS_FEMALE = f"{PEOPLE_ROOT}/F_Business_02/F_Business_02.usd"
CAPTURE_HUMAN_ASSET_CONSTRUCTION_NEW = (
    f"{PEOPLE_ROOT}/original_male_adult_construction_05_new/male_adult_construction_05_new.usd"
)
CAPTURE_HUMAN_ASSET_CONSTRUCTION_02 = (
    f"{PEOPLE_ROOT}/original_male_adult_construction_02/male_adult_construction_02.usd"
)
CAPTURE_HUMAN_ASSET_FEMALE_MEDICAL = f"{PEOPLE_ROOT}/F_Medical_01/F_Medical_01.usd"
CAPTURE_HUMAN_ASSET_MALE_MEDICAL = f"{PEOPLE_ROOT}/M_Medical_01/M_Medical_01.usd"
CAPTURE_HUMAN_ASSET_POLICE_FEMALE = (
    f"{PEOPLE_ROOT}/original_female_adult_police_01/female_adult_police_01.usd"
)
# Active character for capture. Swap to any of the above constants.
CAPTURE_HUMAN_ASSET = CAPTURE_HUMAN_ASSET_BUSINESS_FEMALE
# Stage is Z-up but the 4.2 People assets are authored Y-up. Additionally
# the character's /Root prim has a baked xformOp:rotateXYZ = (-90, 0, 0)
# inside the referenced USD. Our outer rotate composes onto that, so to land
# USD +Y (head) at stage +Z (up) we need net X = +90  ⇒  outer X = +180.
# Verified by simulating wrapper composition with bbox probe:
#   outer (90, 0, *) gives upright (mesh +Z up in stage). outer (180,*,*) lays flat.
# Feet at character-local z=-0.12 after upright rotation, so translate_z = floor(-0.73) - (-0.12) = -0.61.
CAPTURE_HUMAN_TRANSLATE = np.array([0.40, 0.20, -0.61])     # CAPTURE: inside top-camera FOV (x[-0.7,0.7] y[-0.2,0.6]); ignore table clip for capture phase
# CAPTURE_HUMAN_TRANSLATE = np.array([0.47, -1.10, -0.61])  # SCENARIO: next to basket, table-side standing (out of top-FOV — for hazard phase only)
CAPTURE_HUMAN_ROTATION_DEG = np.array([90.0, 0.0, 0.0])     # X+90° → upright; tweak the Z (yaw) to face the robot base
CAPTURE_HUMAN_SCALE = np.array([1.0, 1.0, 1.0])             # NVIDIA People assets are authored in meters

HAZARD_ARM_TRANSLATE = np.array([0.0, 2.50, 0.55])
HAZARD_ARM_ROTATION_DEG = np.array([90.0, 0.0, 0.0])
HAZARD_FOREARM_RADIUS = 0.045
HAZARD_FOREARM_LENGTH = 0.28
HAZARD_HAND_RADIUS = 0.06
HAZARD_ARM_LINEAR_VELOCITY = np.array([0.0, -0.4, 0.0])

HAZARD_OBJECT_PATHS = {
    "HazardBox": "/World/CapstoneAdditions/Hazards/HazardBox",
    "HazardBottle": "/World/CapstoneAdditions/Hazards/HazardBottle",
    "HazardArm": "/World/CapstoneAdditions/Hazards/HazardArm",
}

TOP_CAMERA_POSITION = np.array([0.0, 0.2, 2.0])
TOP_CAMERA_ROTATION_DEG = np.array([0.0, 0.0, 0.0])
TOP_CAMERA_FOCAL_LENGTH_MM = 4.0
# Kinova Gen3 + Robotiq 2F-85 (URDF-imported, sim/kinova_urdf/gen3_isaac.xacro) 의
# 실제 링크 계층은 URDF 체인을 그대로 따라가서 Franka 때보다 훨씬 깊다
# (/Kinova/Geometry/world/base_link/.../end_effector_link/robotiq_85_base_link).
# get_ee_mount_prim() 이 WRIST_MOUNT_CANDIDATES 로 이 경로를 런타임에 찾아내지만,
# EE_CAMERA_PATH 는 create_ee_camera() 가 그 위에 실제로 짓는 경로와 반드시
# 일치해야 하는 별도 상수라 여기도 전체 경로를 그대로 적어둔다.
KINOVA_ARTICULATION_ROOT_PATH = "/Kinova/Geometry/world/base_link"
KINOVA_ROBOTIQ_BASE_LINK_PATH = (
    f"{KINOVA_ARTICULATION_ROOT_PATH}/shoulder_link/half_arm_1_link/half_arm_2_link/"
    "forearm_link/spherical_wrist_1_link/spherical_wrist_2_link/bracelet_link/"
    "end_effector_link/robotiq_85_base_link"
)
EE_CAMERA_PATH = f"{KINOVA_ROBOTIQ_BASE_LINK_PATH}/EEViewCameraMount/CameraRig/CameraFrame/EEViewCamera"
TOP_CAMERA_PATH = "/World/TopViewCamera"
CAMERA_SENSOR_SCOPE = "/World/CameraSensors"
EE_DEPTH_SCOPE = f"{CAMERA_SENSOR_SCOPE}/EEViewDepth"
TOP_DEPTH_SCOPE = f"{CAMERA_SENSOR_SCOPE}/TopViewDepth"
EE_VIEWPORT_NAME = "EE View"
TOP_VIEWPORT_NAME = "Top View"
EE_VIEWPORT_RESOLUTION = (640, 480)
TOP_VIEWPORT_RESOLUTION = (640, 480)
WRIST_MOUNT_CANDIDATES = ["robotiq_85_base_link", "end_effector_link", "bracelet_link"]
EE_MOUNT_FALLBACK_CANDIDATES = ["gripper_center", "tool0", "ee_link", "right_gripper"]
EE_CAMERA_MOUNT_TRANSLATE = np.array([0.000, 0.0, 0.030])
EE_CAMERA_LOCAL_TRANSLATE = np.array([0.095, 0.0, -0.030])
# 2026-09 Kinova Gen3 — robotiq_85_base_link 의 로컬 축 컨벤션이 Panda panda_hand
# 와 달라서 그 값(-160,0,90)을 그대로 못 쓴다. robotiq_85_base_link 의 실측 월드
# 회전행렬 + Home 포즈에서의 카메라 위치로부터, 대략적인 테이블 목표점을 보도록
# 역산한 시작값 — 정밀 조정은 라이브로 EE View 보면서 재검증 필요.
EE_CAMERA_LOCAL_ROTATION_DEG = np.array([108.0, 0.0, 180.0])
TABLETOP_OBJECT_PATHS = {
    "Apple": "/World/CapstoneAdditions/TabletopItems/Apple",
    "Glass": "/World/CapstoneAdditions/TabletopItems/Glass",
    "RedBall": "/World/CapstoneAdditions/TabletopItems/RedBall",
    "Book": "/World/CapstoneAdditions/TabletopItems/Book",
    "Basket": "/World/CapstoneAdditions/TabletopItems/Basket",
    "Phone": "/World/CapstoneAdditions/TabletopItems/Phone",
}

DEPTH_OVERLAY = None
TOP_VIEW_ROS_BRIDGE = None
EE_VIEW_ROS_BRIDGE = None
KINOVA_JOINT_ROS_BRIDGE = None


def set_xform(prim, translate=None, rotate_xyz_deg=None, scale=None):
    xformable = UsdGeom.Xformable(prim)
    ordered_ops = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}
    if translate is not None:
        (ordered_ops.get("xformOp:translate") or xformable.AddTranslateOp()).Set(Gf.Vec3d(*map(float, translate)))
    if rotate_xyz_deg is not None:
        (ordered_ops.get("xformOp:rotateXYZ") or xformable.AddRotateXYZOp()).Set(Gf.Vec3f(*map(float, rotate_xyz_deg)))
    if scale is not None:
        (ordered_ops.get("xformOp:scale") or xformable.AddScaleOp()).Set(Gf.Vec3f(*map(float, scale)))


def define_xform(stage, path, translate=None, rotate_xyz_deg=None, scale=None):
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    set_xform(prim, translate=translate, rotate_xyz_deg=rotate_xyz_deg, scale=scale)
    return prim


def add_visual_reference(stage, path, asset_path, translate=None, rotate_xyz_deg=None, scale=None,
                          prim_path=None):
    """prim_path: 참조 파일의 defaultPrim 대신 그 안의 특정 서브프림 하나만 참조하고
    싶을 때 쓴다 (예: 다중 오브젝트 팩에서 특정 병 하나만 골라 쓰는 경우).

    ★ prim_path 로 가리키는 대상이 Mesh 타입인 경우, path 자체를 Xform 으로 먼저
    선언한 뒤 그 위에 곧바로 레퍼런스를 걸면 로컬 "Xform" 스펙이 참조로 들어온
    "Mesh" 타입을 가려버려(강도상 로컬 레이어가 이김) 자식도 0개, Mesh API 도
    무효가 된다(실측 확인 — UsdGeom.Mesh(prim) 이 False). 그래서 prim_path 를 쓸
    때는 transform 용 Xform 을 만들고, 레퍼런스는 그 자식(Ref)에 걸어 타입 충돌을
    피한다. 파일 전체(defaultPrim=Xform)를 참조하는 기존 경로는 그대로 둔다."""
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().ClearReferences()
    if prim_path:
        ref_prim = stage.DefinePrim(f"{path}/Ref")
        ref_prim.GetReferences().AddReference(assetPath=str(asset_path), primPath=Sdf.Path(prim_path))
    else:
        prim.GetReferences().AddReference(str(asset_path))
    set_xform(prim, translate=translate, rotate_xyz_deg=rotate_xyz_deg, scale=scale)
    return prim


def set_display_color(prim, rgb):
    gprim = UsdGeom.Gprim(prim)
    if gprim:
        gprim.CreateDisplayColorAttr([Gf.Vec3f(*map(float, rgb))])


def set_descendant_display_color(prim, rgb):
    for child in Usd.PrimRange(prim):
        if child == prim:
            continue
        set_display_color(child, rgb)


def iter_collision_prims(root_prim):
    supported = {"Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone"}
    for prim in Usd.PrimRange(root_prim):
        if prim.GetTypeName() in supported:
            yield prim


def set_double_sided(root_prim):
    """Render both faces of every mesh in the subtree.

    UsdGeomGprim.doubleSided defaults to FALSE, and the imported assets do not
    author it. Any mesh whose winding came out inverted in the GLB -> USD
    conversion then has its outward faces culled, so you see straight through
    the near wall into the interior.

    This is not cosmetic: a culled face writes no depth either, so the top
    camera measures the surface BEHIND the wall. For the basket that corrupts
    bbox_3d_world.max.z, which is the rim height the place pose is built on.
    """
    count = 0
    for prim in Usd.PrimRange(root_prim):
        gprim = UsdGeom.Gprim(prim)
        if gprim:
            gprim.CreateDoubleSidedAttr().Set(True)
            count += 1
    return count


def apply_static_collider(root_prim, approximation="convexHull"):
    for prim in iter_collision_prims(root_prim):
        UsdPhysics.CollisionAPI.Apply(prim)
        if prim.GetTypeName() == "Mesh":
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision.CreateApproximationAttr().Set(approximation)


def create_dynamic_body_root(stage, path, translate, mass):
    root = define_xform(stage, path, translate=translate)
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid_body.CreateRigidBodyEnabledAttr(True)
    rigid_body.CreateStartsAsleepAttr(True)
    physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    physx_rigid_body.CreateDisableGravityAttr(False)
    physx_rigid_body.CreateAngularDampingAttr(0.2)
    physx_rigid_body.CreateLinearDampingAttr(0.05)
    physx_rigid_body.CreateSleepThresholdAttr(0.0)
    physx_rigid_body.CreateStabilizationThresholdAttr(0.0)
    # Continuous collision detection. The basket collider is a triangle mesh,
    # which has zero thickness — a discrete step large enough to straddle a wall
    # passes through it. An object dropped into the basket reaches ~0.8 m/s,
    # i.e. ~13 mm per substep, which is the same order as the wall geometry.
    physx_rigid_body.CreateEnableCCDAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(mass))
    return root


def create_kinematic_body_root(stage, path, translate, rotate_xyz_deg=None):
    """Kinematic rigid body — no gravity, scriptable transform."""
    root = define_xform(stage, path, translate=translate, rotate_xyz_deg=rotate_xyz_deg)
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid_body.CreateRigidBodyEnabledAttr(True)
    rigid_body.CreateKinematicEnabledAttr(True)
    physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    physx_rigid_body.CreateDisableGravityAttr(True)
    return root


def apply_initial_motion(root_prim, linear_velocity):
    """Mark a dynamic body awake, disable gravity, and set initial velocity.

    Gravity is disabled so hazards drift across the workspace at constant
    altitude instead of arcing down before they reach the target zone.
    """
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    rigid_body.CreateStartsAsleepAttr(False)
    rigid_body.CreateVelocityAttr(Gf.Vec3f(*[float(v) for v in linear_velocity]))
    physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
    physx_rigid_body.CreateDisableGravityAttr(True)


def build_box_collider(stage, path, size, translate=None, rotate_xyz_deg=None):
    prim = UsdGeom.Cube.Define(stage, path).GetPrim()
    UsdGeom.Cube(prim).CreateSizeAttr(1.0)
    set_xform(prim, translate=translate, rotate_xyz_deg=rotate_xyz_deg, scale=size)
    apply_static_collider(prim, approximation="boundingCube")
    UsdGeom.Imageable(prim).MakeInvisible()
    return prim


def build_sphere_collider(stage, path, radius, translate=None):
    prim = UsdGeom.Sphere.Define(stage, path).GetPrim()
    UsdGeom.Sphere(prim).CreateRadiusAttr(float(radius))
    set_xform(prim, translate=translate)
    apply_static_collider(prim, approximation="boundingSphere")
    UsdGeom.Imageable(prim).MakeInvisible()
    return prim


def build_cylinder_collider(stage, path, radius, height, translate=None, rotate_xyz_deg=None):
    prim = UsdGeom.Cylinder.Define(stage, path).GetPrim()
    cylinder = UsdGeom.Cylinder(prim)
    cylinder.CreateRadiusAttr(float(radius))
    cylinder.CreateHeightAttr(float(height))
    set_xform(prim, translate=translate, rotate_xyz_deg=rotate_xyz_deg)
    apply_static_collider(prim, approximation="convexHull")
    UsdGeom.Imageable(prim).MakeInvisible()
    return prim


def build_apple(stage, path):
    root = create_dynamic_body_root(stage, path, APPLE_TRANSLATE, mass=0.12)
    if USE_APPLE_MESH and APPLE_ASSET.exists():
        add_visual_reference(
            stage,
            f"{path}/Visual",
            APPLE_ASSET,
            translate=(APPLE_VISUAL_TRANSLATE * APPLE_SCALE).tolist(),
            rotate_xyz_deg=APPLE_ROTATION_DEG,
            scale=APPLE_SCALE,
        )
    else:
        body = UsdGeom.Sphere.Define(stage, f"{path}/Visual/Body")
        body.CreateRadiusAttr(0.055)
        set_display_color(body.GetPrim(), [0.80, 0.10, 0.08])

        stem = UsdGeom.Cylinder.Define(stage, f"{path}/Visual/Stem")
        stem.CreateRadiusAttr(0.006)
        stem.CreateHeightAttr(0.045)
        set_xform(stem.GetPrim(), translate=[0.0, 0.0, 0.065])
        set_display_color(stem.GetPrim(), [0.35, 0.22, 0.08])

        leaf = UsdGeom.Cube.Define(stage, f"{path}/Visual/Leaf")
        leaf.CreateSizeAttr(1.0)
        set_xform(leaf.GetPrim(), translate=[0.02, 0.0, 0.07], rotate_xyz_deg=[0.0, 22.0, 35.0], scale=[0.018, 0.008, 0.004])
        set_display_color(leaf.GetPrim(), [0.18, 0.45, 0.12])
    build_sphere_collider(
        stage,
        f"{path}/Collider",
        radius=float(77.5 * APPLE_SCALE[0]),
        translate=(APPLE_COLLIDER_TRANSLATE * APPLE_SCALE).tolist(),
    )
    return root


def build_red_ball(stage, path):
    root = create_dynamic_body_root(stage, path, RED_BALL_TRANSLATE, mass=0.08)
    if RED_BALL_ASSET.exists():
        add_visual_reference(
            stage,
            f"{path}/Visual",
            RED_BALL_ASSET,
            translate=(RED_BALL_VISUAL_TRANSLATE * RED_BALL_SCALE).tolist(),
            rotate_xyz_deg=RED_BALL_ROTATION_DEG,
            scale=RED_BALL_SCALE,
        )
    else:
        ball = UsdGeom.Sphere.Define(stage, f"{path}/Visual/Ball")
        ball.CreateRadiusAttr(0.045)
        set_display_color(ball.GetPrim(), [0.90, 0.08, 0.08])
    build_sphere_collider(
        stage,
        f"{path}/Collider",
        radius=float(1.93 * RED_BALL_SCALE[0]),
        translate=(RED_BALL_COLLIDER_TRANSLATE * RED_BALL_SCALE).tolist(),
    )
    return root


def build_glass(stage, path):
    root = create_dynamic_body_root(stage, path, GLASS_TRANSLATE, mass=0.18)
    if USE_GLASS_MESH and GLASS_ASSET.exists():
        add_visual_reference(
            stage,
            f"{path}/Visual",
            GLASS_ASSET,
            rotate_xyz_deg=GLASS_ROTATION_DEG,
            scale=GLASS_SCALE,
        )
    else:
        glass = UsdGeom.Cylinder.Define(stage, f"{path}/Visual/Glass")
        glass.CreateRadiusAttr(0.04)
        glass.CreateHeightAttr(0.12)
        set_display_color(glass.GetPrim(), [0.75, 0.85, 0.95])
    build_cylinder_collider(
        stage,
        f"{path}/Collider",
        radius=float(3.8 * GLASS_SCALE[0]),
        height=float(9.0 * GLASS_SCALE[2]),
        translate=(GLASS_COLLIDER_TRANSLATE * GLASS_SCALE).tolist(),
        rotate_xyz_deg=GLASS_ROTATION_DEG,
    )
    return root


def bind_omniglass_material(stage, mesh_root, material_path):
    """OmniGlass MDL 셰이더로 새 머티리얼을 만들어 mesh_root 밑의 모든 Mesh 에
    강제로 바인딩한다. 소스 GLB 의 원래 머티리얼이 뭐든 — 불투명 텍스처든, 참조
    범위 밖이라 드롭됐든(GlassBottle 이 실제로 이 케이스였다: primPath 로 leaf
    mesh 하나만 참조하니 그 머티리얼이 있던 /World/Looks/Material 은 참조 범위
    밖이라 "Ignoring" 경고와 함께 빠졌고, 결과적으로 기본 회색 불투명 렌더링이
    됐다) — 무시하고 유리로 렌더링되게 한다. 기존 컵(P_Glassware)도 같은
    OmniGlass.mdl 을 쓴다 (diag_glass_material.py 로 확인한 실제 셰이더 구조).
    파라미터를 하나도 안 건드리면 MDL 기본값(투명, IOR 1.491)이 그대로 적용된다."""
    mat = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.SetSourceAsset("OmniGlass.mdl", "mdl")
    shader.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    for prim in Usd.PrimRange(mesh_root):
        if prim.GetTypeName() == "Mesh":
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    return mat


def _build_glass_prop(stage, path, asset_path, translate, rotate_xyz_deg, scale, mass, prim_path=None):
    """유리 3종(Bottle/Bowl/Jar) 공용 빌더. 콜라이더는 이상화 도형이 아니라
    비주얼 메쉬에서 직접 convexHull 로 뽑는다 — 이 오브젝트들은 피벗/회전 부호가
    아직 실측 검증 전이라, 손으로 지정한 콜라이더 도형은 비주얼과 어긋날 위험이
    크다. convexHull 은 비주얼이 어디 있든 항상 따라가므로 그 문제를 피한다.

    머티리얼은 소스 GLB 것을 신뢰하지 않고 OmniGlass 로 강제 바인딩한다 — 형태만
    보고 고른 에셋들이라(bind_omniglass_material 참고) 처음부터 투명하게 나온다."""
    root = create_dynamic_body_root(stage, path, translate, mass=mass)
    visual = add_visual_reference(
        stage,
        f"{path}/Visual",
        asset_path,
        rotate_xyz_deg=rotate_xyz_deg,
        scale=scale,
        prim_path=prim_path,
    )
    apply_static_collider(visual, approximation="convexHull")
    set_double_sided(visual)
    bind_omniglass_material(stage, visual, f"{path}/Looks/Glass")
    return root


def build_glass_bottle(stage, path):
    return _build_glass_prop(
        stage, path, GLASS_BOTTLE_PACK_ASSET, GLASS_BOTTLE_TRANSLATE,
        GLASS_BOTTLE_ROTATION_DEG, GLASS_BOTTLE_SCALE, GLASS_BOTTLE_MASS,
        prim_path=GLASS_BOTTLE_PRIM_PATH,
    )


def build_glass_bowl(stage, path):
    return _build_glass_prop(
        stage, path, GLASS_BOWL_ASSET, GLASS_BOWL_TRANSLATE,
        GLASS_BOWL_ROTATION_DEG, GLASS_BOWL_SCALE, GLASS_BOWL_MASS,
    )


def build_glass_jar(stage, path):
    return _build_glass_prop(
        stage, path, GLASS_JAR_ASSET, GLASS_JAR_TRANSLATE,
        GLASS_JAR_ROTATION_DEG, GLASS_JAR_SCALE, GLASS_JAR_MASS,
    )


def build_glass_mug(stage, path):
    return _build_glass_prop(
        stage, path, GLASS_MUG_ASSET, GLASS_MUG_TRANSLATE,
        GLASS_MUG_ROTATION_DEG, GLASS_MUG_SCALE, GLASS_MUG_MASS,
    )


def build_basket(stage, path):
    root = define_xform(
        stage,
        path,
        translate=BASKET_TRANSLATE,
        rotate_xyz_deg=BASKET_ROTATION_DEG,
    )
    if BASKET_ASSET.exists():
        visual = add_visual_reference(
            stage,
            f"{path}/Visual",
            BASKET_ASSET,
            scale=BASKET_SCALE,
        )
        # The basket had NO collider at all — visual geometry only — so anything
        # placed in it fell straight through to the table.
        #
        # approximation="none" (exact triangle mesh) rather than the convexHull
        # default: a convex hull caps the opening, so the object would rest on
        # top of the rim instead of going inside. Triangle-mesh colliders are
        # static-only in PhysX, which is fine — the basket is a static prop with
        # no RigidBodyAPI. Same reasoning as the table-top fix below.
        apply_static_collider(visual, approximation="none")
        n = set_double_sided(visual)
        print(f"[basket] triangle-mesh collider + doubleSided on {n} gprim(s)")
    return root


def build_phone(stage, path):
    """Flat-topped stacking destination — static, like the basket.

    Static rather than a dynamic body on purpose: a released ball lands on it,
    and a light rigid body would be shoved out from under its own place pose.
    Nothing picks the phone up, so it never needs to be dynamic.
    """
    root = define_xform(
        stage,
        path,
        translate=PHONE_TRANSLATE,
        rotate_xyz_deg=PHONE_ROTATION_DEG,
    )
    if PHONE_ASSET.exists():
        visual = add_visual_reference(
            stage,
            f"{path}/Visual",
            PHONE_ASSET,
            scale=PHONE_SCALE,
        )
        # Same reason as the basket: a culled face writes no depth, and the top
        # camera's bbox_3d_world.max.z on this prim IS the height the ball is
        # released at. An inverted winding would measure the table underneath.
        n = set_double_sided(visual)
        print(f"[phone] doubleSided on {n} gprim(s)")
    else:
        print(f"[phone] 에셋 없음 — 건너뜀: {PHONE_ASSET}")
        return root
    build_box_collider(
        stage,
        f"{path}/Collider",
        size=(PHONE_COLLIDER_SIZE * PHONE_SCALE).tolist(),
        translate=(PHONE_COLLIDER_TRANSLATE * PHONE_SCALE).tolist(),
    )
    return root


def build_book(stage, path):
    root = create_dynamic_body_root(stage, path, BOOK_TRANSLATE, mass=BOOK_MASS)
    set_xform(root, rotate_xyz_deg=BOOK_ROTATION_DEG)
    physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    physx_rigid_body.CreateAngularDampingAttr(2.5)
    physx_rigid_body.CreateLinearDampingAttr(0.3)
    if BOOK_ASSET.exists():
        add_visual_reference(
            stage,
            f"{path}/Visual",
            BOOK_ASSET,
            rotate_xyz_deg=[0.0, 0.0, 0.0],
            scale=BOOK_SCALE,
        )
    else:
        cover = UsdGeom.Cube.Define(stage, f"{path}/Visual/Cover")
        cover.CreateSizeAttr(1.0)
        set_xform(cover.GetPrim(), scale=[0.13, 0.09, 0.012])
        set_display_color(cover.GetPrim(), [0.14, 0.28, 0.62])

        pages = UsdGeom.Cube.Define(stage, f"{path}/Visual/Pages")
        pages.CreateSizeAttr(1.0)
        set_xform(pages.GetPrim(), translate=[0.0, 0.0, 0.005], scale=[0.118, 0.078, 0.009])
        set_display_color(pages.GetPrim(), [0.94, 0.93, 0.88])
    build_box_collider(
        stage,
        f"{path}/Collider",
        size=(np.array([1.49, 0.38, 2.45]) * BOOK_SCALE).tolist(),
        translate=(BOOK_COLLIDER_TRANSLATE * BOOK_SCALE).tolist(),
        rotate_xyz_deg=[0.0, 0.0, 0.0],
    )
    return root


def build_tabletop_items(stage, root_path):
    if stage.GetPrimAtPath(root_path):
        stage.RemovePrim(root_path)
    props_root = define_xform(stage, root_path, translate=BEDSIDE_TABLE_POSITION, rotate_xyz_deg=BEDSIDE_TABLE_ROTATION_DEG)
    # 2026-09 — 투명물체 depth-restoration 실험을 위해 오파크 클러터(Apple/RedBall/
    # Book/Basket/Phone)는 빼고 유리 오브젝트만 테이블에 올린다. build_apple 등
    # 함수 자체는 지워지지 않았다 — 다른 데모/씬으로 되돌릴 때 이 5줄만 복구하면 됨.
    build_glass(stage, f"{props_root.GetPath()}/Glass")
    build_glass_bottle(stage, f"{props_root.GetPath()}/GlassBottle")
    build_glass_bowl(stage, f"{props_root.GetPath()}/GlassBowl")
    build_glass_jar(stage, f"{props_root.GetPath()}/GlassJar")
    build_glass_mug(stage, f"{props_root.GetPath()}/GlassMug")
    return props_root


def build_hazard_box(stage, path, translate=None, rotation_deg=None, initial_velocity=None):
    """small_box hazard — uses Cardbox_C2 USD from XR Content if available.

    Spawn/motion can be overridden so the box can stand in for the bottle in
    capture mode (ROBOT_CAPSTONE_HAZARD_OBJECT=box): same spawn pose, zero
    initial velocity, and the /hazard/launch_bottle trigger gives it flight.
    """
    if translate is None:
        translate = HAZARD_BOX_TRANSLATE
    if rotation_deg is None:
        rotation_deg = HAZARD_BOX_ROTATION_DEG
    if initial_velocity is None:
        initial_velocity = HAZARD_BOX_LINEAR_VELOCITY
    root = create_dynamic_body_root(stage, path, translate, mass=0.4)
    set_xform(root, rotate_xyz_deg=rotation_deg)
    if CARDBOX_ASSET.exists():
        add_visual_reference(
            stage,
            f"{path}/Visual",
            CARDBOX_ASSET,
            scale=HAZARD_BOX_SCALE,
        )
    else:
        body = UsdGeom.Cube.Define(stage, f"{path}/Visual/Body").GetPrim()
        UsdGeom.Cube(body).CreateSizeAttr(1.0)
        set_xform(body, scale=HAZARD_BOX_FALLBACK_SIZE)
        set_display_color(body, [0.72, 0.55, 0.32])
    build_box_collider(
        stage,
        f"{path}/Collider",
        size=HAZARD_BOX_FALLBACK_SIZE.tolist(),
    )
    apply_initial_motion(root, initial_velocity)
    return root


def build_hazard_bottle(stage, path):
    """pet_bottle hazard — references the NaturalBostonRound PET USD used to
    build the v3 capture dataset, so train and inference share the same render.
    """
    root = create_dynamic_body_root(stage, path, HAZARD_BOTTLE_TRANSLATE, mass=0.25)
    set_xform(root, rotate_xyz_deg=HAZARD_BOTTLE_ROTATION_DEG)
    add_visual_reference(
        stage,
        f"{path}/Visual",
        HAZARD_BOTTLE_ASSET,
        scale=HAZARD_BOTTLE_SCALE,
    )
    build_cylinder_collider(
        stage,
        f"{path}/Collider",
        radius=HAZARD_BOTTLE_RADIUS,
        height=HAZARD_BOTTLE_HEIGHT,
    )
    # Stationary (gravity off, zero velocity) until /hazard/launch_bottle fires —
    # the bottle then gets HAZARD_BOTTLE_LINEAR_VELOCITY so it flies in synced with
    # the arm motion. See _bottle_launch_loop / _apply_bottle_launch_velocity.
    apply_initial_motion(root, np.zeros(3))
    return root


def build_hazard_arm(stage, path):
    """hand/forearm placeholder — dynamic so it flies in with initial velocity."""
    root = create_dynamic_body_root(stage, path, HAZARD_ARM_TRANSLATE, mass=0.6)
    set_xform(root, rotate_xyz_deg=HAZARD_ARM_ROTATION_DEG)
    forearm = UsdGeom.Cylinder.Define(stage, f"{path}/Visual/Forearm").GetPrim()
    UsdGeom.Cylinder(forearm).CreateRadiusAttr(float(HAZARD_FOREARM_RADIUS))
    UsdGeom.Cylinder(forearm).CreateHeightAttr(float(HAZARD_FOREARM_LENGTH))
    set_display_color(forearm, [0.92, 0.76, 0.66])
    hand = UsdGeom.Sphere.Define(stage, f"{path}/Visual/Hand").GetPrim()
    UsdGeom.Sphere(hand).CreateRadiusAttr(float(HAZARD_HAND_RADIUS))
    hand_offset_z = HAZARD_FOREARM_LENGTH * 0.5 + HAZARD_HAND_RADIUS * 0.6
    set_xform(hand, translate=[0.0, 0.0, hand_offset_z])
    set_display_color(hand, [0.96, 0.82, 0.72])
    build_cylinder_collider(
        stage,
        f"{path}/ColliderForearm",
        radius=HAZARD_FOREARM_RADIUS,
        height=HAZARD_FOREARM_LENGTH,
    )
    build_sphere_collider(
        stage,
        f"{path}/ColliderHand",
        radius=HAZARD_HAND_RADIUS,
        translate=[0.0, 0.0, hand_offset_z],
    )
    apply_initial_motion(root, HAZARD_ARM_LINEAR_VELOCITY)
    return root


def build_capture_humans(stage, root_path):
    """Static human prim(s) for hand/forearm capture. Visual reference only —
    no rigid body, no collider, no motion. Move/rotate/scale in the viewport
    (G/R/S keys or gizmos) while capturing top + ee shots. Swap the URL
    constant to cycle characters. Remove this call once dataset is complete.

    Wrapper-Xform pattern: outer rotate lives on Subject, the asset reference
    on /Body. This way Subject's outer rotation COMPOSES with the asset's
    baked /Root rotation instead of overriding it (the hazard builders do the
    same).
    """
    if stage.GetPrimAtPath(root_path):
        stage.RemovePrim(root_path)
    humans_root = define_xform(stage, root_path)
    subject = define_xform(
        stage,
        f"{humans_root.GetPath()}/Subject",
        translate=CAPTURE_HUMAN_TRANSLATE,
        rotate_xyz_deg=CAPTURE_HUMAN_ROTATION_DEG,
        scale=CAPTURE_HUMAN_SCALE,
    )
    add_visual_reference(
        stage,
        f"{subject.GetPath()}/Body",
        CAPTURE_HUMAN_ASSET,
    )
    return humans_root


def build_hazards(stage, root_path):
    if stage.GetPrimAtPath(root_path):
        stage.RemovePrim(root_path)
    hazards_root = define_xform(stage, root_path)
    # Active hazard chosen by ROBOT_CAPSTONE_HAZARD_OBJECT (bottle | box). Both
    # spawn stationary at the bottle's pose and respond to /hazard/launch_bottle
    # so the capture helper / hybrid demos stay identical across assets.
    if HAZARD_OBJECT == "box":
        build_hazard_box(
            stage,
            f"{hazards_root.GetPath()}/HazardBox",
            translate=HAZARD_BOTTLE_TRANSLATE,
            rotation_deg=HAZARD_BOTTLE_ROTATION_DEG,
            initial_velocity=np.zeros(3),
        )
    else:
        build_hazard_bottle(stage, f"{hazards_root.GetPath()}/HazardBottle")
    # build_hazard_arm(stage, f"{hazards_root.GetPath()}/HazardArm")
    return hazards_root


def find_robot_root(stage):
    for prim in stage.Traverse():
        if prim.GetName().lower() == "kinova":
            return prim
    return None


# ── Table collision approximation ─────────────────────────────────────────────
# simple_room.usd 의 상판(table_low) 은 physics:approximation = convexDecomposition
# 으로 저작돼 있다. VHACD 계열 볼록 분해는 메쉬를 복셀화해 볼록 덩어리(최대 64개)
# 로 근사하므로 hull 이 원본 표면보다 위로 부푼다. 실측 결과 컵 자리에서
#   비주얼 상판 = -0.00137   /   충돌 표면 = +0.01526   → 16.6mm 어긋남
# 이 때문에 상판에 정확히 올려둔 물체가 충돌 형상 안에 파묻힌 채로 시작하고,
# 그리퍼가 건드려 깨우는 순간 PhysX 가 침투를 해소하며 물체를 위로 튕겼다("뿅").
#
# 볼록 분해는 *동적* 강체에만 필요한 제약이다. 테이블은 정적이라 삼각형 메쉬
# 충돌을 그대로 쓸 수 있고 그게 정확하다. 에셋에 physxCookedData:triangleMesh
# 가 이미 구워져 있어(195KB) 쿠킹 비용도 추가로 들지 않는다.
TABLE_COLLIDER_PATH = "/background/table_low_327/table_low"


def fix_table_collision(stage):
    """상판 콜라이더를 볼록 분해 → 정확 삼각형 메쉬로 교체."""
    prim = stage.GetPrimAtPath(TABLE_COLLIDER_PATH)
    if not prim or not prim.IsValid():
        print(f"[table-collision] {TABLE_COLLIDER_PATH} 없음 — 건너뜀")
        return
    attr = prim.GetAttribute("physics:approximation")
    if not attr:
        attr = UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr()
    before = attr.Get()
    attr.Set("none")          # "none" = 근사 없음 = 원본 삼각형 메쉬
    print(f"[table-collision] approximation {before} → none (정확 삼각형 메쉬)")


# ── Gripper friction (Robotiq 2F-85 fingertip pads) ───────────────────────────
# Isaac Sim default PhysX material friction (~0.5) is too low for reliable
# top-down grasps of flat/thin objects (book). Boost static/dynamic friction
# on both fingertip pads so closed gripper holds the object during retreat & motion.
# NOTE: static/dynamic friction values below are still the Panda-era numbers,
# ported over structurally but NOT re-verified against the Robotiq pad geometry —
# retune against live grasp tests before trusting them.

FINGER_NAME_CANDIDATES = ("robotiq_85_left_finger_tip_link", "robotiq_85_right_finger_tip_link")
GRIPPER_FRICTION_MATERIAL_PATH = "/World/PhysicsMaterials/GripperHighFriction"
GRIPPER_STATIC_FRICTION  = 4.0   # very high — flat book slips easily otherwise
GRIPPER_DYNAMIC_FRICTION = 3.0   # slightly lower than static
GRIPPER_RESTITUTION      = 0.0   # no bounce


def _ensure_gripper_friction_material(stage):
    """Define (or fetch) the shared physics material applied to both fingers."""
    mat_path = Sdf.Path(GRIPPER_FRICTION_MATERIAL_PATH)
    existing = stage.GetPrimAtPath(mat_path)
    if existing.IsValid():
        return existing
    # Parent /World/PhysicsMaterials Xform (no-op if already exists).
    define_xform(stage, str(mat_path.GetParentPath()))
    material      = UsdShade.Material.Define(stage, mat_path)
    physics_api   = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_api.CreateStaticFrictionAttr().Set(GRIPPER_STATIC_FRICTION)
    physics_api.CreateDynamicFrictionAttr().Set(GRIPPER_DYNAMIC_FRICTION)
    physics_api.CreateRestitutionAttr().Set(GRIPPER_RESTITUTION)
    # PhysxMaterialAPI exposes friction combine mode (max takes the higher
    # of the two contacting materials — ensures gripper friction wins even
    # if the object's own material is slippery).
    physx_api = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_api.CreateFrictionCombineModeAttr().Set("max")
    physx_api.CreateRestitutionCombineModeAttr().Set("min")
    return material.GetPrim()


def _bind_physics_material(target_prim, material_prim):
    """Bind a physics material to a prim (or any of its collision descendants).

    Returns True if at least one binding was applied.
    """
    bound_any = False
    for prim in Usd.PrimRange(target_prim):
        # Bind to anything that participates in physics — collision prims
        # are the surfaces the gripper actually touches. Binding on the
        # finger root alone doesn't always propagate down through nested
        # collision meshes, so we walk the subtree.
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
            binding_api.Bind(
                UsdShade.Material(material_prim),
                bindingStrength=UsdShade.Tokens.weakerThanDescendants,
                materialPurpose="physics",
            )
            bound_any = True
    if not bound_any:
        # Fallback: bind to the root prim directly — covers cases where
        # collisions are defined via PhysxCookedDataAPI or similar without
        # an explicit CollisionAPI marker.
        binding_api = UsdShade.MaterialBindingAPI.Apply(target_prim)
        binding_api.Bind(
            UsdShade.Material(material_prim),
            bindingStrength=UsdShade.Tokens.weakerThanDescendants,
            materialPurpose="physics",
        )
        bound_any = True
    return bound_any


# ── Gripper drive strength — Panda 시절 기록 (2026-09 Kinova Gen3 전환으로 지금
# 아래 코드가 쓰는 값은 아님 — Robotiq 2F-85 튜닝은 더 아래 새 섹션 참고. 다만
# "힘 = min(stiffness×오차, maxForce)"라는 진단 프레임 자체는 그대로 재사용
# 가능해서 지우지 않고 남겨둔다) ─────────────────────────────────────────────
# Isaac 기본 Franka 에셋의 손가락 드라이브는 maxForce=7.2N / stiffness=400 이다.
# 실제 Franka 그리퍼는 연속 70N 을 내므로 한참 약하다. 유리컵(Ø48mm 콜라이더)에서
# 관측된 증상: CLOSE 시 손가락이 pos=0.0257m(폭 51.4mm)에서 멈춤 — 컵보다 3.2mm
# 넓은 지점이다. 요구 힘 = stiffness × 오차 = 400 × 0.0257 = 10.3N 인데 maxForce
# 7.2N 에서 잘려 더 조이지 못한 것. 정상력이 부족하니 μ=4 여도 마찰이 안 나오고,
# 들어 올릴 때 컵이 미끄러져 빠졌다.
#
# stiffness 를 같이 올리는 이유: maxForce 만 올려도 400×0.0257=10.3N 이 상한이라
# 큰 차이가 없다. 힘은 min(stiffness × 오차, maxForce) 로 결정된다.
# damping 은 stiffness/damping 비(=5)를 원본과 동일하게 유지 — 솔버 진동 방지.
#
# ── maxForce 50 → 12 (2026-07-31) ────────────────────────────────────────────
# 50N 은 과했다. gripper_action_server 는 CLOSE 목표를 0.0mm(완전 닫힘)로 주는데
# 컵이 23.7mm 에서 막으므로 오차가 계속 남는다 → 요구력 5000×0.0237 = 118N 이
# maxForce 로 잘려 **50N 이 파지 내내 걸린다**. 컵 무게는 1.8N 이다.
#
# 증상: pick 은 성공(pre_grasp/grasp/CLOSE 전부 SUCCESS, 23.70mm 접촉)하는데
# retreat 에서 local planner 가 "stuck for several iterations" 로 abort. grasp
# (아래로 120mm)와 retreat(위로 150mm)는 제약이 완전히 동일하고 컵을 쥐었는지만
# 다르다 → 계획 문제가 아니라 팔이 물리적으로 못 올라가는 것. 정확 삼각형 메쉬로
# 바꾼 상판(fix_table_collision)에 컵이 간격 0 으로 닿아 있어, 50N 으로 짓누르면
# 컵이 메쉬에 박혀 팔이 못 든다는 가설.
#
# 12N 근거: 필요한 건 컵을 놓치지 않을 만큼이지 최대 악력이 아니다.
# μ=4 이므로 마찰 = 12×4 = 48N, 컵 무게 1.8N 의 26배 여유. 원래 문제였던 기본값
# 7.2N 보다는 67% 강해 조임 부족(51.4mm 정지)도 재발하지 않는다.
#
# ── stiffness 5000 → 1000 (2026-08-06) ───────────────────────────────────────
# stiffness 는 여기서 힘을 전혀 늘리지 못한다. 힘 = min(stiffness × 오차,
# maxForce) 인데 오차가 항상 크기 때문이다:
#   책   손가락 13.3mm → 12N 포화에 필요한 stiffness =  902 N/m
#   컵·사과·공  34mm  →                              353 N/m
# 즉 1000 이면 전부 maxForce 에 포화한다. 5000 이 실제로 바꾸는 건 닫는 속도
# 뿐이다 — 종단 속도 ≈ (stiffness/damping) × 이동거리 이므로 책에서
#   5000/1000 × 26.7mm = 134 mm/s   →   1000/1000 × 26.7mm = 27 mm/s
# 로 5배 느려진다. damping 은 일부러 1000 으로 유지한다(비를 5→1 로 낮춰야
# 속도가 준다).
#
# 증상: 테이블에 자립한 책(104mm 높이, 26.6mm 두께, 0.15kg)을 윗부분에서 물 때
# 양쪽 손가락이 시차를 두고 134mm/s 로 때려 책이 밀리고 계속 떨렸다. 심하면
# 하강 중 손가락에 걸려 넘어졌다. 무게는 원인이 아니다 — 필요 압축력은
# 1.47N/(2×4) = 0.18N 인데 12N 을 걸고 있었다(65배).
# ── Kinova Gen3 / Robotiq 2F-85 (2026-09) ─────────────────────────────────────
# Panda 는 독립 2-프리즈매틱 손가락(drive:linear:...)이었지만 Robotiq 2F-85 는
# 단일 구동 리볼루트 조인트(robotiq_85_left_knuckle_joint, drive:angular:...)
# 하나가 나머지 5개 조인트를 NewtonMimicAPI(mimicJoint/mimicCoef1, USD 임포터가
# 이미 정확히 authoring 해둠 — PhysxMimicJointAPI 는 6.0.1 에서 deprecated 되고
# NewtonMimicAPI 로 대체됨)로 따라가는 구조라 이 구동 조인트 하나만 튜닝하면 된다.
# ★ 단위 함정: UsdPhysics RevoluteJoint 의 limit/target 은 라디안이 아니라
# **도(degree)** 다 — SRDF/URDF 의 Close=0.8rad 은 USD 쪽 upperLimit=45.84° 로
# 이미 변환되어 들어있다(임포터가 처리함). gripper_action_server 등에서 라디안
# ↔ 도 변환을 빠뜨리지 않도록 주의.
# 아래 stiffness/damping 은 임포트 직후 기본값이 0(전혀 안 움직임)이라 최소한
# 동작은 하도록 새로 채운 시작값일 뿐, Panda 때처럼 실측으로 검증되지 않았다 —
# 실제 파지 테스트로 반드시 재튜닝할 것. maxForce=50 은 URDF 임포터가 넣어준
# 값(에셋 기본값)을 그대로 유지.
GRIPPER_DRIVE_JOINT_NAMES = ("robotiq_85_left_knuckle_joint",)
GRIPPER_DRIVE_STIFFNESS = 5.0   # 미검증 시작값 — 실측 필요
GRIPPER_DRIVE_DAMPING   = 5.0   # 미검증 시작값 — 실측 필요
GRIPPER_DRIVE_MAX_FORCE = 50.0  # 에셋 기본값 유지


def fix_gripper_mimic_limits(stage):
    """NewtonMimicAPI follower 조인트 중 physics:lowerLimit/upperLimit 이 비어있는
    것들을 구동 조인트(robotiq_85_left_knuckle_joint) 범위로, mimicCoef1 부호에
    맞춰 채운다.

    URDF 임포터가 robotiq_85_left_inner_knuckle_joint / right_inner_knuckle_joint /
    left_finger_tip_joint / right_finger_tip_joint 4개에는 limit 을 안 채워
    넣었는데, PhysX 는 "NewtonMimicAPI follower joint ... without a finite
    limit" 로 이 4개를 거부한다 — 거부된 조인트는 구속이 안 걸린 채로 남아
    그리퍼 전체가 겉돌며 빙글빙글 도는 원인이 된다.
    """
    robot_root = find_robot_root(stage)
    if robot_root is None:
        print("[gripper-mimic-limits] Kinova root not found — skipped")
        return

    driver = None
    for prim in Usd.PrimRange(robot_root):
        if prim.GetName() in GRIPPER_DRIVE_JOINT_NAMES:
            driver = prim
            break
    if driver is None:
        print("[gripper-mimic-limits] driver joint not found — skipped")
        return
    lower_attr = driver.GetAttribute("physics:lowerLimit")
    upper_attr = driver.GetAttribute("physics:upperLimit")
    lo = lower_attr.Get() if lower_attr else None
    hi = upper_attr.Get() if upper_attr else None
    if lo is None or hi is None:
        print("[gripper-mimic-limits] driver limit not found — skipped")
        return

    fixed = []
    for prim in Usd.PrimRange(robot_root):
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        mimic_rel = prim.GetRelationship("newton:mimicJoint")
        if not mimic_rel or not mimic_rel.GetTargets():
            continue
        existing_lower = prim.GetAttribute("physics:lowerLimit")
        # 주의: 미authoring 상태에서도 Get() 은 None 이 아니라 스키마 기본값
        # (-inf) 를 반환한다 — HasAuthoredValue() 로만 "진짜 authoring 됐는지"
        # 를 구분할 수 있다. Get() is not None 으로 체크하면 항상 참이라
        # 정확히 반대로 동작(고쳐야 할 조인트를 계속 건너뜀)한다.
        if existing_lower and existing_lower.HasAuthoredValue():
            continue  # 이미 채워져 있음(구동 조인트 자신 포함)
        coef_attr = prim.GetAttribute("newton:mimicCoef1")
        coef = coef_attr.Get() if coef_attr and coef_attr.Get() is not None else 1.0
        joint = UsdPhysics.RevoluteJoint(prim)
        if coef < 0:
            joint.CreateLowerLimitAttr().Set(-hi)
            joint.CreateUpperLimitAttr().Set(-lo)
        else:
            joint.CreateLowerLimitAttr().Set(lo)
            joint.CreateUpperLimitAttr().Set(hi)
        fixed.append(prim.GetName())

    if fixed:
        print(f"[gripper-mimic-limits] limit 채움({lo}~{hi} 기준): {fixed}")
    else:
        print("[gripper-mimic-limits] 손볼 조인트 없음")


def boost_gripper_drive(stage):
    """그리퍼 구동 리볼루트 조인트의 드라이브 강성/최대힘을 설정한다."""
    robot_root = find_robot_root(stage)
    if robot_root is None:
        print("[gripper-drive] Kinova root not found — skipped")
        return

    touched = []
    for prim in Usd.PrimRange(robot_root):
        if prim.GetName() not in GRIPPER_DRIVE_JOINT_NAMES:
            continue
        max_force_attr = prim.GetAttribute("drive:angular:physics:maxForce")
        if not max_force_attr:
            print(f"[gripper-drive] {prim.GetName()}: angular drive 없음 — 건너뜀")
            continue
        stiff_attr = prim.GetAttribute("drive:angular:physics:stiffness")
        if not stiff_attr:
            stiff_attr = UsdPhysics.DriveAPI.Apply(prim, "angular").CreateStiffnessAttr()
        damp_attr = prim.GetAttribute("drive:angular:physics:damping")
        if not damp_attr:
            damp_attr = UsdPhysics.DriveAPI.Apply(prim, "angular").CreateDampingAttr()
        before = (stiff_attr.Get(), damp_attr.Get(), max_force_attr.Get())
        stiff_attr.Set(GRIPPER_DRIVE_STIFFNESS)
        damp_attr.Set(GRIPPER_DRIVE_DAMPING)
        max_force_attr.Set(GRIPPER_DRIVE_MAX_FORCE)
        touched.append(f"{prim.GetName()} {before} → "
                       f"({GRIPPER_DRIVE_STIFFNESS}, {GRIPPER_DRIVE_DAMPING}, "
                       f"{GRIPPER_DRIVE_MAX_FORCE})")

    if touched:
        print("[gripper-drive] " + " | ".join(touched))
    else:
        print("[gripper-drive] 구동 조인트를 찾지 못함 — 변경 없음")


# ── Arm joint drive (Kinova Gen3, 2026-09) ────────────────────────────────────
# URDF 임포터가 joint_1..7 에 drive:angular:physics:maxForce 만 채워넣고(1-4=39N·m,
# 5-7=9N·m — Kinova Gen3 실제 관절 정격 토크와 일치, 손대지 않는다) stiffness/
# damping 은 0으로 남겨뒀다. 힘 = stiffness×오차 + damping×속도오차 인데 stiffness
# 가 0이면 목표 위치를 줘도 중력을 버틸 힘이 전혀 안 나온다 — Play 시 팔 전체가
# 흐물흐물 무너지는 원인. Franka 에셋은 Isaac 이 미리 큐레이션해서 이 값이
# 채워져 있었는데, 이번엔 생짜 URDF 임포트라 없다.
# 아래 값은 "오차가 조금만 나도 maxForce 에 바로 포화" 시키는 전략으로 고른
# 시작값일 뿐 — 실측 중력 처짐/떨림으로 반드시 재검증할 것. UsdPhysics
# RevoluteJoint 의 각도 단위는 라디안이 아니라 도(度)라는 점도 그리퍼와 동일.
ARM_JOINT_NAMES = tuple(f"joint_{i}" for i in range(1, 8))
ARM_DRIVE_STIFFNESS = 50.0  # 미검증 시작값 — 실측 필요
ARM_DRIVE_DAMPING   = 50.0  # 미검증 시작값 — 실측 필요
# kortex_gen3_7dof_robotiq_2f_85_moveit_config 의 gen3.srdf "Home" group_state
# (라디안: [0, 0.26, 3.14, -2.27, 0, 0.96, 1.57]) 를 그대로 도(度)로 옮긴 값.
# config/robot_defaults.yaml 의 home_joint_values 와 반드시 같은 값을 가리켜야
# 한다 — 저 파일이 ROS 쪽(BT MoveToHome 등) 소스, 이 상수가 Isaac 쪽(물리
# 드라이브 목표) 소스로 둘이 분리돼 있다. Panda 때는 이 타겟이 Isaac 큐레이션
# 에셋 자체에 미리 박혀 있어서 setup_initial_scene.py 가 손댈 필요가 없었는데,
# 생짜 URDF 임포트는 아무 목표도 없어 Play 해도 처음 자세 그대로 가만히 있다.
# 카메라가 실제로 테이블을 보는지는 라이브로 검증 필요 — Kinova 정격 Home 이지
# 이 씬의 관측 자세로 검증된 값이 아니다.
KINOVA_HOME_JOINT_DEG = {
    "joint_1": 0.0,
    "joint_2": 14.8968,
    "joint_3": 179.9087,
    "joint_4": -130.0613,
    "joint_5": 0.0,
    "joint_6": 55.0035,
    "joint_7": 89.9544,
}


def boost_arm_drive(stage):
    """팔 관절(joint_1..7)의 드라이브 강성/댐핑/목표자세를 채운다(기본값 0 → 중력에 처짐, 목표 없음 → 제자리 고정)."""
    robot_root = find_robot_root(stage)
    if robot_root is None:
        print("[arm-drive] Kinova root not found — skipped")
        return

    touched = []
    for prim in Usd.PrimRange(robot_root):
        name = prim.GetName()
        if name not in ARM_JOINT_NAMES:
            continue
        max_force_attr = prim.GetAttribute("drive:angular:physics:maxForce")
        if not max_force_attr:
            print(f"[arm-drive] {name}: angular drive 없음 — 건너뜀")
            continue
        stiff_attr = prim.GetAttribute("drive:angular:physics:stiffness")
        if not stiff_attr:
            stiff_attr = UsdPhysics.DriveAPI.Apply(prim, "angular").CreateStiffnessAttr()
        damp_attr = prim.GetAttribute("drive:angular:physics:damping")
        if not damp_attr:
            damp_attr = UsdPhysics.DriveAPI.Apply(prim, "angular").CreateDampingAttr()
        target_attr = prim.GetAttribute("drive:angular:physics:targetPosition")
        if not target_attr:
            target_attr = UsdPhysics.DriveAPI.Apply(prim, "angular").CreateTargetPositionAttr()
        before = (stiff_attr.Get(), damp_attr.Get(), max_force_attr.Get(), target_attr.Get())
        target_deg = KINOVA_HOME_JOINT_DEG[name]
        stiff_attr.Set(ARM_DRIVE_STIFFNESS)
        damp_attr.Set(ARM_DRIVE_DAMPING)
        target_attr.Set(target_deg)
        touched.append(f"{name} {before} → ({ARM_DRIVE_STIFFNESS}, {ARM_DRIVE_DAMPING}, {max_force_attr.Get()}, target={target_deg})")

    if touched:
        print("[arm-drive] " + " | ".join(touched))
    else:
        print("[arm-drive] 팔 조인트를 찾지 못함 — 변경 없음")


# PhysX gives a contact ZERO torsional friction by default
# (torsionalPatchRadius = 0), so nothing resists rotation about the contact
# normal. Two flat pads pinching a face therefore let the object spin freely
# about the pinch axis — a real rubber pad deforms into a patch and does resist
# it. That is what made the book swing and hang off the fingertips after the
# lift: friction alone cannot stop rotation (μ 4.0 × 12N = 96N of grip against a
# 1.47N book is never the limit), only a patch can.
#
# minTorsionalPatchRadius forces a floor on the patch even when the geometric
# contact computes to a line or point. Torsional capacity ≈ μ × N × radius =
# 4 × 12 × 0.008 = 0.38 N·m, against a worst-case gravity torque of
# 1.47N × 85mm (grip at the very end of the 171.5mm book) = 0.125 N·m.
GRIPPER_TORSIONAL_PATCH_M = 0.008


def _apply_torsional_patch(target_prim):
    """Give the finger colliders a minimum torsional friction patch."""
    applied = 0
    for prim in Usd.PrimRange(target_prim):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        physx_col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        physx_col.CreateMinTorsionalPatchRadiusAttr().Set(
            GRIPPER_TORSIONAL_PATCH_M)
        physx_col.CreateTorsionalPatchRadiusAttr().Set(
            GRIPPER_TORSIONAL_PATCH_M)
        applied += 1
    return applied


def apply_gripper_friction(stage):
    """Apply the high-friction PhysX material to both Robotiq fingertip pads."""
    robot_root = find_robot_root(stage)
    if robot_root is None:
        print("[gripper-friction] Kinova root not found — skipped")
        return

    material_prim = _ensure_gripper_friction_material(stage)
    applied = []
    patched = 0
    for finger_name in FINGER_NAME_CANDIDATES:
        finger_prim = find_descendant_by_candidates(robot_root, [finger_name])
        if finger_prim is None:
            print(f"[gripper-friction] {finger_name} not found under Kinova")
            continue
        patched += _apply_torsional_patch(finger_prim)
        if _bind_physics_material(finger_prim, material_prim):
            applied.append(str(finger_prim.GetPath()))
    print(f"[gripper-friction] torsional patch "
          f"{GRIPPER_TORSIONAL_PATCH_M * 1000:.0f}mm on {patched} collider(s)")

    if applied:
        print(
            f"[gripper-friction] static={GRIPPER_STATIC_FRICTION} "
            f"dynamic={GRIPPER_DYNAMIC_FRICTION} bound to: {applied}"
        )
    else:
        print("[gripper-friction] No fingers found — material defined but unbound")


def find_descendant_by_candidates(root_prim, candidates):
    """candidates 를 우선순위 순서로 하나씩 찾는다 — 후보 전체를 한 번의 순회에서
    집합으로 묶어 검사하면, 서로 조상-자손 관계인 이름들이 섞였을 때(Gen3는
    bracelet_link ⊃ end_effector_link ⊃ robotiq_85_base_link 로 깊이 중첩)
    리스트 순서와 무관하게 DFS 가 먼저 만나는 "가장 얕은" 조상이 이겨버린다
    (실측: robotiq_85_base_link 를 1순위로 넣어도 bracelet_link 에 마운트됨).
    후보마다 별도로 전체를 순회해서 순서를 제대로 지킨다."""
    if root_prim is None:
        return None
    for name in candidates:
        target = name.lower()
        for prim in Usd.PrimRange(root_prim):
            if prim.GetName().lower() == target:
                return prim
    return None


def create_camera(stage, path, translate, rotate_xyz_deg=None, focal_length_mm=1.93):
    camera = UsdGeom.Camera.Define(stage, path)
    set_xform(camera.GetPrim(), translate=translate, rotate_xyz_deg=rotate_xyz_deg)
    camera.CreateFocalLengthAttr(float(focal_length_mm))
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.02, 1000.0))
    camera.CreateHorizontalApertureAttr(3.84)
    camera.CreateVerticalApertureAttr(2.16)
    return camera.GetPrim()


def get_ee_mount_prim(robot_root):
    return find_descendant_by_candidates(robot_root, WRIST_MOUNT_CANDIDATES) or find_descendant_by_candidates(
        robot_root, EE_MOUNT_FALLBACK_CANDIDATES
    )


def deactivate_legacy_ee_cameras(stage):
    legacy_paths = [
        # Gen3 URDF-import 에 딸려온 실물 내장 손목 카메라(우리는 안 씀 — 우리
        # 자체 EEViewCameraMount 를 robotiq_85_base_link 밑에 새로 짓는다).
        f"{KINOVA_ROBOTIQ_BASE_LINK_PATH.rsplit('/', 1)[0]}/camera_link",
        f"{KINOVA_ROBOTIQ_BASE_LINK_PATH.rsplit('/', 1)[0]}/camera_depth_frame",
        f"{KINOVA_ROBOTIQ_BASE_LINK_PATH.rsplit('/', 1)[0]}/camera_color_frame",
        "/World/CapstoneAdditions/EEViewCamera",
    ]
    for path in legacy_paths:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            prim.SetActive(False)


def create_ee_camera(stage):
    deactivate_legacy_ee_cameras(stage)
    robot_root = find_robot_root(stage)
    pose_source = get_ee_mount_prim(robot_root) if robot_root else None
    if pose_source is None:
        pose_source = stage.GetPrimAtPath(KINOVA_ROBOTIQ_BASE_LINK_PATH)
    if pose_source is None or not pose_source.IsValid():
        pose_source = stage.GetPrimAtPath(KINOVA_ROBOTIQ_BASE_LINK_PATH.rsplit("/", 1)[0])

    parent_path = str(pose_source.GetPath())
    mount_path = f"{parent_path}/EEViewCameraMount"
    if stage.GetPrimAtPath(mount_path):
        stage.RemovePrim(mount_path)
    mount = define_xform(
        stage,
        mount_path,
        translate=EE_CAMERA_MOUNT_TRANSLATE,
    )
    rig = define_xform(
        stage,
        f"{mount.GetPath()}/CameraRig",
        rotate_xyz_deg=EE_CAMERA_LOCAL_ROTATION_DEG,
    )
    camera_frame = define_xform(
        stage,
        f"{rig.GetPath()}/CameraFrame",
        translate=EE_CAMERA_LOCAL_TRANSLATE,
    )

    return create_camera(
        stage,
        f"{camera_frame.GetPath()}/EEViewCamera",
        [0.0, 0.0, 0.0],
        focal_length_mm=1.1,
    )


def force_perspective_view():
    viewport = get_active_viewport()
    if viewport is None:
        return
    try:
        viewport.camera_path = "/OmniverseKit_Persp"
    except Exception:
        pass


def attach_depth_sensor_template(stage, camera_path, scope_path, baseline_mm=None):
    if stage.GetPrimAtPath(scope_path):
        stage.RemovePrim(scope_path)
    stage.DefinePrim(scope_path, "Scope")
    kwargs = {}
    if baseline_mm is not None:
        kwargs["omni:rtx:post:depthSensor:baselineMM"] = baseline_mm
    try:
        SingleViewDepthSensorAsset.add_template_render_product(
            parent_prim_path=scope_path,
            camera_prim_path=camera_path,
            **kwargs,
        )
    except Exception:
        pass


def get_viewport_window_by_name(name):
    for window in get_viewport_window_instances():
        if getattr(window, "name", None) == name:
            return window
    return None


def ensure_viewport_window(name, camera_path, resolution):
    window = get_viewport_window_by_name(name)
    if window is None:
        window = ViewportWindow(name=name, width=resolution[0], height=resolution[1])
    viewport_api = getattr(window, "viewport_api", None)
    if viewport_api is not None:
        try:
            viewport_api.camera_path = camera_path
        except Exception:
            pass
        try:
            viewport_api.set_active_camera(camera_path)
        except Exception:
            pass
        try:
            viewport_api.set_texture_resolution(resolution)
        except Exception:
            pass
    return window


def bind_custom_viewports(ee_camera_path, top_camera_path):
    ensure_viewport_window(EE_VIEWPORT_NAME, ee_camera_path, EE_VIEWPORT_RESOLUTION)
    ensure_viewport_window(TOP_VIEWPORT_NAME, top_camera_path, TOP_VIEWPORT_RESOLUTION)


def save_current_stage(stage):
    root_layer = stage.GetRootLayer()
    root_path = Path(root_layer.realPath or root_layer.identifier)
    target_path = OUTPUT_STAGE.resolve()

    if not root_layer.Save():
        raise RuntimeError(f"Failed to save stage in place: {root_path}")

    if root_path.resolve() != target_path:
        if not root_layer.Export(str(OUTPUT_STAGE)):
            raise RuntimeError(f"Failed to export stage: {OUTPUT_STAGE}")


class TabletopDepthOverlay:
    def __init__(self, stage):
        self._stage = stage
        self._timeline = omni.timeline.get_timeline_interface()
        self._frame_counter = 0
        self._labels = {}
        self._window = None
        self._update_subscription = None
        self._ee_camera = None
        self._top_camera = None
        self._build_ui()
        self._initialize_cameras()
        self._update_subscription = (
            omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self._on_update)
        )

    def destroy(self):
        self._update_subscription = None
        self._ee_camera = None
        self._top_camera = None
        self._window = None
        self._labels = {}

    def _build_ui(self):
        self._window = ui.Window("Tabletop Depth Monitor", width=360, height=230)
        with self._window.frame:
            with ui.VStack(spacing=6):
                ui.Label("Depth updates while the simulation is playing.")
                for name in TABLETOP_OBJECT_PATHS:
                    label = ui.Label(f"{name}: top=-- m | ee=-- m")
                    self._labels[name] = label

    def _initialize_camera(self, prim_path, resolution, name):
        camera = Camera(
            prim_path=prim_path,
            name=name,
            resolution=resolution,
        )
        camera.initialize(attach_rgb_annotator=False)
        camera.add_distance_to_camera_to_frame()
        return camera

    def _initialize_cameras(self):
        self._ee_camera = self._initialize_camera(EE_CAMERA_PATH, EE_VIEWPORT_RESOLUTION, "ee_depth_monitor")
        self._top_camera = self._initialize_camera(TOP_CAMERA_PATH, TOP_VIEWPORT_RESOLUTION, "top_depth_monitor")

    def _get_world_position(self, prim_path):
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = transform.ExtractTranslation()
        return np.array([float(translation[0]), float(translation[1]), float(translation[2])], dtype=np.float32)

    def _sample_depth_at_world_point(self, camera, world_position):
        if world_position is None:
            return None
        frame = camera.get_current_frame()
        depth = frame.get("distance_to_camera")
        if depth is None:
            return None
        depth = np.asarray(depth)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]
        if depth.ndim != 2:
            return None

        image_coords = camera.get_image_coords_from_world_points(np.asarray([world_position], dtype=np.float32))
        if image_coords is None or len(image_coords) == 0:
            return None
        u, v = image_coords[0]
        u = int(round(float(u)))
        v = int(round(float(v)))
        height, width = depth.shape
        if u < 0 or v < 0 or u >= width or v >= height:
            return None

        value = float(depth[v, u])
        if not np.isfinite(value) or value <= 0.0:
            return None
        return value

    def _format_depth(self, value):
        if value is None:
            return "--"
        return f"{value:.3f}"

    def _update_labels(self):
        for name, prim_path in TABLETOP_OBJECT_PATHS.items():
            world_position = self._get_world_position(prim_path)
            top_depth = self._sample_depth_at_world_point(self._top_camera, world_position)
            ee_depth = self._sample_depth_at_world_point(self._ee_camera, world_position)
            self._labels[name].text = (
                f"{name}: top={self._format_depth(top_depth)} m | ee={self._format_depth(ee_depth)} m"
            )

    def _on_update(self, _event):
        if not self._timeline.is_playing():
            return
        self._frame_counter += 1
        if self._frame_counter % 3 != 0:
            return
        self._update_labels()


def apply_scene():
    global TOP_VIEW_ROS_BRIDGE, EE_VIEW_ROS_BRIDGE, KINOVA_JOINT_ROS_BRIDGE
    stage = omni.usd.get_context().get_stage()
    additions_root = define_xform(stage, "/World/CapstoneAdditions")
    bed_prim = stage.GetPrimAtPath(f"{additions_root.GetPath()}/HospitalBed")
    if bed_prim and bed_prim.IsValid():
        bed_prim.SetActive(False)
    table_path = f"{additions_root.GetPath()}/BedsideTable"
    table_prim = stage.GetPrimAtPath(table_path)
    if table_prim and table_prim.IsValid():
        table_prim.SetActive(False)
    build_tabletop_items(stage, f"{additions_root.GetPath()}/TabletopItems")
    fix_table_collision(stage)
    apply_gripper_friction(stage)
    fix_gripper_mimic_limits(stage)
    boost_gripper_drive(stage)
    boost_arm_drive(stage)
    # === Mode toggle ===
    # Capture mode  : `build_capture_humans` ON, `build_hazards` OFF
    # Hazard mode   : `build_capture_humans` OFF, `build_hazards` ON (default flight scenario)
    # build_capture_humans(stage, f"{additions_root.GetPath()}/CaptureHumans")
    build_hazards(stage, f"{additions_root.GetPath()}/Hazards")
    ee_camera = create_ee_camera(stage)
    top_camera = create_camera(
        stage,
        TOP_CAMERA_PATH,
        TOP_CAMERA_POSITION,
        TOP_CAMERA_ROTATION_DEG,
        TOP_CAMERA_FOCAL_LENGTH_MM,
    )
    attach_depth_sensor_template(stage, str(ee_camera.GetPath()), EE_DEPTH_SCOPE, baseline_mm=42)
    attach_depth_sensor_template(stage, str(top_camera.GetPath()), TOP_DEPTH_SCOPE, baseline_mm=42)
    force_perspective_view()
    bind_custom_viewports(str(ee_camera.GetPath()), str(top_camera.GetPath()))
    EE_VIEW_ROS_BRIDGE = build_ee_view_bridge(str(ee_camera.GetPath()))
    TOP_VIEW_ROS_BRIDGE = build_top_view_bridge(str(top_camera.GetPath()))
    KINOVA_JOINT_ROS_BRIDGE = create_ros2_joint_graph(
        # PhysicsArticulationRootAPI 는 /Kinova 자체가 아니라 URDF 임포트 계층
        # 구조상 base_link 에 붙는다(Franka 때는 /Franka 루트에 바로 있었음).
        articulation_path=KINOVA_ARTICULATION_ROOT_PATH,
        graph_path="/World/ROS/KinovaJointGraph",
        # Isaac publishes raw (sim-time) joint states here; joint_state_restamp_node
        # re-stamps them to wall time and republishes on /joint_states, which the
        # MoveIt stack (incl. moveit_cpp's hard-coded 'joint_states') consumes.
        joint_state_topic="/joint_states_isaac",
        joint_command_topic="/joint_command",
    )


# Single cached rigid-prim view for the hazard bottle. Creating a fresh
# RigidPrim/SingleRigidPrim view every frame RE-INITIALISES the physics view and
# zeroes the bottle's velocity (so it never flies). Create the view ONCE and
# reuse it for launch / pose-read / stop.
_BOTTLE_RB = {"view": None, "kind": None}


def _get_bottle_rb():
    """Create (once) and return the cached (view, kind) for the active hazard prim."""
    if _BOTTLE_RB["view"] is not None:
        return _BOTTLE_RB["view"], _BOTTLE_RB["kind"]
    path = HAZARD_OBJECT_PATHS["HazardBox" if HAZARD_OBJECT == "box" else "HazardBottle"]
    try:
        from isaacsim.core.prims import RigidPrim
        _BOTTLE_RB["view"], _BOTTLE_RB["kind"] = RigidPrim(path), "multi"
        return _BOTTLE_RB["view"], _BOTTLE_RB["kind"]
    except Exception as exc:
        print(f"[hazard] RigidPrim create failed: {exc}")
    try:
        from isaacsim.core.prims import SingleRigidPrim
        _BOTTLE_RB["view"], _BOTTLE_RB["kind"] = SingleRigidPrim(path), "single"
        return _BOTTLE_RB["view"], _BOTTLE_RB["kind"]
    except Exception as exc:
        print(f"[hazard] SingleRigidPrim create failed: {exc}")
    return None, None


def _set_bottle_velocity(v):
    """Set the bottle's linear velocity at runtime via the cached view."""
    rb, kind = _get_bottle_rb()
    if rb is None:
        return False
    try:
        if kind == "multi":
            rb.set_velocities(
                np.array([[float(v[0]), float(v[1]), float(v[2]), 0.0, 0.0, 0.0]], dtype=np.float32))
        else:
            rb.set_linear_velocity(np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float32))
        return True
    except Exception as exc:
        print(f"[hazard] set_bottle_velocity failed: {exc}")
        return False


def _apply_bottle_launch_velocity():
    """Give the stationary hazard prim (bottle or box) its flight velocity.

    Picks the velocity vector for the currently-active HAZARD_OBJECT so the
    box / bottle scenarios can be tuned independently. Previously this always
    applied the bottle velocity, which broke the "box flythrough + bottle park"
    pairing once the two scenarios diverged in target speed.
    """
    v = HAZARD_BOX_LINEAR_VELOCITY if HAZARD_OBJECT == "box" else HAZARD_BOTTLE_LINEAR_VELOCITY
    if _set_bottle_velocity(v):
        print(f"[hazard] {HAZARD_OBJECT} launched v={v.tolist()}")
    else:
        print(f"[hazard] ERROR: could not apply {HAZARD_OBJECT} velocity")


def _stop_bottle():
    """Zero the bottle's velocity so it halts in place (gravity off -> stays put)."""
    return _set_bottle_velocity((0.0, 0.0, 0.0))


def _get_bottle_x():
    """Read the bottle's current world x at runtime (None if unavailable)."""
    rb, kind = _get_bottle_rb()
    if rb is None:
        return None
    try:
        if kind == "multi":
            positions, _ = rb.get_world_poses()
            return float(positions[0][0])
        pos, _ = rb.get_world_pose()
        return float(pos[0])
    except Exception:
        return None


async def _bottle_launch_loop():
    """Launch the stationary bottle when /hazard/launch_bottle is received."""
    try:
        import rclpy
        from std_msgs.msg import Empty as _Empty
    except Exception as exc:
        print(f"[hazard] rclpy unavailable in Isaac python; bottle trigger disabled: {exc}")
        return
    try:
        if not rclpy.ok():
            rclpy.init()
    except Exception as exc:
        print(f"[hazard] rclpy.init failed: {exc}")
        return
    node = rclpy.create_node("hazard_bottle_launcher")
    pending = {"go": False}
    node.create_subscription(_Empty, "/hazard/launch_bottle", lambda _m: pending.__setitem__("go", True), 10)
    app = omni.kit.app.get_app()
    state = {"launched": False, "parked": False}

    # Auto-launch on arm motion: subscribe to /joint_states and fire once any
    # joint exceeds HAZARD_AUTO_TRIGGER_RAD from the latched reference pose.
    # During the AUTO_ARM_SEC arming window the reference pose is continuously
    # refreshed so that hybrid startup wiggle / settling don't bake themselves
    # in as the trigger baseline. Coexists with the manual
    # /hazard/launch_bottle path — whichever fires first wins.
    import time as _time
    auto_state = {"initial": None, "arm_at": None}
    if HAZARD_AUTO_LAUNCH:
        try:
            from sensor_msgs.msg import JointState as _JointState
        except Exception as _exc:
            print(f"[hazard] auto-launch disabled — JointState import failed: {_exc}")
        else:
            def _js_cb(msg):
                if state["launched"]:
                    return
                pos = list(msg.position)
                now = _time.monotonic()
                if auto_state["arm_at"] is None:
                    auto_state["arm_at"] = now + HAZARD_AUTO_ARM_SEC
                # Arming window: keep refreshing the reference pose so startup
                # twitching gets absorbed instead of being treated as motion.
                if now < auto_state["arm_at"]:
                    auto_state["initial"] = pos
                    return
                try:
                    max_delta = max(abs(p - p0) for p, p0 in zip(pos, auto_state["initial"]))
                except Exception:
                    return
                if max_delta > HAZARD_AUTO_TRIGGER_RAD:
                    pending["go"] = True
                    print(f"[hazard] auto-launch fired — max joint Δ={max_delta:.3f} rad "
                          f"(threshold={HAZARD_AUTO_TRIGGER_RAD})")
            node.create_subscription(_JointState, "/joint_states", _js_cb, 10)
            print(f"[hazard] auto-launch armed — Δ>{HAZARD_AUTO_TRIGGER_RAD} rad "
                  f"after {HAZARD_AUTO_ARM_SEC}s warm-up on /joint_states")

    print(f"[hazard] {HAZARD_OBJECT} launcher ready (mode={HAZARD_BOTTLE_MODE}, "
          f"auto={'on' if HAZARD_AUTO_LAUNCH else 'off'}) — "
          "trigger: /hazard/launch_bottle")
    while True:
        try:
            rclpy.spin_once(node, timeout_sec=0.0)
        except Exception:
            pass
        if pending["go"]:
            pending["go"] = False
            _apply_bottle_launch_velocity()
            state["launched"] = True
            state["parked"] = False
        # "park" mode: once the flying bottle reaches PARK_X, halt it so it stays
        # in the arm's path as a persistent obstacle (avoidance / replan demo).
        if HAZARD_BOTTLE_MODE == "park" and state["launched"] and not state["parked"]:
            x = _get_bottle_x()
            if x is not None and x <= HAZARD_BOTTLE_PARK_X:
                if _stop_bottle():
                    state["parked"] = True
                    print(f"[hazard] bottle PARKED at x≈{x:.2f} — persistent obstacle")
        await app.next_update_async()


async def main():
    global DEPTH_OVERLAY
    app = omni.kit.app.get_app()
    for _ in range(180):
        await app.next_update_async()
    if not open_stage(str(SOURCE_STAGE)):
        raise RuntimeError(f"Failed to open stage: {SOURCE_STAGE}")
    for _ in range(60):
        await app.next_update_async()
    force_perspective_view()
    apply_scene()
    for _ in range(60):
        await app.next_update_async()
    force_perspective_view()
    bind_custom_viewports(EE_CAMERA_PATH, TOP_CAMERA_PATH)
    stage = omni.usd.get_context().get_stage()
    save_current_stage(stage)
    if DEPTH_OVERLAY is not None:
        DEPTH_OVERLAY.destroy()
    DEPTH_OVERLAY = TabletopDepthOverlay(stage)
    print(f"Saved: {OUTPUT_STAGE}")
    asyncio.ensure_future(_bottle_launch_loop())


asyncio.ensure_future(main())
