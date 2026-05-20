# vgn_grasp_pkg

raw depth images → **signed TSDF** → **VGN grasp 후보 검출** → `/grasp_candidates` (JSON)

---

## 노드

| 노드 | 역할 |
|---|---|
| `vgn_grasp_node` | EE+Top depth ray-casting signed TSDF → VGN inference → bbox 시맨틱 필터 → grasp 발행 |

---

## 전제 조건

### 1. VGN 서브모듈 추가

```bash
cd ~/gsam_ws
git submodule add https://github.com/ethz-asl/vgn external/vgn
```

### 2. pytorch-ignite 설치

GPU 환경 (RTX 3090 등 CUDA 12.1):

```bash
source ~/gsam_ws/gsam_ws_venv/bin/activate
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-ignite tqdm
```

CPU only:

```bash
source ~/gsam_ws/gsam_ws_venv/bin/activate
pip install pytorch-ignite tqdm
```

### 3. VGN 모델 가중치

ethz-asl/vgn GitHub → README의 Google Drive 링크에서 `data.zip` 다운로드:

```
data.zip 압축 해제 → data/models/vgn_conv.pth
cp data/models/vgn_conv.pth ~/gsam_ws/models/vgn_conv.pth
```

> **파일명 규칙**: `vgn_<network>.pth` 형식 필수.  
> `load_network()`가 파일명 두 번째 필드(`vgn_conv.pth` → `conv`)를 파싱해 네트워크 종류를 결정함.  
> 파일명이 다르면 `KeyError` 발생.

---

## 빌드

```bash
source ~/gsam_ws/launch_env.bash
colcon build --packages-select vgn_grasp_pkg
source install/setup.bash
```

---

## 실행

```bash
ros2 launch vgn_grasp_pkg vgn_grasp.launch.py
```

정상 기동 시 출력:

```
[vgn_grasp_node]: Extrinsics loaded: .../config/camera_extrinsics.yaml
[vgn_grasp_node]: VGN device: cpu
[vgn_grasp_node]: VGN model loaded: .../models/vgn_conv.pth
[vgn_grasp_node]: vgn_grasp_node ready  roi=0.3m  reso=40  min_quality=0.5  use_top=True
```

파라미터 오버라이드 예시:

```bash
ros2 launch vgn_grasp_pkg vgn_grasp.launch.py \
  vgn_model_path:=/abs/path/to/vgn_conv.pth \
  min_quality:=0.4 \
  max_candidates:=3 \
  use_top_depth:=false \
  extrinsics_config:=/path/to/camera_extrinsics_isaac.yaml
```

---

## 파라미터 (`config/vgn_params.yaml`)

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `roi_size_m` | `0.30` | ROI 큐브 한 변 길이 (m) — VGN 학습 크기와 일치해야 함 |
| `tsdf_resolution` | `40` | TSDF 복셀 그리드 크기 (40×40×40 고정) |
| `vgn_model_path` | `models/vgn_conv.pth` | 가중치 경로 (절대 or `$GSAM_WS` 기준 상대) |
| `min_quality` | `0.5` | NMS grasp quality 하한값 |
| `max_candidates` | `5` | 발행할 Top-K 후보 수 |
| `min_point_count` | `50` | TARGET point_count 미달 시 skip |
| `ee_depth_topic` | `/ee_camera/depth_image` | EE 카메라 depth 토픽 |
| `ee_camera_info_topic` | `/ee_camera/camera_info` | EE 카메라 intrinsics 토픽 |
| `top_depth_topic` | `/top_camera/depth_image` | Top 카메라 depth 토픽 |
| `top_camera_info_topic` | `/top_camera/camera_info` | Top 카메라 intrinsics 토픽 |
| `use_top_depth` | `true` | Top depth TSDF 적분 여부 (false = EE만 사용) |
| `extrinsics_config` | `""` | R/t YAML 경로. 빈 문자열 시 mask_projection_pkg 기본값 사용 |
| `world_frame` | `world` | 입력 좌표계 |
| `robot_frame` | `panda_link0` | 출력 grasp 좌표계 (TF world→robot_frame) |

`voxel_size = roi_size_m / tsdf_resolution` — 항상 계산, 파라미터 아님 (= 7.5 mm)

> **Extrinsics 규칙**: `extrinsics_config` YAML은 R, t만 포함 (K 없음).  
> K(내부 파라미터)는 `/camera_info` 토픽에서 런타임에 읽음.

---

## 구독 / 발행 토픽

| 토픽 | 타입 | 역할 |
|---|---|---|
| `/ee_camera/depth_image` | `sensor_msgs/Image` | EE depth — **캐시** (32FC1 또는 16UC1) |
| `/ee_camera/camera_info` | `sensor_msgs/CameraInfo` | EE 내부 파라미터 K — **캐시** |
| `/top_camera/depth_image` | `sensor_msgs/Image` | Top depth — **캐시** (`use_top_depth=true` 시) |
| `/top_camera/camera_info` | `sensor_msgs/CameraInfo` | Top 내부 파라미터 K — **캐시** |
| `/world_map_result` | `std_msgs/String` (JSON) | 카테고리별 centroid + bbox_3d_world — **트리거** |
| `/grasp_candidates` | `std_msgs/String` (JSON) | Top-K grasp 후보 출력 |
| `/grasp_markers` | `visualization_msgs/MarkerArray` | RViz2 시각화 |

> `/world_map`은 구독하지 않음 — RViz2 디버그 시각화 전용.

---

## 출력 JSON `/grasp_candidates`

```json
{
  "candidates": [
    {
      "position":   [x, y, z],
      "quaternion": [qx, qy, qz, qw],
      "width":      0.052,
      "quality":    0.87,
      "frame":      "panda_link0"
    }
  ],
  "target_centroid": [x, y, z],
  "stamp": 1234567890.123
}
```

TF 조회 실패 시 `frame`은 `"world"`로 폴백.

---

## RViz2 시각화 (`/grasp_markers`)

```
RViz2 → Add → MarkerArray → Topic: /grasp_markers
Fixed Frame: world
```

| 마커 | 의미 |
|---|---|
| 화살표 | grasp approach 방향 (그리퍼 -Z축), 파란색 (밝을수록 quality 높음) |
| 구 | grasp 위치 |
| 납작한 원통 | gripper width |

---

## 처리 흐름

```
/world_map_result (트리거)
  → target.centroid, target.bbox_3d_world, target.point_count 파싱
  → point_count < min_point_count 이면 skip

ROI 정의
  → roi_min = centroid - 0.15m,  roi_max = centroid + 0.15m  (30cm 큐브)

Signed TSDF 생성 — ray-casting (depth image 기반)
  복셀 중심 64,000개 (40×40×40) → world frame → 각 카메라 frame으로 역변환
  p_cam = R^T @ (p_world - t)          (extrinsics YAML의 R, t 사용)
  u = fx*(px/pz) + cx,  v = fy*(py/pz) + cy   (camera_info의 K 사용)
  sdf = d_obs - pz   (양수=free space, 음수=물체 내부)
  normalize: clip(sdf/trunc, -1, 1) * 0.5 + 0.5
    → 1.0 = 외부/미관측,  0.5 = 표면,  0.0 = 내부
  EE + Top weighted average → (1, 40, 40, 40) float32

VGN inference (ethz-asl/vgn)
  → predict(grid, net, device) → qual_vol, rot_vol, width_vol
  → process(...)  Gaussian smoothing + outside-surface mask + width filter
  → select(..., threshold=min_quality)  NMS → grasps (voxel 좌표)
  → from_voxel_coordinates(grasp, voxel_size)  → roi frame meters 변환

시맨틱 필터링
  → /world_map_result.target.bbox_3d_world 기준
  → grasp 위치가 bbox 내에 없으면 discard

TF world → panda_link0
  → TF 조회 실패 시 world frame 그대로 발행

quality 내림차순 정렬 → Top-K → /grasp_candidates, /grasp_markers 발행
```

---

## VGN 라이브러리 호환성 메모

`ethz-asl/vgn`은 ROS 1(catkin) 기반으로 `vgn.detection`이 `vgn.vis`를 import하고, `vgn.vis`는 `rospy`를 사용함.  
`vgn.networks`와 `vgn.utils.transform`은 rospy 의존성이 없음.

이 패키지는 `predict` / `process` / `select` 함수를 `vgn.detection` 대신 노드 내부에 직접 구현해 rospy 의존성을 우회함. 가중치 로딩(`load_network`)과 좌표 변환(`Transform`, `Rotation`)만 VGN 라이브러리에서 import.

---

## 팀 레포 rsync 방법

```bash
cd ~/capstone_ws/robot_capstone
git checkout main && git pull
git checkout -b sanghyun_MMDD

rsync -a --exclude="__pycache__" --exclude="*.pyc" \
  ~/gsam_ws/src/vgn_grasp_pkg  ros_pkgs/src/

git add ros_pkgs/src/vgn_grasp_pkg
git commit -m "Add vgn_grasp_pkg"
git push origin sanghyun_MMDD
```

기존 패키지(`grounded_sam_pkg`, `mask_projection_pkg`)는 rsync 금지 — [TEAM_DIFF.md](../../../TEAM_DIFF.md) 참고.
