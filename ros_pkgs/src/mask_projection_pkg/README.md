# mask_projection_pkg

2D 세그멘테이션 마스크 + Depth 이미지 → **라벨링된 3D PointCloud2** 변환 패키지.

두 가지 노드를 제공합니다.

| 노드 | 설명 | 카메라 수 |
|---|---|---|
| `mask_projector_node` | 단일 카메라, Gazebo 데모용 (수정 금지) | 1 |
| `multi_view_projector_node` | 두 카메라 (Top + EE) → 월드 좌표 PointCloud2 | 2 |

---

## 모듈 구성

```
mask_projection_pkg/
├── back_projection.py          depth 이미지 → 카메라 좌표계 3D 포인트 (수학 로직만)
├── label_mapper.py             마스크 픽셀값 → 시맨틱 카테고리 + 색상
├── cloud_builder.py            CategoryPoints → PointCloud2 메시지 패킹
├── projector_node.py           단일 카메라 ROS 2 노드 (데모, 수정 금지)
└── multi_view_projector_node.py  두 카메라 월드 맵 ROS 2 노드
launch/
├── mask_projector.launch.py         단일 카메라 launch
└── multi_view_projector.launch.py   두 카메라 launch
```

---

## 시맨틱 카테고리

마스크 픽셀값 → 카테고리 매핑은 `label_mapper.py` 에 정의됩니다.

| 카테고리 ID | 이름 | 색상 (RGB) | 의미 |
|---|---|---|---|
| 0 | FREE | (80, 80, 80) 회색 | 배경 / 빈 공간 |
| 1 | TARGET | (0, 200, 80) 초록 | 잡을 물체 (prompt 첫 번째) |
| 2 | WORKSPACE | (255, 220, 0) 노랑 | 작업 테이블 (prompt 두 번째) |
| 3 | OBSTACLE | (220, 40, 40) 빨강 | 그 외 감지된 물체 |

마스크 픽셀값 = `detections_json` 의 1-based 인덱스.  
pixel=1 → TARGET, pixel=2 → WORKSPACE, pixel=3+ → OBSTACLE.

> **Qwen 연동 예정:** Qwen VLM이 어떤 마스크 ID가 실제 TARGET/WORKSPACE인지 결정하면  
> `label_mapper.py` 의 `MASK_VALUE_TO_CATEGORY` 매핑만 교체하면 됩니다.  
> 다운스트림(cloud_builder, projector 노드)은 변경 불필요.

---

## PointCloud2 메시지 포맷

`cloud_builder.py` 가 생성하는 포인트 레이아웃 (20 bytes/point):

| offset | 필드 | 타입 | 설명 |
|---|---|---|---|
| 0 | x | float32 | X 좌표 (m) |
| 4 | y | float32 | Y 좌표 (m) |
| 8 | z | float32 | Z 좌표 (m) |
| 12 | rgb | float32 | PCL 방식 packed RGB `(r<<16\|g<<8\|b)` |
| 16 | category | uint8 | 카테고리 ID (0=FREE, 1=TARGET, 2=WS, 3=OBS) |

RViz2 에서 `Color Transformer → RGB8` 로 설정하면 카테고리별 색상이 표시됩니다.  
`category` 필드를 기준으로 필터링하면 MoveIt2 충돌 맵 또는 goal pose 추출에 활용 가능합니다.

---

## 노드 1 — `mask_projector_node` (단일 카메라, Gazebo 데모)

> **이 노드와 launch 파일은 수정하지 마세요.** Gazebo 데모 기준 레퍼런스입니다.

### 구독 토픽

| 토픽 | 타입 | 역할 |
|---|---|---|
| `/rgbd_camera/depth_image` | `sensor_msgs/Image` | Depth (32FC1, m) |
| `/rgbd_camera/camera_info` | `sensor_msgs/CameraInfo` | 카메라 내부 파라미터 |
| `/grounded_sam/mask_image` | `sensor_msgs/Image` | 세그멘테이션 마스크 → **트리거** |
| `/grounded_sam/detections_json` | `std_msgs/String` | GSAM 감지 결과 JSON |

### 발행 토픽

| 토픽 | 타입 | 설명 |
|---|---|---|
| `/labeled_points` | `sensor_msgs/PointCloud2` | 라벨링된 포인트클라우드 |
| `/projection_result` | `std_msgs/String` | 카테고리별 centroid JSON |

### 실행

```bash
ros2 launch mask_projection_pkg mask_projector.launch.py \
  initials:=tc
```

---

## 노드 2 — `multi_view_projector_node` (두 카메라, 월드 맵)

Top 카메라 + EE 카메라 두 뷰를 **월드 좌표계**로 변환 후 병합합니다.

- **Top 카메라:** depth만 사용 → 전 포인트 FREE (씬 형상 제공)
- **EE 카메라:** depth + GSAM 마스크 → TARGET / WORKSPACE / OBSTACLE / FREE (시맨틱 제공)

GSAM은 EE 카메라에서만 실행합니다 (CPU 추론 속도 한계).

### 구독 토픽

| 토픽 | 타입 | 역할 |
|---|---|---|
| `/top_camera/depth_image` | `sensor_msgs/Image` | Top 카메라 Depth |
| `/top_camera/camera_info` | `sensor_msgs/CameraInfo` | Top 카메라 내부 파라미터 |
| `/ee_camera/depth_image` | `sensor_msgs/Image` | EE 카메라 Depth |
| `/ee_camera/camera_info` | `sensor_msgs/CameraInfo` | EE 카메라 내부 파라미터 |
| `/grounded_sam/mask_image` | `sensor_msgs/Image` | GSAM 마스크 → **트리거** |
| `/grounded_sam/detections_json` | `std_msgs/String` | GSAM 감지 결과 JSON |

Top 카메라가 없으면 경고 후 EE 뷰만으로 계속 동작합니다.

### 발행 토픽

| 토픽 | 타입 | 설명 |
|---|---|---|
| `/world_map` | `sensor_msgs/PointCloud2` | 월드 좌표계 라벨링 포인트클라우드 |
| `/world_map_result` | `std_msgs/String` | 카테고리별 centroid JSON (월드 좌표) |

### 실행

```bash
# GSAM — EE 카메라 RGB 스트림으로 실행 (image_topic 오버라이드 필수)
ros2 launch grounded_sam_pkg grounded_sam.launch.py \
  image_topic:=/ee_camera/image \
  prompt:="cup, table, object"

# Multi-view projector
ros2 launch mask_projection_pkg multi_view_projector.launch.py \
  initials:=cto
```

---

## 카메라 → 월드 좌표 변환 설정 (TODO)

`multi_view_projector_node.py` 상단의 변환 행렬을 실제 값으로 교체해야 합니다.

**변환 규칙:**
```
p_world = R @ p_cam + t
```

| 변수 | 의미 | 현재 값 |
|---|---|---|
| `_R_TOP` | Top 카메라 좌표계 → 월드 좌표계 회전 (3×3) | `np.eye(3)` (placeholder) |
| `_t_TOP` | Top 카메라 원점의 월드 좌표 (m) | `np.zeros(3)` (placeholder) |
| `_R_EE` | EE 카메라 좌표계 → 월드 좌표계 회전 (home pose 기준, 3×3) | `np.eye(3)` (placeholder) |
| `_t_EE` | EE 카메라 원점의 월드 좌표 (home pose 기준, m) | `np.zeros(3)` (placeholder) |

### Isaac Sim에서 측정하는 방법

1. Isaac Sim USD stage를 엽니다.
2. 카메라 prim(예: `/World/top_camera`)을 선택합니다.
3. Property 패널 → **World Transform** 4×4 행렬을 확인합니다.
4. 좌상단 3×3 블록 = `R` (카메라 X/Y/Z 축의 월드 좌표 표현)
5. 우상단 3×1 열 = `t` (카메라 원점의 월드 좌표, meters)
6. `multi_view_projector_node.py` 의 `_R_TOP/_t_TOP` 에 붙여넣습니다.
7. EE 카메라도 동일하게 반복 (home joint configuration 상태에서).

**예시 (실제 값으로 교체하세요):**
```python
# top camera: 월드 원점 정중앙 위 1.5 m, 아래를 바라보는 경우
_R_TOP = np.array([
    [ 1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0],
    [ 0.0, -1.0,  0.0],
])
_t_TOP = np.array([0.0, 0.0, 1.5])
```

---

## Isaac Sim 씬 구성 및 실행 (현재 타겟 씬)

### 씬 설명

테이블 위에 4개 물체:

| 물체 | 색상 | 카테고리 | 이유 |
|---|---|---|---|
| glass cup (유리컵) | 초록 — TARGET | 잡을 대상 | 이번 작업의 grasp 목표물 |
| table (테이블) | 노랑 — WORKSPACE | 작업 공간 | 조작이 이루어지는 표면 |
| red ball (빨간 공) | 빨강 — OBSTACLE | 장애물 | 충돌 회피 대상 |
| blue cube (파란 큐브) | 빨강 — OBSTACLE | 장애물 | 충돌 회피 대상 |
| book (책) | 빨강 — OBSTACLE | 장애물 | 충돌 회피 대상 |

### 실행 명령

```bash
# 터미널 1 — GSAM (EE 카메라 RGB 스트림, 유리컵을 prompt 첫 번째로)
ros2 launch grounded_sam_pkg grounded_sam.launch.py \
  image_topic:=/ee_camera/image \
  prompt:="glass cup, table, red ball, blue cube, book"

# 터미널 2 — Multi-view projector
ros2 launch mask_projection_pkg multi_view_projector.launch.py \
  initials:=gtrcb
```

**prompt 순서가 카테고리를 결정합니다:**
- 1번 `glass cup` → TARGET (초록)
- 2번 `table` → WORKSPACE (노랑)
- 3번 이후 `red ball`, `blue cube`, `book` → OBSTACLE (빨강)

### RViz2에서 확인하는 방법

1. `PointCloud2` 디스플레이 추가 → Topic: `/world_map`
2. `Color Transformer` → `RGB8` 선택
3. 씬 결과 예시:

```
초록 포인트  →  유리컵 위치 (TARGET)
노랑 포인트  →  테이블 표면 (WORKSPACE)
빨강 포인트  →  공 / 큐브 / 책 (OBSTACLE)
회색 포인트  →  배경 / 빈 공간 (FREE)
```

projection이 제대로 이루어졌다면 각 물체의 3D 형상이 해당 색상으로 표시됩니다.

---

## ⚠️ Isaac Sim 통합 시 수정 체크리스트

아래 항목을 순서대로 처리하면 됩니다.  
**launch 파라미터 오버라이드(1번)는 코드 수정 없음. 나머지는 파일 수정 필요.**

---

### 1. 토픽 이름 오버라이드 — 코드 수정 없음, launch 명령만 변경

Isaac Sim ROS 2 bridge가 발행하는 토픽 이름을 launch 파라미터로 넘기면 됩니다.

```bash
ros2 launch mask_projection_pkg multi_view_projector.launch.py \
  top_depth_topic:=/isaac/top/depth_image \
  top_camera_info_topic:=/isaac/top/camera_info \
  ee_depth_topic:=/isaac/ee/depth_image \
  ee_camera_info_topic:=/isaac/ee/camera_info
```

Isaac Sim bridge 토픽 이름은 동료가 USD stage 설정에서 확인.

---

### 2. 카메라 → 월드 좌표 변환 행렬 채우기

**파일:** `mask_projection_pkg/multi_view_projector_node.py`  
**위치:** 파일 상단, `# TODO (teammate)` 주석 아래

```python
# 현재 (placeholder — 교체 필요)
_R_TOP: np.ndarray = np.eye(3, dtype=np.float64)
_t_TOP: np.ndarray = np.zeros(3, dtype=np.float64)
_R_EE:  np.ndarray = np.eye(3, dtype=np.float64)
_t_EE:  np.ndarray = np.zeros(3, dtype=np.float64)
```

**Isaac Sim에서 값 읽는 방법:**
1. USD stage에서 카메라 prim 선택 (예: `/World/Camera_Top`)
2. Property 패널 → **World Transform** 4×4 행렬 확인
3. 좌상단 3×3 = `R` (카메라 축의 월드 기준 방향)
4. 우상단 3×1 = `t` (카메라 원점의 월드 좌표, 단위 meters)
5. EE 카메라는 로봇 **home pose** 상태에서 동일하게 측정

**변환 규칙:** `p_world = R @ p_cam + t`

**채운 뒤 예시:**
```python
_R_TOP = np.array([
    [ 1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0],
    [ 0.0, -1.0,  0.0],
])
_t_TOP = np.array([0.0, 0.0, 1.5])   # top camera가 월드 원점 위 1.5 m
```

---

### 3. 카테고리 할당 방식 교체 — prompt 순서 → label 기반

**왜 교체해야 하나:**  
현재는 GSAM 감지 결과의 순서가 항상 prompt 순서와 일치한다고 가정합니다.  
실제로는 GSAM이 prompt와 다른 순서로 감지할 수 있어 카테고리가 뒤바뀔 수 있습니다.

**현재 코드 위치:** `mask_projection_pkg/label_mapper.py`

```python
# 현재 (임시 — 교체 필요)
MASK_VALUE_TO_CATEGORY: Dict[int, int] = {
    1: CATEGORY_TARGET,     # mask pixel 1 = prompt 첫 번째 = glass cup 이라고 가정
    2: CATEGORY_WORKSPACE,  # mask pixel 2 = prompt 두 번째 = table 이라고 가정
    # 3+ → OBSTACLE (fallback)
}
```

**교체 방법 — `multi_view_projector_node.py` 의 `_mask_cb` 안에서 동적으로 생성:**

```python
# detections_json 구조 예시 (GSAM 출력):
# [
#   {"label": "red ball",   "confidence": 0.91, "bbox_xyxy": [...]},
#   {"label": "glass cup",  "confidence": 0.87, "bbox_xyxy": [...]},  ← 순서가 바뀔 수 있음
#   {"label": "table",      "confidence": 0.95, "bbox_xyxy": [...]},
# ]

# _mask_cb 에서 아래와 같이 label 기반 매핑을 동적으로 구성:
LABEL_TO_CATEGORY = {
    "glass cup": CATEGORY_TARGET,
    "table":     CATEGORY_WORKSPACE,
    # 나머지는 apply_labels 내부 fallback → CATEGORY_OBSTACLE
}

# detections 리스트를 순회하며 mask pixel value(1-based index)와 label 매핑
mask_value_to_category = {}
for idx, det in enumerate(self._latest_detections):
    mask_val = idx + 1   # mask pixel value는 1-based
    label    = det["label"]
    mask_value_to_category[mask_val] = LABEL_TO_CATEGORY.get(label, CATEGORY_OBSTACLE)
```

그런 다음 `apply_labels()` 호출 시 이 딕셔너리를 사용하도록 수정.  
`cloud_builder.py` 와 다운스트림 노드는 변경 불필요.

---

### 4. (선택) GPU 추론 시 타임스탬프 동기화 방식 교체

Isaac Sim + GPU 환경에서 GSAM 추론이 실시간으로 빨라지면 캐시 방식 대신  
`ApproximateTimeSynchronizer` 로 교체하는 것이 더 정확합니다.

**현재 코드 위치:** `multi_view_projector_node.py` → `_mask_cb` 진입부  
**현재 방식:** depth/camera_info를 최신값으로 캐시, mask 수신 시 즉시 실행  
**교체 방법:** `message_filters.ApproximateTimeSynchronizer` 로 ee_depth + mask 동기화

> CPU 환경(추론 30~40초)에서는 캐시 방식이 맞습니다. GPU 환경으로 전환 후 판단하세요.

---

> **현재는 1번(토픽 오버라이드) + 2번(R/t 채우기) 만으로 동작 검증을 먼저 합니다.**  
> 3번(label 매핑 교체)은 카테고리가 뒤바뀌는 현상이 확인되면 그때 수정합니다.

---

## Isaac Sim 토픽 오버라이드

모든 토픽 이름은 ROS 2 파라미터로 노출되어 있습니다.  
launch 파일에서 오버라이드만 하면 코드 변경 없이 Isaac Sim에 연결됩니다.

```bash
ros2 launch mask_projection_pkg multi_view_projector.launch.py \
  top_depth_topic:=/isaac/top/depth_image \
  top_camera_info_topic:=/isaac/top/camera_info \
  ee_depth_topic:=/isaac/ee/depth_image \
  ee_camera_info_topic:=/isaac/ee/camera_info
```

### 파라미터 전체 목록

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `top_depth_topic` | `/top_camera/depth_image` | Top 카메라 depth 토픽 |
| `top_camera_info_topic` | `/top_camera/camera_info` | Top 카메라 info 토픽 |
| `ee_depth_topic` | `/ee_camera/depth_image` | EE 카메라 depth 토픽 |
| `ee_camera_info_topic` | `/ee_camera/camera_info` | EE 카메라 info 토픽 |
| `mask_topic` | `/grounded_sam/mask_image` | GSAM 마스크 (트리거) |
| `detections_topic` | `/grounded_sam/detections_json` | GSAM 감지 결과 |
| `output_cloud_topic` | `/world_map` | 출력 PointCloud2 |
| `output_result_topic` | `/world_map_result` | 출력 결과 JSON |
| `min_depth` | `0.05` | 유효 depth 최솟값 (m) |
| `max_depth` | `15.0` | 유효 depth 최댓값 (m) |
| `initials` | `''` | 저장 PLY 파일명 접두사 |

---

## 출력 파일

추론 실행 시 `~/gsam_ws/output/` 에 자동 저장됩니다.

| 파일 | 노드 | 설명 |
|---|---|---|
| `cloud_original_{initials}_{stamp}.ply` | `mask_projector_node` | 원본 XYZ 포인트클라우드 |
| `cloud_labeled_{initials}_{stamp}.ply` | `mask_projector_node` | 라벨링된 포인트클라우드 |
| `world_map_{initials}_{stamp}.ply` | `multi_view_projector_node` | 월드 좌표계 라벨링 포인트클라우드 |

PLY 파일은 MeshLab, Open3D, CloudCompare 등으로 열 수 있습니다.

---

## 설계 메모

**왜 ApproximateTimeSynchronizer를 쓰지 않나?**

GSAM이 CPU에서 프레임당 30~40초 소요됩니다. mask_image가 도착할 시점에는 depth 큐가 한참 앞으로 나가 있어 타임스탬프 매칭이 실패합니다. 대신 depth/camera_info를 "최신값"으로 캐시하고 mask_image 도착을 트리거로 삼습니다. GPU 추론(Isaac Sim 환경)으로 전환하면 `ApproximateTimeSynchronizer` 로 교체하는 것이 좋습니다.

**왜 Top 카메라에는 GSAM을 실행하지 않나?**

CPU 추론 속도 한계로 두 뷰를 동시에 처리하기 어렵습니다. Top 카메라는 씬 형상(FREE 포인트)을 제공하고, EE 카메라가 시맨틱 라벨을 담당합니다. 추후 GPU 환경에서 Top 뷰 GSAM이 필요하다면 별도 브랜치/스크립트로 추가합니다.

**QoS**

모든 구독자는 VOLATILE QoS(depth=10)를 사용합니다. Gazebo bridge 및 Isaac Sim bridge 모두 VOLATILE로 발행하기 때문에 TRANSIENT_LOCAL로 구독하면 `incompatible QoS` 경고와 함께 메시지를 수신하지 못합니다.

---

## 향후 계획

- **Qwen VLM 연동:** GSAM 감지 결과 → Qwen이 TARGET/WORKSPACE 결정 → `label_mapper.py` 의 카테고리 매핑 교체. 다운스트림 코드 변경 없음.
- **MoveIt2 연동:** `/world_map_result` 의 TARGET centroid → goal pose, OBSTACLE 포인트 → 충돌 맵.
- **Isaac Sim 연결:** launch 파일 토픽 파라미터 오버라이드만으로 전환 가능 (동료 TODO).
