# mask_projection_pkg

2D 세그멘테이션 마스크 + Depth 이미지 → **라벨링된 3D PointCloud2** 변환 패키지.

---

## 노드 구성

| 노드 | 역할 |
|---|---|
| `mask_projector_node` | 단일 카메라 → `/labeled_points` (Gazebo 데모, 수정 금지) |
| `multi_view_projector_node` | EE + Top 두 카메라 → `/world_map` (Isaac Sim 연동 대상) |

---

## 모듈 구성

```
mask_projection_pkg/
├── multi_view_projector_node.py  ROS 2 노드 — 구독/발행 배선 + 메시지 디코딩만
├── projector_node.py             단일 카메라 Gazebo 데모 레퍼런스 (수정 금지)
├── projection_engine.py          순수 numpy 투영·필터 로직 (ROS 의존성 없음)
│     load_extrinsics / project_labeled / project_unknown
│     collect_seg_points / filter_free_by_unknown
├── ply_utils.py                  PLY 저장 + result JSON 공통 유틸
│     save_ply_xyz / save_ply_labeled / build_result_json
├── cloud_builder.py              CategoryPoints → PointCloud2 메시지 변환
├── label_mapper.py               카테고리 ID / 색상 / mask 픽셀 매핑 (변경 금지)
└── back_projection.py            depth 이미지 → 3D 포인트 역투영
config/
└── camera_extrinsics.yaml        카메라 외부 파라미터 (R, t)
launch/
├── multi_view_projector.launch.py
└── mask_projector.launch.py      (Gazebo 데모용, 수정 금지)
```

---

## 카테고리

| ID | 이름 | 색상 | 출처 |
|---|---|---|---|
| 0 | FREE | 회색 (80,80,80) | EE 뷰 비탐지 픽셀 |
| 1 | TARGET | 초록 (0,200,80) | EE 뷰 GSAM |
| 2 | WORKSPACE | 노랑 (255,220,0) | EE 뷰 GSAM |
| 3 | OBSTACLE | 빨강 (220,40,40) | EE 뷰 GSAM |
| 4 | UNKNOWN | 보라 (150,80,200) | Top 뷰 전체 |

RViz2: `Color Transformer → RGB8`, `Fixed Frame → world`

---

## 카메라 Extrinsics 수정

**파일**: `config/camera_extrinsics.yaml`

```yaml
ee_camera:
  t: [0.5, 0.0, 2.5]   # 카메라 원점 world 좌표 (m)
  R:                     # camera optical frame → world frame 회전 (3×3)
    - [ 0.0,  0.435, -0.900]
    - [ 1.0,  0.0,    0.0  ]
    - [ 0.0, -0.900, -0.435]

top_camera:
  t: [-3.0, 2.0, 2.5]
  R:
    - [ 0.0, -1.0,  0.0]
    - [-1.0,  0.0,  0.0]
    - [ 0.0,  0.0, -1.0]
```

**변환 규칙**: `p_world = R @ p_cam + t`

Isaac Sim 전환: USD stage → 카메라 prim → World Transform 4×4 행렬  
→ 좌상단 3×3 = `R`, 우상단 3×1 열 = `t` (단위 meters)

---

## 동작 방식

```
EE depth + GSAM 마스크  →  TARGET / WORKSPACE / OBSTACLE / FREE
  └─ TARGET 3D bbox 계산 (±5 cm margin)
Top depth              →  UNKNOWN, TARGET bbox 내부 제거
두 뷰 world frame 변환 후 병합  →  /world_map
```

---

## 구독 / 발행 토픽

| 토픽 | 타입 | 역할 |
|---|---|---|
| `<ee_depth_topic>` | `sensor_msgs/Image` | EE depth — 캐시 |
| `<ee_camera_info_topic>` | `sensor_msgs/CameraInfo` | EE 내부 파라미터 — 캐시 |
| `<top_depth_topic>` | `sensor_msgs/Image` | Top depth — 캐시 |
| `<top_camera_info_topic>` | `sensor_msgs/CameraInfo` | Top 내부 파라미터 — 캐시 |
| `<detections_topic>` | `std_msgs/String` | GSAM/Qwen JSON — 캐시 |
| `<mask_topic>` | `sensor_msgs/Image` | 마스크 — **트리거** |
| `/world_map` | `sensor_msgs/PointCloud2` | 출력 포인트클라우드 |
| `/world_map_result` | `std_msgs/String` | 카테고리별 centroid+bbox JSON |

---

## 실행

> **경로 주의**: 아래 명령어의 `~/gsam_ws` 는 본인 워크스페이스 경로로 변경하세요. `output/` 저장 경로도 `ros_node.py` 의 `OUTPUT_DIR` 을 확인하세요.

```bash
cd ~/gsam_ws && source launch_env.bash
ros2 launch mask_projection_pkg multi_view_projector.launch.py
```

Isaac Sim 전환 시 (extrinsics YAML + 토픽 오버라이드):
```bash
ros2 launch mask_projection_pkg multi_view_projector.launch.py \
  extrinsics_config:=/path/to/camera_extrinsics_isaac.yaml \
  ee_depth_topic:=/isaac/ee/depth_image \
  ee_camera_info_topic:=/isaac/ee/camera_info \
  top_depth_topic:=/isaac/top/depth_image \
  top_camera_info_topic:=/isaac/top/camera_info \
  mask_topic:=/qwen/mask_image \
  detections_topic:=/qwen/labeled_detections
```

---

## 출력 파일

추론 시 `~/gsam_ws/output/` 에 자동 저장:  
`world_map_{initials}_{stamp}.ply` — XYZ + RGB + category (MeshLab / Open3D로 열람 가능)

---

## 설계 메모

**타임스탬프 동기화를 안 쓰는 이유**: GSAM CPU 추론 30~40초 소요 → mask_image 도착 시 depth 큐가 한참 앞으로 나가 ApproximateTimeSynchronizer 매칭 실패. depth/camera_info를 최신값으로 캐시하고 mask_image 수신을 트리거로 사용.

**FREE vs UNKNOWN**: octomap_server는 ray casting으로 자유공간 계산. FREE 포인트를 pointcloud에 포함하면 배경이 occupied voxel로 마킹됨 — OBSTACLE + UNKNOWN만 octomap에 넣는 것을 권장.

---

## 향후 계획

| 작업 | 위치 | 내용 |
|---|---|---|
| **Qwen VLM 연동** | `label_mapper.py` | `MASK_VALUE_TO_CATEGORY`를 Qwen 출력의 `label` 필드 기반 딕셔너리로 교체. `detections`에 `"category"` 필드 있으면 자동 우선 적용됨 |
| **MoveIt2 연동** | downstream | `/world_map_result` TARGET centroid → goal pose, OBSTACLE/UNKNOWN → 충돌 맵 |
| **Isaac Sim 전환** | `config/camera_extrinsics_isaac.yaml` | USD stage 카메라 World Transform 4×4 → 좌상단 3×3=R, 우상단=t 로 YAML 작성 후 `extrinsics_config` 인자로 지정 |
| **ApproximateTimeSynchronizer** | `multi_view_projector_node.py` | Isaac Sim GPU 추론으로 전환 시 cache 패턴 → timestamp sync로 교체 |
