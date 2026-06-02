# graspgen_pkg

`vgn_grasp_pkg`의 **드롭인 대체 노드**. EE depth + GSAM 마스크에서 TARGET 포인트 클라우드를 추출해 SSH 터널을 통해 원격 GraspGen 서버(Seraph aurora-g5)로 전송하고, `/grasp_candidates`를 동일한 JSON 형식으로 발행한다.

`bt_pkg` 포함 하위 패키지는 **변경 없이** vgn/graspgen을 교체할 수 있다.

---

## 전체 시스템 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Isaac Sim / Gazebo                               │
│  /ee_camera/{image, depth_image, camera_info}                           │
│  /top_camera/{depth_image, camera_info}                                 │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
               ┌───────────────▼────────────────┐
               │        grounded_sam_node        │  (grounded_sam_pkg)
               │  prompt: "cup, table, ..."      │
               └───────────┬────────────┬────────┘
                           │            │
              /grounded_sam/detections_json
              /grounded_sam/mask_image
                           │            │
               ┌───────────▼────────────┴────────┐
               │         qwen_bridge_node         │  (qwen_pkg)
               │  ← /user_instruction (stdin)     │  ※ 데모 시 qwen_stub_node로 대체 가능
               └───────────┬────────────┬─────────┘
                           │            │
              /qwen/labeled_detections   │   categories: TARGET / DESTINATION / OBSTACLE
              /qwen/grounding_result     │   destination 관계 JSON
              /qwen/mask_image ──────────┘   (trigger — 항상 마지막에 발행)
                           │
               ┌───────────▼────────────────────┐
               │   multi_view_projector_node     │  (mask_projection_pkg)
               │   EE depth + top depth + mask   │
               │   → 라벨링된 3D world point cloud│
               └───┬────────────────────────┬───┘
                   │                        │
          /world_map (PointCloud2)   /world_map_result (JSON trigger)
          /world_cloud_raw                  │
                   │                        │
                   │           ┌────────────▼────────────────────────┐
                   │           │          graspgen_node               │  ← 이 패키지
                   │           │  (또는 vgn_grasp_node — 교체 가능)   │
                   │           │                                      │
                   │           │  EE depth + mask → TARGET cloud      │
                   │           │  ZMQ → SSH tunnel → aurora-g5        │
                   │           │  GraspGen inference (GPU)            │
                   │           └────────────┬────────────────────────┘
                   │                        │
                   │              /grasp_candidates (JSON, Top-K)
                   │              /grasp_markers    (MarkerArray, RViz2)
                   │              /graspgen/target_cloud (PointCloud2, 디버그)
                   │                        │
               ┌───▼────────────────────────▼────────────────────────────┐
               │                    bt_executor_node                      │  (bt_pkg)
               │  WaitForScene: /world_map_result + /grasp_candidates     │
               │  ParseScene  → grasp_index=0, blackboard 채움            │
               │  Pick: SelectGraspCandidate × max_grasp_candidates 회    │
               │        → MoveAction(pre_grasp) → MoveAction(grasp)       │
               │        → GripperAction(close) → MoveAction(retreat)      │
               │  UpdateTargetPose: /qwen/grounding_result + YOLO         │
               │  Place: MoveAction(place) → GripperAction(open)          │
               └─────────────────────────────────────────────────────────┘
```

---

## vgn_grasp_pkg와의 관계 — 선택적 교체

두 패키지는 **동일한 출력 토픽과 JSON 스키마**를 공유한다. 한 번에 하나만 실행하고, 성능을 비교한 뒤 더 나은 쪽을 선택하면 된다.

| | `vgn_grasp_pkg` | `graspgen_pkg` |
|---|---|---|
| 추론 위치 | 로컬 CPU/GPU | 원격 Seraph aurora-g5 |
| 전송 방식 | 없음 (로컬) | ZMQ over SSH tunnel |
| 추가 전제 | 없음 | SSH tunnel 사전 개통 |
| `/grasp_candidates` 형식 | 동일 | 동일 |
| `/grasp_markers` | 발행 | 발행 |
| `bt_pkg` 코드 변경 | 불필요 | 불필요 |

> **주의**: 두 노드를 동시에 켜면 `/grasp_candidates`가 중복 발행되어 bt_pkg 동작이 불안정해진다. 반드시 하나만 실행할 것.

---

## SSH 터널 — GraspGen 서버 접속

GraspGen 서버는 Seraph 클러스터의 `aurora-g5` 노드에서 ZMQ(포트 5556)로 열려 있다. 로컬 PC에서는 SSH 포트 포워딩으로 접근한다.

### 터널 개통 (노드 실행 전 먼저 실행)

```bash
# 학내망
ssh -N -L 5556:aurora-g5:5556 <USERNAME>@aurora.khu.ac.kr

# 외부망 (교외)
ssh -p 30080 -N -L 5556:aurora-g5:5556 <USERNAME>@aurora.khu.ac.kr
```

- `-N` : 명령 실행 없이 터널만 유지
- `-L 5556:aurora-g5:5556` : 로컬 5556 → aurora-g5:5556 포워딩
- 터미널을 닫으면 터널이 끊기므로 별도 탭에서 유지

### 터널 확인

```bash
# 포트가 열려 있으면 터널 정상
nc -zv 127.0.0.1 5556
# 또는
ss -tlnp | grep 5556
```

터널 없이 노드를 켜면 `GraspGen server timeout after 5000 ms` 오류가 발생한다.

### GraspGen 서버 사양

- 서버: Seraph GPU node (aurora-g5 등)
- 모델: `graspgen_franka_panda.yml` (Franka Panda gripper 기준)
- API: ZMQ REQ/REP, msgpack binary 직렬화
- 검증 완료: point cloud (N,3) float32 전송 → grasps (M,4,4), confidences (M,) 반환

> **서버 운용**: 박상현이 Seraph에서 GPU 노드를 잡아 서버를 열어둔다.  
> 사용 전 박상현에게 현재 포트 번호를 확인할 것.  
> 포트 번호를 받은 뒤 터널 명령어의 `5556` 부분과 `zmq_port` 파라미터를 해당 포트로 맞추면 된다.

---

## 빌드 및 의존성

### Python 패키지 설치

```bash
pip install pyzmq msgpack msgpack-numpy
```

### 빌드

```bash
source launch_env.bash
colcon build --packages-select graspgen_pkg
source install/setup.bash
```

---

## 실행 방법

### 옵션 A — 단독 실행 (기존 파이프라인에 붙이기)

GSAM + Projection 파이프라인이 이미 실행 중일 때 graspgen_node만 추가.

```bash
# 터널 먼저 개통 후
ros2 launch graspgen_pkg graspgen.launch.py zmq_port:=5556 topk_num_grasps:=5
```

### 옵션 B — 전체 파이프라인 통합 실행 (Gazebo 데모)

```bash
# T1: Gazebo + RViz
ros2 launch rgbd_projection rgbd_sim.launch.py

# T2: SSH 터널 (별도 탭)
ssh -N -L 5556:aurora-g5:5556 <USERNAME>@aurora.khu.ac.kr

# T3: 전체 파이프라인 (GSAM + Qwen stub + Projection + GraspGen)
ros2 launch graspgen_pkg full_pipeline_graspgen.launch.py \
  prompt:="cup, table" \
  zmq_port:=5556 \
  topk_num_grasps:=5
```

### 옵션 C — qwen_pkg 실제 VLM 연동 시

`full_pipeline_graspgen.launch.py`의 `qwen_stub_node`를 `qwen_pkg`로 교체. `graspgen_node` 자체는 변경 없음.

```bash
# qwen_pkg는 /qwen/labeled_detections, /qwen/mask_image를 동일 형식으로 발행
# → graspgen_node는 그대로 동작
ros2 launch qwen_pkg inst_input_qwen.launch.py
```

---

## 토픽 인터페이스

### 구독

| 토픽 | 타입 | 역할 |
|---|---|---|
| `/ee_camera/depth_image` | `sensor_msgs/Image` | EE 깊이 이미지 (캐시) |
| `/ee_camera/camera_info` | `sensor_msgs/CameraInfo` | EE 내부 파라미터 (캐시) |
| `/qwen/mask_image` | `sensor_msgs/Image` | GSAM 마스크 (캐시) |
| `/qwen/labeled_detections` | `std_msgs/String` | 카테고리 라벨 JSON (캐시) |
| `/world_map_result` | `std_msgs/String` | Projection 결과 **(트리거)** |

`/world_map_result` 수신 시 처리 시작. 나머지 4개가 아직 미도착이면 캐시 후 자동 처리.

### 발행

| 토픽 | 타입 | 소비자 |
|---|---|---|
| `/grasp_candidates` | `std_msgs/String` | `bt_executor_node` (bt_pkg) |
| `/grasp_markers` | `visualization_msgs/MarkerArray` | RViz2 |
| `/graspgen/target_cloud` | `sensor_msgs/PointCloud2` | 디버그 (RViz2) |

### `/grasp_candidates` JSON 스키마

`vgn_grasp_pkg`와 동일.

```json
{
  "candidates": [
    {
      "position":   [x, y, z],
      "quaternion": [qx, qy, qz, qw],
      "width":      0.08,
      "quality":    0.92,
      "frame":      "panda_link0"
    }
  ],
  "target_centroid": [x, y, z],
  "stamp": 1234567890.123
}
```

- `candidates`는 `quality` 내림차순 정렬 (index 0이 best)
- TF(`world` → `panda_link0`) 사용 가능 시 로봇 베이스 프레임으로 변환해 발행
- TF 실패 시 `world` 프레임으로 발행하며 경고 로그 출력

---

## bt_pkg 연동 — `max_grasp_candidates` 커플링

`bt_pkg`는 `/world_map_result` **AND** `/grasp_candidates` 양쪽이 fresh 될 때까지 대기(`WaitForScene`)한 뒤 파이프라인을 진행한다.

`max_grasp_candidates`는 세 곳에서 동일한 값을 바라봐야 한다:

```
config/robot_defaults.yaml
  max_grasp_candidates: 5          ← 이 값 하나만 수정
         ↓                                ↓
graspgen_node                    bt_executor_node
  topk_num_grasps: 5        →    {max_grasp_candidates}=5 (blackboard)
  (graspgen_params.yaml)               ↓
                              pick_and_place.xml
                              RetryUntilSuccessful num_attempts="{max_grasp_candidates}"
```

`graspgen_params.yaml`의 `topk_num_grasps`와 `robot_defaults.yaml`의 `max_grasp_candidates`를 맞춰야 BT retry 횟수와 후보 수가 일치한다. 현재 둘 다 **5**로 설정되어 있다.

---

## 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `zmq_host` | `127.0.0.1` | SSH 터널 로컬 주소 |
| `zmq_port` | `5556` | SSH 터널 로컬 포트 |
| `zmq_timeout_ms` | `5000` | ZMQ 수신 타임아웃 (ms) |
| `num_grasps` | `50` | 서버에 요청할 총 후보 수 |
| `topk_num_grasps` | `5` | 발행할 상위 K개 ← `max_grasp_candidates`와 맞출 것 |
| `min_point_count` | `50` | TARGET 포인트 최소값 (이하 skip) |
| `max_points` | `4096` | 초과 시 랜덤 다운샘플 |
| `gripper_width` | `0.08` | Panda gripper 폭 (m) |
| `extrinsics_config` | `''` | 엑스트린식 YAML 경로 (빈 값 → 패키지 내 기본값) |
| `world_frame` | `world` | 월드 프레임 ID |
| `robot_frame` | `panda_link0` | 로봇 베이스 프레임 ID |

Isaac Sim 전환 시:

```bash
ros2 launch graspgen_pkg graspgen.launch.py \
  extrinsics_config:=/path/to/camera_extrinsics_isaac.yaml
```

---

## qwen_stub_node vs qwen_pkg

| | `qwen_stub_node` (grounded_sam_pkg) | `qwen_bridge_node` (qwen_pkg) |
|---|---|---|
| 추론 | 없음 (mask pass-through) | 원격 Qwen VLM (vLLM endpoint) |
| `/qwen/labeled_detections` | prompt 순서 기반 고정 매핑 | VLM이 TARGET/DESTINATION/OBSTACLE 판별 |
| `/qwen/grounding_result` | 발행 안 함 | 발행 (destination 관계 JSON) |
| `/qwen/mask_image` | 발행 | 발행 |
| 사용 시기 | Gazebo 데모, 빠른 테스트 | Isaac Sim 실제 통합 |

`graspgen_node`는 `/qwen/labeled_detections`에서 `"category": "TARGET"` 필드를 읽어 추출 대상 마스크 값을 결정한다. stub/실제 qwen 모두 동일 형식이므로 graspgen_node 코드 변경 없음.

---

## 내부 처리 흐름

```
/world_map_result 수신 (트리거)
  │
  ├─ 캐시 확인: ee_depth / ee_K / mask 모두 있는가?
  │    없으면 → pending_result에 저장 후 대기
  │    있으면 → 처리 진행
  │
  ├─ find_target_mask_val()
  │    labeled_detections에서 category=="TARGET"인 detection의 idx+1 반환
  │    없으면 fallback: mask_val=1 (첫 번째 detection)
  │
  ├─ extract_target_cloud()
  │    ee_depth → 역투영 → TARGET 마스크 픽셀 선택
  │    → p_world = R_ee @ p_cam + t_ee
  │    → NaN/inf 제거, max_points 초과 시 랜덤 다운샘플
  │
  ├─ point_count < min_point_count → skip
  │
  ├─ ZMQ request → aurora-g5 GraspGen
  │    payload: {point_cloud: (N,3) float32, num_grasps, topk_num_grasps}
  │    response: {grasps: (M,4,4) float32, confidences: (M,) float32}
  │
  ├─ quality 내림차순 정렬 → Top-K 선택
  │
  ├─ TF lookup: world → panda_link0
  │    성공 → 로봇 프레임으로 변환
  │    실패 → world 프레임 그대로 발행 (경고)
  │
  └─ /grasp_candidates 발행 + /grasp_markers 발행
```

---

## 디버깅

```bash
# 후보 확인
ros2 topic echo /grasp_candidates

# TARGET 포인트 클라우드 확인 (RViz2에서 추가 가능)
ros2 topic hz /graspgen/target_cloud

# 트리거 확인
ros2 topic echo /world_map_result

# ZMQ 연결 상태 로그
ros2 run graspgen_pkg graspgen_node --ros-args --log-level DEBUG
```

터널이 끊겼을 때는 노드를 재시작하면 자동 재연결된다.

---

## 파일 구조

```
graspgen_pkg/
  graspgen_pkg/
    graspgen_node.py      메인 ROS 2 노드
    zmq_client.py         ZMQ REQ/REP 클라이언트 (프로토콜 변경 시 이 파일만 수정)
    cloud_extractor.py    EE depth → TARGET world point cloud
    depth_utils.py        depth 디코드, 역투영, extrinsics 로드
    marker_publisher.py   RViz2 마커 생성
  launch/
    graspgen.launch.py              단독 실행
    full_pipeline_graspgen.launch.py 전체 파이프라인 (GSAM + stub + proj + graspgen)
  config/
    graspgen_params.yaml  기본 파라미터 (topk_num_grasps: 5)
```
