# grounded_sam_pkg

GroundingDINO + SAM 기반 객체 세그멘테이션 ROS 2 패키지.  
RGB 이미지를 입력받아 텍스트 프롬프트로 지정한 물체를 감지하고, 각 물체의 세그멘테이션 마스크를 출력합니다.

---

## 노드 구성

| 노드 | 실행 파일 | 역할 |
|---|---|---|
| `grounded_sam_node` | `grounded_sam_node` | GroundingDINO + SAM 추론 |
| `qwen_stub_node` | `qwen_stub_node` | Qwen VLM 임시 대체 — label 기반 category 할당 |
| `instruction_parser_node` | `instruction_parser_node` | 자연어 명령 → Gemini API → GSAM 프롬프트 자동 변환 |

---

## 모듈 구성

```
grounded_sam_pkg/
├── ros_node.py               ROS 2 노드 (구독/발행 wiring) — /dino_prompt 구독으로 런타임 프롬프트 업데이트 지원
├── pipeline.py               GroundingDINO + SAM 추론 orchestration
├── gdino_runner.py           GroundingDINO 추론 래퍼
├── sam_runner.py             SAM 추론 래퍼
├── postprocess.py            감지 결과 → JSON 변환
├── prompt_adapter.py         프롬프트 전처리
├── visualizer.py             annotated_image 생성
├── qwen_stub_node.py         Qwen VLM 임시 stub 노드
└── instruction_parser_node.py  자연어 → Gemini 2.5 Flash → /dino_prompt 발행
config/
├── model_paths.yaml     모델 가중치 경로 + 추론 파라미터
└── runtime.yaml         런타임 옵션
launch/
└── grounded_sam.launch.py
```

---

## 노드 1 — grounded_sam_node

### 구독 토픽

| 토픽 | 타입 | 설명 |
|---|---|---|
| `<image_topic>` | `sensor_msgs/Image` | RGB 입력 이미지 (파라미터로 설정) |

### 발행 토픽

| 토픽 | 타입 | 설명 |
|---|---|---|
| `/grounded_sam/mask_image` | `sensor_msgs/Image` | 세그멘테이션 마스크 (mono8, pixel = 1-based idx) |
| `/grounded_sam/detections_json` | `std_msgs/String` | 감지 결과 JSON 배열 |
| `/grounded_sam/annotated_image` | `sensor_msgs/Image` | bbox + 마스크 오버레이 시각화 이미지 |

### detections_json 포맷

mask pixel 값은 배열의 1-based 인덱스입니다 (`idx=0` → pixel=1).

```json
[
  {"idx": 0, "label": "cup",   "confidence": 0.91, "bbox_xyxy": [10, 20, 80, 90]},
  {"idx": 1, "label": "table", "confidence": 0.95, "bbox_xyxy": [0, 0, 640, 480]},
  {"idx": 2, "label": "object","confidence": 0.82, "bbox_xyxy": [150, 60, 220, 130]}
]
```

### ROS 2 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `image_topic` | `/ee_camera/image` | 구독할 RGB 이미지 토픽 |
| `prompt` | `"object"` | 탐지할 물체 텍스트 프롬프트 (쉼표 구분) |
| `model_config` | launch 기본값 사용 | `model_paths.yaml` 경로 |

### 실행

```bash
ros2 launch grounded_sam_pkg grounded_sam.launch.py \
  prompt:="cup, table, object"
```

---

## 노드 2 — qwen_stub_node

Qwen VLM이 연동되기 전까지 사용하는 임시 노드.  
`detections_json` 의 `label` 필드를 기반으로 `category` 를 추가하고, 마스크를 pass-through 합니다.

**카테고리 할당 규칙 (현재 Gazebo 데모용 하드코딩):**

| label | category |
|---|---|
| `cup` | `TARGET` |
| `table` | `WORKSPACE` |
| 그 외 | `OBSTACLE` |

### 구독 토픽

| 토픽 | 타입 | 역할 |
|---|---|---|
| `/grounded_sam/detections_json` | `std_msgs/String` | GSAM 감지 결과 — 캐시 |
| `/grounded_sam/mask_image` | `sensor_msgs/Image` | GSAM 마스크 — **트리거** |

### 발행 토픽

| 토픽 | 타입 | 설명 |
|---|---|---|
| `/qwen/mask_image` | `sensor_msgs/Image` | 마스크 pass-through |
| `/qwen/labeled_detections` | `std_msgs/String` | `category` 필드 추가된 JSON |

### labeled_detections 포맷

```json
[
  {"idx": 0, "label": "cup",   "category": "TARGET",    "confidence": 0.91, "bbox_xyxy": [10, 20, 80, 90]},
  {"idx": 1, "label": "table", "category": "WORKSPACE", "confidence": 0.95, "bbox_xyxy": [0, 0, 640, 480]},
  {"idx": 2, "label": "object","category": "OBSTACLE",  "confidence": 0.82, "bbox_xyxy": [150, 60, 220, 130]}
]
```

### 실행

```bash
ros2 run grounded_sam_pkg qwen_stub_node
```

---

---

## 노드 3 — instruction_parser_node

자연어 명령을 받아 Gemini 2.5 Flash API로 명사구를 추출하고 `/dino_prompt`로 발행합니다.  
`grounded_sam_node`가 `/dino_prompt`를 구독해 다음 추론 사이클부터 새 프롬프트를 적용합니다.

### 흐름

```
/instruction  (자연어)
      ↓  Gemini 2.5 Flash (~1–2 s)
/dino_prompt  (쉼표 구분 명사구)
      ↓
grounded_sam_node  (다음 추론 사이클에 적용)
```

### 구독 토픽

| 토픽 | 타입 | 설명 |
|---|---|---|
| `/instruction` | `std_msgs/String` | 자연어 명령 |

### 발행 토픽

| 토픽 | 타입 | 설명 |
|---|---|---|
| `/dino_prompt` | `std_msgs/String` | 추출된 명사구 (쉼표 구분) |

### 환경 변수

| 변수 | 설명 |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API 키 (필수) |

### 의존성

```bash
pip install google-genai   # requirements.txt에 포함됨
```

### 실행

```bash
# 터미널 1 — 노드 실행
export GEMINI_API_KEY="your_api_key_here"
ros2 run grounded_sam_pkg instruction_parser_node

# 터미널 2 — 명령 발행
ros2 topic pub /instruction std_msgs/msg/String \
  "data: 'Pick up the glass cup on the table, avoid the red ball'" --once

# 터미널 3 — 결과 확인
ros2 topic echo /dino_prompt
# 출력: data: 'glass cup, table, red ball'
```

### grounded_sam_node 연동

`ros_node.py`는 `/dino_prompt` 구독을 내장하고 있어 별도 설정 없이 연동됩니다.  
새 프롬프트 수신 시 EE/Top 양쪽 뷰가 동시에 업데이트되며, 다음 추론 사이클(`min_process_interval_sec` 이후)부터 반영됩니다.

```
/dino_prompt 수신
  → self.prompt_raw      업데이트  (EE 뷰)
  → self._top_prompt_raw 업데이트  (Top 뷰)
  → 다음 EE 콜백 트리거 시 새 프롬프트로 추론
```

---

## 모델 설정 (config/model_paths.yaml)

가중치를 `models/g-sam/` 아래에 두면 저장소 루트 기준 상대경로로 자동 해석됩니다.

```yaml
grounding_dino:
  config_file: ""       # 비워두면 pip 설치 경로 자동 감지
  checkpoint: "models/g-sam/groundingdino_swint_ogc.pth"
  box_threshold: 0.35
  text_threshold: 0.25
  device: "cuda"

sam:
  model_type: "vit_b"
  checkpoint: "models/g-sam/sam_vit_b_01ec64.pth"
  device: "cuda"
```

---

## 모델 가중치 다운로드

저장소 루트에서 실행:

```bash
mkdir -p models/g-sam
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
     -O models/g-sam/groundingdino_swint_ogc.pth
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
     -O models/g-sam/sam_vit_b_01ec64.pth
```

| 모델 | 파일 | 크기 |
|---|---|---|
| GroundingDINO SwinT | `groundingdino_swint_ogc.pth` | ~662 MB |
| SAM ViT-B | `sam_vit_b_01ec64.pth` | ~375 MB |

---

## Isaac Sim 연동 시 교체할 것

### Qwen VLM 연동

`qwen_stub_node.py` 의 `LABEL_TO_CATEGORY` 딕셔너리 로직을 실제 Qwen API 호출로 교체합니다.  
발행 토픽 이름(`/qwen/mask_image`, `/qwen/labeled_detections`)과 JSON 포맷은 그대로 유지하면  
`mask_projection_pkg` 다운스트림 노드 수정 없이 연동됩니다.

### 이미지 토픽

Isaac Sim bridge 토픽에 맞춰 launch 인자만 변경합니다:

```bash
ros2 launch grounded_sam_pkg grounded_sam.launch.py \
  image_topic:=/isaac/ee/image \
  prompt:="glass cup, table, red ball, blue cube, book"
```

---

## 주의사항

- Gazebo bridge는 VOLATILE QoS로 발행 — `grounded_sam_node` 구독도 VOLATILE(depth=10) 사용.
