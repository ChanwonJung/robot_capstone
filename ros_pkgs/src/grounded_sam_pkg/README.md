# grounded_sam_pkg

RGB 이미지를 **Grounded SAM** (Grounding DINO + SAM) 으로 세그멘테이션하는 ROS 2 노드입니다.  
텍스트 프롬프트로 물체를 지정하면 바운딩박스 + 마스크를 생성하고, 결과를 토픽으로 발행합니다.

> **전체 파이프라인에서의 위치:**  
> 카메라 이미지 입력 → **grounded_sam_pkg** → mask_projection_pkg → MoveIt2

---

## 발행 토픽

| 토픽 | 타입 | 설명 |
|---|---|---|
| `/grounded_sam/mask_image` | `sensor_msgs/Image` (mono8) | 세그멘테이션 마스크 (픽셀값 = 1-based 객체 인덱스) |
| `/grounded_sam/detections_json` | `std_msgs/String` | 탐지 결과 JSON (label, confidence, bbox_xyxy) |
| `/grounded_sam/annotated_image` | `sensor_msgs/Image` | bbox + mask 오버레이 시각화 이미지 |

## 구독 토픽

| 토픽 | 타입 | 설명 |
|---|---|---|
| `/camera/image_raw` (기본값) | `sensor_msgs/Image` | RGB 카메라 이미지 — launch 파라미터로 오버라이드 가능 |

---

## 설치

### 1. Python 가상환경 생성 및 의존성 설치

```bash
cd ~/robot_capstone
python3 -m venv gsam_venv
source gsam_venv/bin/activate
pip install -r ros_pkgs/src/grounded_sam_pkg/requirements.txt
```

### 2. 모델 가중치 다운로드

가중치 파일은 git에 포함되지 않습니다. 직접 다운로드하세요.

```bash
mkdir -p ~/robot_capstone/models/slow_brain/g-sam

# GroundingDINO SwinT (~662 MB)
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
     -O ~/robot_capstone/models/slow_brain/g-sam/groundingdino_swint_ogc.pth

# SAM ViT-H (~2.4 GB)
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
     -O ~/robot_capstone/models/slow_brain/g-sam/sam_vit_h_4b8939.pth
```

| 모델 | 파일명 | 크기 |
|---|---|---|
| GroundingDINO SwinT | `groundingdino_swint_ogc.pth` | ~662 MB |
| SAM ViT-H | `sam_vit_h_4b8939.pth` | ~2.4 GB |

### 3. ROS 2 빌드

```bash
cd ~/robot_capstone
source launch_env.bash
cd ros_pkgs
colcon build
source install/setup.bash
```

---

## 실행

```bash
# 터미널마다 환경 설정 필수
source ~/robot_capstone/launch_env.bash

ros2 launch grounded_sam_pkg grounded_sam.launch.py \
  prompt:="cup, table"
```

### 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `prompt` | `"object"` | 탐지할 물체 (쉼표 구분) |
| `image_topic` | `/camera/image_raw` | 구독할 RGB 이미지 토픽 |
| `model_config` | (필수) | `model_paths.yaml` 절대경로 |

### Isaac Sim 연동 시

Isaac Sim의 카메라 토픽명이 다를 경우 `image_topic` 파라미터만 오버라이드하면 됩니다:

```bash
ros2 launch grounded_sam_pkg grounded_sam.launch.py \
  prompt:="cup, table" \
  image_topic:=/isaac/camera/image_raw
```

---

## 모델 설정 (`config/model_paths.yaml`)

```yaml
grounding_dino:
  config_file: ""       # 비워두면 pip 설치 경로 자동감지
  checkpoint: "~/robot_capstone/models/slow_brain/g-sam/groundingdino_swint_ogc.pth"
  box_threshold: 0.35
  text_threshold: 0.25
  device: "cpu"         # GPU 사용 시 "cuda"

sam:
  model_type: "vit_h"
  checkpoint: "~/robot_capstone/models/slow_brain/g-sam/sam_vit_h_4b8939.pth"
  device: "cpu"         # GPU 사용 시 "cuda"
```

**CPU 환경 주의:** SAM ViT-H + GroundingDINO SwinT CPU 추론 시 프레임당 30~40초 소요.  
속도가 중요하면 `model_type: "vit_b"` (375 MB) 로 변경하거나 GPU 환경에서 실행하세요.

---

## 출력 파일

추론 결과는 `~/robot_capstone/output/`에 자동 저장됩니다.

| 파일 | 설명 |
|---|---|
| `result_{initials}.jpg` | bbox + mask 오버레이 이미지 (매 프레임 덮어씀) |

---

## Gazebo 데모 (개발/테스트용)

Isaac Sim 없이 Gazebo 환경에서 단독으로 테스트하려면 개인 개발 레포를 참고하세요:  
→ https://github.com/tydfuyhf/grounded_sam_ros2_pkg

`rgbd_projection` 패키지가 포함되어 있으며, tabletop 씬 + RGBD 카메라 + ROS bridge 환경을 제공합니다.
