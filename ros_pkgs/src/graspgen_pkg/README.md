# graspgen_pkg

EE depth + GSAM 마스크 → GraspGen 원격 추론 → `/grasp_candidates` 발행.  
`vgn_grasp_pkg`와 동일한 토픽/JSON 형식 — `bt_pkg` 코드 변경 불필요.

---

## 요구사항

### 구독 토픽 (아래 토픽이 발행되고 있어야 동작)

| 토픽 | 발행 노드 |
|---|---|
| `/ee_camera/depth_image` | Isaac Sim |
| `/ee_camera/camera_info` | Isaac Sim |
| `/qwen/mask_image` | `qwen_bridge_node` **(트리거)** |
| `/qwen/labeled_detections` | `qwen_bridge_node` |
| `/world_map_result` | `multi_view_projector_node` **(트리거)** |

### Python 패키지

```bash
pip install pyzmq msgpack msgpack-numpy
```

### SSH 터널 (노드 실행 전 개통)

> 포트 번호는 박상현에게 확인. 아래 `5556` 자리에 해당 포트 입력.

```bash
# 학내망
ssh -N -L 5556:aurora-g5:5556 <USERNAME>@aurora.khu.ac.kr

# 외부망
ssh -p 30080 -N -L 5556:aurora-g5:5556 <USERNAME>@aurora.khu.ac.kr
```

---

## 빌드

```bash
source launch_env.bash
colcon build --packages-select graspgen_pkg
source install/setup.bash
```

## 실행

```bash
ros2 launch graspgen_pkg graspgen.launch.py \
  extrinsics_config:=/path/to/camera_extrinsics_isaac.yaml \
  zmq_port:=5556
```

`extrinsics_config`는 `mask_projection_pkg`와 **동일한 YAML 파일**을 그대로 사용하면 된다.

---

## 발행 토픽

| 토픽 | 소비자 |
|---|---|
| `/grasp_candidates` | `bt_executor_node` |
| `/grasp_markers` | RViz2 |
| `/graspgen/target_cloud` | 디버그 |

### `/grasp_candidates` JSON 형식

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

`candidates`는 `quality` 내림차순 정렬. TF(`world` → `panda_link0`) 실패 시 `frame`은 `"world"`로 발행.

---

## 주의사항

- `topk_num_grasps`(기본 5) = `robot_defaults.yaml`의 `max_grasp_candidates` 값과 반드시 일치시킬 것
- `vgn_grasp_node`와 동시에 켜면 `/grasp_candidates` 중복 발행 → 둘 중 하나만 실행
- 터널 없이 실행하면 5초 후 timeout 오류 발생
