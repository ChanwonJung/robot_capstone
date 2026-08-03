"""
Qwen VLM 자리를 대신하는 stub 노드.
mask_image 수신 시 detections의 label → category 변환 후 발행.

NOTE: labeled_detections 와 mask_image 는 별도 메시지로 짧은 시간차가 있음.
projector_node가 cache 방식이라 CPU 추론(30-40s/frame) 환경에서는 문제없음.
"""
from __future__ import annotations

import json
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

def _stamp_age_sec(node: Node, msg: Image) -> float | None:
    stamp = msg.header.stamp
    stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    if stamp_ns <= 0:
        return None
    age = (node.get_clock().now().nanoseconds - stamp_ns) * 1e-9
    # Isaac/sim headers and ROS/system clock can use different epochs. In that
    # case the computed age becomes huge and is less useful than omitting it.
    if age < 0.0 or age > 3600.0:
        return None
    return age


LABEL_TO_CATEGORY: dict[str, str] = {
    # 투명 유리컵 데모: depth see-through 로 cloud ring 만 잡히지만,
    # graspgen_node 의 transparent_reconstruct 가 마스크+원통 prior 로 복원.
    "cup":        "TARGET",
    "glass cup":  "TARGET",
    "mug":        "TARGET",
    # 빨간 공 및 책 — grasp 대상.
    "apple":      "OBSTACLE",
    "red ball":   "TARGET",
    "ball":       "TARGET",
    # 책은 pick 대상.
    "book":       "TARGET",
    "table":      "DESTINATION",
}


class QwenStubNode(Node):

    def __init__(self) -> None:
        super().__init__("qwen_stub_node")

        # 아래 표는 "빨간 공 데모" 기준이라 apple 이 OBSTACLE 로 고정돼 있다.
        # 대상을 바꿔가며 시험할 때 매번 표를 고치지 않도록 런타임 override 를
        # 둔다. target_label 이 주어지면 그 라벨만 TARGET 이 되고, 원래 TARGET
        # 이던 다른 라벨은 OBSTACLE 로 내려간다 (TARGET 이 둘이면 graspgen 이
        # 어느 쪽을 잡을지 모호해지므로). DESTINATION 은 건드리지 않는다.
        self._target_label = str(
            self.declare_parameter("target_label", "").value).lower().strip()
        if self._target_label:
            self.get_logger().info(
                f"target_label override: '{self._target_label}' 만 TARGET")
        else:
            tgt = [k for k, v in LABEL_TO_CATEGORY.items() if v == "TARGET"]
            self.get_logger().info(
                f"target_label 미지정 — 기본 표 사용 (TARGET: {', '.join(tgt)})")

        self._latest_detections: list[dict] | None = None
        self._pending_mask: Image | None = None

        # QoS depth=10 matches grounded_sam_node's VOLATILE publishers
        self.create_subscription(
            String, "/grounded_sam/detections_json", self._json_cb, 10)
        self.create_subscription(
            Image,  "/grounded_sam/mask_image",       self._mask_cb, 10)

        self._pub_detections = self.create_publisher(
            String, "/qwen/labeled_detections", 10)
        self._pub_mask = self.create_publisher(
            Image,  "/qwen/mask_image", 10)

        self.get_logger().info("QwenStubNode ready")

    def _json_cb(self, msg: String) -> None:
        t0 = time.monotonic()
        try:
            self._latest_detections = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"detections_json parse error: {e}")
            return
        self.get_logger().info(
            f"[PROFILE][qwen_stub][json] parse={time.monotonic() - t0:.6f}s "
            f"detections={len(self._latest_detections)}"
        )
        # mask가 detections_json보다 먼저 도착한 경우 (race condition) 즉시 처리
        if self._pending_mask is not None:
            self.get_logger().info("detections_json 도착 — 대기 중인 mask 처리")
            self._process(self._pending_mask)
            self._pending_mask = None

    def _mask_cb(self, mask_msg: Image) -> None:
        if self._latest_detections is None:
            self._pending_mask = mask_msg
            self.get_logger().warn("detections_json 미도착 — mask 캐시 후 대기")
            return
        self._process(mask_msg)

    def _process(self, mask_msg: Image) -> None:
        t0 = time.monotonic()
        labeled = []
        for det in self._latest_detections:
            label = det.get("label", "").lower().strip()
            category = LABEL_TO_CATEGORY.get(label, "OBSTACLE")
            if self._target_label:
                # override: 지정 라벨만 TARGET, 나머지 TARGET 은 OBSTACLE 로.
                if label == self._target_label:
                    category = "TARGET"
                elif category == "TARGET":
                    category = "OBSTACLE"
            labeled.append({**det, "category": category})

        # publish detections BEFORE mask so projector has latest when triggered
        self._pub_detections.publish(String(data=json.dumps(labeled)))
        self._pub_mask.publish(mask_msg)
        age = _stamp_age_sec(self, mask_msg)
        age_text = f"{age:.3f}s" if age is not None else "n/a"
        self.get_logger().info(
            f"[PROFILE][qwen_stub] total={time.monotonic() - t0:.6f}s "
            f"detections={len(labeled)} mask_input_age={age_text}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = QwenStubNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
