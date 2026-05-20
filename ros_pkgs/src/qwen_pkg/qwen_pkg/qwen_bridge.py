"""
qwen_bridge.py

ROS 2 node that bridges topic inputs to Qwen VLM inference.

Subscribes:
  /grounded_sam/detections_json  (std_msgs/String) — triggers inference on receipt
  /grounded_sam/mask_image       (sensor_msgs/Image) — cached, re-published after inference
  /user_instruction              (std_msgs/String) — natural-language robot command

Publishes:
  /qwen/labeled_detections  (std_msgs/String) — detections JSON enriched with "category"
  /qwen/grounding_result    (std_msgs/String) — GroundingResult JSON with target + destination relation
  /qwen/mask_image          (sensor_msgs/Image) — mask pass-through, published after inference
                                                   to trigger multi_view_projector_node at the
                                                   right time (i.e. after labels are ready)

ROS parameters:
  vllm_endpoint_url  (str) — base URL of the vLLM OpenAI-compat server
  model_name         (str) — model name registered in the vLLM server
"""
from __future__ import annotations

import json
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .qwen_call import classify_detections


class QwenBridgeNode(Node):

    def __init__(self) -> None:
        super().__init__("qwen_bridge_node")

        self.declare_parameter("vllm_endpoint_url", "http://localhost:8000/v1")
        self.declare_parameter("model_name", "qwen-vl")
        self.declare_parameter("instruction", "")

        self._endpoint_url: str = (
            self.get_parameter("vllm_endpoint_url").get_parameter_value().string_value
        )
        self._model_name: str = (
            self.get_parameter("model_name").get_parameter_value().string_value
        )

        seed_instruction = self.get_parameter("instruction").get_parameter_value().string_value
        self._latest_instruction: str | None = seed_instruction if seed_instruction else None
        self._latest_mask: Image | None = None
        self._lock = threading.Lock()
        self._busy = False

        self.create_subscription(
            String, "/grounded_sam/detections_json", self._detections_cb, 10
        )
        self.create_subscription(
            Image, "/grounded_sam/mask_image", self._mask_cb, 10
        )
        self.create_subscription(
            String, "/user_instruction", self._instruction_cb, 10
        )

        self._pub = self.create_publisher(String, "/qwen/labeled_detections", 10)
        self._grounding_pub = self.create_publisher(String, "/qwen/grounding_result", 10)
        self._mask_pub = self.create_publisher(Image, "/qwen/mask_image", 10)

        self.get_logger().info(
            f"QwenBridgeNode ready — endpoint={self._endpoint_url}  model={self._model_name}"
            + (f"  instruction='{self._latest_instruction}'" if self._latest_instruction else "  (waiting for /user_instruction)")
        )

    def _instruction_cb(self, msg: String) -> None:
        with self._lock:
            self._latest_instruction = msg.data

    def _mask_cb(self, msg: Image) -> None:
        with self._lock:
            self._latest_mask = msg

    def _detections_cb(self, msg: String) -> None:
        try:
            detections = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"detections_json parse error: {exc}")
            return

        with self._lock:
            if self._busy:
                self.get_logger().warn("VLM call in progress, dropping detections")
                return
            instruction = self._latest_instruction
            if instruction is None:
                self.get_logger().warn("No instruction cached, skipping detections")
                return
            mask = self._latest_mask
            if mask is None:
                self.get_logger().warn("No mask cached, skipping detections")
                return
            self._busy = True

        threading.Thread(
            target=self._run_inference,
            args=(detections, instruction, mask),
            daemon=True,
        ).start()

    def _run_inference(self, detections: list[dict], instruction: str, mask: Image) -> None:
        try:
            labeled, grounding = classify_detections(
                detections,
                instruction,
                endpoint_url=self._endpoint_url,
                model=self._model_name,
            )
            self._pub.publish(String(data=json.dumps(labeled)))
            grounding_json = (
                grounding.model_dump_json()
                if hasattr(grounding, "model_dump_json")
                else grounding.json()
            )
            self._grounding_pub.publish(String(data=grounding_json))
            # Publish mask last — this triggers multi_view_projector_node,
            # so labeled_detections must be published first.
            self._mask_pub.publish(mask)
            self.get_logger().info(
                f"Classified {len(labeled)} detections for '{instruction}' — "
                f"target={grounding.target_label}(id={grounding.target_id}) "
                f"dest_type={grounding.destination.type} "
                f"ref={grounding.destination.reference_id}"
            )
        except Exception as exc:
            self.get_logger().error(f"classify_detections failed: {exc}")
        finally:
            with self._lock:
                self._busy = False


def main(args=None) -> None:
    rclpy.init(args=args)
    node = QwenBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
