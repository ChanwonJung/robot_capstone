"""GroundedSAMNode — GSAM 추론 결과를 ROS 2 토픽으로 발행."""

import json
import os
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from sensor_msgs.msg import Image
from std_msgs.msg import String

# LATCHED_QOS = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
# ↑ TRANSIENT_LOCAL subscriber is incompatible with Gazebo bridge's VOLATILE publisher

from .pipeline import GroundedSAMPipeline
from .postprocess import format_detections, format_masks, build_label_map
from .prompt_adapter import PromptAdapter
from .visualizer import draw_bboxes, draw_masks, save_result

OUTPUT_DIR = Path(os.environ.get("GSAM_WS", str(Path.home() / "gsam_ws"))) / "output"


class GroundedSAMNode(Node):
    def __init__(self):
        super().__init__("grounded_sam_node")

        self.declare_parameter("model_config", "")
        self.declare_parameter("prompt", "object")
        self.declare_parameter("image_topic", "/camera/image_raw")

        model_config = self.get_parameter("model_config").value
        self.prompt_raw = self.get_parameter("prompt").value
        image_topic = self.get_parameter("image_topic").value

        if not model_config:
            raise ValueError("Parameter 'model_config' must be set to the path of model_paths.yaml")

        self.bridge = CvBridge()
        self.adapter = PromptAdapter()

        self.get_logger().info("Loading models...")
        self.pipeline = GroundedSAMPipeline(model_config)

        # subscriber — use default VOLATILE QoS to match Gazebo bridge publisher
        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self._image_callback,
            qos_profile=10,
        )

        # publishers
        self.pub_annotated = self.create_publisher(Image, "/grounded_sam/annotated_image", 10)
        self.pub_mask = self.create_publisher(Image, "/grounded_sam/mask_image", 10)
        self.pub_json = self.create_publisher(String, "/grounded_sam/detections_json", 10)

        self.get_logger().info(
            f"Ready — subscribed to '{image_topic}', prompt='{self.prompt_raw}'"
        )

    def _image_callback(self, msg: Image) -> None:
        image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        prompt = self.adapter.adapt(self.prompt_raw)

        result = self.pipeline.run(image=image_bgr, prompt=prompt)

        det_list = format_detections(result["detections"], result["phrases"])
        self.get_logger().info(f"Detected {len(det_list)} object(s)")

        # sort detections by prompt keyword order so mask index matches prompt priority
        prompt_keywords = [p.strip().lower() for p in self.prompt_raw.split(",")]

        def _prompt_order(det):
            label = det["label"].lower()
            for i, kw in enumerate(prompt_keywords):
                if kw in label or label in kw:
                    return i
            return len(prompt_keywords)

        sorted_indices = sorted(range(len(det_list)), key=lambda i: _prompt_order(det_list[i]))
        det_list = [det_list[i] for i in sorted_indices]

        # --- annotated image ---
        vis = draw_bboxes(image_bgr, det_list)
        if result["masks"] is not None:
            mask_list = format_masks(result["masks"], result["mask_scores"])
            mask_list = [mask_list[i] for i in sorted_indices]
            vis = draw_masks(vis, mask_list)
            label_map = build_label_map(image_bgr.shape[:2], mask_list)
        else:
            mask_list = []
            label_map = np.zeros(image_bgr.shape[:2], dtype=np.uint8)

        annotated_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
        annotated_msg.header = msg.header   # preserve source timestamp + frame_id
        self.pub_annotated.publish(annotated_msg)

        mask_msg = self.bridge.cv2_to_imgmsg(label_map, encoding="mono8")
        mask_msg.header = msg.header        # ← key: same stamp as depth/camera_info
        self.pub_mask.publish(mask_msg)

        msg_json = String()
        msg_json.data = json.dumps(det_list)
        self.pub_json.publish(msg_json)

        # 파일 저장: result_gcbp.jpg 형식 (각 명사 첫 글자)
        initials = "".join(p.strip()[0] for p in self.prompt_raw.split(",") if p.strip())
        filename = f"result_{initials}.jpg"
        save_result(vis, OUTPUT_DIR / filename)
        self.get_logger().info(f"Saved → {OUTPUT_DIR / filename}")


def main(args=None):
    rclpy.init(args=args)
    node = GroundedSAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
