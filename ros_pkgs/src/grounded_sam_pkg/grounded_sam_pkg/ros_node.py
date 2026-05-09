"""GroundedSAMNode — GSAM 추론 결과를 ROS 2 토픽으로 발행."""

import json
import time
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

def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "sim").exists() and (parent / "ros_pkgs").exists():
            return parent
    return Path.home() / "robot_capstone"


OUTPUT_DIR = _find_project_root() / "output"


class GroundedSAMNode(Node):
    def __init__(self):
        super().__init__("grounded_sam_node")

        self.declare_parameter("model_config", "")
        self.declare_parameter("prompt", "object")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("annotated_topic", "/grounded_sam/annotated_image")
        self.declare_parameter("mask_topic", "/grounded_sam/mask_image")
        self.declare_parameter("detections_topic", "/grounded_sam/detections_json")
        self.declare_parameter("output_subdir", "")
        self.declare_parameter("process_every_n_frames", 30)
        self.declare_parameter("min_process_interval_sec", 1.0)

        model_config = self.get_parameter("model_config").value
        self.prompt_raw = self.get_parameter("prompt").value
        image_topic = self.get_parameter("image_topic").value
        annotated_topic = self.get_parameter("annotated_topic").value
        mask_topic = self.get_parameter("mask_topic").value
        detections_topic = self.get_parameter("detections_topic").value
        output_subdir = str(self.get_parameter("output_subdir").value).strip().strip("/")
        self._process_every_n_frames = max(1, int(self.get_parameter("process_every_n_frames").value))
        self._min_process_interval_sec = float(self.get_parameter("min_process_interval_sec").value)
        self._output_dir = OUTPUT_DIR / output_subdir if output_subdir else OUTPUT_DIR

        if not model_config:
            raise ValueError("Parameter 'model_config' must be set to the path of model_paths.yaml")

        self.bridge = CvBridge()
        self.adapter = PromptAdapter()

        self.get_logger().info("Loading models...")
        self.pipeline = GroundedSAMPipeline(model_config)
        device_info = self.pipeline.describe_devices()
        self.get_logger().info(
            "Device summary: "
            f"torch_cuda_available={device_info['torch_cuda_available']}, "
            f"gdino_config_device={device_info['gdino_config_device']}, "
            f"gdino_model_device={device_info['gdino_model_device']}, "
            f"sam_config_device={device_info['sam_config_device']}, "
            f"sam_model_device={device_info['sam_model_device']}"
        )

        # subscriber — use default VOLATILE QoS to match Gazebo bridge publisher
        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self._image_callback,
            qos_profile=10,
        )

        # publishers
        self.pub_annotated = self.create_publisher(Image, annotated_topic, 10)
        self.pub_mask = self.create_publisher(Image, mask_topic, 10)
        self.pub_json = self.create_publisher(String, detections_topic, 10)
        self._frame_counter = 0
        self._last_process_time = 0.0

        self.get_logger().info(
            f"Ready — subscribed to '{image_topic}', prompt='{self.prompt_raw}', "
            f"process_every_n_frames={self._process_every_n_frames}, "
            f"min_process_interval_sec={self._min_process_interval_sec}, "
            f"output_dir='{self._output_dir}'"
        )

    def _image_callback(self, msg: Image) -> None:
        self._frame_counter += 1
        if self._frame_counter % self._process_every_n_frames != 0:
            return
        now = time.monotonic()
        if now - self._last_process_time < self._min_process_interval_sec:
            return
        self._last_process_time = now

        image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        prompt = self.adapter.adapt(self.prompt_raw)

        result = self.pipeline.run(image=image_bgr, prompt=prompt)
        runtime_devices = result.get("runtime_devices")
        if runtime_devices is not None:
            self.get_logger().info(
                "Runtime devices: "
                f"boxes_device={runtime_devices['boxes_device']}, "
                f"sam_predictor_device={runtime_devices['sam_predictor_device']}"
            )

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
        save_result(vis, self._output_dir / filename)
        self.get_logger().info(f"Saved → {self._output_dir / filename}")


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
