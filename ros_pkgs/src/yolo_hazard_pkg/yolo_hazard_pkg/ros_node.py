"""Fast Brain hazard detection node — YOLO26-seg over a single camera stream."""

import json
import time
from pathlib import Path

import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .postprocess import results_to_hazard_dict
from .yolo_runner import YOLORunner


def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "sim").exists() and (parent / "ros_pkgs").exists():
            return parent
    return Path.home() / "robot_capstone"


PROJECT_ROOT = _find_project_root()


class YoloHazardNode(Node):
    def __init__(self):
        super().__init__("yolo_hazard_node")

        self.declare_parameter("model_config", "")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("detections_topic", "/yolo_hazard/detections_json")
        self.declare_parameter("annotated_topic", "/yolo_hazard/annotated_image")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("conf_threshold", 0.35)
        self.declare_parameter("iou_threshold", 0.5)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("half", True)
        self.declare_parameter("publish_annotated", False)
        self.declare_parameter("filter_by_class", False)
        self.declare_parameter("class_allowlist", [0, 39])
        self.declare_parameter("fps_log_interval_sec", 5.0)

        model_config = str(self.get_parameter("model_config").value)
        image_topic = str(self.get_parameter("image_topic").value)
        detections_topic = str(self.get_parameter("detections_topic").value)
        annotated_topic = str(self.get_parameter("annotated_topic").value)
        device = str(self.get_parameter("device").value)
        conf = float(self.get_parameter("conf_threshold").value)
        iou = float(self.get_parameter("iou_threshold").value)
        imgsz = int(self.get_parameter("imgsz").value)
        half = bool(self.get_parameter("half").value)
        self._publish_annotated = bool(self.get_parameter("publish_annotated").value)
        filter_by_class = bool(self.get_parameter("filter_by_class").value)
        self._class_allowlist = (
            list(self.get_parameter("class_allowlist").value)
            if filter_by_class
            else None
        )
        self._fps_log_interval = float(self.get_parameter("fps_log_interval_sec").value)

        weights = self._resolve_weights(model_config)
        self.get_logger().info(f"Loading YOLO weights: {weights}")

        self.runner = YOLORunner(
            weights=weights,
            device=device,
            imgsz=imgsz,
            conf_threshold=conf,
            iou_threshold=iou,
            half=half,
        )

        self.bridge = CvBridge()

        self.sub_image = self.create_subscription(
            Image, image_topic, self._image_callback, qos_profile=10
        )
        self.pub_json = self.create_publisher(String, detections_topic, 10)
        self.pub_annotated = (
            self.create_publisher(Image, annotated_topic, 10)
            if self._publish_annotated
            else None
        )

        self._frame_count = 0
        self._fps_window_start = time.monotonic()

        self.get_logger().info(
            f"Ready — subscribed to '{image_topic}', publishing detections on '{detections_topic}', "
            f"device={device}, imgsz={imgsz}, conf={conf}, half={half}, "
            f"class_filter={'ON ' + str(self._class_allowlist) if self._class_allowlist is not None else 'OFF (all classes)'}"
        )

    @staticmethod
    def _resolve_weights(model_config: str) -> str:
        if not model_config:
            return "yolo26s-seg.pt"
        cfg_path = Path(model_config)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"model_config not found: {cfg_path}")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        weights = cfg.get("weights", "yolo26s-seg.pt")
        weights_path = Path(weights)
        if not weights_path.is_absolute():
            candidate = PROJECT_ROOT / weights
            if candidate.exists():
                return str(candidate)
        return weights

    def _image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"cv_bridge convert failed: {exc}")
            return

        result = self.runner.infer(frame)

        timestamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        hazard = results_to_hazard_dict(
            result,
            timestamp_ns=timestamp_ns,
            frame_id=msg.header.frame_id,
            image_width=frame.shape[1],
            image_height=frame.shape[0],
            class_allowlist=self._class_allowlist,
        )

        out_msg = String()
        out_msg.data = json.dumps(hazard)
        self.pub_json.publish(out_msg)

        if self.pub_annotated is not None:
            annotated = result.plot()
            ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            ann_msg.header = msg.header
            self.pub_annotated.publish(ann_msg)

        self._frame_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_window_start
        if elapsed >= self._fps_log_interval:
            fps = self._frame_count / elapsed
            self.get_logger().info(
                f"FPS={fps:.1f} ({self._frame_count} frames / {elapsed:.1f}s)"
            )
            self._frame_count = 0
            self._fps_window_start = now


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloHazardNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
