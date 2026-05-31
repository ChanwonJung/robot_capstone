"""GroundedSAMNode — GSAM 추론 결과를 ROS 2 토픽으로 발행.

단일 모델 인스턴스로 EE 뷰만, 또는 EE + Top 두 뷰를 함께 처리합니다.
top_image_topic 파라미터가 비어 있으면 단일 뷰(기존 동작), 값이 설정되면
EE 콜백 trigger 시점에 캐시된 Top 이미지도 같은 모델로 순차 처리합니다.
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

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

        # ── EE view parameters (primary) ──────────────────────────────────────
        self.declare_parameter("model_config", "")
        self.declare_parameter("prompt", "object")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("annotated_topic", "/grounded_sam/annotated_image")
        self.declare_parameter("mask_topic", "/grounded_sam/mask_image")
        self.declare_parameter("detections_topic", "/grounded_sam/detections_json")
        self.declare_parameter("output_subdir", "")
        self.declare_parameter("process_every_n_frames", 30)
        self.declare_parameter("min_process_interval_sec", 1.0)
        # Reject detections whose bbox area exceeds this fraction of the image.
        # Catches "glass cup → entire table" style false positives without any
        # change to the prompt. Set to 1.0 (or higher) to disable.
        self.declare_parameter("max_bbox_area_ratio", 0.4)

        # ── Top view parameters (optional dual-view) ──────────────────────────
        # Setting top_image_topic to a non-empty string enables dual-view mode:
        # the same pipeline instance also runs on the cached Top image at every
        # EE-triggered cycle. Each view publishes to its own topic set.
        self.declare_parameter("top_image_topic", "")
        self.declare_parameter("top_prompt", "")  # empty → reuse EE prompt
        self.declare_parameter("top_annotated_topic", "/top/grounded_sam/annotated_image")
        self.declare_parameter("top_mask_topic", "/top/grounded_sam/mask_image")
        self.declare_parameter("top_detections_topic", "/top/grounded_sam/detections_json")
        # Top depth gating — pre-mask BGR image with a depth range to exclude
        # things like the robot arm from the Top view before g-sam sees it.
        # Leave top_depth_topic empty to skip depth masking.
        self.declare_parameter("top_depth_topic", "")
        self.declare_parameter("top_min_depth", 0.0)
        self.declare_parameter("top_max_depth", 100.0)

        model_config = self.get_parameter("model_config").value
        self.prompt_raw = self.get_parameter("prompt").value
        image_topic = self.get_parameter("image_topic").value
        annotated_topic = self.get_parameter("annotated_topic").value
        mask_topic = self.get_parameter("mask_topic").value
        detections_topic = self.get_parameter("detections_topic").value
        output_subdir = str(self.get_parameter("output_subdir").value).strip().strip("/")
        self._process_every_n_frames = max(1, int(self.get_parameter("process_every_n_frames").value))
        self._min_process_interval_sec = float(self.get_parameter("min_process_interval_sec").value)
        self._max_bbox_area_ratio = float(self.get_parameter("max_bbox_area_ratio").value)
        self._output_dir = OUTPUT_DIR / output_subdir if output_subdir else OUTPUT_DIR

        top_image_topic = str(self.get_parameter("top_image_topic").value).strip()
        self._dual_view = bool(top_image_topic)
        top_prompt = str(self.get_parameter("top_prompt").value).strip()
        self._top_prompt_raw = top_prompt if top_prompt else self.prompt_raw

        top_depth_topic = str(self.get_parameter("top_depth_topic").value).strip()
        self._top_depth_enabled = bool(top_depth_topic)
        self._top_min_depth = float(self.get_parameter("top_min_depth").value)
        self._top_max_depth = float(self.get_parameter("top_max_depth").value)

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

        # ── EE view subs / pubs ───────────────────────────────────────────────
        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self._ee_image_callback,
            qos_profile=10,
        )
        self.pub_annotated = self.create_publisher(Image, annotated_topic, 10)
        self.pub_mask = self.create_publisher(Image, mask_topic, 10)
        self.pub_json = self.create_publisher(String, detections_topic, 10)

        # ── Top view subs / pubs (only if dual-view enabled) ──────────────────
        self._top_latest: Optional[Image] = None
        self._top_latest_depth: Optional[Image] = None
        self.pub_top_annotated = None
        self.pub_top_mask = None
        self.pub_top_json = None
        if self._dual_view:
            top_annotated_topic = self.get_parameter("top_annotated_topic").value
            top_mask_topic = self.get_parameter("top_mask_topic").value
            top_detections_topic = self.get_parameter("top_detections_topic").value
            self.top_subscription = self.create_subscription(
                Image,
                top_image_topic,
                self._top_image_callback,
                qos_profile=10,
            )
            self.pub_top_annotated = self.create_publisher(Image, top_annotated_topic, 10)
            self.pub_top_mask = self.create_publisher(Image, top_mask_topic, 10)
            self.pub_top_json = self.create_publisher(String, top_detections_topic, 10)
            if self._top_depth_enabled:
                self.top_depth_subscription = self.create_subscription(
                    Image,
                    top_depth_topic,
                    self._top_depth_callback,
                    qos_profile=10,
                )

        self._frame_counter = 0
        self._last_process_time = 0.0

        self.create_subscription(String, "/dino_prompt", self._dino_prompt_cb, 10)

        self.get_logger().info(
            f"Ready — EE='{image_topic}' prompt='{self.prompt_raw}'"
            + (f", Top='{top_image_topic}' prompt='{self._top_prompt_raw}'"
               if self._dual_view else ", Top=disabled")
            + (f", Top depth gating=[{self._top_min_depth}, {self._top_max_depth}]m on '{top_depth_topic}'"
               if (self._dual_view and self._top_depth_enabled) else "")
            + f", process_every_n_frames={self._process_every_n_frames}, "
              f"min_process_interval_sec={self._min_process_interval_sec}, "
              f"output_dir='{self._output_dir}'"
        )

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _dino_prompt_cb(self, msg: String) -> None:
        new_prompt = msg.data.strip()
        if not new_prompt:
            self.get_logger().warn("/dino_prompt 빈 메시지 무시")
            return
        self.prompt_raw = new_prompt
        self._top_prompt_raw = new_prompt
        self.get_logger().info(f"Prompt updated: '{new_prompt}'")

    def _top_image_callback(self, msg: Image) -> None:
        # Top images are cached; processing is driven by the EE callback so
        # both views advance together (one model, one cycle).
        self._top_latest = msg

    def _top_depth_callback(self, msg: Image) -> None:
        # Top depth cached for optional depth-range masking of the Top RGB
        # image before it reaches the g-sam pipeline.
        self._top_latest_depth = msg

    def _ee_image_callback(self, msg: Image) -> None:
        self._frame_counter += 1
        if self._frame_counter % self._process_every_n_frames != 0:
            return
        now = time.monotonic()
        if now - self._last_process_time < self._min_process_interval_sec:
            return
        self._last_process_time = now

        # EE view always runs (this is the trigger)
        self._process_view(
            msg, self.prompt_raw, "ee",
            self.pub_annotated, self.pub_mask, self.pub_json,
        )

        # Top view runs only if dual-view enabled and a Top image has arrived
        if self._dual_view and self._top_latest is not None:
            top_image_override = None
            if self._top_depth_enabled and self._top_latest_depth is not None:
                top_bgr = self.bridge.imgmsg_to_cv2(self._top_latest, desired_encoding="bgr8")
                top_image_override = self._apply_top_depth_mask(top_bgr, self._top_latest_depth)
            self._process_view(
                self._top_latest, self._top_prompt_raw, "top",
                self.pub_top_annotated, self.pub_top_mask, self.pub_top_json,
                image_bgr_override=top_image_override,
            )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _apply_top_depth_mask(self, image_bgr: np.ndarray, depth_msg: Image) -> np.ndarray:
        """Mask out Top-view pixels whose depth is outside [top_min_depth, top_max_depth].

        Out-of-range / non-finite depth pixels are filled with the median colour
        of the in-range region (≈ table background) so the masked silhouette
        does not create a shape the detector can latch onto.
        Returns a new BGR array (input not modified).
        """
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
        if depth.shape[:2] != image_bgr.shape[:2]:
            self.get_logger().warn(
                f"[top] depth shape {depth.shape[:2]} != image shape {image_bgr.shape[:2]}; "
                "skipping depth mask this cycle."
            )
            return image_bgr
        in_range = (
            np.isfinite(depth)
            & (depth >= self._top_min_depth)
            & (depth <= self._top_max_depth)
        )
        masked = image_bgr.copy()
        if in_range.any() and (~in_range).any():
            fill = np.median(image_bgr[in_range], axis=0).astype(np.uint8)
            masked[~in_range] = fill
        return masked

    # ── per-view processing (shared by EE and Top) ────────────────────────────

    def _process_view(
        self,
        img_msg: Image,
        prompt_raw: str,
        view_name: str,
        pub_annotated,
        pub_mask,
        pub_json,
        image_bgr_override: Optional[np.ndarray] = None,
    ) -> None:
        if image_bgr_override is not None:
            image_bgr = image_bgr_override
        else:
            image_bgr = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        prompt = self.adapter.adapt(prompt_raw)

        result = self.pipeline.run(image=image_bgr, prompt=prompt)
        runtime_devices = result.get("runtime_devices")
        if runtime_devices is not None:
            self.get_logger().info(
                f"[{view_name}] Runtime devices: "
                f"boxes_device={runtime_devices['boxes_device']}, "
                f"sam_predictor_device={runtime_devices['sam_predictor_device']}"
            )

        det_list = format_detections(result["detections"], result["phrases"])

        # Drop detections whose bbox covers more than max_bbox_area_ratio of
        # the image — e.g. "glass cup → entire table" false positives.
        if self._max_bbox_area_ratio < 1.0 and det_list:
            img_h, img_w = image_bgr.shape[:2]
            img_area = float(img_h * img_w)
            max_area = self._max_bbox_area_ratio * img_area
            kept_idx, dropped = [], []
            for i, det in enumerate(det_list):
                x1, y1, x2, y2 = det["bbox_xyxy"]
                area = max(0.0, (float(x2) - float(x1)) * (float(y2) - float(y1)))
                if area <= max_area:
                    kept_idx.append(i)
                else:
                    dropped.append((det.get("label", "?"), area / img_area))
            if dropped:
                summary = ", ".join(f"{lbl}@{frac:.0%}" for lbl, frac in dropped)
                self.get_logger().info(
                    f"[{view_name}] Dropped {len(dropped)} oversized detection(s): {summary}"
                )
            if kept_idx:
                det_list = [det_list[i] for i in kept_idx]
                if result["masks"] is not None:
                    result["masks"] = result["masks"][kept_idx]
                    if result["mask_scores"] is not None:
                        result["mask_scores"] = result["mask_scores"][kept_idx]
            else:
                det_list = []
                result["masks"] = None
                result["mask_scores"] = None

        self.get_logger().info(f"[{view_name}] Detected {len(det_list)} object(s)")

        # sort detections by prompt keyword order so mask index matches prompt priority
        prompt_keywords = [p.strip().lower() for p in prompt_raw.split(",")]

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
        annotated_msg.header = img_msg.header
        pub_annotated.publish(annotated_msg)

        mask_msg = self.bridge.cv2_to_imgmsg(label_map, encoding="mono8")
        mask_msg.header = img_msg.header
        pub_mask.publish(mask_msg)

        msg_json = String()
        msg_json.data = json.dumps(det_list)
        pub_json.publish(msg_json)

        # 파일 저장: result_{view}_{initials}.jpg
        initials = "".join(p.strip()[0] for p in prompt_raw.split(",") if p.strip())
        filename = f"result_{view_name}_{initials}.jpg" if initials else f"result_{view_name}.jpg"
        save_result(vis, self._output_dir / filename)
        self.get_logger().info(f"[{view_name}] Saved → {self._output_dir / filename}")


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
