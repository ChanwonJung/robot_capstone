#!/usr/bin/env python3
"""Capture camera frames into PNGs during /hazard/launch_bottle windows for YOLO fine-tune data."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Empty


class ImageCaptureHelper(Node):
    def __init__(self, image_topic, trigger_topic, out_dir, save_every_n, window_sec):
        super().__init__("image_capture_helper")
        self._out_dir = Path(out_dir).expanduser().resolve()
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._save_every_n = max(1, int(save_every_n))
        self._window_sec = float(window_sec)

        self._bridge = CvBridge()
        self._capture_until = 0.0
        self._frame_counter = 0
        self._saved_counter = len(list(self._out_dir.glob("frame_*.png")))
        self._trigger_idx = 0
        self._was_in_window = False

        self.create_subscription(Image, image_topic, self._image_cb, 10)
        self.create_subscription(Empty, trigger_topic, self._trigger_cb, 10)

        self.get_logger().info(
            f"ready: image={image_topic}, trigger={trigger_topic}, "
            f"out={self._out_dir}, save_every_n={self._save_every_n}, "
            f"window={self._window_sec}s, resuming at saved_idx={self._saved_counter}"
        )

    def _trigger_cb(self, _msg) -> None:
        self._trigger_idx += 1
        self._capture_until = time.monotonic() + self._window_sec
        self._frame_counter = 0
        self.get_logger().info(
            f"trigger #{self._trigger_idx}: capture window open {self._window_sec:.1f}s"
        )

    def _image_cb(self, msg: Image) -> None:
        in_window = time.monotonic() < self._capture_until
        if not in_window:
            if self._was_in_window:
                self.get_logger().info(
                    f"window #{self._trigger_idx} closed — total saved={self._saved_counter}"
                )
            self._was_in_window = False
            return

        self._was_in_window = True
        if self._frame_counter % self._save_every_n == 0:
            try:
                frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception as exc:
                self.get_logger().warn(f"cv_bridge convert failed: {exc}")
                self._frame_counter += 1
                return
            path = self._out_dir / f"frame_{self._saved_counter:05d}.png"
            if cv2.imwrite(str(path), frame):
                self._saved_counter += 1
            else:
                self.get_logger().warn(f"imwrite failed: {path}")
        self._frame_counter += 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-topic", default="/camera/image_raw")
    parser.add_argument("--trigger-topic", default="/hazard/launch_bottle")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--save-every-n", type=int, default=10)
    parser.add_argument("--window-sec", type=float, default=12.0)
    args, ros_argv = parser.parse_known_args(argv)

    rclpy.init(args=ros_argv)
    node = ImageCaptureHelper(
        args.image_topic, args.trigger_topic, args.out_dir,
        args.save_every_n, args.window_sec,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"shutdown — final saved_idx={node._saved_counter}")
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
