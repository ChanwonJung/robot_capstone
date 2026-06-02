#!/usr/bin/env python3
"""hazard_level_translator_node.py

Translates raw YOLO JSON detection streams into a single /bt/hazard_level Int8.

Level rules:
  3  (HALT)  — class_id in halt_class_ids AND confidence >= halt_conf_threshold
  1  (SLOW)  — any detection with confidence >= slow_conf_threshold
  0  (CLEAR) — no qualifying detections, or decay_sec elapsed since last detection

Level 3 wins over level 1. Level decays to 0 after decay_sec with no input.
Both /yolo_hazard/top/detections_json and /yolo_hazard/ee/detections_json
are consumed; the highest level from either is published.
"""
from __future__ import annotations

import json
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8, String


class HazardLevelTranslatorNode(Node):
    def __init__(self) -> None:
        super().__init__("hazard_level_translator_node")

        self.declare_parameter("halt_class_ids",      [0])
        self.declare_parameter("halt_conf_threshold",  0.6)
        self.declare_parameter("slow_conf_threshold",  0.5)
        self.declare_parameter("decay_sec",            0.3)
        self.declare_parameter("publish_rate_hz",     30.0)
        self.declare_parameter("detection_topics", [
            "/yolo_hazard/top/detections_json",
            "/yolo_hazard/ee/detections_json",
        ])

        self._halt_ids   = set(int(x) for x in self.get_parameter("halt_class_ids").value)
        self._halt_conf  = float(self.get_parameter("halt_conf_threshold").value)
        self._slow_conf  = float(self.get_parameter("slow_conf_threshold").value)
        self._decay_sec  = float(self.get_parameter("decay_sec").value)

        self._pub = self.create_publisher(Int8, "/bt/hazard_level", 10)

        # Separate level + timestamp per topic so one slow camera doesn't
        # mask a detection on the other.
        topics: list[str] = list(self.get_parameter("detection_topics").value)
        self._state: dict[str, dict] = {
            t: {"level": 0, "last_seen": 0.0} for t in topics
        }

        for topic in topics:
            self.create_subscription(String, topic, self._make_cb(topic), 10)

        rate = float(self.get_parameter("publish_rate_hz").value)
        self.create_timer(1.0 / rate, self._publish)

        self.get_logger().info(
            f"HazardLevelTranslator ready: halt_ids={self._halt_ids} "
            f"halt_conf>={self._halt_conf}  slow_conf>={self._slow_conf}  "
            f"decay={self._decay_sec}s  topics={topics}"
        )

    # ── callbacks ──────────────────────────────────────────────────────────

    def _make_cb(self, topic: str):
        def _cb(msg: String) -> None:
            level = self._classify(msg.data)
            state = self._state[topic]
            if level > 0:
                state["level"]     = level
                state["last_seen"] = time.monotonic()
            # If level == 0 the state stays as-is; decay timer handles clearing.
        return _cb

    def _classify(self, raw: str) -> int:
        """Return the highest hazard level found in the JSON payload."""
        try:
            data = json.loads(raw)
            detections = data["hazard_detections"]["detections"]
        except (KeyError, ValueError, TypeError):
            return 0

        level = 0
        for det in detections:
            try:
                cls_id = int(det["class_id"])
                conf   = float(det["confidence"])
            except (KeyError, ValueError, TypeError):
                continue

            if cls_id in self._halt_ids and conf >= self._halt_conf:
                return 3  # can't get higher — return immediately
            if conf >= self._slow_conf:
                level = max(level, 1)

        return level

    def _publish(self) -> None:
        now = time.monotonic()
        combined = 0
        for state in self._state.values():
            if state["level"] > 0:
                age = now - state["last_seen"]
                if age > self._decay_sec:
                    state["level"] = 0  # decay
                else:
                    combined = max(combined, state["level"])

        msg = Int8()
        msg.data = combined
        self._pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HazardLevelTranslatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
