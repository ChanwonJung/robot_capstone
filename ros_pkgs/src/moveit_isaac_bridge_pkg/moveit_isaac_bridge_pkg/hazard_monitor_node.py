"""Hazard monitor: cancel the active MoveGroup motion when a Fast Brain hazard appears.

Subscribes to the YOLO hazard detection JSON topics (top-view + eye-in-hand).
When a detection whose class is in the trigger set crosses the confidence
threshold, this node calls the MoveGroup action's cancel service to stop the
arm. Cancelling the ``/move_action`` goal propagates down to the
FollowJointTrajectory goal on the Isaac bridge, which holds the last commanded
joint positions — i.e. the arm stops in place.

This is the interim "simple cancel" reaction. The full pipeline will instead
inject the hazard as a dynamic collision object and let hybrid planning replan
around it.
"""
from __future__ import annotations

import json
from typing import Optional

import rclpy
from action_msgs.srv import CancelGoal
from rclpy.node import Node
from std_msgs.msg import String


class HazardMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__("hazard_monitor_node")

        self.declare_parameter(
            "detection_topics",
            ["/yolo_hazard/top/detections_json", "/yolo_hazard/ee/detections_json"],
        )
        # Fast Brain 3-class model: 0=arm, 1=bottle, 2=box. All are hazards here.
        self.declare_parameter("trigger_class_ids", [0, 1, 2])
        # A bit above the detector conf_threshold (0.35) to avoid stopping on
        # low-confidence flickers.
        self.declare_parameter("conf_threshold", 0.5)
        self.declare_parameter("move_action_name", "/move_action")
        # Detection runs ~20+ fps; without a cooldown we would spam cancel every
        # frame while a hazard is in view. Cancel-all is idempotent, so one call
        # per cooldown window is enough to keep the arm stopped.
        self.declare_parameter("cancel_cooldown_sec", 1.0)

        self._topics = list(self.get_parameter("detection_topics").value)
        self._trigger = set(int(c) for c in self.get_parameter("trigger_class_ids").value)
        self._conf = float(self.get_parameter("conf_threshold").value)
        self._action_name = str(self.get_parameter("move_action_name").value)
        self._cooldown = float(self.get_parameter("cancel_cooldown_sec").value)

        self._cancel_srv = f"{self._action_name}/_action/cancel_goal"
        self._cancel_client = self.create_client(CancelGoal, self._cancel_srv)

        self._last_cancel_time: Optional[rclpy.time.Time] = None

        for topic in self._topics:
            self.create_subscription(String, topic, self._make_cb(topic), 10)

        self.get_logger().info(
            f"HazardMonitor ready: topics={self._topics}, "
            f"trigger_classes={sorted(self._trigger)}, conf>={self._conf}, "
            f"cancel_service={self._cancel_srv}, cooldown={self._cooldown}s"
        )

    def _make_cb(self, topic: str):
        def _cb(msg: String) -> None:
            self._on_detections(topic, msg)

        return _cb

    def _on_detections(self, topic: str, msg: String) -> None:
        try:
            detections = json.loads(msg.data)["hazard_detections"]["detections"]
        except (ValueError, KeyError, TypeError):
            return

        for det in detections:
            try:
                cls_id = int(det["class_id"])
                conf = float(det["confidence"])
            except (KeyError, ValueError, TypeError):
                continue
            if cls_id in self._trigger and conf >= self._conf:
                self._maybe_cancel(topic, det.get("class_name", str(cls_id)), conf)
                return

    def _maybe_cancel(self, topic: str, class_name: str, conf: float) -> None:
        now = self.get_clock().now()
        if self._last_cancel_time is not None:
            elapsed = (now - self._last_cancel_time).nanoseconds * 1e-9
            if elapsed < self._cooldown:
                return
        self._last_cancel_time = now

        if not self._cancel_client.service_is_ready():
            self.get_logger().warn(
                f"Hazard '{class_name}' ({conf:.2f}) on {topic}, but cancel "
                f"service '{self._cancel_srv}' not available yet"
            )
            return

        self.get_logger().warn(
            f"HAZARD '{class_name}' ({conf:.2f}) on {topic} → cancelling MoveGroup goals"
        )
        # Empty goal_info (zero uuid + zero stamp) tells the action server to
        # cancel ALL active goals.
        future = self._cancel_client.call_async(CancelGoal.Request())
        future.add_done_callback(self._cancel_done_cb)

    def _cancel_done_cb(self, future) -> None:
        try:
            resp = future.result()
        except Exception as exc:
            self.get_logger().error(f"Cancel call failed: {exc}")
            return

        n = len(resp.goals_canceling)
        if resp.return_code == CancelGoal.Response.ERROR_NONE and n > 0:
            self.get_logger().info(f"Cancel accepted; {n} goal(s) cancelling")
        else:
            self.get_logger().info(
                f"No active goal to cancel (return_code={resp.return_code})"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HazardMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
