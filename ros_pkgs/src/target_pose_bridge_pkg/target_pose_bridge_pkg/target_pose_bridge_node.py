from __future__ import annotations

import json
import math
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String


def _quaternion_from_euler(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


class TargetPoseBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("target_pose_bridge_node")

        self.declare_parameter("input_topic", "/world_map_result")
        self.declare_parameter("pre_grasp_topic", "/pre_grasp_target_pose")
        self.declare_parameter("grasp_topic", "/grasp_target_pose")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("min_point_count", 100)
        self.declare_parameter("grasp_z_offset", 0.03)
        self.declare_parameter("pre_grasp_z_offset", 0.12)
        self.declare_parameter("min_z", 0.0)
        self.declare_parameter("max_z", 2.0)
        self.declare_parameter("roll", math.pi)
        self.declare_parameter("pitch", 0.0)
        self.declare_parameter("yaw", 0.0)
        self.declare_parameter("republish_position_threshold", 0.005)

        input_topic = str(self.get_parameter("input_topic").value)
        pre_grasp_topic = str(self.get_parameter("pre_grasp_topic").value)
        grasp_topic = str(self.get_parameter("grasp_topic").value)

        self._world_frame = str(self.get_parameter("world_frame").value)
        self._min_point_count = int(self.get_parameter("min_point_count").value)
        self._grasp_z_offset = float(self.get_parameter("grasp_z_offset").value)
        self._pre_grasp_z_offset = float(self.get_parameter("pre_grasp_z_offset").value)
        self._min_z = float(self.get_parameter("min_z").value)
        self._max_z = float(self.get_parameter("max_z").value)
        self._roll = float(self.get_parameter("roll").value)
        self._pitch = float(self.get_parameter("pitch").value)
        self._yaw = float(self.get_parameter("yaw").value)
        self._republish_position_threshold = float(
            self.get_parameter("republish_position_threshold").value
        )

        self._quat = _quaternion_from_euler(self._roll, self._pitch, self._yaw)
        self._last_published_centroid: Optional[Tuple[float, float, float]] = None

        self._pre_grasp_pub = self.create_publisher(PoseStamped, pre_grasp_topic, 10)
        self._grasp_pub = self.create_publisher(PoseStamped, grasp_topic, 10)
        self.create_subscription(String, input_topic, self._result_cb, 10)

        self.get_logger().info(
            f"Target pose bridge ready: input={input_topic}, pre_grasp={pre_grasp_topic}, "
            f"grasp={grasp_topic}, frame={self._world_frame}"
        )

    def _result_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Failed to parse target JSON: {exc}")
            return

        target = payload.get("target")
        if not isinstance(target, dict):
            self.get_logger().warn("No 'target' entry present in world_map_result")
            return

        label = str(target.get("label", "target"))
        point_count = int(target.get("point_count", 0))
        if point_count < self._min_point_count:
            self.get_logger().warn(
                f"Ignoring target '{label}' because point_count={point_count} < {self._min_point_count}"
            )
            return

        centroid = target.get("centroid")
        if not self._is_valid_centroid(centroid):
            self.get_logger().warn(f"Ignoring target '{label}' due to invalid centroid: {centroid}")
            return

        centroid_tuple = (float(centroid[0]), float(centroid[1]), float(centroid[2]))
        if not (self._min_z <= centroid_tuple[2] <= self._max_z):
            self.get_logger().warn(
                f"Ignoring target '{label}' because z={centroid_tuple[2]:.4f} is outside "
                f"[{self._min_z}, {self._max_z}]"
            )
            return

        if self._last_published_centroid and self._distance(
            centroid_tuple, self._last_published_centroid
        ) < self._republish_position_threshold:
            return

        grasp_pose = self._make_pose(
            x=centroid_tuple[0],
            y=centroid_tuple[1],
            z=centroid_tuple[2] + self._grasp_z_offset,
        )
        pre_grasp_pose = self._make_pose(
            x=centroid_tuple[0],
            y=centroid_tuple[1],
            z=centroid_tuple[2] + self._pre_grasp_z_offset,
        )

        self._grasp_pub.publish(grasp_pose)
        self._pre_grasp_pub.publish(pre_grasp_pose)
        self._last_published_centroid = centroid_tuple

        self.get_logger().info(
            f"Published target poses for '{label}': centroid={centroid_tuple}, "
            f"grasp_z={grasp_pose.pose.position.z:.4f}, "
            f"pre_grasp_z={pre_grasp_pose.pose.position.z:.4f}"
        )

    def _make_pose(self, *, x: float, y: float, z: float) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._world_frame
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.x = self._quat[0]
        msg.pose.orientation.y = self._quat[1]
        msg.pose.orientation.z = self._quat[2]
        msg.pose.orientation.w = self._quat[3]
        return msg

    @staticmethod
    def _is_valid_centroid(value) -> bool:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return False
        return all(isinstance(component, (int, float)) for component in value)

    @staticmethod
    def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return math.sqrt(sum((lhs - rhs) ** 2 for lhs, rhs in zip(a, b)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TargetPoseBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
