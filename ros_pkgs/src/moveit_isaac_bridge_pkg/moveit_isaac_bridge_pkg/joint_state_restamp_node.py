"""Re-stamp Isaac joint states with ROS wall time.

Isaac stamps /joint_states with non-wall time (sim time / app uptime — small
second values) and the scene publishes no /clock, so the ROS stack runs on wall
time (use_sim_time:=false). MoveIt's state monitor then rejects the joint state
as ancient ("recent timestamp within 1.0s"), which makes the hybrid global
planner fail to configure its planning scene monitor.

This node republishes the joint state with header.stamp = wall now(), so every
wall-time consumer (robot_state_publisher TF, the hybrid planners) sees a fresh
timestamp. Whatever Isaac stamps becomes irrelevant.
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStateRestampNode(Node):
    def __init__(self) -> None:
        super().__init__("joint_state_restamp_node")
        self.declare_parameter("input_topic", "/joint_states_isaac")
        self.declare_parameter("output_topic", "/joint_states")
        self._in = str(self.get_parameter("input_topic").value)
        self._out = str(self.get_parameter("output_topic").value)

        self._pub = self.create_publisher(JointState, self._out, 10)
        self.create_subscription(JointState, self._in, self._cb, 10)
        self.get_logger().info(
            f"JointStateRestamp: {self._in} -> {self._out} (re-stamped with wall now())"
        )

    def _cb(self, msg: JointState) -> None:
        msg.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JointStateRestampNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
