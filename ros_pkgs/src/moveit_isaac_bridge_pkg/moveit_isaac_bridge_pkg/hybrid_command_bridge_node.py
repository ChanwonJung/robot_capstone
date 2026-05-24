"""Bridge the hybrid LocalPlanner output to Isaac Sim joint commands.

The MoveIt hybrid LocalPlanner (ForwardTrajectory) streams joint *positions* as a
std_msgs/Float64MultiArray, ordered by the planning group's joints. Isaac Sim is
driven by a sensor_msgs/JointState on /joint_command (same interface the
FollowJointTrajectory bridge uses). This node converts one to the other.

We use this instead of ros2_control because the rest of the stack commands Isaac
directly via /joint_command. The hybrid execution path therefore bypasses the
FollowJointTrajectory bridge / target_pose_executor entirely.
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class HybridCommandBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("hybrid_command_bridge_node")

        self.declare_parameter("input_topic", "/hybrid/joint_position_command")
        self.declare_parameter("command_topic", "/joint_command")
        # Must match the panda_arm group's active joint order, which is also the
        # order the LocalPlanner packs positions into the Float64MultiArray.
        self.declare_parameter(
            "joint_names",
            [
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
            ],
        )

        self._input_topic = str(self.get_parameter("input_topic").value)
        self._command_topic = str(self.get_parameter("command_topic").value)
        self._joints = list(self.get_parameter("joint_names").value)

        self._warned_len = False
        self._pub = self.create_publisher(JointState, self._command_topic, 10)
        self.create_subscription(Float64MultiArray, self._input_topic, self._cb, 10)

        self.get_logger().info(
            f"HybridCommandBridge ready: {self._input_topic} (Float64MultiArray) "
            f"-> {self._command_topic} (JointState), joints={self._joints}"
        )

    def _cb(self, msg: Float64MultiArray) -> None:
        positions = list(msg.data)
        if len(positions) != len(self._joints):
            if not self._warned_len:
                self.get_logger().warn(
                    f"Command length {len(positions)} != {len(self._joints)} joints; "
                    f"ignoring. (check group joint order / joint_names param)"
                )
                self._warned_len = True
            return

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(self._joints)
        out.position = positions
        self._pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HybridCommandBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
