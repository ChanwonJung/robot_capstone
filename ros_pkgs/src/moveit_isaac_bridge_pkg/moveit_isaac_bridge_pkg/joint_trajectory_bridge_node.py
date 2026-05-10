from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, List, Optional

import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


@dataclass
class JointStateSnapshot:
    positions: Dict[str, float]
    stamp: rclpy.time.Time


def _duration_msg_to_duration(msg: DurationMsg) -> Duration:
    return Duration(seconds=msg.sec, nanoseconds=msg.nanosec)


class JointTrajectoryBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("joint_trajectory_bridge_node")

        self.declare_parameter("command_topic", "/joint_command")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("action_name", "/panda_arm_controller/follow_joint_trajectory")
        self.declare_parameter(
            "controlled_joints",
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
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("goal_tolerance", 0.03)
        self.declare_parameter("goal_timeout_sec", 5.0)
        self.declare_parameter("allow_partial_joints_goal", False)

        self._command_topic = str(self.get_parameter("command_topic").value)
        self._joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self._action_name = str(self.get_parameter("action_name").value)
        self._controlled_joints = list(self.get_parameter("controlled_joints").value)
        self._publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self._goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self._goal_timeout_sec = float(self.get_parameter("goal_timeout_sec").value)
        self._allow_partial = bool(self.get_parameter("allow_partial_joints_goal").value)

        qos = QoSProfile(depth=10)
        self._joint_state_sub = self.create_subscription(
            JointState,
            self._joint_state_topic,
            self._joint_state_cb,
            qos,
        )
        self._joint_command_pub = self.create_publisher(JointState, self._command_topic, qos)
        self._latest_joint_state: Optional[JointStateSnapshot] = None

        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            self._action_name,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            execute_callback=self._execute_callback,
        )

        self.get_logger().info(
            f"Joint trajectory bridge ready: action={self._action_name}, "
            f"command_topic={self._command_topic}, joint_state_topic={self._joint_state_topic}"
        )

    def destroy_node(self) -> bool:
        self._action_server.destroy()
        return super().destroy_node()

    def _joint_state_cb(self, msg: JointState) -> None:
        positions = {}
        for index, name in enumerate(msg.name):
            if index < len(msg.position):
                positions[name] = msg.position[index]
        self._latest_joint_state = JointStateSnapshot(
            positions=positions,
            stamp=rclpy.time.Time.from_msg(msg.header.stamp) if msg.header.stamp.sec or msg.header.stamp.nanosec else self.get_clock().now(),
        )

    def _goal_callback(self, goal_request: FollowJointTrajectory.Goal) -> int:
        trajectory = goal_request.trajectory
        if not trajectory.joint_names:
            self.get_logger().warn("Rejected trajectory goal with no joint names")
            return GoalResponse.REJECT
        if not trajectory.points:
            self.get_logger().warn("Rejected trajectory goal with no trajectory points")
            return GoalResponse.REJECT

        missing = [name for name in trajectory.joint_names if name not in self._controlled_joints]
        if missing and not self._allow_partial:
            self.get_logger().warn(f"Rejected trajectory goal with unsupported joints: {missing}")
            return GoalResponse.REJECT

        return GoalResponse.ACCEPT

    def _cancel_callback(self, _goal_handle) -> int:
        self.get_logger().info("Received cancel request for active trajectory goal")
        return CancelResponse.ACCEPT

    def _execute_callback(self, goal_handle) -> FollowJointTrajectory.Result:
        trajectory = goal_handle.request.trajectory
        joint_names = list(trajectory.joint_names)

        if self._latest_joint_state is None:
            goal_handle.abort()
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = "No joint state received yet from Isaac Sim"
            return result

        if any(len(point.positions) != len(joint_names) for point in trajectory.points):
            goal_handle.abort()
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = "Trajectory point position dimensions do not match joint_names"
            return result

        self.get_logger().info(
            f"Executing trajectory with {len(trajectory.points)} point(s) for joints {joint_names}"
        )

        publish_period = 1.0 / max(self._publish_rate_hz, 1.0)
        start_time = self.get_clock().now()
        previous_target = self._build_target_map_from_latest()

        for point in trajectory.points:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = FollowJointTrajectory.Result()
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                result.error_string = "Trajectory goal canceled"
                return result

            target_map = previous_target.copy()
            for name, position in zip(joint_names, point.positions):
                if name in self._controlled_joints:
                    target_map[name] = position

            target_time = start_time + _duration_msg_to_duration(point.time_from_start)
            while self.get_clock().now() < target_time:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result = FollowJointTrajectory.Result()
                    result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                    result.error_string = "Trajectory goal canceled"
                    return result
                self._publish_joint_command(target_map)
                self._publish_feedback(goal_handle, joint_names, point)
                time.sleep(publish_period)

            self._publish_joint_command(target_map)
            self._publish_feedback(goal_handle, joint_names, point)
            previous_target = target_map

        reached = self._wait_until_goal_reached(goal_handle, previous_target)
        result = FollowJointTrajectory.Result()
        if reached:
            goal_handle.succeed()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            result.error_string = "Trajectory execution completed"
        else:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
            result.error_string = "Timed out waiting for Isaac Sim joints to reach the commanded goal"
        return result

    def _build_target_map_from_latest(self) -> Dict[str, float]:
        target_map = {name: 0.0 for name in self._controlled_joints}
        if self._latest_joint_state is not None:
            for name in self._controlled_joints:
                if name in self._latest_joint_state.positions:
                    target_map[name] = self._latest_joint_state.positions[name]
        return target_map

    def _publish_joint_command(self, target_map: Dict[str, float]) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self._controlled_joints)
        msg.position = [float(target_map[name]) for name in self._controlled_joints]
        self._joint_command_pub.publish(msg)

    def _publish_feedback(
        self,
        goal_handle,
        joint_names: List[str],
        desired_point: JointTrajectoryPoint,
    ) -> None:
        feedback = FollowJointTrajectory.Feedback()
        feedback.joint_names = joint_names
        feedback.desired = desired_point
        feedback.actual = JointTrajectoryPoint()
        feedback.error = JointTrajectoryPoint()

        actual_positions = []
        error_positions = []
        for index, name in enumerate(joint_names):
            desired_position = desired_point.positions[index]
            actual_position = desired_position
            if self._latest_joint_state and name in self._latest_joint_state.positions:
                actual_position = self._latest_joint_state.positions[name]
            actual_positions.append(actual_position)
            error_positions.append(desired_position - actual_position)

        feedback.actual.positions = actual_positions
        feedback.error.positions = error_positions
        goal_handle.publish_feedback(feedback)

    def _wait_until_goal_reached(self, goal_handle, target_map: Dict[str, float]) -> bool:
        deadline = self.get_clock().now() + Duration(seconds=self._goal_timeout_sec)
        while self.get_clock().now() < deadline:
            if goal_handle.is_cancel_requested:
                return False
            self._publish_joint_command(target_map)
            if self._is_goal_within_tolerance(target_map):
                return True
            time.sleep(1.0 / max(self._publish_rate_hz, 1.0))
        return self._is_goal_within_tolerance(target_map)

    def _is_goal_within_tolerance(self, target_map: Dict[str, float]) -> bool:
        if self._latest_joint_state is None:
            return False
        for name in self._controlled_joints:
            actual = self._latest_joint_state.positions.get(name)
            if actual is None:
                return False
            if abs(actual - target_map[name]) > self._goal_tolerance:
                return False
        return True

def main(args=None) -> None:
    rclpy.init(args=args)
    node = JointTrajectoryBridgeNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
