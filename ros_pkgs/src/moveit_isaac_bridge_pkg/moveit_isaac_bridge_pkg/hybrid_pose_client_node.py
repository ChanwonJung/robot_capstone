"""Send a target pose to the hybrid planning architecture.

Mirrors target_pose_executor_node, but instead of the MoveGroup action it builds a
HybridPlanner goal (a one-item MotionSequenceRequest) and sends it to the hybrid
planning manager's /run_hybrid_planning action. The hybrid manager then runs the
global planner once and executes via the local planner — which stops/resumes
around hazard collision objects injected on /collision_object.

Use this in the hybrid (stop+resume) pipeline instead of target_pose_executor.
"""
from __future__ import annotations

import threading
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import HybridPlanner
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    MotionPlanRequest,
    MotionSequenceItem,
    MotionSequenceRequest,
    OrientationConstraint,
    PositionConstraint,
    WorkspaceParameters,
)
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Empty


class HybridPoseClientNode(Node):
    def __init__(self) -> None:
        super().__init__("hybrid_pose_client_node")

        self.declare_parameter("target_pose_topic", "/grasp_target_pose")
        self.declare_parameter("action_name", "/run_hybrid_planning")
        self.declare_parameter("planning_group", "panda_arm")
        self.declare_parameter("end_effector_link", "panda_link8")
        self.declare_parameter("planning_frame", "panda_link0")
        self.declare_parameter("planner_id", "RRTConnectkConfigDefault")
        self.declare_parameter("pipeline_id", "ompl")
        self.declare_parameter("num_planning_attempts", 10)
        self.declare_parameter("allowed_planning_time", 5.0)
        self.declare_parameter("position_tolerance", 0.01)
        self.declare_parameter("orientation_tolerance", 0.05)
        # Slow arm so the hazard bottle crosses the path during the motion.
        self.declare_parameter("max_velocity_scaling_factor", 0.015)
        self.declare_parameter("max_acceleration_scaling_factor", 0.05)
        # Stop the EE above the target instead of driving into it (the glass is
        # the grasp target, not a MoveIt collision object, so nothing stops the
        # arm from slamming into it — which physically blocks the robot and makes
        # the local planner "stuck" at the end). Raise/lower to tune approach height.
        self.declare_parameter("link_target_z_offset", 0.20)
        self.declare_parameter("ignore_while_busy", True)
        # One motion per command: target_pose_bridge republishes /grasp_target_pose
        # repeatedly, and re-sending preempts/cancels the in-flight hybrid goal
        # (manager thrashes -> "Unknown event" -> crash). Send once; publish
        # std_msgs/Empty on the reset topic to re-arm for another run.
        self.declare_parameter("one_shot", True)
        self.declare_parameter("reset_topic", "/hybrid_pose_client/reset")

        self._target_topic = str(self.get_parameter("target_pose_topic").value)
        self._action_name = str(self.get_parameter("action_name").value)
        self._group = str(self.get_parameter("planning_group").value)
        self._ee_link = str(self.get_parameter("end_effector_link").value)
        self._planning_frame = str(self.get_parameter("planning_frame").value)
        self._planner_id = str(self.get_parameter("planner_id").value)
        self._pipeline_id = str(self.get_parameter("pipeline_id").value)
        self._num_attempts = int(self.get_parameter("num_planning_attempts").value)
        self._planning_time = float(self.get_parameter("allowed_planning_time").value)
        self._pos_tol = float(self.get_parameter("position_tolerance").value)
        self._ori_tol = float(self.get_parameter("orientation_tolerance").value)
        self._vel_scale = float(self.get_parameter("max_velocity_scaling_factor").value)
        self._acc_scale = float(self.get_parameter("max_acceleration_scaling_factor").value)
        self._link_z_offset = float(self.get_parameter("link_target_z_offset").value)
        self._ignore_while_busy = bool(self.get_parameter("ignore_while_busy").value)
        self._one_shot = bool(self.get_parameter("one_shot").value)
        self._reset_topic = str(self.get_parameter("reset_topic").value)

        self._busy_lock = threading.Lock()
        self._busy = False
        self._done = False
        self._goal_handle = None

        self._action_client = ActionClient(self, HybridPlanner, self._action_name)
        self.create_subscription(PoseStamped, self._target_topic, self._pose_cb, 1)
        self.create_subscription(Empty, self._reset_topic, self._reset_cb, 1)
        # Current joints, used to populate the goal's start_state (see _build_goal).
        self._latest_joint_state: Optional[JointState] = None
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)
        # Fired when a goal is sent (arm starts moving) so Isaac launches the
        # stationary hazard bottle in sync with the motion.
        self.declare_parameter("bottle_trigger_topic", "/hazard/launch_bottle")
        self._bottle_trigger_pub = self.create_publisher(
            Empty, str(self.get_parameter("bottle_trigger_topic").value), 10
        )

        # The hybrid manager (moveit_cpp) comes up slowly, often after the first
        # target arrives. Cache the latest target and dispatch it from a timer
        # once the action server is ready, instead of failing on first contact.
        self._pending_pose: Optional[PoseStamped] = None
        self._warned_no_server = False
        self.create_timer(0.5, self._try_send)

        self.get_logger().info(
            f"HybridPoseClient ready: topic={self._target_topic}, "
            f"action={self._action_name}, group={self._group}, "
            f"ee_link={self._ee_link} (+{self._link_z_offset:.3f}m z), one_shot={self._one_shot}"
        )

    def _joint_state_cb(self, msg: JointState) -> None:
        self._latest_joint_state = msg

    def _reset_cb(self, _msg) -> None:
        with self._busy_lock:
            self._done = False
        self.get_logger().info("one_shot latch reset — accepting new targets")

    def _pose_cb(self, msg: PoseStamped) -> None:
        # Cache only — dispatch happens in _try_send once the server is ready.
        self._pending_pose = msg

    def _try_send(self) -> None:
        with self._busy_lock:
            if self._pending_pose is None:
                return
            if self._one_shot and self._done:
                self._pending_pose = None
                return
            if self._busy:
                return
            if not self._action_client.server_is_ready():
                if not self._warned_no_server:
                    self.get_logger().warn(
                        f"Waiting for HybridPlanner action server '{self._action_name}'..."
                    )
                    self._warned_no_server = True
                return
            msg = self._pending_pose
            self._pending_pose = None
            self._busy = True

        target_frame = msg.header.frame_id or self._planning_frame
        target_pose = PoseStamped()
        target_pose.header.frame_id = target_frame
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.pose = msg.pose
        target_pose.pose.position.z = msg.pose.position.z + self._link_z_offset

        goal = self._build_goal(target_pose)
        self.get_logger().info(
            f"Sending HybridPlanner goal: pos=({target_pose.pose.position.x:.3f}, "
            f"{target_pose.pose.position.y:.3f}, {target_pose.pose.position.z:.3f}) "
            f"frame={target_frame}"
        )
        send_future = self._action_client.send_goal_async(goal)
        send_future.add_done_callback(self._goal_response_cb)

        # Launch the hazard bottle in sync with the motion start.
        self._bottle_trigger_pub.publish(Empty())
        self.get_logger().info("Published /hazard/launch_bottle (motion started)")

    def _build_goal(self, target_pose: PoseStamped) -> HybridPlanner.Goal:
        position_constraint = PositionConstraint()
        position_constraint.header = target_pose.header
        position_constraint.link_name = self._ee_link
        position_constraint.target_point_offset.x = 0.0
        position_constraint.target_point_offset.y = 0.0
        position_constraint.target_point_offset.z = 0.0

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [self._pos_tol]

        region = BoundingVolume()
        region.primitives.append(sphere)
        region.primitive_poses.append(target_pose.pose)
        position_constraint.constraint_region = region
        position_constraint.weight = 1.0

        orientation_constraint = OrientationConstraint()
        orientation_constraint.header = target_pose.header
        orientation_constraint.link_name = self._ee_link
        orientation_constraint.orientation = target_pose.pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = self._ori_tol
        orientation_constraint.absolute_y_axis_tolerance = self._ori_tol
        orientation_constraint.absolute_z_axis_tolerance = self._ori_tol
        orientation_constraint.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(position_constraint)
        constraints.orientation_constraints.append(orientation_constraint)

        workspace = WorkspaceParameters()
        workspace.header.frame_id = self._planning_frame
        workspace.min_corner.x = -1.5
        workspace.min_corner.y = -1.5
        workspace.min_corner.z = -0.5
        workspace.max_corner.x = 1.5
        workspace.max_corner.y = 1.5
        workspace.max_corner.z = 2.0

        req = MotionPlanRequest()
        req.workspace_parameters = workspace
        # Populate start_state with the actual current joints. An empty start
        # state propagates to the global solution's trajectory_start, which makes
        # the local planner see an empty JointState and immediately get "stuck".
        if self._latest_joint_state is not None:
            req.start_state.joint_state = self._latest_joint_state
        else:
            req.start_state.is_diff = True
        req.group_name = self._group
        req.pipeline_id = self._pipeline_id
        req.planner_id = self._planner_id
        req.num_planning_attempts = self._num_attempts
        req.allowed_planning_time = self._planning_time
        req.max_velocity_scaling_factor = self._vel_scale
        req.max_acceleration_scaling_factor = self._acc_scale
        req.goal_constraints.append(constraints)

        item = MotionSequenceItem()
        item.req = req
        item.blend_radius = 0.0

        sequence = MotionSequenceRequest()
        sequence.items.append(item)

        goal = HybridPlanner.Goal()
        goal.planning_group = self._group
        goal.motion_sequence = sequence
        return goal

    def _goal_response_cb(self, future) -> None:
        try:
            handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"Goal send failed: {exc}")
            with self._busy_lock:
                self._busy = False
            return

        if not handle.accepted:
            self.get_logger().warn("HybridPlanner goal rejected")
            with self._busy_lock:
                self._busy = False
            return

        self._goal_handle = handle
        self.get_logger().info("HybridPlanner goal accepted; executing")
        handle.get_result_async().add_done_callback(self._result_cb)

    def _result_cb(self, future) -> None:
        try:
            result = future.result().result
        except Exception as exc:
            self.get_logger().error(f"HybridPlanner result error: {exc}")
            with self._busy_lock:
                self._busy = False
            return

        code = result.error_code.val
        if code == 1:
            self.get_logger().info("Hybrid planning SUCCESS")
        else:
            self.get_logger().warn(
                f"Hybrid planning ended, error_code={code} "
                f"({result.error_message or 'no message'})"
            )

        with self._busy_lock:
            self._busy = False
            self._goal_handle = None
            # Latch on ANY completion (success or failure), not just success, so a
            # failed/aborted goal doesn't make us re-send and preempt/thrash the
            # manager. Re-arm explicitly via the reset topic.
            if self._one_shot:
                self._done = True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HybridPoseClientNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
