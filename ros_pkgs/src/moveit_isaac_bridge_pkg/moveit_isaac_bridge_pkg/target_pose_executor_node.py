"""Minimal MoveIt executor for grounded-SAM target poses.

Listens on a target PoseStamped topic (default ``/grasp_target_pose`` produced
by ``target_pose_bridge``) and asks MoveIt to plan + execute a motion of the
Panda arm so that ``end_effector_link`` reaches that pose.

This is the **motion-generation validation** node for the Slow Brain pipeline:
no gripper, no pick&place sequence, no pre/post-grasp choreography. Those
belong to the future Qwen behavior-tree + graspnet layer.
"""
from __future__ import annotations

import threading
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    MotionPlanRequest,
    OrientationConstraint,
    PlanningOptions,
    PositionConstraint,
    WorkspaceParameters,
)
from rclpy.action import ActionClient
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive


class TargetPoseExecutorNode(Node):
    def __init__(self) -> None:
        super().__init__("target_pose_executor_node")

        self.declare_parameter("target_pose_topic", "/grasp_target_pose")
        self.declare_parameter("planning_group", "panda_arm")
        self.declare_parameter("end_effector_link", "panda_link8")
        self.declare_parameter("planning_frame", "panda_link0")
        self.declare_parameter("planner_id", "RRTConnectkConfigDefault")
        self.declare_parameter("pipeline_id", "ompl")
        self.declare_parameter("num_planning_attempts", 10)
        self.declare_parameter("allowed_planning_time", 5.0)
        self.declare_parameter("position_tolerance", 0.01)
        self.declare_parameter("orientation_tolerance", 0.05)
        self.declare_parameter("max_velocity_scaling_factor", 0.3)
        self.declare_parameter("max_acceleration_scaling_factor", 0.3)
        # Offset added to target z so that `end_effector_link` (panda_link8 by
        # default) is placed above the object centroid, leaving room for the
        # gripper fingers to extend down. Tune for the actual gripper geometry.
        self.declare_parameter("link_target_z_offset", 0.10)
        self.declare_parameter("auto_execute", True)
        self.declare_parameter("action_name", "/move_action")
        self.declare_parameter("ignore_while_busy", True)
        # Stop accepting targets after the first SUCCESS. Aligns with Slow
        # Brain semantics ("one observation per command, one motion"). Reset
        # by publishing std_msgs/Empty to the reset topic or by restarting.
        self.declare_parameter("one_shot", True)
        self.declare_parameter("reset_topic", "/target_pose_executor/reset")

        self._target_topic = str(self.get_parameter("target_pose_topic").value)
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
        self._auto_execute = bool(self.get_parameter("auto_execute").value)
        self._action_name = str(self.get_parameter("action_name").value)
        self._ignore_while_busy = bool(self.get_parameter("ignore_while_busy").value)
        self._one_shot = bool(self.get_parameter("one_shot").value)
        self._reset_topic = str(self.get_parameter("reset_topic").value)

        self._busy_lock = threading.Lock()
        self._busy = False
        self._done = False
        self._goal_handle = None

        self._action_client = ActionClient(self, MoveGroup, self._action_name)

        self.create_subscription(PoseStamped, self._target_topic, self._pose_cb, 1)
        from std_msgs.msg import Empty
        self.create_subscription(Empty, self._reset_topic, self._reset_cb, 1)

        self.get_logger().info(
            f"TargetPoseExecutor ready: topic={self._target_topic}, group={self._group}, "
            f"ee_link={self._ee_link} (+{self._link_z_offset:.3f}m z), "
            f"plan_only={not self._auto_execute}, one_shot={self._one_shot}"
        )

    def _reset_cb(self, _msg) -> None:
        with self._busy_lock:
            self._done = False
        self.get_logger().info("one_shot latch reset — accepting new targets")

    def _pose_cb(self, msg: PoseStamped) -> None:
        with self._busy_lock:
            if self._one_shot and self._done:
                self.get_logger().info(
                    "one_shot: motion already completed for this command; "
                    "ignoring new target (publish std_msgs/Empty on "
                    f"{self._reset_topic} to re-arm)"
                )
                return
            if self._busy and self._ignore_while_busy:
                self.get_logger().info("Plan/execute already in progress — ignoring new target")
                return

        if not self._action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(
                f"MoveGroup action server '{self._action_name}' not available"
            )
            return

        target_frame = msg.header.frame_id or self._planning_frame
        if target_frame != self._planning_frame:
            self.get_logger().warn(
                f"Target frame '{target_frame}' differs from planning frame "
                f"'{self._planning_frame}'. Using target frame as-is; expect a "
                f"static TF identity between them (e.g. world↔panda_link0)."
            )

        target_pose = PoseStamped()
        target_pose.header.frame_id = target_frame
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.pose = msg.pose
        target_pose.pose.position.z = msg.pose.position.z + self._link_z_offset

        goal = self._build_goal(target_pose)

        with self._busy_lock:
            self._busy = True

        self.get_logger().info(
            f"Sending MoveGroup goal: pos=({target_pose.pose.position.x:.3f}, "
            f"{target_pose.pose.position.y:.3f}, {target_pose.pose.position.z:.3f}) "
            f"frame={target_frame}, plan_only={not self._auto_execute}"
        )

        send_future = self._action_client.send_goal_async(goal)
        send_future.add_done_callback(self._goal_response_cb)

    def _build_goal(self, target_pose: PoseStamped) -> MoveGroup.Goal:
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
        req.group_name = self._group
        req.pipeline_id = self._pipeline_id
        req.planner_id = self._planner_id
        req.num_planning_attempts = self._num_attempts
        req.allowed_planning_time = self._planning_time
        req.max_velocity_scaling_factor = self._vel_scale
        req.max_acceleration_scaling_factor = self._acc_scale
        req.goal_constraints.append(constraints)

        opts = PlanningOptions()
        opts.plan_only = not self._auto_execute
        opts.look_around = False
        opts.replan = False
        opts.replan_attempts = 0

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options = opts
        return goal

    def _goal_response_cb(self, future) -> None:
        try:
            handle = future.result()
        except Exception as exc:  # action call failure
            self.get_logger().error(f"Goal send failed: {exc}")
            with self._busy_lock:
                self._busy = False
            return

        if not handle.accepted:
            self.get_logger().warn("MoveGroup goal rejected by server")
            with self._busy_lock:
                self._busy = False
            return

        self._goal_handle = handle
        self.get_logger().info("MoveGroup goal accepted; waiting for result")
        handle.get_result_async().add_done_callback(self._result_cb)

    def _result_cb(self, future) -> None:
        try:
            result_wrapper = future.result()
        except Exception as exc:
            self.get_logger().error(f"MoveGroup result error: {exc}")
            with self._busy_lock:
                self._busy = False
            return

        result = result_wrapper.result
        code = result.error_code.val
        if code == 1:  # SUCCESS
            self.get_logger().info("MoveIt plan+execute SUCCESS")
        else:
            self.get_logger().warn(f"MoveIt plan+execute FAILED, error_code={code}")

        with self._busy_lock:
            self._busy = False
            self._goal_handle = None
            if self._one_shot and code == 1:
                self._done = True
                self.get_logger().info(
                    f"one_shot latch armed; further targets on {self._target_topic} "
                    "will be ignored until reset"
                )


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = TargetPoseExecutorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
