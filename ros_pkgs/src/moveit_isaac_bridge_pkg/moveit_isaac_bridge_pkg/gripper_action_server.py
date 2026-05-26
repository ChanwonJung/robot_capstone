"""gripper_action_server.py

Serves control_msgs/action/GripperCommand on /gripper_command.

The BT's GripperAction node calls this to open or close the Panda hand.
Contact detection is done by monitoring panda_finger_joint1/2 from /joint_states:
  - CLOSE goal: command fingers to 0.0 m.
    • If the fingers stall above contact_threshold_m → object grasped → result.stalled = True
    • If fingers reach ≈ 0.0 m → no object → result.stalled = False
  - OPEN goal: command fingers to open_position_m each. Always succeeds.

Motion is sent via MoveIt's /move_action (MoveGroup) targeting the panda_hand group.
This reuses the existing MoveIt infrastructure and doesn't require a separate
hand trajectory controller action server.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import rclpy
from control_msgs.action import GripperCommand
from moveit_msgs.action import MoveGroup as MoveGroupAction
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
)
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

FINGER_JOINTS   = ["panda_finger_joint1", "panda_finger_joint2"]
OPEN_POS_M      = 0.04   # m per finger (0.08 m total opening)
CLOSE_POS_M     = 0.0    # m per finger (fully closed)


class GripperActionServer(Node):
    def __init__(self) -> None:
        super().__init__("gripper_action_server")

        self.declare_parameter("contact_threshold_m", 0.008)
        self.declare_parameter("timeout_sec",         3.0)
        self.declare_parameter("move_action",         "/move_action")
        self.declare_parameter("planning_group",      "panda_hand")
        self.declare_parameter("joint_tolerance",     0.005)
        self.declare_parameter("poll_rate_hz",        20.0)

        self._contact_thr    = float(self.get_parameter("contact_threshold_m").value)
        self._timeout        = float(self.get_parameter("timeout_sec").value)
        self._planning_group = str(self.get_parameter("planning_group").value)
        self._joint_tol      = float(self.get_parameter("joint_tolerance").value)
        self._poll_dt        = 1.0 / float(self.get_parameter("poll_rate_hz").value)
        move_action          = str(self.get_parameter("move_action").value)

        # /joint_states — thread-safe finger position cache
        self._finger_pos: dict[str, float] = {j: OPEN_POS_M for j in FINGER_JOINTS}
        self._js_lock = threading.Lock()
        self.create_subscription(JointState, "/joint_states", self._js_cb, 10)

        # MoveIt action client for panda_hand group
        self._cb_group = ReentrantCallbackGroup()
        self._move_client = ActionClient(
            self, MoveGroupAction, move_action,
            callback_group=self._cb_group)

        # GripperCommand action server
        self._server = ActionServer(
            self,
            GripperCommand,
            "/gripper_command",
            execute_callback=self._execute_cb,
            goal_callback=lambda _: GoalResponse.ACCEPT,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            f"GripperActionServer ready on /gripper_command "
            f"(contact_thr={self._contact_thr*1000:.1f}mm, "
            f"timeout={self._timeout}s, group='{self._planning_group}')"
        )

    # ── /joint_states callback ─────────────────────────────────────────────

    def _js_cb(self, msg: JointState) -> None:
        with self._js_lock:
            for name, pos in zip(msg.name, msg.position):
                if name in FINGER_JOINTS:
                    self._finger_pos[name] = float(pos)

    def _avg_finger_pos(self) -> float:
        with self._js_lock:
            return sum(self._finger_pos.values()) / len(self._finger_pos)

    # ── MoveIt helper ─────────────────────────────────────────────────────

    def _move_fingers(self, target_pos: float) -> bool:
        """Send MoveGroup goal for panda_hand. Returns True if planning succeeded."""
        if not self._move_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("MoveGroup action server not available")
            return False

        # Build joint constraints for both finger joints
        constraints = Constraints()
        for joint in FINGER_JOINTS:
            jc = JointConstraint()
            jc.joint_name      = joint
            jc.position        = target_pos
            jc.tolerance_above = self._joint_tol
            jc.tolerance_below = self._joint_tol
            jc.weight          = 1.0
            constraints.joint_constraints.append(jc)

        req = MotionPlanRequest()
        req.group_name                      = self._planning_group
        req.pipeline_id                     = "ompl"
        req.num_planning_attempts           = 5
        req.allowed_planning_time           = 3.0
        req.max_velocity_scaling_factor     = 1.0
        req.max_acceleration_scaling_factor = 1.0
        req.start_state.is_diff            = True
        req.goal_constraints.append(constraints)

        goal = MoveGroupAction.Goal()
        goal.request = req

        # Blocking send (this callback runs on ReentrantCallbackGroup thread)
        future = self._move_client.send_goal_async(goal)
        # Spin until future resolves
        while not future.done():
            time.sleep(0.02)

        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().warn("Finger trajectory goal rejected")
            return False

        result_future = handle.get_result_async()
        while not result_future.done():
            time.sleep(0.02)

        code = result_future.result().result.error_code.val
        if code != 1:
            self.get_logger().warn(f"Finger trajectory ended with code {code}")
        return code == 1

    # ── Action execute callback ────────────────────────────────────────────

    def _execute_cb(self, goal_handle) -> GripperCommand.Result:
        commanded_pos = goal_handle.request.command.position
        is_close      = commanded_pos < 0.01

        result = GripperCommand.Result()

        target = CLOSE_POS_M if is_close else OPEN_POS_M
        self.get_logger().info(
            f"GripperCommand: {'CLOSE' if is_close else 'OPEN'} "
            f"(target={target*1000:.1f}mm)"
        )

        # Send the finger trajectory. Even if MoveIt planning fails we still
        # poll joint_states because Isaac Sim may have executed a partial move.
        ok = self._move_fingers(target)
        if not ok:
            self.get_logger().warn("Finger MoveIt planning failed — polling contact anyway")

        # Poll joint_states until settled or timeout
        deadline = time.monotonic() + self._timeout
        prev_pos = self._avg_finger_pos()
        settle_count = 0
        SETTLE_THRESHOLD = 0.0005   # m — considered not moving if delta < this
        SETTLE_TICKS     = 4        # consecutive stable readings

        while time.monotonic() < deadline:
            time.sleep(self._poll_dt)
            cur_pos = self._avg_finger_pos()
            delta   = abs(cur_pos - prev_pos)
            prev_pos = cur_pos

            if delta < SETTLE_THRESHOLD:
                settle_count += 1
            else:
                settle_count = 0

            if settle_count >= SETTLE_TICKS:
                break  # motion has settled

        final_pos = self._avg_finger_pos()
        result.position = final_pos

        if is_close:
            # Stalled on object if fingers stopped above contact threshold
            grasped = final_pos > self._contact_threshold
            result.stalled       = grasped
            result.reached_goal  = not grasped
            result.effort        = 0.0
            if grasped:
                self.get_logger().info(
                    f"CLOSE: contact at {final_pos*1000:.2f}mm — GRASPED (stalled=True)")
            else:
                self.get_logger().warn(
                    f"CLOSE: fully closed ({final_pos*1000:.2f}mm) — MISSED (stalled=False)")
        else:
            result.stalled      = False
            result.reached_goal = True
            result.effort       = 0.0
            self.get_logger().info(f"OPEN: final pos {final_pos*1000:.2f}mm")

        goal_handle.succeed()
        return result

    @property
    def _contact_threshold(self) -> float:
        return self._contact_thr


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GripperActionServer()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
