#include "bt_pkg/action_nodes.hpp"

namespace bt_pkg {

GripperAction::GripperAction(const std::string& name,
                             const BT::NodeConfig& config,
                             const BT::RosNodeParams& params)
  : BT::RosActionNode<GripperCommand>(name, config, params)
{}

bool GripperAction::setGoal(Goal& goal)
{
  std::string cmd = "close";
  getInput("command", cmd);

  if (cmd == "close") {
    goal.command.position   = 0.0;   // fully closed
    goal.command.max_effort = 50.0;
    RCLCPP_INFO(logger(), "GripperAction: CLOSE (contact detection active)");
  } else {
    goal.command.position   = 0.08;  // fully open (0.04m per finger)
    goal.command.max_effort = 20.0;
    RCLCPP_INFO(logger(), "GripperAction: OPEN");
  }
  return true;
}

BT::NodeStatus GripperAction::onResultReceived(const WrappedResult& wr)
{
  std::string cmd = "close";
  getInput("command", cmd);

  const auto& res = *wr.result;

  if (cmd == "open") {
    // Open always succeeds regardless of result fields
    RCLCPP_INFO(logger(), "GripperAction OPEN: SUCCESS (pos=%.4fm)", res.position);
    return BT::NodeStatus::SUCCESS;
  }

  // CLOSE: gripper_action_server sets stalled=true when it detected contact
  if (res.stalled) {
    RCLCPP_INFO(logger(),
      "GripperAction CLOSE: SUCCESS — contact at pos=%.4fm (object grasped)", res.position);
    return BT::NodeStatus::SUCCESS;
  }

  // Fingers fully closed (res.reached_goal=true) means nothing was there
  RCLCPP_WARN(logger(),
    "GripperAction CLOSE: FAILURE — fingers fully closed, no contact (missed)");
  return BT::NodeStatus::FAILURE;
}

BT::NodeStatus GripperAction::onFailure(BT::ActionNodeErrorCode err)
{
  RCLCPP_ERROR(logger(), "GripperAction action error: %s", BT::toStr(err));
  return BT::NodeStatus::FAILURE;
}

}  // namespace bt_pkg
