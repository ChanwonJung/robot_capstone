#include "bt_pkg/action_nodes.hpp"

#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/joint_constraint.hpp>
#include <moveit_msgs/msg/motion_plan_request.hpp>

namespace bt_pkg {

MoveToHome::MoveToHome(const std::string& name,
                       const BT::NodeConfig& config,
                       const BT::RosNodeParams& params,
                       std::shared_ptr<SceneData> scene,
                       const std::string& planning_group,
                       const std::vector<std::string>& joint_names,
                       const std::vector<double>& joint_values,
                       double joint_tolerance)
  : BT::RosActionNode<MoveGroup>(name, config, params)
  , scene_(std::move(scene))
  , planning_group_(planning_group)
  , joint_names_(joint_names)
  , joint_values_(joint_values)
  , joint_tolerance_(joint_tolerance)
{}

bool MoveToHome::setGoal(Goal& goal)
{
  double speed = 0.5;
  getInput("speed", speed);

  // Build joint constraints for the "ready" state
  moveit_msgs::msg::Constraints constraints;
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    moveit_msgs::msg::JointConstraint jc;
    jc.joint_name     = joint_names_[i];
    jc.position       = joint_values_[i];
    jc.tolerance_above = joint_tolerance_;
    jc.tolerance_below = joint_tolerance_;
    jc.weight         = 1.0;
    constraints.joint_constraints.push_back(jc);
  }

  moveit_msgs::msg::MotionPlanRequest req;
  req.group_name            = planning_group_;
  req.pipeline_id           = "ompl";
  req.planner_id            = "RRTConnectkConfigDefault";
  req.num_planning_attempts = 10;
  req.allowed_planning_time = 10.0;  // more time for recovery
  req.max_velocity_scaling_factor     = speed;
  req.max_acceleration_scaling_factor = 0.1;
  req.goal_constraints.push_back(constraints);

  {
    std::lock_guard<std::mutex> lk(scene_->mtx);
    if (scene_->has_joint_state) {
      req.start_state.joint_state = scene_->latest_joint_state;
    } else {
      req.start_state.is_diff = true;
    }
  }

  goal.request = req;
  RCLCPP_INFO(logger(), "MoveToHome: moving to 'ready' state at speed %.2f", speed);
  return true;
}

BT::NodeStatus MoveToHome::onResultReceived(const WrappedResult& wr)
{
  int code = wr.result->error_code.val;
  if (code == 1) {
    RCLCPP_INFO(logger(), "MoveToHome: SUCCESS");
    return BT::NodeStatus::SUCCESS;
  }
  RCLCPP_WARN(logger(), "MoveToHome: FAILURE (error_code=%d)", code);
  return BT::NodeStatus::FAILURE;
}

BT::NodeStatus MoveToHome::onFailure(BT::ActionNodeErrorCode err)
{
  RCLCPP_ERROR(logger(), "MoveToHome action error: %s", BT::toStr(err));
  return BT::NodeStatus::FAILURE;
}

}  // namespace bt_pkg
