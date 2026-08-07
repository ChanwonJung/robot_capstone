#include "bt_pkg/action_nodes.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/joint_constraint.hpp>
#include <moveit_msgs/msg/motion_plan_request.hpp>
#include <moveit_msgs/msg/motion_sequence_item.hpp>
#include <moveit_msgs/msg/motion_sequence_request.hpp>

namespace bt_pkg {

MoveToJointGoal::MoveToJointGoal(const std::string& name,
                                 const BT::NodeConfig& config,
                                 const BT::RosNodeParams& params,
                                 std::shared_ptr<SceneData> scene,
                                 const std::string& planning_group,
                                 const std::vector<std::string>& joint_names,
                                 const std::vector<double>& joint_values,
                                 double joint_tolerance,
                                 const std::string& label)
  : BT::RosActionNode<HybridPlanner>(name, config, params)
  , scene_(std::move(scene))
  , planning_group_(planning_group)
  , joint_names_(joint_names)
  , joint_values_(joint_values)
  , joint_tolerance_(joint_tolerance)
  , label_(label)
{}

bool MoveToJointGoal::setGoal(Goal& goal)
{
  // Empty configured values = use the pose latched from the first /joint_states.
  std::vector<double> values = joint_values_;
  if (values.empty()) {
    std::lock_guard<std::mutex> lk(scene_->mtx);
    if (!scene_->has_observation_pose) {
      RCLCPP_ERROR(logger(),
        "%s: observation pose not latched yet — no /joint_states received",
        label_.c_str());
      return false;
    }
    values = scene_->observation_joint_values;
  }

  if (joint_names_.size() != values.size()) {
    RCLCPP_ERROR(logger(),
      "%s: %zu joint names but %zu values — check robot_defaults.yaml",
      label_.c_str(), joint_names_.size(), values.size());
    return false;
  }

  last_target_ = values;

  double speed = 0.5;
  getInput("speed", speed);

  moveit_msgs::msg::Constraints constraints;
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    moveit_msgs::msg::JointConstraint jc;
    jc.joint_name      = joint_names_[i];
    jc.position        = values[i];
    jc.tolerance_above = joint_tolerance_;
    jc.tolerance_below = joint_tolerance_;
    jc.weight          = 1.0;
    constraints.joint_constraints.push_back(jc);
  }

  moveit_msgs::msg::MotionPlanRequest req;
  req.group_name            = planning_group_;
  req.pipeline_id           = "ompl";
  req.planner_id            = "RRTConnectkConfigDefault";
  req.num_planning_attempts = 10;
  req.allowed_planning_time = 10.0;
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

  moveit_msgs::msg::MotionSequenceItem item;
  item.req = req;
  item.blend_radius = 0.0;

  moveit_msgs::msg::MotionSequenceRequest seq;
  seq.items.push_back(item);

  goal.planning_group  = planning_group_;
  goal.motion_sequence = seq;

  std::string joined;
  for (size_t i = 0; i < values.size(); ++i) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%s%.3f", i ? ", " : "", values[i]);
    joined += buf;
  }
  RCLCPP_INFO(logger(), "%s: moving to [%s] at speed %.2f",
    label_.c_str(), joined.c_str(), speed);
  return true;
}

// Worst per-joint |achieved - commanded|, or -1 if it cannot be measured.
// The planner reporting SUCCESS only means it landed inside the goal
// constraint, so this is what actually says how close the arm got.
double MoveToJointGoal::goal_residual(const std::vector<double>& target) const
{
  std::lock_guard<std::mutex> lk(scene_->mtx);
  if (!scene_->has_joint_state) return -1.0;

  const auto& js = scene_->latest_joint_state;
  double worst = 0.0;
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    auto it = std::find(js.name.begin(), js.name.end(), joint_names_[i]);
    if (it == js.name.end()) return -1.0;
    const double got = js.position[std::distance(js.name.begin(), it)];
    worst = std::max(worst, std::abs(got - target[i]));
  }
  return worst;
}

BT::NodeStatus MoveToJointGoal::onResultReceived(const WrappedResult& wr)
{
  int code = wr.result->error_code.val;
  if (code == 1) {
    const double residual = goal_residual(last_target_);
    if (residual < 0.0) {
      RCLCPP_INFO(logger(), "%s: SUCCESS", label_.c_str());
    } else {
      RCLCPP_INFO(logger(), "%s: SUCCESS (worst joint residual %.4f rad, "
        "tolerance %.4f)", label_.c_str(), residual, joint_tolerance_);
    }
    return BT::NodeStatus::SUCCESS;
  }
  RCLCPP_WARN(logger(), "%s: FAILURE (error_code=%d)", label_.c_str(), code);
  return BT::NodeStatus::FAILURE;
}

BT::NodeStatus MoveToJointGoal::onFailure(BT::ActionNodeErrorCode err)
{
  RCLCPP_ERROR(logger(), "%s action error: %s", label_.c_str(), BT::toStr(err));
  return BT::NodeStatus::FAILURE;
}

}  // namespace bt_pkg
