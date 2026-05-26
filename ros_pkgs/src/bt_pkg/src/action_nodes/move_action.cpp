#include "bt_pkg/action_nodes.hpp"

#include <moveit_msgs/msg/bounding_volume.hpp>
#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/motion_plan_request.hpp>
#include <moveit_msgs/msg/motion_sequence_item.hpp>
#include <moveit_msgs/msg/motion_sequence_request.hpp>
#include <moveit_msgs/msg/orientation_constraint.hpp>
#include <moveit_msgs/msg/position_constraint.hpp>
#include <moveit_msgs/msg/workspace_parameters.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

namespace bt_pkg {

MoveAction::MoveAction(const std::string& name,
                       const BT::NodeConfig& config,
                       const BT::RosNodeParams& params,
                       std::shared_ptr<SceneData> scene,
                       const std::string& planning_group,
                       const std::string& ee_link,
                       const std::string& planning_frame,
                       double pos_tol,
                       double ori_tol)
  : BT::RosActionNode<HybridPlanner>(name, config, params)
  , scene_(std::move(scene))
  , planning_group_(planning_group)
  , ee_link_(ee_link)
  , planning_frame_(planning_frame)
  , pos_tol_(pos_tol)
  , ori_tol_(ori_tol)
{}

bool MoveAction::setGoal(Goal& goal)
{
  // Read pose from blackboard via pose_key port
  std::string pose_key;
  if (!getInput("pose_key", pose_key)) {
    RCLCPP_ERROR(logger(), "MoveAction: missing pose_key port");
    return false;
  }

  geometry_msgs::msg::PoseStamped target_pose;
  try {
    target_pose = config().blackboard->get<geometry_msgs::msg::PoseStamped>(pose_key);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(logger(), "MoveAction: blackboard key '%s' not found: %s",
      pose_key.c_str(), e.what());
    return false;
  }

  double speed = 0.3;
  getInput("speed", speed);

  // ── Build MotionPlanRequest (mirrors hybrid_pose_client_node.py _build_goal) ──

  // Position constraint
  moveit_msgs::msg::PositionConstraint pos_c;
  pos_c.header = target_pose.header;
  if (pos_c.header.frame_id.empty()) pos_c.header.frame_id = planning_frame_;
  pos_c.link_name = ee_link_;

  shape_msgs::msg::SolidPrimitive sphere;
  sphere.type = shape_msgs::msg::SolidPrimitive::SPHERE;
  sphere.dimensions = {pos_tol_};

  moveit_msgs::msg::BoundingVolume region;
  region.primitives.push_back(sphere);
  region.primitive_poses.push_back(target_pose.pose);

  pos_c.constraint_region = region;
  pos_c.weight = 1.0;

  // Orientation constraint
  moveit_msgs::msg::OrientationConstraint ori_c;
  ori_c.header = pos_c.header;
  ori_c.link_name = ee_link_;
  ori_c.orientation = target_pose.pose.orientation;
  ori_c.absolute_x_axis_tolerance = ori_tol_;
  ori_c.absolute_y_axis_tolerance = ori_tol_;
  ori_c.absolute_z_axis_tolerance = ori_tol_;
  ori_c.weight = 1.0;

  moveit_msgs::msg::Constraints constraints;
  constraints.position_constraints.push_back(pos_c);
  constraints.orientation_constraints.push_back(ori_c);

  // Workspace
  moveit_msgs::msg::WorkspaceParameters ws;
  ws.header.frame_id = planning_frame_;
  ws.min_corner.x = -1.5; ws.min_corner.y = -1.5; ws.min_corner.z = -0.5;
  ws.max_corner.x =  1.5; ws.max_corner.y =  1.5; ws.max_corner.z =  2.0;

  // Motion plan request
  moveit_msgs::msg::MotionPlanRequest req;
  req.workspace_parameters = ws;
  req.group_name            = planning_group_;
  req.pipeline_id           = "ompl";
  req.planner_id            = "RRTConnectkConfigDefault";
  req.num_planning_attempts = 10;
  req.allowed_planning_time = 5.0;
  req.max_velocity_scaling_factor     = speed;
  req.max_acceleration_scaling_factor = 0.1;
  req.goal_constraints.push_back(constraints);

  // Populate start state from latest joint state to avoid "stuck local planner" bug
  // (same as hybrid_pose_client_node.py — empty start_state breaks hybrid planning)
  {
    std::lock_guard<std::mutex> lk(scene_->mtx);
    if (scene_->has_joint_state) {
      req.start_state.joint_state = scene_->latest_joint_state;
    } else {
      req.start_state.is_diff = true;
    }
  }

  // Sequence wrapper
  moveit_msgs::msg::MotionSequenceItem item;
  item.req = req;
  item.blend_radius = 0.0;

  moveit_msgs::msg::MotionSequenceRequest seq;
  seq.items.push_back(item);

  goal.planning_group  = planning_group_;
  goal.motion_sequence = seq;

  RCLCPP_INFO(logger(),
    "MoveAction '%s': target=(%.3f, %.3f, %.3f) speed=%.2f",
    pose_key.c_str(),
    target_pose.pose.position.x,
    target_pose.pose.position.y,
    target_pose.pose.position.z,
    speed);

  return true;
}

BT::NodeStatus MoveAction::onResultReceived(const WrappedResult& wr)
{
  int code = wr.result->error_code.val;
  if (code == 1) {
    RCLCPP_INFO(logger(), "MoveAction: SUCCESS (error_code=1)");
    return BT::NodeStatus::SUCCESS;
  }
  RCLCPP_WARN(logger(), "MoveAction: FAILURE (error_code=%d: %s)",
    code, wr.result->error_message.c_str());
  return BT::NodeStatus::FAILURE;
}

BT::NodeStatus MoveAction::onFailure(BT::ActionNodeErrorCode err)
{
  RCLCPP_ERROR(logger(), "MoveAction action error: %s", BT::toStr(err));
  return BT::NodeStatus::FAILURE;
}

}  // namespace bt_pkg
