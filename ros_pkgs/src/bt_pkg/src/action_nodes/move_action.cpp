#include "bt_pkg/action_nodes.hpp"

#include <moveit_msgs/msg/bounding_volume.hpp>
#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/joint_constraint.hpp>
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
  // Planning effort 증가: OMPL RRTConnect 의 sampling-based path 가 random
  // curve 인 문제 → attempts/time 늘려 simplify 후 짧은 path 선택 확률↑.
  req.num_planning_attempts = 30;
  req.allowed_planning_time = 10.0;
  req.max_velocity_scaling_factor     = speed;
  req.max_acceleration_scaling_factor = 0.1;
  req.goal_constraints.push_back(constraints);

  // Path orientation constraint: lock_orientation 가 true 이면 trajectory 중
  // EE orientation 을 target 과 거의 동일하게 강제 → OMPL 이 wrist 회전
  // path 만들지 않음. pre_grasp → grasp 직선 descent 에 필수.
  bool lock_orientation = false;
  getInput("lock_orientation", lock_orientation);
  if (lock_orientation) {
    double ori_path_tol = 0.3;
    getInput("lock_orientation_tol", ori_path_tol);
    moveit_msgs::msg::OrientationConstraint path_ori_c;
    path_ori_c.header = pos_c.header;
    path_ori_c.link_name = ee_link_;
    path_ori_c.orientation = target_pose.pose.orientation;
    path_ori_c.absolute_x_axis_tolerance = ori_path_tol;
    path_ori_c.absolute_y_axis_tolerance = ori_path_tol;
    path_ori_c.absolute_z_axis_tolerance = ori_path_tol;
    path_ori_c.weight = 1.0;
    req.path_constraints.orientation_constraints.push_back(path_ori_c);
    RCLCPP_INFO(logger(),
      "MoveAction: lock_orientation tol=±%.2frad", ori_path_tol);
  }

  // Path z-floor: planning_scene 에 책 collision object 가 없어서 OMPL 이
  // 책 옆/아래로 path 를 만드는 문제 차단. EE link 가 (z ≥ min_path_z) 영역
  // 안에만 sampling 되도록 큰 box constraint.
  double min_path_z = 0.0;
  getInput("min_path_z", min_path_z);
  if (min_path_z > 1e-6) {
    moveit_msgs::msg::PositionConstraint path_pos_c;
    path_pos_c.header.frame_id = planning_frame_;
    path_pos_c.link_name = ee_link_;
    // Workspace 보다 살짝 작은 box, z 만 lift.
    const double top_z = 2.0;
    const double half_z = (top_z - min_path_z) * 0.5;
    const double center_z = (top_z + min_path_z) * 0.5;
    shape_msgs::msg::SolidPrimitive safe_box;
    safe_box.type = shape_msgs::msg::SolidPrimitive::BOX;
    safe_box.dimensions = {3.0, 3.0, 2.0 * half_z};  // FULL dimensions
    moveit_msgs::msg::BoundingVolume safe_region;
    safe_region.primitives.push_back(safe_box);
    geometry_msgs::msg::Pose region_pose;
    region_pose.position.x = 0.0;
    region_pose.position.y = 0.0;
    region_pose.position.z = center_z;
    region_pose.orientation.w = 1.0;
    safe_region.primitive_poses.push_back(region_pose);
    path_pos_c.constraint_region = safe_region;
    path_pos_c.weight = 1.0;
    req.path_constraints.position_constraints.push_back(path_pos_c);
    RCLCPP_INFO(logger(),
      "MoveAction: min_path_z active (EE z ≥ %.3f m along path)", min_path_z);
  }

  // Base joints lock: 7-DOF redundancy 로 OMPL 이 base joints 를 회전시키는
  // path 만드는 문제 차단. 현재 base joint 값들을 path 동안 ±0.3rad 안 유지.
  // 끝쪽 joints (4~7) 는 자유 → wrist 미세 조정만으로 grasp 도달.
  bool lock_base_joints = false;
  getInput("lock_base_joints", lock_base_joints);
  if (lock_base_joints) {
    sensor_msgs::msg::JointState current_js;
    bool has_js = false;
    {
      std::lock_guard<std::mutex> lk(scene_->mtx);
      if (scene_->has_joint_state) {
        current_js = scene_->latest_joint_state;
        has_js = true;
      }
    }
    if (has_js) {
      const std::vector<std::string> base_joint_names = {
        "panda_joint1", "panda_joint2", "panda_joint3"
      };
      const double base_tol = 0.3;  // ≈17° 여유
      for (const auto& jname : base_joint_names) {
        // 현재 값 찾기
        auto it = std::find(current_js.name.begin(), current_js.name.end(), jname);
        if (it == current_js.name.end()) continue;
        size_t idx = std::distance(current_js.name.begin(), it);
        if (idx >= current_js.position.size()) continue;
        moveit_msgs::msg::JointConstraint jc;
        jc.joint_name      = jname;
        jc.position        = current_js.position[idx];
        jc.tolerance_above = base_tol;
        jc.tolerance_below = base_tol;
        jc.weight          = 1.0;
        req.path_constraints.joint_constraints.push_back(jc);
      }
      RCLCPP_INFO(logger(),
        "MoveAction: lock_base_joints active (joint_1,2,3 ±%.2frad)", base_tol);
    } else {
      RCLCPP_WARN(logger(),
        "MoveAction: lock_base_joints requested but no joint_state cached");
    }
  }

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
