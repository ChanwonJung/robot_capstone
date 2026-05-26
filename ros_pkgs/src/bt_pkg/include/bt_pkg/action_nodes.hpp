#pragma once

#include <memory>
#include <string>
#include <vector>

#include <behaviortree_ros2/bt_action_node.hpp>
#include <behaviortree_cpp/action_node.h>
#include <behaviortree_cpp/bt_factory.h>
#include <control_msgs/action/gripper_command.hpp>
#include <moveit_msgs/action/hybrid_planner.hpp>
#include <moveit_msgs/action/move_group.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/empty.hpp>

#include "bt_pkg/destination_calculator.hpp"
#include "bt_pkg/scene_data.hpp"

namespace bt_pkg {

// ─────────────────────────────────────────────────────────────────────────────
// WaitForScene
// Blocks (RUNNING) until /world_map_result AND /grasp_candidates are both
// fresher than the last processed timestamp.
// ─────────────────────────────────────────────────────────────────────────────
class WaitForScene : public BT::StatefulActionNode {
public:
  WaitForScene(const std::string& name, const BT::NodeConfig& config,
               std::shared_ptr<SceneData> scene);

  static BT::PortsList providedPorts() { return {}; }

  BT::NodeStatus onStart()   override;
  BT::NodeStatus onRunning() override;
  void           onHalted()  override {}

private:
  bool is_fresh() const;  // must be called with scene_->mtx held
  std::shared_ptr<SceneData> scene_;
};

// ─────────────────────────────────────────────────────────────────────────────
// ParseScene
// Reads SceneData into the BT blackboard. Resets grasp_index to 0.
// Marks last_processed_stamp so WaitForScene won't re-trigger on same data.
// ─────────────────────────────────────────────────────────────────────────────
class ParseScene : public BT::SyncActionNode {
public:
  ParseScene(const std::string& name, const BT::NodeConfig& config,
             std::shared_ptr<SceneData> scene,
             double pre_grasp_z_offset, double retreat_z_offset);

  static BT::PortsList providedPorts() { return {}; }
  BT::NodeStatus tick() override;

private:
  std::shared_ptr<SceneData> scene_;
  double pre_grasp_z_offset_;
  double retreat_z_offset_;
};

// ─────────────────────────────────────────────────────────────────────────────
// SelectGraspCandidate
// Reads grasp_index from blackboard, sets pre_grasp_pose + grasp_pose,
// increments index. Returns FAILURE when all candidates are exhausted.
// ─────────────────────────────────────────────────────────────────────────────
class SelectGraspCandidate : public BT::SyncActionNode {
public:
  SelectGraspCandidate(const std::string& name, const BT::NodeConfig& config,
                       double pre_grasp_z_offset);

  static BT::PortsList providedPorts() { return {}; }
  BT::NodeStatus tick() override;

private:
  double pre_grasp_z_offset_;
};

// ─────────────────────────────────────────────────────────────────────────────
// MoveAction
// Wraps moveit_msgs/HybridPlanner action. Reads pose from blackboard via
// pose_key port; velocity scaling from speed port.
// ─────────────────────────────────────────────────────────────────────────────
class MoveAction
  : public BT::RosActionNode<moveit_msgs::action::HybridPlanner>
{
public:
  using HybridPlanner = moveit_msgs::action::HybridPlanner;

  MoveAction(const std::string& name, const BT::NodeConfig& config,
             const BT::RosNodeParams& params,
             std::shared_ptr<SceneData> scene,
             const std::string& planning_group,
             const std::string& ee_link,
             const std::string& planning_frame,
             double pos_tol, double ori_tol);

  static BT::PortsList providedPorts() {
    return BT::RosActionNode<HybridPlanner>::providedBasicPorts({
      BT::InputPort<std::string>("pose_key", "grasp_pose",
                                 "Blackboard key for target PoseStamped"),
      BT::InputPort<double>("speed", 0.3, "max_velocity_scaling_factor"),
    });
  }

  bool          setGoal(Goal& goal)                          override;
  BT::NodeStatus onResultReceived(const WrappedResult& wr)   override;
  BT::NodeStatus onFailure(BT::ActionNodeErrorCode err)      override;
  void           onHalt()                                    override {}

private:
  std::shared_ptr<SceneData> scene_;
  std::string planning_group_, ee_link_, planning_frame_;
  double pos_tol_, ori_tol_;
};

// ─────────────────────────────────────────────────────────────────────────────
// MoveToHome
// Wraps moveit_msgs/MoveGroup action. Sends the Panda "ready" joint state.
// Used only in recovery paths (no obstacle avoidance needed).
// ─────────────────────────────────────────────────────────────────────────────
class MoveToHome
  : public BT::RosActionNode<moveit_msgs::action::MoveGroup>
{
public:
  using MoveGroup = moveit_msgs::action::MoveGroup;

  MoveToHome(const std::string& name, const BT::NodeConfig& config,
             const BT::RosNodeParams& params,
             std::shared_ptr<SceneData> scene,
             const std::string& planning_group,
             const std::vector<std::string>& joint_names,
             const std::vector<double>& joint_values,
             double joint_tolerance);

  static BT::PortsList providedPorts() {
    return BT::RosActionNode<MoveGroup>::providedBasicPorts({
      BT::InputPort<double>("speed", 0.5, "max_velocity_scaling_factor"),
    });
  }

  bool          setGoal(Goal& goal)                          override;
  BT::NodeStatus onResultReceived(const WrappedResult& wr)   override;
  BT::NodeStatus onFailure(BT::ActionNodeErrorCode err)      override;
  void           onHalt()                                    override {}

private:
  std::shared_ptr<SceneData> scene_;
  std::string planning_group_;
  std::vector<std::string> joint_names_;
  std::vector<double>      joint_values_;
  double joint_tolerance_;
};

// ─────────────────────────────────────────────────────────────────────────────
// GripperAction
// Wraps control_msgs/GripperCommand action (served by gripper_action_server).
// command port: "close" | "open"
// CLOSE: SUCCESS = stalled (object grasped), FAILURE = fully closed (missed)
// OPEN:  always SUCCESS
// ─────────────────────────────────────────────────────────────────────────────
class GripperAction
  : public BT::RosActionNode<control_msgs::action::GripperCommand>
{
public:
  using GripperCommand = control_msgs::action::GripperCommand;

  GripperAction(const std::string& name, const BT::NodeConfig& config,
                const BT::RosNodeParams& params);

  static BT::PortsList providedPorts() {
    return BT::RosActionNode<GripperCommand>::providedBasicPorts({
      BT::InputPort<std::string>("command", "close", "\"close\" or \"open\""),
    });
  }

  bool          setGoal(Goal& goal)                          override;
  BT::NodeStatus onResultReceived(const WrappedResult& wr)   override;
  BT::NodeStatus onFailure(BT::ActionNodeErrorCode err)      override;
  void           onHalt()                                    override {}
};

// ─────────────────────────────────────────────────────────────────────────────
// UpdateTargetPose
// Reads destination_spec from blackboard + live yolo_objects from SceneData.
// Computes place_pose and writes it to the blackboard.
// ─────────────────────────────────────────────────────────────────────────────
class UpdateTargetPose : public BT::SyncActionNode {
public:
  UpdateTargetPose(const std::string& name, const BT::NodeConfig& config,
                   std::shared_ptr<SceneData> scene,
                   const PlacePoseParams& params,
                   double dest_match_radius_m);

  static BT::PortsList providedPorts() { return {}; }
  BT::NodeStatus tick() override;

private:
  std::shared_ptr<SceneData> scene_;
  PlacePoseParams params_;
  double dest_match_radius_m_;
};

// ─────────────────────────────────────────────────────────────────────────────
// RequestReplan
// Publishes /bt/replan_request and resets scene freshness flags.
// Intentionally returns FAILURE to restart the Main Pipeline Sequence from
// WaitForScene (standard BT "restart" pattern).
// ─────────────────────────────────────────────────────────────────────────────
class RequestReplan : public BT::SyncActionNode {
public:
  RequestReplan(const std::string& name, const BT::NodeConfig& config,
                std::shared_ptr<SceneData> scene,
                rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr pub);

  static BT::PortsList providedPorts() { return {}; }
  BT::NodeStatus tick() override;

private:
  std::shared_ptr<SceneData> scene_;
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr pub_;
};

}  // namespace bt_pkg
