#include "bt_pkg/action_nodes.hpp"
#include "bt_pkg/destination_calculator.hpp"

namespace bt_pkg {

SelectGraspCandidate::SelectGraspCandidate(const std::string& name,
                                           const BT::NodeConfig& config,
                                           double pre_grasp_z_offset)
  : BT::SyncActionNode(name, config)
  , pre_grasp_z_offset_(pre_grasp_z_offset)
{}

BT::NodeStatus SelectGraspCandidate::tick()
{
  auto& bb = *config().blackboard;

  auto candidates = bb.get<std::vector<GraspCandidate>>("grasp_candidates");
  int  idx        = bb.get<int>("grasp_index");

  if (idx >= static_cast<int>(candidates.size())) {
    RCLCPP_WARN(rclcpp::get_logger("SelectGraspCandidate"),
      "All %zu candidates exhausted — FAILURE", candidates.size());
    return BT::NodeStatus::FAILURE;
  }

  const auto& c = candidates[idx];
  auto pre_grasp = lift_z(c.pose, pre_grasp_z_offset_);

  bb.set<geometry_msgs::msg::PoseStamped>("grasp_pose",     c.pose);
  bb.set<geometry_msgs::msg::PoseStamped>("pre_grasp_pose", pre_grasp);
  bb.set<int>("grasp_index", idx + 1);  // advance for next retry

  RCLCPP_INFO(rclcpp::get_logger("SelectGraspCandidate"),
    "Selected candidate %d/%zu  quality=%.3f width=%.3fm",
    idx + 1, candidates.size(), c.quality, c.width);

  return BT::NodeStatus::SUCCESS;
}

}  // namespace bt_pkg
