#include "bt_pkg/condition_nodes.hpp"

namespace bt_pkg {

HasDestinationCentroid::HasDestinationCentroid(const std::string& name,
                                               const BT::NodeConfig& config)
  : BT::ConditionNode(name, config)
{}

BT::NodeStatus HasDestinationCentroid::tick()
{
  auto& bb = *config().blackboard;

  // Absent key = ParseScene has not run yet. Fail closed: no place phase.
  bool has = false;
  if (!bb.get<bool>("has_destination_centroid", has)) {
    RCLCPP_WARN(rclcpp::get_logger("HasDestinationCentroid"),
      "has_destination_centroid not on the blackboard — ParseScene has not run");
    return BT::NodeStatus::FAILURE;
  }

  return has ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
}

}  // namespace bt_pkg
