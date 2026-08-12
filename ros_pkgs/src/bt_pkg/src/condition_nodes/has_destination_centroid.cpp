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

  if (has) return BT::NodeStatus::SUCCESS;

  // Say WHY the place phase is being skipped. Failing here drops the tree into
  // <AlwaysSuccess/>, so the arm silently finishes the pick, returns to the
  // observation pose, and looks for all the world like a BT with no place logic
  // at all — the two causes below are indistinguishable from the outside
  // without this. Ticked once per pipeline pass, so it cannot spam.
  DestinationSpec spec;
  const bool named = bb.get<DestinationSpec>("destination_spec", spec)
                     && !spec.reference_label.empty();
  if (named) {
    // The case CLAUDE.md calls "named but not in view": the VLM understood the
    // instruction and filled the spec, but nothing segmented the object, so
    // /world_map_result carries no destination centroid to place at. Common
    // failure is the overhead view not returning a DESTINATION detection.
    RCLCPP_WARN(rclcpp::get_logger("HasDestinationCentroid"),
      "destination '%s' (type='%s'%s%s) was named but never measured — "
      "/world_map_result has no destination centroid. SKIPPING the place phase "
      "and keeping the object. Check whether the overhead view segmented it.",
      spec.reference_label.c_str(), spec.type.c_str(),
      spec.region.empty()   ? "" : (" region=" + spec.region).c_str(),
      spec.relation.empty() ? "" : (" relation=" + spec.relation).c_str());
  } else {
    RCLCPP_INFO(rclcpp::get_logger("HasDestinationCentroid"),
      "pick-only instruction — no destination named, skipping the place phase");
  }
  return BT::NodeStatus::FAILURE;
}

}  // namespace bt_pkg
