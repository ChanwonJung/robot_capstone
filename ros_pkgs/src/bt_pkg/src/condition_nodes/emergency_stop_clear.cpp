#include "bt_pkg/condition_nodes.hpp"

namespace bt_pkg {

EmergencyStopClear::EmergencyStopClear(const std::string& name,
                                       const BT::NodeConfig& config,
                                       std::shared_ptr<SceneData> scene)
  : BT::ConditionNode(name, config), scene_(std::move(scene))
{}

BT::NodeStatus EmergencyStopClear::tick()
{
  std::lock_guard<std::mutex> lk(scene_->mtx);
  if (scene_->hazard_level >= 3) {
    RCLCPP_ERROR_THROTTLE(rclcpp::get_logger("EmergencyStopClear"),
      *rclcpp::Clock::make_shared(), 2000,
      "EMERGENCY STOP: hazard_level=%d (arm/person detected) — tree halted",
      scene_->hazard_level);
    return BT::NodeStatus::FAILURE;
  }
  return BT::NodeStatus::SUCCESS;
}

}  // namespace bt_pkg
