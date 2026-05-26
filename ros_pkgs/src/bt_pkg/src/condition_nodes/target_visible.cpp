#include "bt_pkg/condition_nodes.hpp"
#include "bt_pkg/destination_calculator.hpp"  // for dist3

namespace bt_pkg {

TargetVisible::TargetVisible(const std::string& name,
                             const BT::NodeConfig& config,
                             std::shared_ptr<SceneData> scene,
                             rclcpp::Clock::SharedPtr clock,
                             double search_radius_m,
                             double staleness_sec)
  : BT::ConditionNode(name, config)
  , scene_(std::move(scene))
  , clock_(std::move(clock))
  , search_radius_m_(search_radius_m)
  , staleness_sec_(staleness_sec)
{}

BT::NodeStatus TargetVisible::tick()
{
  std::lock_guard<std::mutex> lk(scene_->mtx);

  // Check staleness: /yolo/target_centroid must have been published recently.
  auto age = (clock_->now() - scene_->target_centroid_stamp).seconds();
  if (age > staleness_sec_) {
    RCLCPP_WARN_THROTTLE(rclcpp::get_logger("TargetVisible"),
      *clock_, 2000, "Target centroid stale (%.1fs) — FAILURE", age);
    return BT::NodeStatus::FAILURE;
  }

  // Check proximity to initial centroid set by ParseScene.
  const auto& live = scene_->target_centroid_live.point;
  const auto& seed = scene_->target_centroid;
  std::array<double, 3> live_arr = {live.x, live.y, live.z};

  double d = dist3(live_arr, seed);
  if (d > search_radius_m_) {
    RCLCPP_WARN_THROTTLE(rclcpp::get_logger("TargetVisible"),
      *clock_, 2000,
      "Target moved %.3fm from seed (radius=%.3fm) — FAILURE", d, search_radius_m_);
    return BT::NodeStatus::FAILURE;
  }

  return BT::NodeStatus::SUCCESS;
}

}  // namespace bt_pkg
