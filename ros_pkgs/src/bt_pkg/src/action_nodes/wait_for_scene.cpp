#include "bt_pkg/action_nodes.hpp"

namespace bt_pkg {

WaitForScene::WaitForScene(const std::string& name,
                           const BT::NodeConfig& config,
                           std::shared_ptr<SceneData> scene)
  : BT::StatefulActionNode(name, config), scene_(std::move(scene))
{}

bool WaitForScene::is_fresh() const
{
  // Both topics must have arrived AND be newer than what we last consumed.
  return scene_->world_map_fresh
      && scene_->grasp_candidates_fresh
      && scene_->world_map_stamp      > scene_->last_processed_stamp
      && scene_->grasp_candidates_stamp > scene_->last_processed_stamp;
}

BT::NodeStatus WaitForScene::onStart()
{
  std::lock_guard<std::mutex> lk(scene_->mtx);
  if (is_fresh()) {
    RCLCPP_INFO(rclcpp::get_logger("WaitForScene"), "Scene data already fresh — proceeding");
    return BT::NodeStatus::SUCCESS;
  }
  RCLCPP_INFO(rclcpp::get_logger("WaitForScene"),
    "Waiting for /world_map_result + /grasp_candidates ...");
  return BT::NodeStatus::RUNNING;
}

BT::NodeStatus WaitForScene::onRunning()
{
  std::lock_guard<std::mutex> lk(scene_->mtx);
  if (is_fresh()) {
    RCLCPP_INFO(rclcpp::get_logger("WaitForScene"), "Scene data received — proceeding");
    return BT::NodeStatus::SUCCESS;
  }
  return BT::NodeStatus::RUNNING;
}

}  // namespace bt_pkg
