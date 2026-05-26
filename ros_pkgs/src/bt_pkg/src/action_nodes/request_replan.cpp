#include "bt_pkg/action_nodes.hpp"

namespace bt_pkg {

RequestReplan::RequestReplan(
  const std::string& name,
  const BT::NodeConfig& config,
  std::shared_ptr<SceneData> scene,
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr pub)
  : BT::SyncActionNode(name, config)
  , scene_(std::move(scene))
  , pub_(std::move(pub))
{}

BT::NodeStatus RequestReplan::tick()
{
  // Signal the Slow Brain to re-run GSAM + Qwen.
  pub_->publish(std_msgs::msg::Empty{});

  // Invalidate cached scene so WaitForScene blocks until new data arrives.
  {
    std::lock_guard<std::mutex> lk(scene_->mtx);
    scene_->world_map_fresh        = false;
    scene_->grasp_candidates_fresh = false;
    scene_->awaiting_replan        = true;
    // Reset last_processed_stamp to a zero time so even if timestamps happen to
    // match the old batch, WaitForScene will re-block correctly.
    scene_->last_processed_stamp   = rclcpp::Time(0, 0, RCL_ROS_TIME);
  }

  RCLCPP_INFO(rclcpp::get_logger("RequestReplan"),
    "Published /bt/replan_request — waiting for fresh scene data");

  // Intentional FAILURE: causes the enclosing Sequence (Pick Recovery) and
  // Fallback (Pick or Recover) to both fail, which propagates up through
  // Main Pipeline → RepeatForever re-ticks from WaitForScene.
  return BT::NodeStatus::FAILURE;
}

}  // namespace bt_pkg
