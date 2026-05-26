#include "bt_pkg/action_nodes.hpp"
#include "bt_pkg/destination_calculator.hpp"

namespace bt_pkg {

ParseScene::ParseScene(const std::string& name,
                       const BT::NodeConfig& config,
                       std::shared_ptr<SceneData> scene,
                       double pre_grasp_z_offset,
                       double retreat_z_offset)
  : BT::SyncActionNode(name, config)
  , scene_(std::move(scene))
  , pre_grasp_z_offset_(pre_grasp_z_offset)
  , retreat_z_offset_(retreat_z_offset)
{}

BT::NodeStatus ParseScene::tick()
{
  auto& bb = *config().blackboard;
  std::vector<GraspCandidate> candidates;
  std::array<double, 3> target_centroid      = {};
  std::array<double, 3> destination_centroid = {};
  std::string           target_label;
  DestinationSpec       destination_spec;
  rclcpp::Time          world_map_stamp;

  {
    std::lock_guard<std::mutex> lk(scene_->mtx);
    candidates         = scene_->grasp_candidates;
    target_centroid    = scene_->target_centroid;
    destination_centroid = scene_->destination_centroid;
    target_label       = scene_->target_label;
    destination_spec   = scene_->destination_spec;
    world_map_stamp    = scene_->world_map_stamp;

    // Stamp the data so WaitForScene won't re-trigger on the same batch.
    scene_->last_processed_stamp   = world_map_stamp;
    scene_->world_map_fresh        = false;
    scene_->grasp_candidates_fresh = false;
    scene_->awaiting_replan        = false;
  }

  // If VGN returned no candidates, fall back to a single centroid-based pose
  // with a down-pointing gripper. This keeps the pipeline alive while VGN
  // is still being tuned. (Experimental multi-view recovery is tracked separately.)
  if (candidates.empty()) {
    RCLCPP_WARN(rclcpp::get_logger("ParseScene"),
      "No VGN candidates — falling back to centroid pose");
    GraspCandidate fallback;
    fallback.pose.header.frame_id = "panda_link0";
    fallback.pose.pose.position.x = target_centroid[0];
    fallback.pose.pose.position.y = target_centroid[1];
    fallback.pose.pose.position.z = target_centroid[2];
    // Gripper pointing down: RPY = (π, 0, 0)
    fallback.pose.pose.orientation.w = 0.0;
    fallback.pose.pose.orientation.x = 1.0;
    fallback.pose.pose.orientation.y = 0.0;
    fallback.pose.pose.orientation.z = 0.0;
    fallback.quality = 0.0;
    fallback.width   = 0.08;
    candidates.push_back(fallback);
  }

  // Pre-compute retreat pose: first candidate z + retreat_offset.
  auto retreat_pose = lift_z(candidates[0].pose, retreat_z_offset_);

  // Write to blackboard
  bb.set<std::vector<GraspCandidate>>("grasp_candidates",       candidates);
  bb.set<int>                        ("grasp_index",            0);
  bb.set<std::array<double,3>>       ("target_centroid",        target_centroid);
  bb.set<std::string>                ("target_label",           target_label);
  bb.set<DestinationSpec>            ("destination_spec",       destination_spec);
  bb.set<std::array<double,3>>       ("destination_centroid_init", destination_centroid);
  bb.set<geometry_msgs::msg::PoseStamped>("retreat_pose",       retreat_pose);

  RCLCPP_INFO(rclcpp::get_logger("ParseScene"),
    "Parsed scene: target='%s' dest_type='%s' candidates=%zu",
    target_label.c_str(), destination_spec.type.c_str(), candidates.size());

  return BT::NodeStatus::SUCCESS;
}

}  // namespace bt_pkg
