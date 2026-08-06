#include "bt_pkg/action_nodes.hpp"
#include "bt_pkg/destination_calculator.hpp"

#include <algorithm>
#include <cstdio>

namespace bt_pkg {

ParseScene::ParseScene(const std::string& name,
                       const BT::NodeConfig& config,
                       std::shared_ptr<SceneData> scene,
                       const std::vector<std::string>& arm_joint_names,
                       bool latch_observation_pose)
  : BT::SyncActionNode(name, config)
  , scene_(std::move(scene))
  , arm_joint_names_(arm_joint_names)
  , latch_observation_pose_(latch_observation_pose)
{}

// The arm has not moved yet on the first scan, so the current configuration is
// the one the cameras just observed from — the pose MoveToObservation must
// return to. Latched once and never updated: later cycles start from the
// returned pose, and re-latching would let drift accumulate.
void ParseScene::latch_observation_pose()
{
  if (!latch_observation_pose_ || scene_->has_observation_pose) return;
  if (!scene_->has_joint_state) return;

  const auto& js = scene_->latest_joint_state;
  std::vector<double> vals;
  for (const auto& want : arm_joint_names_) {
    // /joint_states also carries the fingers, and the order is not guaranteed.
    auto it = std::find(js.name.begin(), js.name.end(), want);
    if (it == js.name.end()) {
      RCLCPP_WARN(rclcpp::get_logger("ParseScene"),
        "Cannot latch observation pose: '%s' missing from /joint_states",
        want.c_str());
      return;
    }
    vals.push_back(js.position[std::distance(js.name.begin(), it)]);
  }

  scene_->observation_joint_values = vals;
  scene_->has_observation_pose     = true;

  std::string joined;
  for (size_t i = 0; i < vals.size(); ++i) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%s%.4f", i ? ", " : "", vals[i]);
    joined += buf;
  }
  RCLCPP_INFO(rclcpp::get_logger("ParseScene"),
    "Observation pose latched from the first scan: [%s]", joined.c_str());
}

BT::NodeStatus ParseScene::tick()
{
  auto& bb = *config().blackboard;
  std::vector<GraspCandidate> candidates;
  ObjectGeometry        target_geom;
  ObjectGeometry        dest_geom;
  std::string           target_label;
  std::string           destination_label;
  DestinationSpec       destination_spec;
  bool                  has_destination = false;
  rclcpp::Time          world_map_stamp;

  {
    std::lock_guard<std::mutex> lk(scene_->mtx);
    candidates         = scene_->grasp_candidates;
    target_geom        = scene_->target;
    dest_geom          = scene_->destination;
    target_label       = scene_->target_label;
    destination_label  = scene_->destination_label;
    destination_spec   = scene_->destination_spec;
    has_destination    = scene_->has_destination;
    world_map_stamp    = scene_->world_map_stamp;

    latch_observation_pose();

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
    fallback.pose.pose.position.x = target_geom.centroid[0];
    fallback.pose.pose.position.y = target_geom.centroid[1];
    fallback.pose.pose.position.z = target_geom.centroid[2];
    // Gripper pointing down: RPY = (π, 0, 0)
    fallback.pose.pose.orientation.w = 0.0;
    fallback.pose.pose.orientation.x = 1.0;
    fallback.pose.pose.orientation.y = 0.0;
    fallback.pose.pose.orientation.z = 0.0;
    fallback.quality = 0.0;
    fallback.width   = 0.08;
    candidates.push_back(fallback);
  }

  // retreat_pose belongs to SelectGraspCandidate — it must follow the candidate
  // actually selected, not candidates[0] on every retry.

  // Place gates on the measured centroid, not on the spec: "put it in the box"
  // with no box in view fills the spec and leaves the centroid at {0,0,0}.
  const bool has_destination_centroid = has_destination && dest_geom.centroid_valid;

  // Write to blackboard
  bb.set<std::vector<GraspCandidate>>("grasp_candidates",       candidates);
  bb.set<int>                        ("grasp_index",            0);
  bb.set<std::array<double,3>>       ("target_centroid",        target_geom.centroid);
  bb.set<ObjectGeometry>             ("target_geometry",        target_geom);
  bb.set<std::string>                ("target_label",           target_label);
  bb.set<DestinationSpec>            ("destination_spec",       destination_spec);
  bb.set<ObjectGeometry>             ("destination_geometry",   dest_geom);
  bb.set<bool>                       ("has_destination_centroid", has_destination_centroid);

  RCLCPP_INFO(rclcpp::get_logger("ParseScene"),
    "Parsed scene: target='%s' dest='%s' dest_type='%s' candidates=%zu place=%s",
    target_label.c_str(), destination_label.c_str(),
    destination_spec.type.c_str(), candidates.size(),
    has_destination_centroid ? "enabled" : "SKIPPED (no destination centroid)");

  if (has_destination && !dest_geom.centroid_valid) {
    RCLCPP_WARN(rclcpp::get_logger("ParseScene"),
      "Destination '%s' named but /world_map_result carried no centroid — was "
      "it in the top camera's view? Place skipped; the pick still runs.",
      destination_spec.reference_label.c_str());
  }

  return BT::NodeStatus::SUCCESS;
}

}  // namespace bt_pkg
