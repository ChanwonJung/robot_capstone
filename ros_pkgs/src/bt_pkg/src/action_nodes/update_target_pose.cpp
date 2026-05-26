#include "bt_pkg/action_nodes.hpp"
#include "bt_pkg/destination_calculator.hpp"

namespace bt_pkg {

UpdateTargetPose::UpdateTargetPose(const std::string& name,
                                   const BT::NodeConfig& config,
                                   std::shared_ptr<SceneData> scene,
                                   const PlacePoseParams& params,
                                   double dest_match_radius_m)
  : BT::SyncActionNode(name, config)
  , scene_(std::move(scene))
  , params_(params)
  , dest_match_radius_m_(dest_match_radius_m)
{}

BT::NodeStatus UpdateTargetPose::tick()
{
  auto& bb = *config().blackboard;

  auto spec      = bb.get<DestinationSpec>           ("destination_spec");
  auto dest_init = bb.get<std::array<double, 3>>     ("destination_centroid_init");

  // Try to get a fresher 3D position from the live YOLO world map.
  // Match by class_name first, then nearest to the initial centroid.
  std::array<double, 3> dest_centroid = dest_init;
  {
    std::lock_guard<std::mutex> lk(scene_->mtx);
    double best_d = dest_match_radius_m_;
    for (const auto& obj : scene_->yolo_objects) {
      if (obj.class_name != spec.reference_label) continue;
      double d = dist3(obj.centroid, dest_init);
      if (d < best_d) {
        best_d        = d;
        dest_centroid = obj.centroid;
      }
    }
  }

  auto place_pose = compute_place_pose(spec, dest_centroid, params_);
  // Stamp with current time so MoveIt accepts it
  place_pose.header.stamp = rclcpp::Clock().now();

  bb.set<geometry_msgs::msg::PoseStamped>("place_pose", place_pose);

  RCLCPP_INFO(rclcpp::get_logger("UpdateTargetPose"),
    "Place pose: type='%s' rel='%s%s' → (%.3f, %.3f, %.3f)",
    spec.type.c_str(),
    spec.relation.empty() ? spec.region.c_str() : spec.relation.c_str(),
    spec.reference_label.empty() ? "" : (" of '" + spec.reference_label + "'").c_str(),
    place_pose.pose.position.x,
    place_pose.pose.position.y,
    place_pose.pose.position.z);

  return BT::NodeStatus::SUCCESS;
}

}  // namespace bt_pkg
