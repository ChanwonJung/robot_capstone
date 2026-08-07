#include "bt_pkg/action_nodes.hpp"
#include "bt_pkg/destination_calculator.hpp"

namespace bt_pkg {

UpdateTargetPose::UpdateTargetPose(const std::string& name,
                                   const BT::NodeConfig& config,
                                   std::shared_ptr<SceneData> scene,
                                   const PlacePoseParams& params,
                                   double dest_match_radius_m,
                                   double retreat_z_offset)
  : BT::SyncActionNode(name, config)
  , scene_(std::move(scene))
  , params_(params)
  , dest_match_radius_m_(dest_match_radius_m)
  , retreat_z_offset_(retreat_z_offset)
{}

BT::NodeStatus UpdateTargetPose::tick()
{
  auto& bb = *config().blackboard;

  auto spec       = bb.get<DestinationSpec>("destination_spec");
  auto dest       = bb.get<ObjectGeometry> ("destination_geometry");
  auto target     = bb.get<ObjectGeometry> ("target_geometry");
  auto grasp_pose = bb.get<geometry_msgs::msg::PoseStamped>("grasp_pose");

  // HasDestinationCentroid already gates this in the tree; belt and braces so a
  // hand-edited XML can't aim the arm at the robot's own base.
  if (!dest.centroid_valid) {
    RCLCPP_ERROR(rclcpp::get_logger("UpdateTargetPose"),
      "No destination centroid — refusing to compute a place pose");
    return BT::NodeStatus::FAILURE;
  }

  // Fresher 3D position from the live YOLO world map: match by class_name, then
  // nearest to the scan's centroid. The bbox rides the same translation so the
  // rim height tracks the move.
  {
    std::lock_guard<std::mutex> lk(scene_->mtx);
    double best_d = dest_match_radius_m_;
    const YoloObject* best = nullptr;
    for (const auto& obj : scene_->yolo_objects) {
      if (obj.class_name != spec.reference_label) continue;
      double d = dist3(obj.centroid, dest.centroid);
      if (d < best_d) { best_d = d; best = &obj; }
    }
    if (best) {
      for (int i = 0; i < 3; ++i) {
        const double delta = best->centroid[i] - dest.centroid[i];
        dest.centroid[i] = best->centroid[i];
        dest.bbox_min[i] += delta;
        dest.bbox_max[i] += delta;
      }
    }
  }

  auto place_pose = compute_place_pose(spec, dest, target, grasp_pose, params_);
  // Stamp with current time so MoveIt accepts it
  place_pose.header.stamp = rclcpp::Clock().now();

  // Lift off the release point. The pick's retreat_pose sits over where the
  // object was PICKED UP — reusing it drags the open gripper across the scene.
  auto post_place_pose = retract_along_approach(place_pose, retreat_z_offset_);

  bb.set<geometry_msgs::msg::PoseStamped>("place_pose",      place_pose);
  bb.set<geometry_msgs::msg::PoseStamped>("post_place_pose", post_place_pose);

  RCLCPP_INFO(rclcpp::get_logger("UpdateTargetPose"),
    "Place pose: type='%s' rel='%s%s' dest_top_z=%.3f → (%.3f, %.3f, %.3f)",
    spec.type.c_str(),
    spec.relation.empty() ? spec.region.c_str() : spec.relation.c_str(),
    spec.reference_label.empty() ? "" : (" of '" + spec.reference_label + "'").c_str(),
    dest.bbox_valid ? dest.bbox_max[2] : dest.centroid[2],
    place_pose.pose.position.x,
    place_pose.pose.position.y,
    place_pose.pose.position.z);

  return BT::NodeStatus::SUCCESS;
}

}  // namespace bt_pkg
