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

  // Steer off whatever is already standing there. The geometric offset knows
  // only directions and footprint sizes, so "the left side of the table" is
  // just as happy to land on the apple.
  {
    std::vector<TabletopObstacle> others;
    {
      std::lock_guard<std::mutex> lk(scene_->mtx);
      others.reserve(scene_->obstacles.size());
      for (const auto& o : scene_->obstacles) {
        // The extractor separates by height, not identity, so the target and
        // the destination appear in this list too. The target is in the gripper
        // by now and its old footprint is stale; the destination is where we
        // are deliberately aiming (a basket to drop into).
        if (target.centroid_valid
            && dist3(o.centroid, target.centroid) < o.xy_radius + 0.02) continue;
        if (dest.centroid_valid
            && dist3(o.centroid, dest.centroid) < o.xy_radius + 0.02) continue;
        others.push_back(o);
      }
    }
    const auto h = target.half_extent();
    const double obj_r = std::max(h[0], h[1]);
    const auto fs = nudge_to_free_space(
      place_pose.pose.position.x, place_pose.pose.position.y, obj_r, others,
      params_.place_clearance_m, params_.place_search_radius_m, params_.max_place_reach_m);

    if (!fs.found) {
      // Deliberately not a FAILURE: the arm is holding the object and dropping
      // the place phase leaves it held, which is recoverable. Placing it on top
      // of something is not.
      RCLCPP_WARN(rclcpp::get_logger("UpdateTargetPose"),
        "no clear spot within %.2f m of (%.3f, %.3f) — %zu obstacles. Placing "
        "at the requested point anyway; it may land on something.",
        params_.place_search_radius_m, place_pose.pose.position.x,
        place_pose.pose.position.y, others.size());
    } else if (fs.moved) {
      RCLCPP_INFO(rclcpp::get_logger("UpdateTargetPose"),
        "  clutter: shifted %.3f m off (%.3f, %.3f) → (%.3f, %.3f), "
        "%zu obstacles, object r=%.3f",
        std::hypot(fs.x - place_pose.pose.position.x,
                   fs.y - place_pose.pose.position.y),
        place_pose.pose.position.x, place_pose.pose.position.y,
        fs.x, fs.y, others.size(), obj_r);
    }
    place_pose.pose.position.x = fs.x;
    place_pose.pose.position.y = fs.y;
  }

  // Stamp with current time so MoveIt accepts it
  place_pose.header.stamp = rclcpp::Clock().now();

  // Lift off the release point. The pick's retreat_pose sits over where the
  // object was PICKED UP — reusing it drags the open gripper across the scene.
  auto post_place_pose = retract_along_approach(place_pose, retreat_z_offset_);

  // MoveAction's min_path_z constrains link8, but the thing that has to clear
  // the scene is the object dangling below it: a book hangs `lift` below link8,
  // so a constant 0.15 dragged its underside along at table height and through
  // the glass. The floor that gives the OBJECT carry_clearance_m is therefore
  // per-object.
  const double lift      = carry_lift(target, grasp_pose);
  const double transit_z = params_.carry_clearance_m + lift;

  // ...but that floor is a box constraint over the WHOLE path, target included,
  // and the release point is deliberately BELOW it — 0.277 transit against a
  // 0.206 release. Constraining the descent to stay above its own destination
  // is unsatisfiable, and the planner can only report ACTION_ABORTED.
  //
  // So split the move exactly as the pick does (pre_grasp → grasp): fly across
  // at transit height, then descend straight down with no floor at all. The
  // constraint now applies only to the leg that has to clear the scene.
  double up = transit_z - place_pose.pose.position.z;
  if (up < retreat_z_offset_) up = retreat_z_offset_;
  auto pre_place_pose = retract_along_approach(place_pose, up);

  bb.set<geometry_msgs::msg::PoseStamped>("pre_place_pose",  pre_place_pose);
  bb.set<geometry_msgs::msg::PoseStamped>("place_pose",      place_pose);
  bb.set<geometry_msgs::msg::PoseStamped>("post_place_pose", post_place_pose);
  bb.set<double>("carry_path_z", transit_z);

  RCLCPP_INFO(rclcpp::get_logger("UpdateTargetPose"),
    "Place pose: type='%s' rel='%s%s' dest_top_z=%.3f → (%.3f, %.3f, %.3f)",
    spec.type.c_str(),
    spec.relation.empty() ? spec.region.c_str() : spec.relation.c_str(),
    spec.reference_label.empty() ? "" : (" of '" + spec.reference_label + "'").c_str(),
    dest.bbox_valid ? dest.bbox_max[2] : dest.centroid[2],
    place_pose.pose.position.x,
    place_pose.pose.position.y,
    place_pose.pose.position.z);

  // The XY shift is no longer a constant: named directions clear both
  // footprints, and surface regions scale with the destination's. Both read
  // the measured bbox, which for a table is the top-view mask clipped by the
  // workspace crop — print the half-extent that drove it so a nonsense
  // footprint is visible here rather than inferred from where the object lands.
  {
    const auto h = dest.half_extent();
    RCLCPP_INFO(rclcpp::get_logger("UpdateTargetPose"),
      "  dest half-extent (%.3f, %.3f) → applied XY offset (%+.3f, %+.3f)",
      h[0], h[1],
      place_pose.pose.position.x - dest.centroid[0],
      place_pose.pose.position.y - dest.centroid[1]);

    // Print the reach the pose actually asks for, next to the ceiling that
    // bounds it. A clamped shift is otherwise invisible until the arm stops
    // short of where the instruction pointed.
    const double reach = std::hypot(place_pose.pose.position.x,
                                    place_pose.pose.position.y);
    RCLCPP_INFO(rclcpp::get_logger("UpdateTargetPose"),
      "  XY reach %.3f m (max %.3f%s) · carry lift %.3f → transit z %.3f "
      "(object underside %.3f), then descend %.3f to release",
      reach, params_.max_place_reach_m,
      reach >= params_.max_place_reach_m - 1e-3 ? ", CLAMPED" : "",
      lift, transit_z, transit_z - lift,
      pre_place_pose.pose.position.z - place_pose.pose.position.z);
  }

  return BT::NodeStatus::SUCCESS;
}

}  // namespace bt_pkg
