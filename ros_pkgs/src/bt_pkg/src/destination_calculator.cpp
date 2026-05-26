#include "bt_pkg/destination_calculator.hpp"

#include <cmath>

namespace bt_pkg {

geometry_msgs::msg::PoseStamped compute_place_pose(
  const DestinationSpec&       spec,
  const std::array<double, 3>& dest_centroid,
  const PlacePoseParams&       params)
{
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = "panda_link0";

  double x = dest_centroid[0];
  double y = dest_centroid[1];
  double z = dest_centroid[2];

  if (spec.type == "container") {
    // Drop object slightly above container centroid.
    z += params.container_drop_z;

  } else if (spec.type == "surface") {
    z += params.place_height_m;
    const auto& r = spec.region;
    if      (r == "left_edge")  x -= params.side_offset_m;
    else if (r == "right_edge") x += params.side_offset_m;
    else if (r == "far_end")    y += params.side_offset_m;
    else if (r == "near_end")   y -= params.side_offset_m;
    // "center" → no XY offset

  } else if (spec.type == "relation") {
    const auto& rel = spec.relation;
    if      (rel == "left_of")     { z += params.place_height_m; x -= params.side_offset_m; }
    else if (rel == "right_of")    { z += params.place_height_m; x += params.side_offset_m; }
    else if (rel == "in_front_of") { z += params.place_height_m; y -= params.side_offset_m; }
    else if (rel == "behind")      { z += params.place_height_m; y += params.side_offset_m; }
    else if (rel == "on_top_of")   { z += params.place_height_m * 2.0; }  // stack on top
    else if (rel == "near")        { z += params.place_height_m; x += params.near_offset_m; }
    else                           { z += params.place_height_m; }  // unknown → just lift
  }

  pose.pose.position.x = x;
  pose.pose.position.y = y;
  pose.pose.position.z = z;

  // Gripper pointing straight down: RPY = (π, 0, 0)
  // quaternion = (w=0, x=1, y=0, z=0)
  pose.pose.orientation.w = 0.0;
  pose.pose.orientation.x = 1.0;
  pose.pose.orientation.y = 0.0;
  pose.pose.orientation.z = 0.0;

  return pose;
}

geometry_msgs::msg::PoseStamped lift_z(
  const geometry_msgs::msg::PoseStamped& in, double dz)
{
  auto out = in;
  out.pose.position.z += dz;
  return out;
}

}  // namespace bt_pkg
