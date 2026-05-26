#pragma once

#include <array>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>

#include "bt_pkg/scene_data.hpp"

namespace bt_pkg {

// Parameters loaded from bt_params.yaml and passed to compute_place_pose().
struct PlacePoseParams {
  double side_offset_m    = 0.08;   // XY shift for left_of / right_of / etc.
  double place_height_m   = 0.05;   // Z above destination centroid
  double container_drop_z = 0.03;   // Z above container centroid
  double near_offset_m    = 0.08;   // XY shift for "near"
};

// Pure geometry: convert a DestinationSpec + destination centroid into a
// PoseStamped the arm should move to before releasing the object.
// Frame is always "panda_link0".
// Orientation: gripper pointing down (RPY = π, 0, 0).
geometry_msgs::msg::PoseStamped compute_place_pose(
  const DestinationSpec&       spec,
  const std::array<double, 3>& dest_centroid,
  const PlacePoseParams&       params);

// Euclidean distance between two 3-element arrays.
inline double dist3(const std::array<double, 3>& a, const std::array<double, 3>& b)
{
  double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// Return pose with z raised by dz (all other fields unchanged).
geometry_msgs::msg::PoseStamped lift_z(
  const geometry_msgs::msg::PoseStamped& in, double dz);

}  // namespace bt_pkg
