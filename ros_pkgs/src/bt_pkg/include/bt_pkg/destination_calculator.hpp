#pragma once

#include <array>
#include <cmath>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>

#include "bt_pkg/scene_data.hpp"

namespace bt_pkg {

// From bt_params.yaml. Offsets describe the RELEASED OBJECT, not the link8 pose.
struct PlacePoseParams {
  double side_offset_m     = 0.08;  // XY shift for left_of / right_of / etc.
  double place_height_m    = 0.05;  // clearance under the object before release
  double container_drop_z  = 0.03;  // clearance of the object's underside over the rim
  double near_offset_m     = 0.08;  // minimum radial "near" offset
  double near_clearance_m  = 0.05;  // gap between the two XY footprints for "near"
  // Keep the pick's yaw so the wrist doesn't twist the held object in transit.
  // false = fixed straight-down, for when a place is unreachable.
  bool   keep_grasp_yaw    = true;
};

// DestinationSpec + measured scene → panda_link8 pose to reach before release.
// Frame is always "panda_link0". See CLAUDE.md "Place phase".
//
// Z is measured, not assumed: (grasp_pose.z - target.centroid.z) is how high
// link8 rides above the object's centre while carrying it, and
// (target.centroid.z - target.bbox_min.z) its half-height. Both are added to
// the desired underside height, so one clearance value fits book and cup alike.
// Degrades to centroid + place_height_m when a bbox/centroid is missing.
//
// Caller must still gate on dest.centroid_valid — see HasDestinationCentroid.
geometry_msgs::msg::PoseStamped compute_place_pose(
  const DestinationSpec&                 spec,
  const ObjectGeometry&                  dest,
  const ObjectGeometry&                  target,
  const geometry_msgs::msg::PoseStamped& grasp_pose,
  const PlacePoseParams&                 params);

// Euclidean distance between two 3-element arrays.
inline double dist3(const std::array<double, 3>& a, const std::array<double, 3>& b)
{
  double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// Back the pose off by `dist` along its OWN approach axis (the gripper frame's
// +Z, so the retracted pose sits behind the jaws), orientation unchanged.
//
// For a straight top-down grasp the approach is world -Z, so this is exactly
// "raise z by dist" — the behaviour this replaced. It differs only when the
// grasp is tilted (graspgen's align_tilt profile): there, lifting along world Z
// puts the pre-grasp off the gripper's own axis, so the descent enters the
// object at an angle instead of sliding down alongside it.
geometry_msgs::msg::PoseStamped retract_along_approach(
  const geometry_msgs::msg::PoseStamped& in, double dist);

}  // namespace bt_pkg
