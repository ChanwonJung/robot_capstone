#pragma once

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>

#include "bt_pkg/scene_data.hpp"

namespace bt_pkg {

// From bt_params.yaml. Offsets describe the RELEASED OBJECT, not the link8 pose.
struct PlacePoseParams {
  double side_offset_m     = 0.08;  // XY shift for left_of / right_of / etc.
  double place_height_m    = 0.05;  // clearance under the object before release
  double container_drop_z  = 0.03;  // clearance of the object's underside over the rim
  double near_offset_m     = 0.08;  // minimum radial "near" offset
  double near_clearance_m  = 0.05;  // gap between the two XY footprints, all directions
  // A surface region ("the right side of the table") is a fraction of the
  // destination's OWN footprint, not a fixed shift — 8 cm from a table's
  // centroid is still the table's middle. Clamped because the far side of a
  // large table is outside the arm's reach.
  double region_frac       = 0.6;   // how far toward the named edge, 0..1 of half-extent
  double region_max_offset_m = 0.45;  // ceiling on that fraction
  // Hard XY envelope for the released pose. The ceiling above bounds the shift;
  // this bounds the RESULT, which is what the arm actually has to reach, and it
  // is what lets region_max_offset_m be generous. Measured: a place at XY radius
  // 0.777 succeeded, 0.853 aborted five times (Panda spec reach 0.855).
  double max_place_reach_m = 0.80;
  // Clearance under the CARRIED OBJECT along the transit path, over the tallest
  // thing it flies across. Converted to a link8 floor by adding the carry lift.
  double carry_clearance_m = 0.12;
  // Gap to leave between the placed object's footprint and any tabletop
  // clutter, and how far to look for a clear spot before giving up.
  double place_clearance_m     = 0.04;
  double place_search_radius_m = 0.20;
  // Keep the pick's yaw so the wrist doesn't twist the held object in transit.
  // false = fixed straight-down, for when a place is unreachable.
  bool   keep_grasp_yaw    = true;
};

// Nudge a place pose off the tabletop clutter, keeping it as close to `ideal`
// as possible. Returns the adjusted XY, or `ideal` unchanged when nothing is
// in the way — or when nothing clear exists within `search_radius_m`, in which
// case `moved` is set false so the caller can say so rather than pretend.
//
// The ideal point is a pure geometric offset and knows nothing about what is
// already sitting there: "the left side of the table" happily lands on the
// apple. Obstacles are XY footprints (see TabletopObstacle) because the
// overhead scan measures footprints reliably and heights only sometimes.
//
// Search is a widening ring scan, so the first free point found is the closest
// one — no gradient, no local minima, and deterministic for a given scan.
struct FreeSpaceResult {
  double x = 0.0;
  double y = 0.0;
  bool   moved = false;      // an obstacle was avoided
  bool   found = true;       // false = gave up, x/y are the ideal point
};

FreeSpaceResult nudge_to_free_space(
  double ideal_x, double ideal_y,
  double object_radius,                            // the carried object's own footprint
  const std::vector<TabletopObstacle>& obstacles,
  double clearance_m,
  double search_radius_m,
  double max_reach_m,
  double step_m = 0.02);

// How far the object's underside hangs below link8 while carried:
// (link8 above the object's centre at closing time) + (its half-height).
// Both clamped at 0, so an unmeasured object contributes nothing rather than
// lifting the path by a negative amount.
//
// A path constraint on link8 is NOT a clearance for the object: with a book
// this is 0.157 m, so min_path_z 0.15 dragged the book's underside along at
// z = -0.007 — table height — straight through a 0.1 m glass on the way.
double carry_lift(const ObjectGeometry&                  target,
                  const geometry_msgs::msg::PoseStamped& grasp_pose);

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
