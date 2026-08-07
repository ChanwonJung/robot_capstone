#include "bt_pkg/destination_calculator.hpp"

#include <algorithm>
#include <cmath>

namespace bt_pkg {
namespace {

// Box's XY support function: centre → boundary along unit (ux, uy).
// Zero without a bbox, degrading the caller to a fixed offset.
double xy_support(const ObjectGeometry& g, double ux, double uy)
{
  const auto h = g.half_extent();
  return std::abs(ux) * h[0] + std::abs(uy) * h[1];
}

// Lid of the object — a container's rim.
double top_z(const ObjectGeometry& g)
{
  return g.bbox_valid ? g.bbox_max[2] : g.centroid[2];
}

// Floor of the object = the surface it stands on, where "beside it" belongs.
double base_z(const ObjectGeometry& g)
{
  return g.bbox_valid ? g.bbox_min[2] : g.centroid[2];
}

// Exact for q = Rz(psi)·Rx(pi); nearest upright yaw for align_tilt grasps.
double yaw_of(const geometry_msgs::msg::Quaternion& q)
{
  return std::atan2(2.0 * (q.x * q.y + q.w * q.z),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

// Rz(yaw) · Rx(pi) — straight down, twisted by `yaw` about world Z.
geometry_msgs::msg::Quaternion down_with_yaw(double yaw)
{
  geometry_msgs::msg::Quaternion q;
  q.w = 0.0;
  q.x = std::cos(0.5 * yaw);
  q.y = std::sin(0.5 * yaw);
  q.z = 0.0;
  return q;
}

}  // namespace

geometry_msgs::msg::PoseStamped compute_place_pose(
  const DestinationSpec&                 spec,
  const ObjectGeometry&                  dest,
  const ObjectGeometry&                  target,
  const geometry_msgs::msg::PoseStamped& grasp_pose,
  const PlacePoseParams&                 params)
{
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = "panda_link0";

  // How the object sits in the gripper. hold_offset = link8 above the object's
  // centre while carrying; half_height = centre above its own underside. Both
  // 0 when unmeasured, which puts link8 itself where the object should go.
  double hold_offset = 0.0;
  double half_height = 0.0;
  if (target.centroid_valid) {
    hold_offset = grasp_pose.pose.position.z - target.centroid[2];
    if (target.bbox_valid) {
      half_height = target.centroid[2] - target.bbox_min[2];
    }
  }
  // Negative = link8 below the object's centre at closing time: impossible for
  // a top-down grasp, so the scan and the grasp disagree. Don't drive down.
  hold_offset = std::max(0.0, hold_offset);
  half_height = std::max(0.0, half_height);

  double x = dest.centroid[0];
  double y = dest.centroid[1];
  // The released object's UNDERSIDE. Converted to a link8 height at the end.
  double underside_z = dest.centroid[2] + params.place_height_m;

  // Offset to the side the object is already on: shortest carry, and the arm
  // never crosses over the destination. Without bboxes the supports are 0 and
  // this collapses to near_offset_m — only now in the right direction.
  auto offset_toward_target = [&]() {
    double dx = target.centroid[0] - dest.centroid[0];
    double dy = target.centroid[1] - dest.centroid[1];
    double n  = std::hypot(dx, dy);
    if (!target.centroid_valid || n < 1e-6) {
      x += params.near_offset_m;  // no direction — historical fixed +X shove
      return;
    }
    const double ux = dx / n, uy = dy / n;
    double d = xy_support(dest, ux, uy) + xy_support(target, ux, uy)
             + params.near_clearance_m;
    if (d < params.near_offset_m) d = params.near_offset_m;
    x += ux * d;
    y += uy * d;
  };

  if (spec.type == "container") {
    // Rim, not centroid — a basket's centroid sits ~90 mm below its opening,
    // so centroid + 3 cm released into the wall.
    underside_z = top_z(dest) + params.container_drop_z;

  } else if (spec.type == "surface") {
    underside_z = top_z(dest) + params.place_height_m;
    const auto& r = spec.region;
    if      (r == "left_edge")  x -= params.side_offset_m;
    else if (r == "right_edge") x += params.side_offset_m;
    else if (r == "far_end")    y += params.side_offset_m;
    else if (r == "near_end")   y -= params.side_offset_m;
    // "center" → no XY offset

  } else if (spec.type == "relation") {
    const auto& rel = spec.relation;
    if (rel == "on_top_of") {
      underside_z = top_z(dest) + params.place_height_m;
    } else {
      // Beside the reference → lands on whatever surface the reference is on.
      underside_z = base_z(dest) + params.place_height_m;
      if      (rel == "left_of")     x -= params.side_offset_m;
      else if (rel == "right_of")    x += params.side_offset_m;
      else if (rel == "in_front_of") y -= params.side_offset_m;
      else if (rel == "behind")      y += params.side_offset_m;
      else                           offset_toward_target();  // "near" + unknown
    }

  } else {
    // Unknown/empty type — same under-specified case "near" covers. Used to
    // take NO branch and return the raw centroid, burying the object inside.
    underside_z = base_z(dest) + params.place_height_m;
    offset_toward_target();
  }

  pose.pose.position.x = x;
  pose.pose.position.y = y;
  pose.pose.position.z = underside_z + half_height + hold_offset;

  pose.pose.orientation = params.keep_grasp_yaw
    ? down_with_yaw(yaw_of(grasp_pose.pose.orientation))
    : down_with_yaw(0.0);  // (w=0, x=1, y=0, z=0) — RPY (pi, 0, 0)

  return pose;
}

geometry_msgs::msg::PoseStamped retract_along_approach(
  const geometry_msgs::msg::PoseStamped& in, double dist)
{
  auto out = in;
  // Gripper frame +Z is the approach direction, so backing off is -Z local.
  // Rotate (0, 0, -dist) into the world frame with the pose's own quaternion:
  //   v' = v + 2 * qv x (qv x v + w*v)
  const auto& q = in.pose.orientation;
  const double vx = 0.0, vy = 0.0, vz = -dist;
  const double tx = 2.0 * (q.y * vz - q.z * vy);
  const double ty = 2.0 * (q.z * vx - q.x * vz);
  const double tz = 2.0 * (q.x * vy - q.y * vx);
  out.pose.position.x += vx + q.w * tx + (q.y * tz - q.z * ty);
  out.pose.position.y += vy + q.w * ty + (q.z * tx - q.x * tz);
  out.pose.position.z += vz + q.w * tz + (q.x * ty - q.y * tx);
  return out;
}

}  // namespace bt_pkg
