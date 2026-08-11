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

// The plane a surface region places onto — read off the TARGET, which is
// already standing on it, never off the surface's own geometry.
//
// No statistic of a surface mask survives. Two live scans of the same table
// minutes apart disagreed in opposite directions, because SAM segmented a
// different thing each time:
//
//   label            px      centroid z    bbox max z
//   "table surface"  196547   0.007 ok      0.517  back wall
//   "table"           23549  -0.768 legs    0.002  ok
//
// (true tabletop: -0.00137). Pick the max and the first scan releases into the
// air; pick the centroid and the second drives 72 cm through the table. The
// target's own underside was within 1.4 mm on BOTH — it is physics, not
// statistics, so mask contamination cannot reach it.
//
// Limitation: an object standing on another object measures that object's top,
// not the table. Acceptable for regions, which are about where on a surface to
// put something already resting on one.
//
// The fallback is top_z, not the centroid: with no measured target both dest
// estimates are unreliable, and releasing too high merely drops the object
// while releasing too low drives the arm into the furniture.
double standing_plane_z(const ObjectGeometry& target, const ObjectGeometry& dest)
{
  return target.bbox_valid ? target.bbox_min[2] : top_z(dest);
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

// link8 above the object's centre while carrying, and the centre above its own
// underside. Both 0 when unmeasured, which puts link8 itself where the object
// should go rather than driving below it.
void carry_split(const ObjectGeometry&                  target,
                 const geometry_msgs::msg::PoseStamped& grasp_pose,
                 double& hold_offset, double& half_height)
{
  hold_offset = 0.0;
  half_height = 0.0;
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
}
}  // namespace

FreeSpaceResult nudge_to_free_space(
  double ideal_x, double ideal_y,
  double object_radius,
  const std::vector<TabletopObstacle>& obstacles,
  double clearance_m,
  double search_radius_m,
  double max_reach_m,
  double step_m)
{
  FreeSpaceResult r{ideal_x, ideal_y, false, true};
  if (obstacles.empty() || search_radius_m <= 0.0) return r;

  auto blocked = [&](double x, double y) {
    for (const auto& o : obstacles) {
      const double need = o.xy_radius + object_radius + clearance_m;
      const double dx = x - o.centroid[0], dy = y - o.centroid[1];
      if (dx * dx + dy * dy < need * need) return true;
    }
    return false;
  };
  auto reachable = [&](double x, double y) {
    return max_reach_m <= 1e-6 || std::hypot(x, y) <= max_reach_m;
  };

  if (!blocked(ideal_x, ideal_y)) return r;      // nothing in the way

  // Widening rings: the first hit is the nearest free point. Arc step is tied
  // to the radial step so the angular sampling stays about as fine as the
  // radial one instead of thinning out as the ring grows.
  if (step_m <= 1e-6) step_m = 0.02;
  for (double rad = step_m; rad <= search_radius_m + 1e-9; rad += step_m) {
    const int n = std::max(8, static_cast<int>(std::ceil(2.0 * M_PI * rad / step_m)));
    for (int i = 0; i < n; ++i) {
      const double a = 2.0 * M_PI * i / n;
      const double x = ideal_x + rad * std::cos(a);
      const double y = ideal_y + rad * std::sin(a);
      if (!reachable(x, y) || blocked(x, y)) continue;
      return FreeSpaceResult{x, y, true, true};
    }
  }
  // Everything within reach of the ideal point is occupied. Report it: placing
  // on top of another object is worse than admitting there is nowhere to put it.
  return FreeSpaceResult{ideal_x, ideal_y, false, false};
}

double carry_lift(const ObjectGeometry&                  target,
                  const geometry_msgs::msg::PoseStamped& grasp_pose)
{
  double hold_offset = 0.0, half_height = 0.0;
  carry_split(target, grasp_pose, hold_offset, half_height);
  return hold_offset + half_height;
}

geometry_msgs::msg::PoseStamped compute_place_pose(
  const DestinationSpec&                 spec,
  const ObjectGeometry&                  dest,
  const ObjectGeometry&                  target,
  const geometry_msgs::msg::PoseStamped& grasp_pose,
  const PlacePoseParams&                 params)
{
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = "panda_link0";

  double hold_offset = 0.0, half_height = 0.0;
  carry_split(target, grasp_pose, hold_offset, half_height);

  double x = dest.centroid[0];
  double y = dest.centroid[1];
  // Where the shift started, for the reach clamp at the end: pulling back along
  // anchor→place preserves the direction the instruction named. Regions
  // overwrite this with the target; relations keep the destination.
  double anchor_x = x, anchor_y = y;
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

  // Beside the destination along a NAMED direction. Same spacing rule as
  // "near" — clear both footprints, then leave a gap — only the direction is
  // dictated by the word rather than by where the target already is. A fixed
  // side_offset_m ignored both bboxes, so "to the right of the book" dropped
  // the cup onto a book wider than 16 cm. It survives as the floor for objects
  // too small or unmeasured for the supports to matter.
  auto place_beside = [&](double ux, double uy) {
    double d = xy_support(dest, ux, uy) + xy_support(target, ux, uy)
             + params.near_clearance_m;
    if (d < params.side_offset_m) d = params.side_offset_m;
    x += ux * d;
    y += uy * d;
  };

  // Shift the object toward a named side of the surface, ANCHORED ON THE
  // TARGET's current position rather than on the destination's centroid.
  //
  // This is deliberately weaker than "put it at the table's left edge", and the
  // weaker reading is the only one the data supports. A surface mask is scene
  // background, so its centroid is wherever SAM's blob happened to land: two
  // live scans of one table gave (0.816, -0.004) and (1.608, 0.383), the second
  // already past the arm's reach before any offset was added. Anchoring there
  // teleported the object onto that phantom point and then shifted sideways —
  // (0.588, 0.633) for a book sitting at (0.588, -0.159), an unreachable pose
  // five ACTION_ABORTED retries could not diagnose.
  //
  // The destination still sets HOW FAR to shift — region_frac of its
  // half-extent, less the target's own support so the object lands fully on the
  // surface — but no longer WHERE FROM. Clamped because the far half of a table
  // is outside the arm's reach; on a real table the clamp is what comes out, so
  // region_frac's scaling is exercised by gtest rather than by the live scene.
  auto place_at_edge = [&](double ux, double uy) {
    double d  = params.region_frac * xy_support(dest, ux, uy)
              - xy_support(target, ux, uy);
    double hi = std::max(params.side_offset_m, params.region_max_offset_m);
    d = std::clamp(d, params.side_offset_m, hi);
    if (target.centroid_valid) {   // else keep the dest centroid seeded above
      x = anchor_x = target.centroid[0];
      y = anchor_y = target.centroid[1];
    }
    x += ux * d;
    y += uy * d;
  };

  if (spec.type == "container") {
    // Rim, not centroid — a basket's centroid sits ~90 mm below its opening,
    // so centroid + 3 cm released into the wall.
    underside_z = top_z(dest) + params.container_drop_z;

  // Which world axis "left"/"right"/"front"/"behind" mean. Poses are in
  // panda_link0: +X points away from the base across the table, +Y is the
  // robot's LEFT, +Z is up. So left/right are ±Y and near/far are ±X.
  //
  // Both blocks below had these swapped — left/right on ±X, front/behind on ±Y
  // — which sent "to the right of the book" straight back toward the base. The
  // overhead camera agrees with the robot here rather than contradicting it:
  // its extrinsic R = [[0,-1,0],[-1,0,0],[0,0,-1]] maps image-right to -Y and
  // image-up to +X, so an operator reading the top view and the robot mean the
  // same thing by "right". Do not "fix" one convention into the other.
  } else if (spec.type == "surface") {
    underside_z = standing_plane_z(target, dest) + params.place_height_m;
    const auto& r = spec.region;
    if      (r == "left_edge")  place_at_edge( 0.0,  1.0);
    else if (r == "right_edge") place_at_edge( 0.0, -1.0);
    else if (r == "far_end")    place_at_edge( 1.0,  0.0);
    else if (r == "near_end")   place_at_edge(-1.0,  0.0);
    // "center" names no direction, so there is nothing to shift the target by
    // and it is the one region still anchored on the destination centroid —
    // with all of that centroid's unreliability. Untested live.

  } else if (spec.type == "relation") {
    const auto& rel = spec.relation;
    if (rel == "on_top_of") {
      underside_z = top_z(dest) + params.place_height_m;
    } else {
      // Beside the reference → lands on whatever surface the reference is on.
      underside_z = base_z(dest) + params.place_height_m;
      if      (rel == "left_of")     place_beside( 0.0,  1.0);
      else if (rel == "right_of")    place_beside( 0.0, -1.0);
      else if (rel == "in_front_of") place_beside(-1.0,  0.0);
      else if (rel == "behind")      place_beside( 1.0,  0.0);
      else                           offset_toward_target();  // "near" + unknown
    }

  } else {
    // Unknown/empty type — same under-specified case "near" covers. Used to
    // take NO branch and return the raw centroid, burying the object inside.
    underside_z = base_z(dest) + params.place_height_m;
    offset_toward_target();
  }

  // Reach clamp. Every branch above sizes its shift from measured geometry, and
  // measured geometry has repeatedly put the result outside the workspace: a
  // phone at XY radius 0.681 plus its own 0.098 support sent "behind the phone"
  // to 0.853, five ACTION_ABORTED retries that reported nothing but a failed
  // plan. Shorten the shift instead — walk back along anchor→place so the named
  // direction survives — rather than scaling XY toward the base, which would
  // also drag the axis the instruction never mentioned.
  //
  // Solves |anchor + t*(place-anchor)| = R for the largest t in [0,1]. An
  // anchor already outside R has no solution in range and collapses to t = 0:
  // nothing here can rescue a destination the arm cannot reach, and refusing to
  // shift at least keeps the object where it is.
  if (params.max_place_reach_m > 1e-6) {
    const double R = params.max_place_reach_m;
    if (std::hypot(x, y) > R) {
      const double dx = x - anchor_x, dy = y - anchor_y;
      const double a  = dx * dx + dy * dy;
      double t = 0.0;
      if (a > 1e-12) {
        const double b    = 2.0 * (anchor_x * dx + anchor_y * dy);
        const double c    = anchor_x * anchor_x + anchor_y * anchor_y - R * R;
        const double disc = b * b - 4.0 * a * c;
        if (disc >= 0.0) {
          t = std::clamp((-b + std::sqrt(disc)) / (2.0 * a), 0.0, 1.0);
        }
      }
      x = anchor_x + t * dx;
      y = anchor_y + t * dy;
    }
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
