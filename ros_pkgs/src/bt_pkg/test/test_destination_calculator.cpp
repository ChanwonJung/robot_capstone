// Geometry tests for compute_place_pose(). The failure modes here are "arm
// drives into the basket wall" and "arm dives at its own base", so each case is
// pinned with real Isaac numbers: basket rim z 0.18 at (0.480, -0.420),
// book ~20 mm thick.

#include <gtest/gtest.h>

#include "bt_pkg/destination_calculator.hpp"

using bt_pkg::DestinationSpec;
using bt_pkg::ObjectGeometry;
using bt_pkg::PlacePoseParams;
using bt_pkg::compute_place_pose;

namespace {

// Basket: centroid sits well below the rim.
ObjectGeometry basket()
{
  ObjectGeometry g;
  g.centroid       = {0.480, -0.420, 0.090};
  g.bbox_min       = {0.363, -0.527, 0.000};
  g.bbox_max       = {0.597, -0.313, 0.180};   // 234 x 214 mm opening, 180 mm rim
  g.centroid_valid = true;
  g.bbox_valid     = true;
  return g;
}

// Book on the table, 20 mm thick → centre 10 mm above its base.
ObjectGeometry book()
{
  ObjectGeometry g;
  g.centroid       = {0.450, -0.151, 0.110};
  g.bbox_min       = {0.370, -0.211, 0.100};
  g.bbox_max       = {0.530, -0.091, 0.120};
  g.centroid_valid = true;
  g.bbox_valid     = true;
  return g;
}

geometry_msgs::msg::PoseStamped grasp_at(double x, double y, double z,
                                         double yaw = 0.0)
{
  geometry_msgs::msg::PoseStamped p;
  p.pose.position.x = x;
  p.pose.position.y = y;
  p.pose.position.z = z;
  p.pose.orientation.w = 0.0;
  p.pose.orientation.x = std::cos(0.5 * yaw);
  p.pose.orientation.y = std::sin(0.5 * yaw);
  p.pose.orientation.z = 0.0;
  return p;
}

DestinationSpec container_spec()
{
  DestinationSpec s;
  s.type            = "container";
  s.reference_label = "basket";
  return s;
}

}  // namespace

// The bug this change exists to fix: a basket's centroid is 90 mm below its rim.
TEST(ComputePlacePose, ContainerClearsTheRimNotTheCentroid)
{
  PlacePoseParams params;  // container_drop_z 0.03
  const auto b = book();
  auto pose = compute_place_pose(container_spec(), basket(), b,
                                 grasp_at(0.450, -0.151, 0.210), params);

  // underside = rim 0.180 + 0.030 = 0.210
  // + book half-height 0.010 + hold offset 0.100 = 0.320
  EXPECT_NEAR(pose.pose.position.z, 0.320, 1e-6);

  // At or below the rim would clip the wall on the way in.
  const double object_underside = pose.pose.position.z - 0.010 - 0.100;
  EXPECT_GT(object_underside, basket().bbox_max[2]);

  // XY is the container centroid — drop it straight in.
  EXPECT_NEAR(pose.pose.position.x, 0.480, 1e-6);
  EXPECT_NEAR(pose.pose.position.y, -0.420, 1e-6);
}

// Same clearance for tall and flat: the object's height is measured, not assumed.
TEST(ComputePlacePose, TallerObjectIsReleasedHigher)
{
  PlacePoseParams params;
  ObjectGeometry cup;
  cup.centroid       = {0.450, 0.000, 0.050};
  cup.bbox_min       = {0.416, -0.034, 0.000};
  cup.bbox_max       = {0.484,  0.034, 0.100};   // 100 mm tall
  cup.centroid_valid = true;
  cup.bbox_valid     = true;

  auto pose = compute_place_pose(container_spec(), basket(), cup,
                                 grasp_at(0.450, 0.0, 0.115), params);

  // underside 0.210 + half-height 0.050 + hold offset 0.065 = 0.325
  EXPECT_NEAR(pose.pose.position.z, 0.325, 1e-6);
}

// No bbox → no rim to measure; fall back rather than invent a height.
TEST(ComputePlacePose, ContainerWithoutBboxFallsBackToCentroid)
{
  PlacePoseParams params;
  ObjectGeometry g;
  g.centroid       = {0.480, -0.420, 0.090};
  g.centroid_valid = true;   // bbox_valid stays false

  ObjectGeometry t;
  t.centroid       = {0.450, -0.151, 0.110};
  t.centroid_valid = true;   // no bbox either → half-height 0

  auto pose = compute_place_pose(container_spec(), g, t,
                                 grasp_at(0.450, -0.151, 0.210), params);
  // centroid 0.090 + drop 0.030 + 0 + hold offset 0.100
  EXPECT_NEAR(pose.pose.position.z, 0.220, 1e-6);
}

// "near" used to shove +X unconditionally. Must land on the side the object is
// already on, so the arm never carries it across the destination.
TEST(ComputePlacePose, NearOffsetsTowardTheTargetsCurrentSide)
{
  PlacePoseParams params;
  DestinationSpec spec;
  spec.type     = "relation";
  spec.relation = "near";

  auto dest = basket();
  auto t    = book();
  // Book is at y = -0.151, basket at y = -0.420 → the book is on the +Y side.
  auto pose = compute_place_pose(spec, dest, t, grasp_at(0.450, -0.151, 0.210),
                                 params);

  EXPECT_GT(pose.pose.position.y, dest.centroid[1])
    << "placed on the far side — the arm would cross over the destination";

  // Outside the basket's own footprint, not on top of it.
  EXPECT_GT(pose.pose.position.y, dest.bbox_max[1]);
}

// Mirror image — the direction must follow the target, not a constant.
TEST(ComputePlacePose, NearFlipsWithTheTarget)
{
  PlacePoseParams params;
  DestinationSpec spec;
  spec.type     = "relation";
  spec.relation = "near";

  auto dest = basket();
  auto t    = book();
  t.centroid[1] = -0.700;              // move the book to the -Y side
  t.bbox_min[1] = -0.760;
  t.bbox_max[1] = -0.640;

  auto pose = compute_place_pose(spec, dest, t, grasp_at(0.450, -0.700, 0.210),
                                 params);
  EXPECT_LT(pose.pose.position.y, dest.centroid[1]);
  EXPECT_LT(pose.pose.position.y, dest.bbox_min[1]);
}

// No target centroid → no direction; keep the fixed +X rather than divide by 0.
TEST(ComputePlacePose, NearWithoutTargetFallsBackToFixedX)
{
  PlacePoseParams params;
  DestinationSpec spec;
  spec.type     = "relation";
  spec.relation = "near";

  auto dest = basket();
  ObjectGeometry t;  // nothing valid

  auto pose = compute_place_pose(spec, dest, t, grasp_at(0.0, 0.0, 0.0), params);
  EXPECT_NEAR(pose.pose.position.x, dest.centroid[0] + params.near_offset_m, 1e-6);
  EXPECT_NEAR(pose.pose.position.y, dest.centroid[1], 1e-6);
}

// Unknown/empty type used to take NO branch and return the raw centroid.
TEST(ComputePlacePose, UnknownTypeStillLiftsAndOffsets)
{
  PlacePoseParams params;
  DestinationSpec spec;
  spec.type = "";   // VLM emitted something outside the vocabulary

  auto dest = basket();
  auto t    = book();
  auto pose = compute_place_pose(spec, dest, t, grasp_at(0.450, -0.151, 0.210),
                                 params);

  EXPECT_GT(pose.pose.position.z, dest.centroid[2])
    << "unknown type buried the object in the destination";
  const double dxy = std::hypot(pose.pose.position.x - dest.centroid[0],
                                pose.pose.position.y - dest.centroid[1]);
  EXPECT_GT(dxy, 0.0) << "unknown type placed on top of the destination centre";
}

// Beside-relations land on what the reference stands on, not on top of it.
TEST(ComputePlacePose, BesideRelationUsesTheSupportSurface)
{
  PlacePoseParams params;
  DestinationSpec spec;
  spec.type     = "relation";
  spec.relation = "left_of";

  auto dest = basket();      // bbox_min.z = 0.0 (standing on the table)
  auto t    = book();
  auto pose = compute_place_pose(spec, dest, t, grasp_at(0.450, -0.151, 0.210),
                                 params);

  // underside = 0.000 + 0.050, + half-height 0.010 + hold offset 0.100
  EXPECT_NEAR(pose.pose.position.z, 0.160, 1e-6);
  EXPECT_NEAR(pose.pose.position.x, dest.centroid[0] - params.side_offset_m, 1e-6);
}

// on_top_of stacks on the lid → top, not floor.
TEST(ComputePlacePose, OnTopOfUsesTheReferenceTop)
{
  PlacePoseParams params;
  DestinationSpec spec;
  spec.type     = "relation";
  spec.relation = "on_top_of";

  auto pose = compute_place_pose(spec, basket(), book(),
                                 grasp_at(0.450, -0.151, 0.210), params);
  // rim 0.180 + 0.050 + 0.010 + 0.100
  EXPECT_NEAR(pose.pose.position.z, 0.340, 1e-6);
}

// The wrist must not twist the held object between pick and place.
TEST(ComputePlacePose, KeepsTheGraspYaw)
{
  PlacePoseParams params;   // keep_grasp_yaw defaults true
  const double yaw = 0.6;

  auto pose = compute_place_pose(container_spec(), basket(), book(),
                                 grasp_at(0.450, -0.151, 0.210, yaw), params);

  const auto& q = pose.pose.orientation;
  const double got = std::atan2(2.0 * (q.x * q.y + q.w * q.z),
                                1.0 - 2.0 * (q.y * q.y + q.z * q.z));
  EXPECT_NEAR(got, yaw, 1e-6);
  EXPECT_NEAR(q.w, 0.0, 1e-9) << "not pointing straight down";
  EXPECT_NEAR(q.z, 0.0, 1e-9);
}

TEST(ComputePlacePose, KeepGraspYawCanBeDisabled)
{
  PlacePoseParams params;
  params.keep_grasp_yaw = false;

  auto pose = compute_place_pose(container_spec(), basket(), book(),
                                 grasp_at(0.450, -0.151, 0.210, 0.6), params);
  const auto& q = pose.pose.orientation;
  EXPECT_NEAR(q.w, 0.0, 1e-9);
  EXPECT_NEAR(q.x, 1.0, 1e-9);
  EXPECT_NEAR(q.y, 0.0, 1e-9);
  EXPECT_NEAR(q.z, 0.0, 1e-9);
}

// A scan and a grasp that disagree must not drive the arm downward.
TEST(ComputePlacePose, NegativeHoldOffsetIsIgnored)
{
  PlacePoseParams params;
  auto t = book();
  // link8 below the object's centre — impossible for a top-down grasp.
  auto pose = compute_place_pose(container_spec(), basket(), t,
                                 grasp_at(0.450, -0.151, 0.050), params);

  // Falls back to hold offset 0: 0.180 + 0.030 + 0.010
  EXPECT_NEAR(pose.pose.position.z, 0.220, 1e-6);
}

TEST(RetractAlongApproach, TopDownGraspLiftsStraightUp)
{
  auto in = grasp_at(0.4, 0.1, 0.2);
  auto out = bt_pkg::retract_along_approach(in, 0.15);
  EXPECT_NEAR(out.pose.position.x, 0.4, 1e-9);
  EXPECT_NEAR(out.pose.position.y, 0.1, 1e-9);
  EXPECT_NEAR(out.pose.position.z, 0.35, 1e-9);
}
