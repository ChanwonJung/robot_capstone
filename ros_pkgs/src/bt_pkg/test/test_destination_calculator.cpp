// Geometry tests for compute_place_pose(). The failure modes here are "arm
// drives into the basket wall" and "arm dives at its own base", so each case is
// pinned with real Isaac numbers: basket rim z 0.18 at (0.480, -0.420),
// book ~20 mm thick.

#include <gtest/gtest.h>

#include "bt_pkg/destination_calculator.hpp"

using bt_pkg::DestinationSpec;
using bt_pkg::ObjectGeometry;
using bt_pkg::PlacePoseParams;
using bt_pkg::TabletopObstacle;
using bt_pkg::carry_lift;
using bt_pkg::compute_place_pose;
using bt_pkg::nudge_to_free_space;

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
  // "left of" is +Y in panda_link0, not -X. This assertion used to encode the
  // -X version, so the whole suite passed while every left/right place went
  // toward or away from the base instead of sideways.
  // Basket half-Y 0.107 + book half-Y 0.060 + clearance 0.050 = 0.217, well
  // past the 0.080 floor — a fixed side_offset_m would have dropped the book
  // inside the basket's own footprint.
  EXPECT_NEAR(pose.pose.position.y, dest.centroid[1] + 0.217, 1e-6);
  EXPECT_NEAR(pose.pose.position.x, dest.centroid[0], 1e-6);
}

// Every left/right/near/far spelling, on both the surface and relation
// branches, against the panda_link0 convention: +X away from the base, +Y left.
TEST(ComputePlacePose, SideOffsetsUseTheRightAxis)
{
  PlacePoseParams params;
  // Axis mapping only. The basket sits at XY radius 0.638, so "behind" it would
  // otherwise trip the reach clamp and mix two concerns into one expectation;
  // the clamp has its own tests below.
  params.max_place_reach_m = 0.0;
  const auto dest = basket();
  const auto t    = book();
  const auto grasp = grasp_at(0.450, -0.151, 0.210);

  // Distances are hand-computed rather than re-derived from the formula, which
  // would make the test agree with any implementation including a wrong one.
  // Basket half (0.117, 0.107), book half (0.080, 0.060), clearance 0.050:
  //   relation, ±Y: 0.107 + 0.060 + 0.050 = 0.217
  //   relation, ±X: 0.117 + 0.080 + 0.050 = 0.247
  //   surface,  ±Y: 0.6*0.107 - 0.060 = 0.004 → floors at side_offset_m 0.080
  //   surface,  ±X: 0.6*0.117 - 0.080 < 0    → floors at side_offset_m 0.080
  // A basket is too small for its own regions to mean anything, so the surface
  // rows still land on the old constant. Scaling is covered separately below.
  //
  // The two families differ in what they are anchored on, so expected positions
  // are absolute rather than deltas off one origin:
  //   relation → the destination; "beside the basket" is a spot in the world
  //   surface  → the target's own position, shifted along the named axis; a
  //              plane's centroid is a phantom point, not a spot to place at
  const double bx = dest.centroid[0], by = dest.centroid[1];   // basket
  const double tx = t.centroid[0],    ty = t.centroid[1];      // book
  struct Case {
    const char* type;
    const char* key;      // region for "surface", relation for "relation"
    double      x;
    double      y;
  };
  const Case cases[] = {
    {"surface",  "left_edge",   tx,         ty + 0.080},
    {"surface",  "right_edge",  tx,         ty - 0.080},
    {"surface",  "far_end",     tx + 0.080, ty},
    {"surface",  "near_end",    tx - 0.080, ty},
    // "center" names no direction, so there is nothing to shift the target by:
    // the one region still anchored on the destination.
    {"surface",  "center",      bx,         by},
    {"relation", "left_of",     bx,         by + 0.217},
    {"relation", "right_of",    bx,         by - 0.217},
    {"relation", "behind",      bx + 0.247, by},
    {"relation", "in_front_of", bx - 0.247, by},
  };

  for (const auto& c : cases) {
    DestinationSpec spec;
    spec.type = c.type;
    if (spec.type == "surface") spec.region = c.key;
    else                        spec.relation = c.key;

    const auto pose = compute_place_pose(spec, dest, t, grasp, params);
    EXPECT_NEAR(pose.pose.position.x, c.x, 1e-6) << c.type << " / " << c.key;
    EXPECT_NEAR(pose.pose.position.y, c.y, 1e-6) << c.type << " / " << c.key;
  }
}

// --- Named directions clear both footprints -------------------------------
//
// The bug: side_offset_m was a fixed 0.08 from the destination's CENTROID, so
// "to the right of the book" put the cup on top of any book wider than 16 cm.

// A book laid out along Y, wide enough that a fixed 8 cm lands on it.
ObjectGeometry wide_book()
{
  ObjectGeometry g;
  g.centroid       = {0.450, -0.151, 0.110};
  g.bbox_min       = {0.410, -0.271, 0.100};
  g.bbox_max       = {0.490, -0.031, 0.120};   // 80 x 240 mm
  g.centroid_valid = true;
  g.bbox_valid     = true;
  return g;
}

// Glass cup, 68 mm across and 100 mm tall (the measured Isaac asset).
ObjectGeometry cup()
{
  ObjectGeometry g;
  g.centroid       = {0.300, 0.269, 0.050};
  g.bbox_min       = {0.266, 0.235, 0.000};
  g.bbox_max       = {0.334, 0.303, 0.100};
  g.centroid_valid = true;
  g.bbox_valid     = true;
  return g;
}

TEST(ComputePlacePose, BesideRelationClearsBothFootprints)
{
  PlacePoseParams params;
  DestinationSpec spec;
  spec.type     = "relation";
  spec.relation = "right_of";

  const auto dest = wide_book();   // half-Y 0.120
  const auto t    = cup();         // half-Y 0.034
  const auto pose = compute_place_pose(spec, dest, t,
                                       grasp_at(0.300, 0.269, 0.150), params);

  // 0.120 + 0.034 + 0.050 = 0.204, and "right" is -Y.
  EXPECT_NEAR(pose.pose.position.y, dest.centroid[1] - 0.204, 1e-6);

  // The property that actually matters: no overlap between the two footprints.
  // The old fixed 0.08 fails this — 0.08 < 0.120 + 0.034.
  const double gap = std::abs(pose.pose.position.y - dest.centroid[1])
                   - dest.half_extent()[1] - t.half_extent()[1];
  EXPECT_GT(gap, 0.0);
  EXPECT_NEAR(gap, params.near_clearance_m, 1e-6);
}

TEST(ComputePlacePose, BesideRelationFloorsAtSideOffsetForSmallObjects)
{
  PlacePoseParams params;
  DestinationSpec spec;
  spec.type     = "relation";
  spec.relation = "left_of";

  // Two objects small enough that the supports + clearance stay under 0.08.
  ObjectGeometry small_dest;
  small_dest.centroid       = {0.450, 0.000, 0.020};
  small_dest.bbox_min       = {0.440, -0.010, 0.000};
  small_dest.bbox_max       = {0.460,  0.010, 0.040};   // half-Y 0.010
  small_dest.centroid_valid = true;
  small_dest.bbox_valid     = true;

  ObjectGeometry small_target = small_dest;
  small_target.centroid = {0.300, 0.200, 0.020};

  const auto pose = compute_place_pose(spec, small_dest, small_target,
                                       grasp_at(0.300, 0.200, 0.120), params);
  // 0.010 + 0.010 + 0.050 = 0.070 < 0.080, so the floor wins.
  EXPECT_NEAR(pose.pose.position.y,
              small_dest.centroid[1] + params.side_offset_m, 1e-6);
}

// --- Surface regions scale with the surface --------------------------------

// A table, 1.2 x 0.8 m, top at z = 0. Half-extent (0.600, 0.400).
ObjectGeometry table()
{
  ObjectGeometry g;
  g.centroid       = { 0.450,  0.000, -0.010};
  g.bbox_min       = {-0.150, -0.400, -0.020};
  g.bbox_max       = { 1.050,  0.400,  0.000};
  g.centroid_valid = true;
  g.bbox_valid     = true;
  return g;
}

TEST(ComputePlacePose, SurfaceRegionScalesWithTheSurface)
{
  PlacePoseParams params;   // region_frac 0.6, region_max_offset_m 0.25
  DestinationSpec spec;
  spec.type   = "surface";
  spec.region = "right_edge";

  const auto dest = table();
  const auto t    = cup();
  const auto pose = compute_place_pose(spec, dest, t,
                                       grasp_at(0.300, 0.269, 0.150), params);

  // The table sets HOW FAR: 0.6 * 0.400 - 0.034 = 0.206, inside the 0.25 cap.
  // The cup sets WHERE FROM, on both axes — "right" is -Y, X is untouched.
  EXPECT_NEAR(pose.pose.position.y, t.centroid[1] - 0.206, 1e-6);
  EXPECT_NEAR(pose.pose.position.x, t.centroid[0], 1e-6);

  // The point of the change: a shift that scales with the table, not the old
  // fixed 8 cm.
  EXPECT_GT(std::abs(pose.pose.position.y - t.centroid[1]),
            2.0 * params.side_offset_m);

  // The shift plus the cup's own body stays within the table's half-extent, so
  // a cup starting at the middle would land on it. Note this bounds the SHIFT,
  // not the final pose: anchored on the target, a cup already near the edge can
  // be pushed off it. Nothing measures the surface reliably enough to prevent
  // that — see standing_plane_z().
  EXPECT_LE(std::abs(pose.pose.position.y - t.centroid[1]) + t.half_extent()[1],
            dest.half_extent()[1]);
}

TEST(ComputePlacePose, SurfaceRegionIsClampedToReach)
{
  PlacePoseParams params;
  DestinationSpec spec;
  spec.type   = "surface";
  spec.region = "far_end";

  auto dest = table();
  dest.bbox_min[0] = -0.750;   // 3 m deep, as the measured table's 0.833 half
  dest.bbox_max[0] =  1.650;   // -extent already was → half-X 1.200
  const auto t    = cup();
  const auto pose = compute_place_pose(spec, dest, t,
                                       grasp_at(0.300, 0.269, 0.150), params);

  // 0.6 * 1.200 - 0.034 = 0.686, past the cap — nobody asked for the object to
  // be flung to the far end of a 3 m table. Applied from the cup's own
  // position, not the table's centroid.
  EXPECT_NEAR(pose.pose.position.x,
              t.centroid[0] + params.region_max_offset_m, 1e-6);
}

// The cap bounds the SHIFT; this bounds the RESULT. Live, "behind the phone"
// sized a legitimate 0.208 m shift off the phone's own footprint and landed at
// XY radius 0.853 — past the Panda's 0.855 m spec reach once height was added.
// Five retries reported only "ACTION_ABORTED".
TEST(ComputePlacePose, OverReachingShiftIsShortenedNotAbandoned)
{
  PlacePoseParams params;
  DestinationSpec spec;
  spec.type     = "relation";
  spec.relation = "behind";        // +X, straight away from the base

  ObjectGeometry phone;            // as measured 2026-08-11
  phone.centroid = {0.535, 0.420, 0.015};
  phone.bbox_min = {0.435, 0.373, -0.001};
  phone.bbox_max = {0.633, 0.467, 0.022};
  phone.centroid_valid = phone.bbox_valid = true;

  const auto t = cup();
  const auto pose = compute_place_pose(spec, phone, t,
                                       grasp_at(0.300, 0.269, 0.150), params);

  const double reach = std::hypot(pose.pose.position.x, pose.pose.position.y);
  EXPECT_NEAR(reach, params.max_place_reach_m, 1e-6);

  // Shortened along the named direction, not scaled toward the base: "behind"
  // is +X only, so Y must not move and X must still have advanced.
  EXPECT_NEAR(pose.pose.position.y, phone.centroid[1], 1e-6);
  EXPECT_GT(pose.pose.position.x, phone.centroid[0]);
}

// An anchor already outside the envelope cannot be rescued by shortening. Leave
// the object where it is rather than shifting it further out.
TEST(ComputePlacePose, UnreachableAnchorProducesNoShift)
{
  PlacePoseParams params;
  DestinationSpec spec;
  spec.type     = "relation";
  spec.relation = "behind";

  ObjectGeometry far_dest;
  far_dest.centroid = {1.200, 0.500, 0.050};
  far_dest.centroid_valid = true;

  const auto pose = compute_place_pose(spec, far_dest, cup(),
                                       grasp_at(0.300, 0.269, 0.150), params);
  EXPECT_NEAR(pose.pose.position.x, far_dest.centroid[0], 1e-6);
  EXPECT_NEAR(pose.pose.position.y, far_dest.centroid[1], 1e-6);
}

// ── placing around tabletop clutter ──────────────────────────────────────────

namespace {
TabletopObstacle obst(double x, double y, double r, double top = 0.07)
{
  TabletopObstacle o;
  o.centroid  = {x, y, 0.03};
  o.xy_radius = r;
  o.top_z     = top;
  o.top_valid = true;
  return o;
}
}  // namespace

TEST(NudgeToFreeSpace, LeavesAClearPointAlone)
{
  const std::vector<TabletopObstacle> obs{obst(0.60, 0.30, 0.04)};
  const auto r = nudge_to_free_space(0.50, -0.10, 0.05, obs, 0.04, 0.20, 0.80);
  EXPECT_FALSE(r.moved);
  EXPECT_TRUE(r.found);
  EXPECT_NEAR(r.x, 0.50, 1e-9);
  EXPECT_NEAR(r.y, -0.10, 1e-9);
}

TEST(NudgeToFreeSpace, StepsOffAnOccupiedPointAndClearsIt)
{
  const std::vector<TabletopObstacle> obs{obst(0.60, 0.10, 0.04)};
  const double obj_r = 0.05, clear = 0.04;
  const auto r = nudge_to_free_space(0.60, 0.10, obj_r, obs, clear, 0.30, 0.80);
  ASSERT_TRUE(r.found);
  EXPECT_TRUE(r.moved);

  // Far enough that the two footprints plus the gap no longer overlap...
  const double d = std::hypot(r.x - 0.60, r.y - 0.10);
  EXPECT_GE(d, 0.04 + obj_r + clear - 1e-6);
  // ...and no further than it had to go. The ring scan returns the first free
  // point, so the answer sits just outside the required separation.
  EXPECT_LE(d, 0.04 + obj_r + clear + 0.03);
}

TEST(NudgeToFreeSpace, NeverReturnsAPointOutsideReach)
{
  // Obstacle sitting right at the edge of the envelope: every escape route
  // outward is unreachable, so the search must come back inward.
  const std::vector<TabletopObstacle> obs{obst(0.78, 0.00, 0.06)};
  const auto r = nudge_to_free_space(0.78, 0.00, 0.04, obs, 0.04, 0.30, 0.80);
  ASSERT_TRUE(r.found);
  EXPECT_LE(std::hypot(r.x, r.y), 0.80 + 1e-9);
}

TEST(NudgeToFreeSpace, ReportsFailureRatherThanStackingObjects)
{
  // Ideal point boxed in by obstacles out past the search radius.
  std::vector<TabletopObstacle> obs;
  for (int i = 0; i < 16; ++i) {
    const double a = 2.0 * M_PI * i / 16;
    obs.push_back(obst(0.50 + 0.10 * std::cos(a), 0.10 * std::sin(a), 0.10));
  }
  obs.push_back(obst(0.50, 0.00, 0.06));
  const auto r = nudge_to_free_space(0.50, 0.00, 0.05, obs, 0.04, 0.08, 0.80);
  EXPECT_FALSE(r.found);
  EXPECT_FALSE(r.moved);
  // Falls back to the requested point rather than inventing one.
  EXPECT_NEAR(r.x, 0.50, 1e-9);
  EXPECT_NEAR(r.y, 0.00, 1e-9);
}

TEST(NudgeToFreeSpace, NoObstaclesIsANoOp)
{
  const auto r = nudge_to_free_space(0.50, 0.20, 0.05, {}, 0.04, 0.20, 0.80);
  EXPECT_FALSE(r.moved);
  EXPECT_TRUE(r.found);
  EXPECT_NEAR(r.x, 0.50, 1e-9);
}

// Regression from the 2026-08-11 scan, obstacle extractor output verbatim.
// "put the book on the left side of the table" placed at (0.588, 0.179) — and
// the red ball measured at (0.646, 0.108) with a 0.041 m footprint is 0.091 m
// away, closer than book(0.086) + ball(0.041) + clearance(0.04) = 0.167.
TEST(NudgeToFreeSpace, ClearsTheMeasuredTabletop)
{
  const std::vector<TabletopObstacle> obs{
    obst(0.480, -0.425, 0.164, 0.126),   // basket
    obst(0.602,  0.272, 0.036, 0.071),   // apple
    obst(0.646,  0.108, 0.041, 0.067),   // red ball
    obst(0.602,  0.452, 0.017, 0.021),   // phone
  };
  const double book_r = 0.086;
  const auto r = nudge_to_free_space(0.588, 0.179, book_r, obs, 0.04, 0.20, 0.80);
  ASSERT_TRUE(r.found);
  EXPECT_TRUE(r.moved);

  for (const auto& o : obs) {
    EXPECT_GE(std::hypot(r.x - o.centroid[0], r.y - o.centroid[1]),
              o.xy_radius + book_r + 0.04 - 1e-6);
  }
  EXPECT_LE(std::hypot(r.x, r.y), 0.80 + 1e-9);
}

// The path floor must describe the OBJECT, not link8. A book hangs 0.157 m
// below link8, so a constant 0.15 dragged its underside along at table height
// and through a 0.1 m glass.
TEST(CarryLift, MeasuresTheObjectHangingBelowLink8)
{
  ObjectGeometry book_g;           // as measured 2026-08-11
  book_g.centroid = {0.588, -0.159, 0.075};
  book_g.bbox_min = {0.537, -0.261, -0.001};
  book_g.bbox_max = {0.710, -0.080, 0.105};
  book_g.centroid_valid = book_g.bbox_valid = true;

  // hold offset 0.156 - 0.075 = 0.081, half-height 0.075 - -0.001 = 0.076.
  const double lift = carry_lift(book_g, grasp_at(0.588, -0.159, 0.156));
  EXPECT_NEAR(lift, 0.157, 1e-6);

  PlacePoseParams params;          // carry_clearance_m 0.12
  const double floor_z = params.carry_clearance_m + lift;
  EXPECT_NEAR(floor_z, 0.277, 1e-6);
  // The old constant put the book's underside below the tabletop.
  EXPECT_LT(0.150 - lift, 0.0);
  // The derived floor clears a 0.10 m glass.
  EXPECT_GT(floor_z - lift, 0.10);

  // Unmeasured object → no lift, so the floor degrades to the bare clearance
  // rather than to a negative number.
  EXPECT_NEAR(carry_lift(ObjectGeometry{}, grasp_at(0.5, 0.0, 0.3)), 0.0, 1e-6);
}

// Every region is a shift of the TARGET, along the one axis it names. Nothing
// is read from the surface centroid, which is a phantom point: SAM put the same
// table at (0.816, -0.004) and (1.608, 0.383) minutes apart.
TEST(ComputePlacePose, SurfaceRegionShiftsTheTargetAlongOneAxisOnly)
{
  PlacePoseParams params;
  const auto dest = table();          // half-extent (0.600, 0.400)
  const auto t    = cup();            // half-extent (0.034, 0.034)
  const auto g    = grasp_at(0.300, 0.269, 0.150);

  // ±Y: 0.6*0.400 - 0.034 = 0.206.  ±X: 0.6*0.600 - 0.034 = 0.326.
  // Both under the 0.45 cap, and the cup is central enough not to trip the
  // reach clamp — this test is about which axis moves.
  struct Case { const char* region; double dx; double dy; };
  const Case cases[] = {
    {"left_edge",  0.000, +0.206},
    {"right_edge", 0.000, -0.206},
    {"far_end",   +0.326,  0.000},
    {"near_end",  -0.326,  0.000},
  };
  for (const auto& c : cases) {
    DestinationSpec spec;
    spec.type   = "surface";
    spec.region = c.region;
    const auto p = compute_place_pose(spec, dest, t, g, params);
    EXPECT_NEAR(p.pose.position.x, t.centroid[0] + c.dx, 1e-6) << c.region;
    EXPECT_NEAR(p.pose.position.y, t.centroid[1] + c.dy, 1e-6) << c.region;
  }
}

// The whole point: a destination centroid parked outside the workspace must not
// drag the place pose out with it. Only the offset SIZE comes from the surface.
TEST(ComputePlacePose, SurfaceCentroidOutsideTheWorkspaceIsNotInherited)
{
  PlacePoseParams params;
  DestinationSpec spec;
  spec.type   = "surface";
  spec.region = "left_edge";

  auto dest = table();
  const auto t = cup();
  const auto g = grasp_at(0.300, 0.269, 0.150);
  const auto before = compute_place_pose(spec, dest, t, g, params);

  dest.centroid[0] += 1.2;            // shove the phantom point out of reach
  dest.centroid[1] += 0.4;
  const auto after = compute_place_pose(spec, dest, t, g, params);

  EXPECT_NEAR(after.pose.position.x, before.pose.position.x, 1e-6);
  EXPECT_NEAR(after.pose.position.y, before.pose.position.y, 1e-6);
}

// No measured target → no current position to preserve, so the destination
// centroid is all that is left.
TEST(ComputePlacePose, SurfaceRegionFallsBackToDestinationWithoutATarget)
{
  PlacePoseParams params;
  DestinationSpec spec;
  spec.type   = "surface";
  spec.region = "right_edge";

  const auto dest = table();
  ObjectGeometry t;             // centroid_valid stays false
  const auto pose = compute_place_pose(spec, dest, t,
                                       grasp_at(0.300, 0.269, 0.150), params);
  EXPECT_NEAR(pose.pose.position.x, dest.centroid[0], 1e-6);
}

// Both live scans of 2026-08-11, verbatim. Same table, minutes apart, same
// instruction shape ("... on the left side of the table") — and SAM returned
// two irreconcilable geometries. Each broke a different single-statistic guess
// at the tabletop height, which is why the height now comes off the target.
//
//   label            px      centroid          bbox max z
//   "table surface"  196547  (0.816,-0.004, 0.007)   0.517  ← back wall
//   "table"           23549  (1.608, 0.383,-0.768)   0.002  ← legs and floor
//
// True tabletop: -0.00137. Both targets' own undersides were within 1.4 mm.
struct LiveScan {
  const char*    name;
  ObjectGeometry dest;
  ObjectGeometry target;
  double         grasp_z;
  double         expect_x, expect_z;
  // The shift the region asks for, before the reach clamp. Whether it survives
  // is the point of `clamped`: the ball's 0.392 m lands at radius 0.802 and is
  // shortened to exactly max_place_reach_m, the book's capped 0.45 fits.
  double         shift_y;
  bool           clamped;
};

class MeasuredTableSurface : public ::testing::TestWithParam<LiveScan> {};

TEST_P(MeasuredTableSurface, StaysReachableAndOnThePlane)
{
  const auto& s = GetParam();
  PlacePoseParams params;
  DestinationSpec spec;
  spec.type   = "surface";
  spec.region = "left_edge";

  const auto pose = compute_place_pose(
    spec, s.dest, s.target,
    grasp_at(s.target.centroid[0], s.target.centroid[1], s.grasp_z), params);

  EXPECT_NEAR(pose.pose.position.x, s.expect_x, 1e-6) << s.name;
  EXPECT_NEAR(pose.pose.position.z, s.expect_z, 1e-6) << s.name;

  const double xy = std::hypot(pose.pose.position.x, pose.pose.position.y);
  if (s.clamped) {
    EXPECT_NEAR(xy, params.max_place_reach_m, 1e-6) << s.name;
    // Shortened, not reversed: still left of where it started, still short of
    // what the region asked for.
    EXPECT_GT(pose.pose.position.y, s.target.centroid[1]) << s.name;
    EXPECT_LT(pose.pose.position.y, s.target.centroid[1] + s.shift_y) << s.name;
  } else {
    EXPECT_NEAR(pose.pose.position.y, s.target.centroid[1] + s.shift_y, 1e-6)
      << s.name;
    EXPECT_LE(xy, params.max_place_reach_m) << s.name;
  }

  // Neither contaminated centroid may drag the pose out of the workspace.
  const double reach = std::sqrt(pose.pose.position.x * pose.pose.position.x
                               + pose.pose.position.y * pose.pose.position.y
                               + pose.pose.position.z * pose.pose.position.z);
  EXPECT_LT(reach, 0.855) << s.name;

  // And the object is released above the plane it was standing on, not through
  // it: the "table" scan drove to z = -0.561, 56 cm below the tabletop.
  EXPECT_GT(pose.pose.position.z - params.place_height_m, s.target.bbox_min[2])
    << s.name;
}

INSTANTIATE_TEST_SUITE_P(
  LiveScans20260811, MeasuredTableSurface, ::testing::Values(
    // Red ball. d = 0.6*0.713 - 0.036 = 0.392, under the 0.45 cap but the ball
    // already sits at x 0.629, so the result overshoots and is shortened.
    // z = underside -0.000 + 0.05 + half-height 0.053 + hold offset 0.100.
    LiveScan{"table surface / red ball",
             ObjectGeometry{{ 0.816, -0.004,  0.007},
                            { 0.135, -0.714, -0.001},
                            { 1.446,  0.709,  0.517}, true, true},
             ObjectGeometry{{ 0.629,  0.105,  0.053},
                            { 0.606,  0.080,  0.000},
                            { 0.701,  0.141,  0.068}, true, true},
             0.153, 0.629, 0.203, 0.392, true},
    // Book. d = 0.6*0.946 - 0.079 = 0.489 → cap 0.45, and it fits.
    // z = underside -0.001 + 0.05 + half-height 0.076 + hold offset 0.081.
    LiveScan{"table / book",
             ObjectGeometry{{ 1.608,  0.383, -0.768},
                            { 0.171, -1.134, -0.770},
                            { 1.837,  1.329,  0.002}, true, true},
             ObjectGeometry{{ 0.588, -0.159,  0.075},
                            { 0.537, -0.261, -0.001},
                            { 0.710, -0.080,  0.105}, true, true},
             0.156, 0.588, 0.206, 0.450, false}));

// Height comes off the target's underside, so no contamination of the
// surface's own geometry — top, centroid, or both — can move it. Containers
// still read top_z, where the extreme genuinely is the rim.
TEST(ComputePlacePose, SurfaceHeightComesFromTheTargetNotTheSurface)
{
  PlacePoseParams params;
  DestinationSpec spec;
  spec.type   = "surface";
  spec.region = "center";

  auto dest = table();
  const auto clean = compute_place_pose(spec, dest, cup(),
                                        grasp_at(0.300, 0.269, 0.150), params);

  dest.bbox_max[2] += 0.5;    // back wall, as in the "table surface" scan
  dest.centroid[2] -= 0.8;    // legs and floor, as in the "table" scan
  const auto dirty = compute_place_pose(spec, dest, cup(),
                                        grasp_at(0.300, 0.269, 0.150), params);
  EXPECT_NEAR(dirty.pose.position.z, clean.pose.position.z, 1e-6);

  // Cup underside 0.000 + 0.05 clearance + half-height 0.050 + hold offset
  // 0.100.
  EXPECT_NEAR(clean.pose.position.z, 0.200, 1e-6);
}

TEST(ComputePlacePose, SurfaceCenterIgnoresTheFootprint)
{
  PlacePoseParams params;
  DestinationSpec spec;
  spec.type   = "surface";
  spec.region = "center";

  const auto dest = table();
  const auto pose = compute_place_pose(spec, dest, cup(),
                                       grasp_at(0.300, 0.269, 0.150), params);
  EXPECT_NEAR(pose.pose.position.x, dest.centroid[0], 1e-6);
  EXPECT_NEAR(pose.pose.position.y, dest.centroid[1], 1e-6);
}

// Without a bbox both supports are 0, so everything degrades to the constants
// rather than to 0 — the arm must not stack the object onto the destination.
TEST(ComputePlacePose, DirectionsFallBackToConstantsWithoutBboxes)
{
  PlacePoseParams params;
  ObjectGeometry dest;
  dest.centroid       = {0.450, 0.000, 0.050};
  dest.centroid_valid = true;   // bbox_valid stays false

  ObjectGeometry t;
  t.centroid       = {0.300, 0.200, 0.050};
  t.centroid_valid = true;

  DestinationSpec rel;
  rel.type     = "relation";
  rel.relation = "right_of";
  auto p1 = compute_place_pose(rel, dest, t, grasp_at(0.300, 0.200, 0.150),
                               params);
  // 0 + 0 + near_clearance_m 0.05 < side_offset_m 0.08 → the floor.
  EXPECT_NEAR(p1.pose.position.y, dest.centroid[1] - params.side_offset_m, 1e-6);

  DestinationSpec surf;
  surf.type   = "surface";
  surf.region = "right_edge";
  auto p2 = compute_place_pose(surf, dest, t, grasp_at(0.300, 0.200, 0.150),
                               params);
  // 0.6 * 0 - 0 = 0 → clamped up to the floor, and applied from the target, so
  // the object still moves rather than being stacked onto the destination.
  EXPECT_NEAR(p2.pose.position.y, t.centroid[1] - params.side_offset_m, 1e-6);
  EXPECT_NEAR(p2.pose.position.x, t.centroid[0], 1e-6);
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
