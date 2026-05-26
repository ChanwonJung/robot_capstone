#pragma once

#include <array>
#include <mutex>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

namespace bt_pkg {

// ── Sub-structs ──────────────────────────────────────────────────────────────

struct GraspCandidate {
  geometry_msgs::msg::PoseStamped pose;  // frame: panda_link0
  double quality = 0.0;
  double width   = 0.0;
};

// Mirrors qwen_call.py GroundingResult.destination
struct DestinationSpec {
  std::string type;             // "container" | "surface" | "relation"
  std::string reference_label;  // semantic label of the reference object
  std::string relation;         // "left_of" | "right_of" | "on_top_of" | "near" | ...
  std::string region;           // "left_edge" | "right_edge" | "center" | ... (surface only)
};

// One entry from /yolo/world_map
struct YoloObject {
  std::string class_name;
  std::array<double, 3> centroid = {};  // world frame (panda_link0)
  double confidence = 0.0;
};

// ── Main shared data struct ───────────────────────────────────────────────────
// Owned by bt_executor_node; passed as shared_ptr to every BT node.
// All fields are guarded by `mtx`.

struct SceneData {
  mutable std::mutex mtx;

  // ── /world_map_result ────────────────────────────────────────────────────
  bool world_map_fresh = false;
  rclcpp::Time world_map_stamp;
  std::array<double, 3> target_centroid      = {};
  std::array<double, 3> destination_centroid = {};
  std::string target_label;
  std::string destination_label;

  // ── /grasp_candidates (vgn_grasp_node) ──────────────────────────────────
  bool grasp_candidates_fresh = false;
  rclcpp::Time grasp_candidates_stamp;
  std::vector<GraspCandidate> grasp_candidates;  // sorted best-first by VGN

  // ── /qwen/grounding_result ───────────────────────────────────────────────
  bool grounding_result_fresh = false;
  DestinationSpec destination_spec;

  // ── /yolo/world_map (yolo_world_map_node) ───────────────────────────────
  std::vector<YoloObject> yolo_objects;
  rclcpp::Time yolo_world_map_stamp;

  // ── /yolo/target_centroid ────────────────────────────────────────────────
  geometry_msgs::msg::PointStamped target_centroid_live;
  rclcpp::Time target_centroid_stamp;

  // ── /bt/hazard_level ─────────────────────────────────────────────────────
  // 0=clear  1=slow (obstacle detected)  3=halt (arm/person detected)
  int hazard_level = 0;

  // ── /joint_states ────────────────────────────────────────────────────────
  sensor_msgs::msg::JointState latest_joint_state;
  bool has_joint_state = false;

  // ── Replan / cycle coordination ──────────────────────────────────────────
  // WaitForScene only unblocks when both stamps are newer than this.
  rclcpp::Time last_processed_stamp;
  bool awaiting_replan = false;
};

}  // namespace bt_pkg
