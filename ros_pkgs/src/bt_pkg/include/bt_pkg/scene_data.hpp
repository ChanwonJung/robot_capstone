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

  void clear() { type.clear(); reference_label.clear(); relation.clear(); region.clear(); }
  bool empty() const { return type.empty() && reference_label.empty(); }
};

// One category out of /world_map_result. Check the valid flags before posing:
// an unset centroid is {0,0,0} = the robot's own base.
struct ObjectGeometry {
  std::array<double, 3> centroid = {};
  std::array<double, 3> bbox_min = {};
  std::array<double, 3> bbox_max = {};
  bool centroid_valid = false;
  bool bbox_valid     = false;

  void clear() { *this = ObjectGeometry{}; }

  // Zero when no bbox was published.
  std::array<double, 3> half_extent() const {
    if (!bbox_valid) return {};
    return {0.5 * (bbox_max[0] - bbox_min[0]),
            0.5 * (bbox_max[1] - bbox_min[1]),
            0.5 * (bbox_max[2] - bbox_min[2])};
  }
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
  // Stamped from node->get_clock()->now() (RCL_ROS_TIME). Default-constructed
  // rclcpp::Time is RCL_SYSTEM_TIME, so comparing the two throws
  // "can't compare times with different time sources" — pin the clock type here.
  rclcpp::Time world_map_stamp{0, 0, RCL_ROS_TIME};
  ObjectGeometry target;
  ObjectGeometry destination;
  std::string target_label;
  std::string destination_label;

  // ── /grasp_candidates (vgn_grasp_node) ──────────────────────────────────
  bool grasp_candidates_fresh = false;
  rclcpp::Time grasp_candidates_stamp{0, 0, RCL_ROS_TIME};
  std::vector<GraspCandidate> grasp_candidates;  // sorted best-first by VGN

  // ── /qwen/grounding_result ───────────────────────────────────────────────
  bool grounding_result_fresh = false;
  // False for pick-only ("pick up the book"), where qwen omits the key. The
  // spec is cleared to match, or the previous command's destination survives.
  bool has_destination = false;
  DestinationSpec destination_spec;

  // ── /yolo/world_map (yolo_world_map_node) ───────────────────────────────
  std::vector<YoloObject> yolo_objects;
  rclcpp::Time yolo_world_map_stamp{0, 0, RCL_ROS_TIME};

  // ── /yolo/target_centroid ────────────────────────────────────────────────
  geometry_msgs::msg::PointStamped target_centroid_live;
  rclcpp::Time target_centroid_stamp{0, 0, RCL_ROS_TIME};

  // ── /bt/hazard_level ─────────────────────────────────────────────────────
  // 0=clear  1=slow (obstacle detected)  3=halt (arm/person detected)
  int hazard_level = 0;

  // ── /joint_states ────────────────────────────────────────────────────────
  sensor_msgs::msg::JointState latest_joint_state;
  bool has_joint_state = false;

  // Observation pose, latched from the first /joint_states. The BT starts
  // before the arm has moved, so that first sample IS the pose the scene loads
  // at — which is the pose camera_extrinsics.yaml was captured at. Overridden
  // by the observation_joint_values parameter when it is set.
  std::vector<double> observation_joint_values;
  bool has_observation_pose = false;

  // ── Replan / cycle coordination ──────────────────────────────────────────
  // WaitForScene only unblocks when both stamps are newer than this.
  rclcpp::Time last_processed_stamp{0, 0, RCL_ROS_TIME};
  bool awaiting_replan = false;
};

}  // namespace bt_pkg
