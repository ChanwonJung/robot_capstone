#pragma once

#include <memory>
#include <string>

#include <behaviortree_cpp/condition_node.h>
#include <behaviortree_cpp/bt_factory.h>
#include <rclcpp/clock.hpp>

#include "bt_pkg/scene_data.hpp"

namespace bt_pkg {

// ─────────────────────────────────────────────────────────────────────────────
// EmergencyStopClear
// Reads hazard_level from SceneData (populated by /bt/hazard_level sub).
// SUCCESS when level < 3 (normal and slow-down hazards handled by hybrid planner).
// FAILURE when level == 3 (arm / person detected) — halts the whole tree via
// the wrapping ReactiveSequence.
// ─────────────────────────────────────────────────────────────────────────────
class EmergencyStopClear : public BT::ConditionNode {
public:
  EmergencyStopClear(const std::string& name, const BT::NodeConfig& config,
                     std::shared_ptr<SceneData> scene);

  static BT::PortsList providedPorts() { return {}; }
  BT::NodeStatus tick() override;

private:
  std::shared_ptr<SceneData> scene_;
};

// ─────────────────────────────────────────────────────────────────────────────
// TargetVisible
// Checks that /yolo/target_centroid has been published within staleness_sec
// AND the live centroid is within search_radius_m of the initial target centroid
// stored on the blackboard ("target_centroid" key).
// SUCCESS = target still where we expect it.
// FAILURE = target lost or hasn't been seen recently → triggers Pick Recovery.
// ─────────────────────────────────────────────────────────────────────────────
class TargetVisible : public BT::ConditionNode {
public:
  TargetVisible(const std::string& name, const BT::NodeConfig& config,
                std::shared_ptr<SceneData> scene,
                rclcpp::Clock::SharedPtr clock,
                double search_radius_m,
                double staleness_sec);

  static BT::PortsList providedPorts() { return {}; }
  BT::NodeStatus tick() override;

private:
  std::shared_ptr<SceneData> scene_;
  rclcpp::Clock::SharedPtr   clock_;
  double search_radius_m_;
  double staleness_sec_;
};

}  // namespace bt_pkg
