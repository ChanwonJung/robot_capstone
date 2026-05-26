/**
 * bt_executor_node.cpp
 *
 * Behavior tree executor for the robot capstone pick-and-place pipeline.
 *
 * Owns all ROS subscriptions (updates SceneData), constructs the BT factory,
 * registers every node type, loads pick_and_place.xml, and ticks the tree at
 * 10 Hz via a wall timer.
 *
 * Hazard handling summary:
 *   Level 0 – clear         : normal operation
 *   Level 1 – slow          : hybrid planner + hazard_collision_injector handle it
 *   Level 3 – halt          : EmergencyStopClear returns FAILURE → tree suspends
 */

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_ros2/bt_action_node.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/int8.hpp>
#include <std_msgs/msg/string.hpp>

#include <nlohmann/json.hpp>

#include "bt_pkg/action_nodes.hpp"
#include "bt_pkg/condition_nodes.hpp"
#include "bt_pkg/destination_calculator.hpp"
#include "bt_pkg/scene_data.hpp"

using json = nlohmann::json;

// ── JSON parsers ─────────────────────────────────────────────────────────────

static void parse_world_map_result(const std::string& raw,
                                   bt_pkg::SceneData& scene)
{
  try {
    auto j = json::parse(raw);

    if (j.contains("target")) {
      auto& t = j["target"];
      scene.target_label = t.value("label", "");
      if (t.contains("centroid")) {
        scene.target_centroid = {t["centroid"][0], t["centroid"][1], t["centroid"][2]};
      }
    }

    if (j.contains("destination")) {
      auto& d = j["destination"];
      scene.destination_label = d.value("label", "");
      if (d.contains("centroid")) {
        scene.destination_centroid = {
          d["centroid"][0], d["centroid"][1], d["centroid"][2]};
      }
    }
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("bt_executor"),
      "parse_world_map_result failed: %s", e.what());
  }
}

static void parse_grasp_candidates(const std::string& raw,
                                   bt_pkg::SceneData& scene)
{
  try {
    auto j = json::parse(raw);
    scene.grasp_candidates.clear();

    for (const auto& c : j.value("candidates", json::array())) {
      bt_pkg::GraspCandidate gc;
      gc.pose.header.frame_id = c.value("frame", "panda_link0");

      auto& pos = c["position"];
      gc.pose.pose.position.x = pos[0];
      gc.pose.pose.position.y = pos[1];
      gc.pose.pose.position.z = pos[2];

      auto& q = c["quaternion"];
      gc.pose.pose.orientation.x = q[0];
      gc.pose.pose.orientation.y = q[1];
      gc.pose.pose.orientation.z = q[2];
      gc.pose.pose.orientation.w = q[3];

      gc.quality = c.value("quality", 0.0);
      gc.width   = c.value("width",   0.08);
      scene.grasp_candidates.push_back(gc);
    }
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("bt_executor"),
      "parse_grasp_candidates failed: %s", e.what());
  }
}

static void parse_grounding_result(const std::string& raw,
                                   bt_pkg::SceneData& scene)
{
  // Mirrors GroundingResult / DestinationSpec from qwen_call.py
  try {
    auto j = json::parse(raw);
    auto& spec = scene.destination_spec;

    if (j.contains("destination")) {
      auto& d = j["destination"];
      spec.type             = d.value("type", "");
      spec.reference_label  = d.value("reference_label", "");
      spec.relation         = d.value("relation", "");
      spec.region           = d.value("region", "");
    }
    scene.grounding_result_fresh = true;
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("bt_executor"),
      "parse_grounding_result failed: %s", e.what());
  }
}

static void parse_yolo_world_map(const std::string& raw,
                                 bt_pkg::SceneData& scene,
                                 const rclcpp::Time& stamp)
{
  try {
    auto j = json::parse(raw);
    scene.yolo_objects.clear();

    for (const auto& obj : j.value("objects", json::array())) {
      bt_pkg::YoloObject yo;
      yo.class_name  = obj.value("class_name", "");
      yo.confidence  = obj.value("confidence", 0.0);
      if (obj.contains("centroid")) {
        yo.centroid = {obj["centroid"][0], obj["centroid"][1], obj["centroid"][2]};
      }
      scene.yolo_objects.push_back(yo);
    }
    scene.yolo_world_map_stamp = stamp;
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("bt_executor"),
      "parse_yolo_world_map failed: %s", e.what());
  }
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>("bt_executor_node");

  // ── Parameters ────────────────────────────────────────────────────────────
  auto tree_file = node->declare_parameter<std::string>(
    "tree_file",
    ament_index_cpp::get_package_share_directory("bt_pkg")
      + "/behavior_trees/pick_and_place.xml");

  auto planning_group  = node->declare_parameter<std::string>("planning_group",  "panda_arm");
  auto ee_link         = node->declare_parameter<std::string>("end_effector_link","panda_link8");
  auto planning_frame  = node->declare_parameter<std::string>("planning_frame",  "panda_link0");
  auto move_action_srv = node->declare_parameter<std::string>("move_action",     "/run_hybrid_planning");
  auto move_home_srv   = node->declare_parameter<std::string>("move_home_action","/move_action");
  auto gripper_srv     = node->declare_parameter<std::string>("gripper_action",  "/gripper_command");

  auto pos_tol         = node->declare_parameter<double>("position_tolerance",    0.01);
  auto max_grasp_cands = node->declare_parameter<int>("max_grasp_candidates", 5);
  auto ori_tol         = node->declare_parameter<double>("orientation_tolerance", 0.05);
  auto pre_grasp_z     = node->declare_parameter<double>("pre_grasp_z_offset",    0.12);
  auto retreat_z       = node->declare_parameter<double>("retreat_z_offset",      0.15);
  auto target_radius   = node->declare_parameter<double>("target_search_radius_m",0.25);
  auto target_staleness= node->declare_parameter<double>("target_staleness_sec",  1.0);
  auto dest_radius     = node->declare_parameter<double>("dest_match_radius_m",   0.3);

  // "ready" state for MoveToHome
  auto home_joint_names = node->declare_parameter<std::vector<std::string>>(
    "home_joint_names",
    {"panda_joint1","panda_joint2","panda_joint3","panda_joint4",
     "panda_joint5","panda_joint6","panda_joint7"});
  auto home_joint_values = node->declare_parameter<std::vector<double>>(
    "home_joint_values", {0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785});
  auto home_joint_tol = node->declare_parameter<double>("home_joint_tolerance", 0.05);

  // Place pose geometry
  bt_pkg::PlacePoseParams place_params;
  place_params.side_offset_m    = node->declare_parameter<double>("side_offset_m",    0.08);
  place_params.place_height_m   = node->declare_parameter<double>("place_height_m",   0.05);
  place_params.container_drop_z = node->declare_parameter<double>("container_drop_z", 0.03);
  place_params.near_offset_m    = node->declare_parameter<double>("near_offset_m",    0.08);

  // ── Shared state ──────────────────────────────────────────────────────────
  auto scene = std::make_shared<bt_pkg::SceneData>();

  // ── Subscriptions ─────────────────────────────────────────────────────────
  auto sub_world_map = node->create_subscription<std_msgs::msg::String>(
    "/world_map_result", 10,
    [scene, node](const std_msgs::msg::String::SharedPtr msg) {
      std::lock_guard<std::mutex> lk(scene->mtx);
      parse_world_map_result(msg->data, *scene);
      scene->world_map_fresh = true;
      scene->world_map_stamp = node->get_clock()->now();
    });

  auto sub_grasp = node->create_subscription<std_msgs::msg::String>(
    "/grasp_candidates", 10,
    [scene, node](const std_msgs::msg::String::SharedPtr msg) {
      std::lock_guard<std::mutex> lk(scene->mtx);
      parse_grasp_candidates(msg->data, *scene);
      scene->grasp_candidates_fresh = true;
      scene->grasp_candidates_stamp = node->get_clock()->now();
    });

  auto sub_grounding = node->create_subscription<std_msgs::msg::String>(
    "/qwen/grounding_result", 10,
    [scene](const std_msgs::msg::String::SharedPtr msg) {
      std::lock_guard<std::mutex> lk(scene->mtx);
      parse_grounding_result(msg->data, *scene);
    });

  auto sub_yolo_map = node->create_subscription<std_msgs::msg::String>(
    "/yolo/world_map", 10,
    [scene, node](const std_msgs::msg::String::SharedPtr msg) {
      std::lock_guard<std::mutex> lk(scene->mtx);
      parse_yolo_world_map(msg->data, *scene, node->get_clock()->now());
    });

  auto sub_target = node->create_subscription<geometry_msgs::msg::PointStamped>(
    "/yolo/target_centroid", 10,
    [scene, node](const geometry_msgs::msg::PointStamped::SharedPtr msg) {
      std::lock_guard<std::mutex> lk(scene->mtx);
      scene->target_centroid_live  = *msg;
      scene->target_centroid_stamp = node->get_clock()->now();
    });

  auto sub_hazard = node->create_subscription<std_msgs::msg::Int8>(
    "/bt/hazard_level", 10,
    [scene](const std_msgs::msg::Int8::SharedPtr msg) {
      std::lock_guard<std::mutex> lk(scene->mtx);
      scene->hazard_level = msg->data;
    });

  auto sub_joints = node->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states", 10,
    [scene](const sensor_msgs::msg::JointState::SharedPtr msg) {
      std::lock_guard<std::mutex> lk(scene->mtx);
      scene->latest_joint_state = *msg;
      scene->has_joint_state    = true;
    });

  // ── Publisher (used by RequestReplan) ────────────────────────────────────
  auto pub_replan = node->create_publisher<std_msgs::msg::Empty>(
    "/bt/replan_request", 10);

  // ── BT Factory ────────────────────────────────────────────────────────────
  BT::BehaviorTreeFactory factory;

  // RosNodeParams per action server (action name set via default_port_value
  // so the XML can override it without touching C++).
  BT::RosNodeParams params_move;
  params_move.nh = node;
  params_move.default_port_value = move_action_srv;

  BT::RosNodeParams params_home;
  params_home.nh = node;
  params_home.default_port_value = move_home_srv;

  BT::RosNodeParams params_gripper;
  params_gripper.nh = node;
  params_gripper.default_port_value = gripper_srv;

  // ── Register action nodes ─────────────────────────────────────────────────

  factory.registerBuilder<bt_pkg::WaitForScene>(
    "WaitForScene",
    [scene](const std::string& n, const BT::NodeConfig& c) {
      return std::make_unique<bt_pkg::WaitForScene>(n, c, scene);
    });

  factory.registerBuilder<bt_pkg::ParseScene>(
    "ParseScene",
    [scene, pre_grasp_z, retreat_z](const std::string& n, const BT::NodeConfig& c) {
      return std::make_unique<bt_pkg::ParseScene>(n, c, scene, pre_grasp_z, retreat_z);
    });

  factory.registerBuilder<bt_pkg::SelectGraspCandidate>(
    "SelectGraspCandidate",
    [pre_grasp_z](const std::string& n, const BT::NodeConfig& c) {
      return std::make_unique<bt_pkg::SelectGraspCandidate>(n, c, pre_grasp_z);
    });

  factory.registerBuilder<bt_pkg::MoveAction>(
    "MoveAction",
    [params_move, scene, planning_group, ee_link, planning_frame, pos_tol, ori_tol](
      const std::string& n, const BT::NodeConfig& c)
    {
      return std::make_unique<bt_pkg::MoveAction>(
        n, c, params_move, scene,
        planning_group, ee_link, planning_frame, pos_tol, ori_tol);
    });

  factory.registerBuilder<bt_pkg::MoveToHome>(
    "MoveToHome",
    [params_home, scene, planning_group, home_joint_names, home_joint_values, home_joint_tol](
      const std::string& n, const BT::NodeConfig& c)
    {
      return std::make_unique<bt_pkg::MoveToHome>(
        n, c, params_home, scene,
        planning_group, home_joint_names, home_joint_values, home_joint_tol);
    });

  factory.registerBuilder<bt_pkg::GripperAction>(
    "GripperAction",
    [params_gripper](const std::string& n, const BT::NodeConfig& c) {
      return std::make_unique<bt_pkg::GripperAction>(n, c, params_gripper);
    });

  factory.registerBuilder<bt_pkg::UpdateTargetPose>(
    "UpdateTargetPose",
    [scene, place_params, dest_radius](const std::string& n, const BT::NodeConfig& c) {
      return std::make_unique<bt_pkg::UpdateTargetPose>(n, c, scene, place_params, dest_radius);
    });

  factory.registerBuilder<bt_pkg::RequestReplan>(
    "RequestReplan",
    [scene, pub_replan](const std::string& n, const BT::NodeConfig& c) {
      return std::make_unique<bt_pkg::RequestReplan>(n, c, scene, pub_replan);
    });

  // ── Register condition nodes ──────────────────────────────────────────────

  factory.registerBuilder<bt_pkg::EmergencyStopClear>(
    "EmergencyStopClear",
    [scene](const std::string& n, const BT::NodeConfig& c) {
      return std::make_unique<bt_pkg::EmergencyStopClear>(n, c, scene);
    });

  factory.registerBuilder<bt_pkg::TargetVisible>(
    "TargetVisible",
    [scene, node, target_radius, target_staleness](
      const std::string& nm, const BT::NodeConfig& c)
    {
      return std::make_unique<bt_pkg::TargetVisible>(
        nm, c, scene, node->get_clock(), target_radius, target_staleness);
    });

  // ── Load tree ─────────────────────────────────────────────────────────────
  RCLCPP_INFO(node->get_logger(), "Loading BT from: %s", tree_file.c_str());
  auto tree = factory.createTreeFromFile(tree_file);
  RCLCPP_INFO(node->get_logger(), "BT loaded — ticking at 10 Hz");

  // Seed blackboard with shared params so the XML can reference them as ports.
  tree.rootBlackboard()->set<int>("max_grasp_candidates", max_grasp_cands);
  RCLCPP_INFO(node->get_logger(),
    "Blackboard: max_grasp_candidates=%d", max_grasp_cands);

  // ── Tick timer (10 Hz) ───────────────────────────────────────────────────
  auto tick_timer = node->create_wall_timer(
    std::chrono::milliseconds(100),
    [&tree, &node]() {
      auto status = tree.tickOnce();
      if (status == BT::NodeStatus::SUCCESS) {
        RCLCPP_INFO(node->get_logger(), "BT cycle complete — waiting for next command");
      } else if (status == BT::NodeStatus::FAILURE) {
        // RepeatForever wraps the top level, so FAILURE here is unexpected.
        RCLCPP_ERROR(node->get_logger(), "BT returned FAILURE at root — this is a bug");
      }
    });

  // ── Spin ─────────────────────────────────────────────────────────────────
  // MultiThreadedExecutor lets RosActionNode callbacks and the tick timer run
  // concurrently without deadlocking each other.
  rclcpp::executors::MultiThreadedExecutor exec(
    rclcpp::ExecutorOptions{}, 4 /*threads*/);
  exec.add_node(node);

  try {
    exec.spin();
  } catch (const std::exception& e) {
    RCLCPP_FATAL(node->get_logger(), "bt_executor_node crashed: %s", e.what());
  }

  rclcpp::shutdown();
  return 0;
}
