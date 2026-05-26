cmake_minimum_required(VERSION 3.16)
project(bt_pkg)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── dependencies ────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(behaviortree_ros2 REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(nlohmann_json REQUIRED)

# ── bt_executor_node ─────────────────────────────────────────────────────────
add_executable(bt_executor_node
  src/bt_executor_node.cpp
  src/destination_calculator.cpp
  src/action_nodes/wait_for_scene.cpp
  src/action_nodes/parse_scene.cpp
  src/action_nodes/select_grasp_candidate.cpp
  src/action_nodes/move_action.cpp
  src/action_nodes/move_to_home.cpp
  src/action_nodes/gripper_action.cpp
  src/action_nodes/update_target_pose.cpp
  src/action_nodes/request_replan.cpp
  src/condition_nodes/emergency_stop_clear.cpp
  src/condition_nodes/target_visible.cpp
)

target_include_directories(bt_executor_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${behaviortree_cpp_INCLUDE_DIRS}
)

ament_target_dependencies(bt_executor_node
  rclcpp
  rclcpp_action
  ament_index_cpp
  behaviortree_ros2
  behaviortree_cpp
  moveit_msgs
  control_msgs
  geometry_msgs
  std_msgs
  sensor_msgs
  shape_msgs
  trajectory_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

target_link_libraries(bt_executor_node
  behaviortree_ros2::behaviortree_ros2
  nlohmann_json::nlohmann_json
)

# ── install ──────────────────────────────────────────────────────────────────
install(TARGETS bt_executor_node
  DESTINATION lib/${PROJECT_NAME}
)

# Python companion scripts (hazard translator + yolo world map)
install(PROGRAMS
  scripts/hazard_level_translator_node.py
  scripts/yolo_world_map_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  behavior_trees
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
