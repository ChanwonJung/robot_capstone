"""bt_system.launch.py

Launches the three nodes that run on the development/teammate node as the
behavior tree package:

  1. hazard_level_translator_node  (immediate — safety node)
  2. yolo_world_map_node           (immediate — perception support)
  3. bt_executor_node              (delayed 5 s — waits for action servers)

Prerequisites (must be running before this launch):
  • slow_brain launch  → /world_map_result, /grasp_candidates, /qwen/grounding_result
  • moveit_bridge launch → /run_hybrid_planning, /move_action, /gripper_command
  • yolo_hazard launch → /yolo_hazard/top/detections_json, /yolo_hazard/ee/detections_json
  • Isaac Sim / joint_trajectory_bridge → /joint_states, cameras, depth

Usage:
  ros2 launch bt_pkg bt_system.launch.py \\
    extrinsics_config:=/abs/path/camera_extrinsics_isaac.yaml
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _robot_defaults() -> str:
    """Return absolute path to config/robot_defaults.yaml at repo root."""
    root = os.environ.get(
        "ROBOT_CAPSTONE_ROOT",
        os.path.realpath(os.path.join(
            get_package_share_directory("bt_pkg"), *([".."] * 4))),
    )
    return os.path.join(root, "config", "robot_defaults.yaml")


def generate_launch_description():
    pkg_share = get_package_share_directory("bt_pkg")
    params_file = os.path.join(pkg_share, "config", "bt_params.yaml")
    defaults_file = _robot_defaults()

    # ── Launch arguments ────────────────────────────────────────────────────
    ext_arg = DeclareLaunchArgument(
        "extrinsics_config",
        default_value="",
        description="Absolute path to camera_extrinsics_isaac.yaml",
    )

    tree_arg = DeclareLaunchArgument(
        "tree_file",
        default_value=os.path.join(pkg_share, "behavior_trees", "pick_and_place.xml"),
        description="Absolute path to BT XML file (override for custom trees)",
    )

    # ── Nodes ───────────────────────────────────────────────────────────────

    # Safety first — up before anything else so the E-stop check is live
    # even before the BT starts ticking.
    hazard_translator = Node(
        package="bt_pkg",
        executable="hazard_level_translator_node.py",
        name="hazard_level_translator_node",
        output="screen",
        parameters=[defaults_file, params_file],
    )

    # YOLO 3D tracker — feeds UpdateTargetPose and TargetVisible.
    yolo_world_map = Node(
        package="bt_pkg",
        executable="yolo_world_map_node.py",
        name="yolo_world_map_node",
        output="screen",
        parameters=[
            defaults_file,
            params_file,
            {"extrinsics_config": LaunchConfiguration("extrinsics_config")},
        ],
    )

    # BT executor — delayed to give the hybrid planner and gripper server
    # time to finish their own startup sequences.
    bt_executor = TimerAction(
        period=5.0,
        actions=[
            Node(
                package="bt_pkg",
                executable="bt_executor_node",
                name="bt_executor_node",
                output="screen",
                parameters=[
                    defaults_file,
                    params_file,
                    {"tree_file": LaunchConfiguration("tree_file")},
                ],
            )
        ],
    )

    return LaunchDescription([
        ext_arg,
        tree_arg,
        hazard_translator,
        yolo_world_map,
        bt_executor,
    ])
