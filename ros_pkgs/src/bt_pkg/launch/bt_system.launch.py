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
from launch_ros.parameter_descriptions import ParameterValue


def _robot_defaults() -> str:
    """Return absolute path to config/robot_defaults.yaml at repo root."""
    root = os.environ.get(
        "ROBOT_CAPSTONE_ROOT",
        os.path.realpath(os.path.join(
            get_package_share_directory("bt_pkg"), *([".."] * 4))),
    )
    return os.path.join(root, "config", "robot_defaults.yaml")


def _default_extrinsics() -> str:
    """mask_projection_pkg 의 camera_extrinsics.yaml — Isaac 씬 실측 R/t."""
    root = os.environ.get(
        "ROBOT_CAPSTONE_ROOT",
        os.path.realpath(os.path.join(
            get_package_share_directory("bt_pkg"), *([".."] * 4))),
    )
    return os.path.join(
        root, "ros_pkgs", "src", "mask_projection_pkg",
        "config", "camera_extrinsics.yaml")


def generate_launch_description():
    pkg_share = get_package_share_directory("bt_pkg")
    params_file = os.path.join(pkg_share, "config", "bt_params.yaml")
    defaults_file = _robot_defaults()

    # ── Launch arguments ────────────────────────────────────────────────────
    ext_arg = DeclareLaunchArgument(
        "extrinsics_config",
        default_value=_default_extrinsics(),
        description=("camera_extrinsics.yaml 경로. 기본값은 mask_projection_pkg "
                     "안의 Isaac 실측본이라 보통 넘길 필요가 없다."),
    )

    tree_arg = DeclareLaunchArgument(
        "tree_file",
        default_value=os.path.join(pkg_share, "behavior_trees", "pick_and_place.xml"),
        description="Absolute path to BT XML file (override for custom trees)",
    )

    # bt_params.yaml 의 0.12 를 덮어쓴다. SelectGraspCandidate 가 이 거리만큼
    # grasp pose 에서 접근축 반대로 물러난 지점을 pre_grasp 으로 쓰는데, 길수록
    # 하강 구간에서 OMPL 이 옆으로 새어 물체를 칠 여지가 커진다. 책(높이 104mm,
    # 두께 26.6mm)처럼 조 사이로 좁게 밀어 넣어야 하는 대상에서 0.12 는 과했다.
    # 0.03 은 실기 검증값 — 짧게 바꾸는 게 목적이지 물체별 튜닝값은 아니므로
    # 도달이 빠듯하면 인자로 늘리면 된다.
    pre_grasp_arg = DeclareLaunchArgument(
        "pre_grasp_z_offset",
        default_value="0.03",
        description=("grasp pose 에서 접근축 반대로 물러나는 거리(m). "
                     "bt_params.yaml 값을 덮어쓴다."),
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
                    {
                        "tree_file": LaunchConfiguration("tree_file"),
                        # params_file 뒤에 와야 덮어쓴다.
                        "pre_grasp_z_offset": ParameterValue(
                            LaunchConfiguration("pre_grasp_z_offset"),
                            value_type=float),
                    },
                ],
            )
        ],
    )

    return LaunchDescription([
        ext_arg,
        tree_arg,
        pre_grasp_arg,
        hazard_translator,
        yolo_world_map,
        bt_executor,
    ])
