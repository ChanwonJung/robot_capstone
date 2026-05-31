"""
vgn_grasp.launch.py

Launch vgn_grasp_node.

Prerequisites:
  1. git submodule add https://github.com/ethz-asl/vgn external/vgn
  2. source launch_env.bash  (sets ROBOT_CAPSTONE_ROOT, activates venv)
  3. pip install pytorch-ignite  (inside venv)
  4. Place VGN weights at: $ROBOT_CAPSTONE_ROOT/models/vgn_conv.pth

Usage:
  ros2 launch vgn_grasp_pkg vgn_grasp.launch.py

Override example:
  ros2 launch vgn_grasp_pkg vgn_grasp.launch.py \\
    vgn_model_path:=/abs/path/to/model.pt \\
    min_quality:=0.4 \\
    max_grasp_candidates:=3 \\
    use_top_depth:=false \\
    extrinsics_config:=/path/to/camera_extrinsics_isaac.yaml
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _robot_defaults() -> str:
    root = os.environ.get(
        "ROBOT_CAPSTONE_ROOT",
        os.path.realpath(os.path.join(
            get_package_share_directory("vgn_grasp_pkg"), *([".."] * 4))),
    )
    return os.path.join(root, "config", "robot_defaults.yaml")


def generate_launch_description() -> LaunchDescription:
    pkg_share  = get_package_share_directory("vgn_grasp_pkg")
    vgn_params = os.path.join(pkg_share, "config", "vgn_params.yaml")
    defaults   = _robot_defaults()

    args = [
        DeclareLaunchArgument('roi_size_m',             default_value='0.30'),
        DeclareLaunchArgument('tsdf_resolution',        default_value='40'),
        DeclareLaunchArgument('vgn_model_path',         default_value='models/vgn_conv.pth'),
        DeclareLaunchArgument('min_quality',            default_value='0.5'),
        DeclareLaunchArgument('max_grasp_candidates',   default_value='5'),
        DeclareLaunchArgument('min_point_count',        default_value='50'),
        DeclareLaunchArgument('ee_depth_topic',         default_value='/ee_camera/depth_image'),
        DeclareLaunchArgument('ee_camera_info_topic',   default_value='/ee_camera/camera_info'),
        DeclareLaunchArgument('top_depth_topic',        default_value='/top_camera/depth_image'),
        DeclareLaunchArgument('top_camera_info_topic',  default_value='/top_camera/camera_info'),
        DeclareLaunchArgument('world_map_result_topic', default_value='/world_map_result'),
        DeclareLaunchArgument('grasp_candidates_topic', default_value='/grasp_candidates'),
        DeclareLaunchArgument('extrinsics_config',      default_value=''),
        DeclareLaunchArgument('use_top_depth',          default_value='true'),
        DeclareLaunchArgument('top_occlude_filter',     default_value='true'),
        DeclareLaunchArgument('trunc_factor',           default_value='4.0'),
        DeclareLaunchArgument('world_frame',            default_value='world'),
        DeclareLaunchArgument('robot_frame',            default_value='panda_link0'),
    ]

    node = Node(
        package    = 'vgn_grasp_pkg',
        executable = 'vgn_grasp_node',
        name       = 'vgn_grasp_node',
        output     = 'screen',
        parameters = [
            defaults,
            vgn_params,
            {
                'roi_size_m':             LaunchConfiguration('roi_size_m'),
                'tsdf_resolution':        LaunchConfiguration('tsdf_resolution'),
                'vgn_model_path':         LaunchConfiguration('vgn_model_path'),
                'min_quality':            LaunchConfiguration('min_quality'),
                'max_grasp_candidates':   LaunchConfiguration('max_grasp_candidates'),
                'min_point_count':        LaunchConfiguration('min_point_count'),
                'ee_depth_topic':         LaunchConfiguration('ee_depth_topic'),
                'ee_camera_info_topic':   LaunchConfiguration('ee_camera_info_topic'),
                'top_depth_topic':        LaunchConfiguration('top_depth_topic'),
                'top_camera_info_topic':  LaunchConfiguration('top_camera_info_topic'),
                'world_map_result_topic': LaunchConfiguration('world_map_result_topic'),
                'grasp_candidates_topic': LaunchConfiguration('grasp_candidates_topic'),
                'extrinsics_config':      LaunchConfiguration('extrinsics_config'),
                'use_top_depth':          LaunchConfiguration('use_top_depth'),
                'top_occlude_filter':     LaunchConfiguration('top_occlude_filter'),
                'trunc_factor':           LaunchConfiguration('trunc_factor'),
                'world_frame':            LaunchConfiguration('world_frame'),
                'robot_frame':            LaunchConfiguration('robot_frame'),
            },
        ],
    )

    return LaunchDescription(args + [node])
