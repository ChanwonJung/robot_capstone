from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("input_topic", default_value="/world_map_result"),
            DeclareLaunchArgument("pre_grasp_topic", default_value="/pre_grasp_target_pose"),
            DeclareLaunchArgument("grasp_topic", default_value="/grasp_target_pose"),
            DeclareLaunchArgument("world_frame", default_value="world"),
            DeclareLaunchArgument("min_point_count", default_value="100"),
            DeclareLaunchArgument("grasp_z_offset", default_value="0.03"),
            DeclareLaunchArgument("pre_grasp_z_offset", default_value="0.12"),
            DeclareLaunchArgument("min_z", default_value="-0.5"),
            DeclareLaunchArgument("max_z", default_value="2.0"),
            Node(
                package="target_pose_bridge_pkg",
                executable="target_pose_bridge_node",
                output="screen",
                parameters=[
                    {
                        "input_topic": LaunchConfiguration("input_topic"),
                        "pre_grasp_topic": LaunchConfiguration("pre_grasp_topic"),
                        "grasp_topic": LaunchConfiguration("grasp_topic"),
                        "world_frame": LaunchConfiguration("world_frame"),
                        "min_point_count": LaunchConfiguration("min_point_count"),
                        "grasp_z_offset": LaunchConfiguration("grasp_z_offset"),
                        "pre_grasp_z_offset": LaunchConfiguration("pre_grasp_z_offset"),
                        "min_z": LaunchConfiguration("min_z"),
                        "max_z": LaunchConfiguration("max_z"),
                    }
                ],
            ),
        ]
    )
