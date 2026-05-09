import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("grounded_sam_pkg")
    default_config = os.path.join(pkg_share, "config", "model_paths.yaml")

    return LaunchDescription([
        DeclareLaunchArgument(
            "model_config",
            default_value=default_config,
            description="Path to model_paths.yaml",
        ),
        DeclareLaunchArgument(
            "prompt",
            default_value="object",
            description="Object noun phrase, e.g. 'bottle, cup'",
        ),
        DeclareLaunchArgument(
            "image_topic",
            default_value="/ee_camera/image",
            description="Image topic to subscribe to",
        ),
        Node(
            package="grounded_sam_pkg",
            executable="grounded_sam_node",
            name="grounded_sam_node",
            parameters=[{
                "model_config": LaunchConfiguration("model_config"),
                "prompt": LaunchConfiguration("prompt"),
                "image_topic": LaunchConfiguration("image_topic"),
            }],
            output="screen",
        ),
    ])
