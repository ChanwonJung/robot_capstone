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
        DeclareLaunchArgument(
            "annotated_topic",
            default_value="/grounded_sam/annotated_image",
            description="Annotated image topic to publish",
        ),
        DeclareLaunchArgument(
            "mask_topic",
            default_value="/grounded_sam/mask_image",
            description="Mask image topic to publish",
        ),
        DeclareLaunchArgument(
            "detections_topic",
            default_value="/grounded_sam/detections_json",
            description="Detections JSON topic to publish",
        ),
        DeclareLaunchArgument(
            "output_subdir",
            default_value="",
            description="Subdirectory under robot_capstone/output for saved images",
        ),
        DeclareLaunchArgument(
            "process_every_n_frames",
            default_value="30",
            description="Run Grounded SAM once every N incoming frames",
        ),
        DeclareLaunchArgument(
            "min_process_interval_sec",
            default_value="1.0",
            description="Minimum time gap between inference runs",
        ),
        # Optional dual-view (Top) inputs/outputs. Leave top_image_topic empty
        # for single-view (EE-only) mode.
        DeclareLaunchArgument(
            "top_image_topic",
            default_value="",
            description="Top camera image topic. Empty disables dual-view.",
        ),
        DeclareLaunchArgument(
            "top_prompt",
            default_value="",
            description="Optional Top-view prompt. Empty reuses 'prompt'.",
        ),
        DeclareLaunchArgument(
            "top_annotated_topic",
            default_value="/top/grounded_sam/annotated_image",
        ),
        DeclareLaunchArgument(
            "top_mask_topic",
            default_value="/top/grounded_sam/mask_image",
        ),
        DeclareLaunchArgument(
            "top_detections_topic",
            default_value="/top/grounded_sam/detections_json",
        ),
        # Top depth gating — exclude pixels outside [top_min_depth, top_max_depth]
        # from the Top RGB image before g-sam runs (e.g. to hide the robot arm).
        DeclareLaunchArgument(
            "top_depth_topic",
            default_value="",
            description="Top depth topic. Empty disables Top depth masking.",
        ),
        DeclareLaunchArgument(
            "top_min_depth",
            default_value="0.0",
            description="Top depth lower bound in meters (inclusive).",
        ),
        DeclareLaunchArgument(
            "top_max_depth",
            default_value="100.0",
            description="Top depth upper bound in meters (inclusive).",
        ),
        Node(
            package="grounded_sam_pkg",
            executable="grounded_sam_node",
            name="grounded_sam_node",
            parameters=[{
                "model_config": LaunchConfiguration("model_config"),
                "prompt": LaunchConfiguration("prompt"),
                "image_topic": LaunchConfiguration("image_topic"),
                "annotated_topic": LaunchConfiguration("annotated_topic"),
                "mask_topic": LaunchConfiguration("mask_topic"),
                "detections_topic": LaunchConfiguration("detections_topic"),
                "output_subdir": LaunchConfiguration("output_subdir"),
                "process_every_n_frames": LaunchConfiguration("process_every_n_frames"),
                "min_process_interval_sec": LaunchConfiguration("min_process_interval_sec"),
                "top_image_topic": LaunchConfiguration("top_image_topic"),
                "top_prompt": LaunchConfiguration("top_prompt"),
                "top_annotated_topic": LaunchConfiguration("top_annotated_topic"),
                "top_mask_topic": LaunchConfiguration("top_mask_topic"),
                "top_detections_topic": LaunchConfiguration("top_detections_topic"),
                "top_depth_topic": LaunchConfiguration("top_depth_topic"),
                "top_min_depth": LaunchConfiguration("top_min_depth"),
                "top_max_depth": LaunchConfiguration("top_max_depth"),
            }],
            output="screen",
        ),
    ])
