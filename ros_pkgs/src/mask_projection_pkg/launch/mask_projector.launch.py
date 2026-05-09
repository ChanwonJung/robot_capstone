"""
mask_projector.launch.py

Launches MaskProjectorNode with configurable topic names.

Default values match the Gazebo Harmonic + ros_gz_bridge setup.
Override any argument to adapt to a different simulator (e.g. Isaac Sim).

Example — Gazebo (default, no overrides needed):
  ros2 launch mask_projection_pkg mask_projector.launch.py

Example — Isaac Sim with different topic names:
  ros2 launch mask_projection_pkg mask_projector.launch.py \
    depth_topic:=/isaac/depth \
    camera_info_topic:=/isaac/camera_info \
    output_frame_id:=camera_frame
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # ── simulator adapter topics ───────────────────────────────────────────
        DeclareLaunchArgument(
            'depth_topic',
            default_value='/rgbd_camera/depth_image',
            description='Depth image topic (sensor_msgs/Image, 32FC1 meters)',
        ),
        DeclareLaunchArgument(
            'camera_info_topic',
            default_value='/rgbd_camera/camera_info',
            description='CameraInfo topic (sensor_msgs/CameraInfo)',
        ),
        # ── upstream segmentation topics ──────────────────────────────────────
        DeclareLaunchArgument(
            'mask_topic',
            default_value='/qwen/mask_image',
            description='Mask label map topic (sensor_msgs/Image, mono8, 1-based index)',
        ),
        DeclareLaunchArgument(
            'detections_topic',
            default_value='/qwen/labeled_detections',
            description='Detection JSON topic (std_msgs/String)',
        ),
        # ── output topics ─────────────────────────────────────────────────────
        DeclareLaunchArgument(
            'output_cloud_topic',
            default_value='/labeled_points',
            description='Labeled PointCloud2 output topic',
        ),
        DeclareLaunchArgument(
            'output_result_topic',
            default_value='/projection_result',
            description='Projection result JSON output topic',
        ),
        DeclareLaunchArgument(
            'output_frame_id',
            default_value='',
            description=(
                'Override PointCloud2 frame_id. '
                'Empty = use frame_id from incoming depth message.'
            ),
        ),
        # ── depth filter ──────────────────────────────────────────────────────
        DeclareLaunchArgument(
            'min_depth', default_value='0.05',
            description='Minimum valid depth (m)',
        ),
        DeclareLaunchArgument(
            'max_depth', default_value='15.0',
            description='Maximum valid depth (m)',
        ),
        DeclareLaunchArgument(
            'initials', default_value='',
            description='Prompt initials for output filenames, e.g. "tc" for "table, cup"',
        ),
        # ── node ──────────────────────────────────────────────────────────────
        Node(
            package='mask_projection_pkg',
            executable='mask_projector_node',
            name='mask_projector_node',
            parameters=[{
                'depth_topic':         LaunchConfiguration('depth_topic'),
                'camera_info_topic':   LaunchConfiguration('camera_info_topic'),
                'mask_topic':          LaunchConfiguration('mask_topic'),
                'detections_topic':    LaunchConfiguration('detections_topic'),
                'output_cloud_topic':  LaunchConfiguration('output_cloud_topic'),
                'output_result_topic': LaunchConfiguration('output_result_topic'),
                'output_frame_id':     LaunchConfiguration('output_frame_id'),
                'min_depth':           LaunchConfiguration('min_depth'),
                'max_depth':           LaunchConfiguration('max_depth'),
                'initials':            LaunchConfiguration('initials'),
            }],
            output='screen',
        ),
    ])
