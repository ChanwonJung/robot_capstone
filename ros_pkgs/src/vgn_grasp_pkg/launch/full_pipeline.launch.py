"""
full_pipeline.launch.py

GSAM 추론 파이프라인만 실행 (Gazebo/RViz 제외).

  T1: ros2 launch rgbd_projection rgbd_sim.launch.py   ← Gazebo + RViz (별도)
  T2: ros2 launch vgn_grasp_pkg full_pipeline.launch.py

Usage:
  ros2 launch vgn_grasp_pkg full_pipeline.launch.py
  ros2 launch vgn_grasp_pkg full_pipeline.launch.py prompt:="cup, table, object"
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:

    pkg_gsam = get_package_share_directory('grounded_sam_pkg')
    pkg_proj = get_package_share_directory('mask_projection_pkg')
    pkg_vgn  = get_package_share_directory('vgn_grasp_pkg')

    # ── Launch arguments ──────────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument(
            'prompt',
            default_value='cup, table, object',
            description='GSAM detection prompt (comma-separated object names)',
        ),
        DeclareLaunchArgument(
            'vgn_model_path',
            default_value='models/vgn_conv.pth',
            description='VGN weight file path (absolute or $GSAM_WS-relative)',
        ),
        DeclareLaunchArgument(
            'min_quality',
            default_value='0.5',
            description='VGN grasp quality threshold',
        ),
        DeclareLaunchArgument(
            'max_candidates',
            default_value='5',
            description='Top-K grasp candidates to publish',
        ),
        DeclareLaunchArgument(
            'process_once',
            default_value='true',
            description='GSAM: 첫 탐지 성공 후 구독 해제 (데모용)',
        ),
    ]

    gsam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gsam, 'launch', 'grounded_sam.launch.py')
        ),
        launch_arguments={
            'prompt':       LaunchConfiguration('prompt'),
            'image_topic':  '/ee_camera/image',
            'process_once': LaunchConfiguration('process_once'),
        }.items(),
    )

    qwen_stub = Node(
        package    = 'grounded_sam_pkg',
        executable = 'qwen_stub_node',
        name       = 'qwen_stub_node',
        output     = 'screen',
    )

    projector = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_proj, 'launch', 'multi_view_projector.launch.py')
        ),
    )

    vgn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_vgn, 'launch', 'vgn_grasp.launch.py')
        ),
        launch_arguments={
            'vgn_model_path': LaunchConfiguration('vgn_model_path'),
            'min_quality':    LaunchConfiguration('min_quality'),
            'max_candidates': LaunchConfiguration('max_candidates'),
        }.items(),
    )

    return LaunchDescription(args + [gsam, qwen_stub, projector, vgn])
