"""
full_pipeline_4cam.launch.py

4-camera TSDF 전체 파이프라인 (Gazebo/RViz 제외).

  T1: ros2 launch rgbd_projection rgbd_sim_4cam.launch.py   ← 4cam Gazebo (별도)
  T2: ros2 launch vgn_grasp_pkg full_pipeline_4cam.launch.py

Usage:
  ros2 launch vgn_grasp_pkg full_pipeline_4cam.launch.py \\
    vgn_model_path:=models/vgn_conv.pth

Override examples:
  # 사이드 카메라 끄고 비교
  ros2 launch vgn_grasp_pkg full_pipeline_4cam.launch.py \\
    vgn_model_path:=models/vgn_conv.pth use_side_depth:=false

  # top occlude filter 끄고 테스트
  ros2 launch vgn_grasp_pkg full_pipeline_4cam.launch.py \\
    vgn_model_path:=models/vgn_conv.pth top_occlude_filter:=false
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

    extrinsics_4cam = os.path.join(
        get_package_share_directory('mask_projection_pkg'),
        'config', 'camera_extrinsics_4cam.yaml',
    )

    args = [
        DeclareLaunchArgument(
            'prompt',
            default_value='cup, table',
            description='GSAM detection prompt',
        ),
        DeclareLaunchArgument(
            'vgn_model_path',
            default_value='models/vgn_conv.pth',
            description='VGN weight path (absolute or $ROBOT_CAPSTONE_ROOT-relative)',
        ),
        DeclareLaunchArgument(
            'min_quality',
            default_value='0.5',
            description='VGN grasp quality threshold',
        ),
        DeclareLaunchArgument(
            'max_grasp_candidates',
            default_value='5',
            description='Top-K grasp candidates',
        ),
        DeclareLaunchArgument(
            'process_once',
            default_value='true',
            description='GSAM: 첫 탐지 후 구독 해제 (데모용)',
        ),
        DeclareLaunchArgument(
            'use_side_depth',
            default_value='true',
            description='right/left 사이드 카메라 TSDF 적분 여부',
        ),
        DeclareLaunchArgument(
            'use_top_depth',
            default_value='true',
            description='top 카메라 TSDF 적분 여부',
        ),
        DeclareLaunchArgument(
            'top_occlude_filter',
            default_value='true',
            description='top 카메라 occlude filter (컵 측면 벽 보호)',
        ),
        DeclareLaunchArgument(
            'trunc_factor',
            default_value='4.0',
            description='TSDF truncation = trunc_factor × voxel_size',
        ),
        DeclareLaunchArgument(
            'ee_weight',
            default_value='6.0',
            description='EE 카메라 TSDF 가중치',
        ),
        DeclareLaunchArgument(
            'top_weight',
            default_value='4.0',
            description='Top 카메라 TSDF 가중치',
        ),
        DeclareLaunchArgument(
            'side_weight',
            default_value='4.0',
            description='Right/Left 사이드 카메라 TSDF 가중치',
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
            os.path.join(pkg_vgn, 'launch', 'vgn_grasp_4cam.launch.py')
        ),
        launch_arguments={
            'vgn_model_path':       LaunchConfiguration('vgn_model_path'),
            'min_quality':          LaunchConfiguration('min_quality'),
            'max_grasp_candidates': LaunchConfiguration('max_grasp_candidates'),
            'use_side_depth':       LaunchConfiguration('use_side_depth'),
            'use_top_depth':        LaunchConfiguration('use_top_depth'),
            'top_occlude_filter':   LaunchConfiguration('top_occlude_filter'),
            'trunc_factor':         LaunchConfiguration('trunc_factor'),
            'ee_weight':            LaunchConfiguration('ee_weight'),
            'top_weight':           LaunchConfiguration('top_weight'),
            'side_weight':          LaunchConfiguration('side_weight'),
            'extrinsics_config':    extrinsics_4cam,
        }.items(),
    )

    return LaunchDescription(args + [gsam, qwen_stub, projector, vgn])
