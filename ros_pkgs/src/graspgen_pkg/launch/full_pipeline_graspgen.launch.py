"""
full_pipeline_graspgen.launch.py — GSAM 추론 파이프라인 전체 (GraspGen 버전)

vgn_grasp_pkg 대신 graspgen_pkg를 사용. 기존 full_pipeline.launch.py와 동일 구조.

  T1: ros2 launch rgbd_projection rgbd_sim.launch.py    ← Gazebo + RViz (별도 터미널)
  T2: ros2 launch graspgen_pkg full_pipeline_graspgen.launch.py

SSH tunnel (T2 실행 전):
  ssh -N -L 5556:aurora-g5:5556 <user>@aurora.khu.ac.kr

Usage:
  ros2 launch graspgen_pkg full_pipeline_graspgen.launch.py
  ros2 launch graspgen_pkg full_pipeline_graspgen.launch.py \
    prompt:="glass cup, table" zmq_port:=5556 topk_num_grasps:=5
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:

    pkg_gsam      = get_package_share_directory('grounded_sam_pkg')
    pkg_proj      = get_package_share_directory('mask_projection_pkg')
    pkg_graspgen  = get_package_share_directory('graspgen_pkg')

    args = [
        # ── scene / GSAM ──────────────────────────────────────────────────────
        DeclareLaunchArgument(
            'prompt',
            default_value='cup, table, object',
            description='GSAM detection prompt (comma-separated object names)',
        ),
        DeclareLaunchArgument(
            'process_once',
            default_value='true',
            description='GSAM: 첫 탐지 성공 후 구독 해제 (데모용)',
        ),
        # ── ZMQ / GraspGen ────────────────────────────────────────────────────
        DeclareLaunchArgument(
            'zmq_host',
            default_value='127.0.0.1',
            description='GraspGen server host (SSH tunnel endpoint)',
        ),
        DeclareLaunchArgument(
            'zmq_port',
            default_value='5556',
            description='GraspGen server port (SSH tunnel local port)',
        ),
        DeclareLaunchArgument(
            'zmq_timeout_ms',
            default_value='5000',
            description='ZMQ recv timeout (ms)',
        ),
        DeclareLaunchArgument(
            'num_grasps',
            default_value='50',
            description='Total grasp candidates to request from server',
        ),
        DeclareLaunchArgument(
            'topk_num_grasps',
            default_value='5',
            description='Top-K grasp candidates to publish',
        ),
        DeclareLaunchArgument(
            'extrinsics_config',
            default_value='',
            description='Path to camera_extrinsics.yaml (empty = auto)',
        ),
        DeclareLaunchArgument(
            'world_frame',
            default_value='world',
            description='World frame ID',
        ),
        DeclareLaunchArgument(
            'robot_frame',
            default_value='panda_link0',
            description='Robot base frame for output poses',
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

    graspgen = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_graspgen, 'launch', 'graspgen.launch.py')
        ),
        launch_arguments={
            'zmq_host':         LaunchConfiguration('zmq_host'),
            'zmq_port':         LaunchConfiguration('zmq_port'),
            'zmq_timeout_ms':   LaunchConfiguration('zmq_timeout_ms'),
            'num_grasps':       LaunchConfiguration('num_grasps'),
            'topk_num_grasps':  LaunchConfiguration('topk_num_grasps'),
            'extrinsics_config':LaunchConfiguration('extrinsics_config'),
            'world_frame':      LaunchConfiguration('world_frame'),
            'robot_frame':      LaunchConfiguration('robot_frame'),
        }.items(),
    )

    return LaunchDescription(args + [gsam, qwen_stub, projector, graspgen])
