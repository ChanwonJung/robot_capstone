"""
graspgen.launch.py — graspgen_node 단독 실행 (기존 파이프라인 끝에 붙이기용)

전제: GSAM + Qwen stub + Projection 파이프라인이 이미 실행 중이어야 함.
전제: SSH tunnel이 열려 있어야 함.

Usage:
  ros2 launch graspgen_pkg graspgen.launch.py
  ros2 launch graspgen_pkg graspgen.launch.py zmq_host:=127.0.0.1 zmq_port:=5556
  ros2 launch graspgen_pkg graspgen.launch.py \
    zmq_host:=127.0.0.1 zmq_port:=5558 topk_num_grasps:=3

SSH tunnel:
  # 학내망
  ssh -N -L 5556:aurora-g5:5556 <user>@aurora.khu.ac.kr
  # 외부망
  ssh -p 30080 -N -L 5556:aurora-g5:5556 <user>@aurora.khu.ac.kr
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:

    pkg = get_package_share_directory('graspgen_pkg')

    args = [
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
            'min_point_count',
            default_value='50',
            description='Minimum TARGET point count to trigger inference',
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

    node = Node(
        package    = 'graspgen_pkg',
        executable = 'graspgen_node',
        name       = 'graspgen_node',
        output     = 'screen',
        parameters = [
            os.path.join(pkg, 'config', 'graspgen_params.yaml'),
            {
                'zmq_host':         LaunchConfiguration('zmq_host'),
                'zmq_port':         LaunchConfiguration('zmq_port'),
                'zmq_timeout_ms':   LaunchConfiguration('zmq_timeout_ms'),
                'num_grasps':       LaunchConfiguration('num_grasps'),
                'topk_num_grasps':  LaunchConfiguration('topk_num_grasps'),
                'min_point_count':  LaunchConfiguration('min_point_count'),
                'extrinsics_config':LaunchConfiguration('extrinsics_config'),
                'world_frame':      LaunchConfiguration('world_frame'),
                'robot_frame':      LaunchConfiguration('robot_frame'),
            },
        ],
    )

    return LaunchDescription(args + [node])
