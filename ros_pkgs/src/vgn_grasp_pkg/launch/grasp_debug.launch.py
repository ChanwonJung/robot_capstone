"""
grasp_debug.launch.py

/tsdf_debug, /grasp_markers, /world_map 확인용 RViz2만 실행.

Usage:
  T1: ros2 launch rgbd_projection rgbd_sim.launch.py
  T2: ros2 launch vgn_grasp_pkg full_pipeline.launch.py [vgn_model_path:=...]
  T3: ros2 launch vgn_grasp_pkg grasp_debug.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:

    rviz_cfg = os.path.join(
        get_package_share_directory('vgn_grasp_pkg'), 'rviz', 'grasp_demo.rviz'
    )

    return LaunchDescription([
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2_grasp_debug',
            arguments=['-d', rviz_cfg],
            output='screen',
        )
    ])
