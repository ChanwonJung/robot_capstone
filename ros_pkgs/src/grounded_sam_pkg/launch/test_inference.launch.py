"""
테스트용 통합 런치 파일.
test_image_pub + grounded_sam_node 를 동시에 띄운다.

이미지 경로: grounded_sam_pkg/test_image_pub.py 상단 IMAGE_PATH 수정
prompt   : 런치 인수로 전달

사용법:
    source launch_env.bash
    ros2 launch grounded_sam_pkg test_inference.launch.py prompt:="glass, chair, person"
"""
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
            "prompt",
            default_value="object",
            description="찾을 객체. 예: 'glass, chair, person'",
        ),

        # 테스트 이미지 퍼블리셔
        Node(
            package="grounded_sam_pkg",
            executable="test_image_pub",
            name="test_image_publisher",
            output="screen",
        ),

        # 추론 노드
        Node(
            package="grounded_sam_pkg",
            executable="grounded_sam_node",
            name="grounded_sam_node",
            parameters=[{
                "model_config": default_config,
                "prompt": LaunchConfiguration("prompt"),
                "image_topic": "/camera/image_raw",
            }],
            output="screen",
        ),
    ])
