"""Launch qwen_bridge_node plus the stdin instruction prompt.

    ros2 launch qwen_a100 qwen_a100.launch.py

The prompt node runs inside an xterm because `ros2 launch` does not forward
stdin to child processes. Set use_xterm:=false when driving /user_instruction
from elsewhere (a replay bag, another node, or `ros2 topic pub`).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _robot_defaults() -> list:
    """Repo-wide defaults, loaded first so package params can override them."""
    root = os.environ.get("ROBOT_CAPSTONE_ROOT", "")
    path = os.path.join(root, "config", "robot_defaults.yaml") if root else ""
    return [path] if path and os.path.isfile(path) else []


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("qwen_a100")
    params = os.path.join(pkg_share, "config", "qwen_a100_params.yaml")

    args = [
        DeclareLaunchArgument("vllm_endpoint_url", default_value="http://localhost:8000/v1"),
        DeclareLaunchArgument("model_name", default_value="qwen35-local",
                              description="served model id (Qwen3.5-27B)"),
        DeclareLaunchArgument("image_topic", default_value="/ee_camera/image_raw"),
        DeclareLaunchArgument("image_path", default_value="",
                              description="use this file as the frame instead of "
                                          "image_topic (offline test)"),
        DeclareLaunchArgument("bbox_convention", default_value="normalized_1000"),
        DeclareLaunchArgument("instruction", default_value=""),
        DeclareLaunchArgument("use_xterm", default_value="true"),
    ]

    overrides = {
        "vllm_endpoint_url": LaunchConfiguration("vllm_endpoint_url"),
        "model_name": LaunchConfiguration("model_name"),
        "image_topic": LaunchConfiguration("image_topic"),
        "image_path": LaunchConfiguration("image_path"),
        "bbox_convention": LaunchConfiguration("bbox_convention"),
        "instruction": LaunchConfiguration("instruction"),
    }

    bridge = Node(
        package="qwen_a100",
        executable="qwen_bridge_node",
        name="qwen_bridge_node",
        output="screen",
        parameters=[*_robot_defaults(), params, overrides],
    )

    prompt_xterm = Node(
        package="qwen_a100",
        executable="instruction_prompt_node",
        name="instruction_prompt_node",
        output="screen",
        parameters=[*_robot_defaults(), params],
        prefix="xterm -geometry 100x16 -T 'Slow Brain instruction' -e",
        condition=IfCondition(LaunchConfiguration("use_xterm")),
    )

    prompt_plain = Node(
        package="qwen_a100",
        executable="instruction_prompt_node",
        name="instruction_prompt_node",
        output="screen",
        parameters=[*_robot_defaults(), params],
        condition=UnlessCondition(LaunchConfiguration("use_xterm")),
    )

    return LaunchDescription([*args, bridge, prompt_xterm, prompt_plain])
