"""
inst_input_qwen.launch.py

Launches the full Qwen slow-brain input pipeline:
  1. instruction_prompt_node — opens an xterm for the user to type natural-language
     commands; each entry is published to /user_instruction (std_msgs/String).
  2. qwen_bridge_node — subscribes to /user_instruction and /grounded_sam/detections_json,
     runs Qwen VLM inference, and publishes /qwen/labeled_detections.

Usage:
  ros2 launch qwen_pkg inst_input_qwen.launch.py
  ros2 launch qwen_pkg inst_input_qwen.launch.py vllm_endpoint_url:=http://<host>:8000/v1
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            "vllm_endpoint_url",
            default_value="http://localhost:8000/v1",
            description="Base URL of the vLLM OpenAI-compatible server",
        ),
        DeclareLaunchArgument(
            "model_name",
            default_value="qwen-vl",
            description="Model name registered in the vLLM server",
        ),
        # xterm is required because ros2 launch does not forward stdin to child
        # processes — input() in instruction_prompt_node would never receive
        # keystrokes without a dedicated terminal window.
        ExecuteProcess(
            cmd=[
                "xterm",
                "-title", "Robot Command Input",
                "-e", "ros2 run qwen_pkg instruction_prompt_node",
            ],
            output="screen",
        ),
        Node(
            package="qwen_pkg",
            executable="qwen_bridge_node",
            name="qwen_bridge",
            output="screen",
            parameters=[{
                "vllm_endpoint_url": LaunchConfiguration("vllm_endpoint_url"),
                "model_name": LaunchConfiguration("model_name"),
            }],
        ),
    ])
