from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("command_topic", default_value="/joint_command"),
            DeclareLaunchArgument("joint_state_topic", default_value="/joint_states"),
            DeclareLaunchArgument(
                "action_name",
                default_value="/panda_arm_controller/follow_joint_trajectory",
            ),
            Node(
                package="moveit_isaac_bridge_pkg",
                executable="joint_trajectory_bridge_node",
                output="screen",
                parameters=[
                    {
                        "command_topic": LaunchConfiguration("command_topic"),
                        "joint_state_topic": LaunchConfiguration("joint_state_topic"),
                        "action_name": LaunchConfiguration("action_name"),
                    }
                ],
            ),
        ]
    )
