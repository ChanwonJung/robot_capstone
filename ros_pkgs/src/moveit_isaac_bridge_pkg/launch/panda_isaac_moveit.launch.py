import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value="moveit.rviz",
        description="RViz configuration file from the Panda MoveIt config package",
    )
    start_rviz_arg = DeclareLaunchArgument(
        "start_rviz",
        default_value="true",
        description="Whether to start RViz with the Panda MoveIt config",
    )
    command_topic_arg = DeclareLaunchArgument(
        "command_topic",
        default_value="/joint_command",
        description="Isaac Sim joint command topic used by the trajectory bridge",
    )
    joint_state_topic_arg = DeclareLaunchArgument(
        "joint_state_topic",
        default_value="/joint_states",
        description="Joint state topic published by Isaac Sim",
    )

    bridge_pkg_share = get_package_share_directory("moveit_isaac_bridge_pkg")
    moveit_controllers = os.path.join(bridge_pkg_share, "config", "moveit_controllers.yaml")

    moveit_config = (
        MoveItConfigsBuilder(
            "moveit_resources_panda",
            package_name="moveit_resources_panda_moveit_config",
        )
        .robot_description(file_path="config/panda.urdf.xacro")
        .robot_description_semantic(file_path="config/panda.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .trajectory_execution(file_path=moveit_controllers)
        .planning_pipelines(
            pipelines=["ompl", "pilz_industrial_motion_planner", "chomp", "stomp"]
        )
        .to_moveit_configs()
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict(), {"use_sim_time": True}],
    )

    rviz_config = PathJoinSubstitution(
        [FindPackageShare("moveit_resources_panda_moveit_config"), "launch", LaunchConfiguration("rviz_config")]
    )
    rviz_node = Node(
        condition=IfCondition(LaunchConfiguration("start_rviz")),
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            moveit_config.planning_pipelines,
            {"use_sim_time": True},
        ],
    )

    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "panda_link0"],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description, {"use_sim_time": True}],
    )

    joint_trajectory_bridge = Node(
        package="moveit_isaac_bridge_pkg",
        executable="joint_trajectory_bridge_node",
        output="screen",
        parameters=[
            {
                "command_topic": LaunchConfiguration("command_topic"),
                "joint_state_topic": LaunchConfiguration("joint_state_topic"),
                "action_name": "/panda_arm_controller/follow_joint_trajectory",
                "use_sim_time": True,
            }
        ],
    )

    return LaunchDescription(
        [
            rviz_config_arg,
            start_rviz_arg,
            command_topic_arg,
            joint_state_topic_arg,
            static_tf_node,
            robot_state_publisher,
            joint_trajectory_bridge,
            move_group_node,
            rviz_node,
        ]
    )
