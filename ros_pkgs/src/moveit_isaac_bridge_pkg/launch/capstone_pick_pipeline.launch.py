from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    target_pose_launch = PathJoinSubstitution(
        [FindPackageShare("target_pose_bridge_pkg"), "launch", "target_pose_bridge.launch.py"]
    )
    moveit_launch = PathJoinSubstitution(
        [FindPackageShare("moveit_isaac_bridge_pkg"), "launch", "panda_isaac_moveit.launch.py"]
    )

    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(target_pose_launch),
                launch_arguments={
                    "input_topic": "/world_map_result",
                    "pre_grasp_topic": "/pre_grasp_target_pose",
                    "grasp_topic": "/grasp_target_pose",
                    "world_frame": "world",
                    "min_point_count": "100",
                    "grasp_z_offset": "0.03",
                    "pre_grasp_z_offset": "0.12",
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(moveit_launch),
                launch_arguments={
                    "command_topic": "/joint_command",
                    "joint_state_topic": "/joint_states",
                    "start_rviz": "true",
                }.items(),
            ),
        ]
    )
