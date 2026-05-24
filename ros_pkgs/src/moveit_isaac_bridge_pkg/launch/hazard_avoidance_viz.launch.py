"""One-shot bring-up for visualising hazard collision objects in RViz.

Starts move_group + RViz (panda_isaac_moveit) and the hazard collision injector.
Collision objects published by the injector on /collision_object show up in the
RViz MotionPlanning "Scene Geometry" as the bottle flies through the top view.

Still run separately:
  - Isaac Sim scene (cameras, depth, /joint_states, the flying bottle)
  - YOLO hazard nodes:  ros2 launch yolo_hazard_pkg yolo_hazard_top.launch.py
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    moveit_launch = PathJoinSubstitution(
        [FindPackageShare("moveit_isaac_bridge_pkg"), "launch", "panda_isaac_moveit.launch.py"]
    )

    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(moveit_launch),
                launch_arguments={"start_rviz": "true"}.items(),
            ),
            Node(
                package="moveit_isaac_bridge_pkg",
                executable="hazard_collision_injector_node",
                name="hazard_collision_injector_node",
                output="screen",
            ),
        ]
    )
