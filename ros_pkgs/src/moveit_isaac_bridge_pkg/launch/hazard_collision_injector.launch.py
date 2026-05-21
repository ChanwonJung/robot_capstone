"""Launch the hazard collision injector.

Turns top-view YOLO hazard detections into MoveIt collision objects on
/collision_object. Run alongside the YOLO hazard nodes (yolo_hazard_pkg), the
Isaac camera bridge (depth + camera_info), and move_group. Tuning lives in node
parameters (trigger_class_ids, conf_threshold, default_obstacle_height, etc.).
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="moveit_isaac_bridge_pkg",
                executable="hazard_collision_injector_node",
                name="hazard_collision_injector_node",
                output="screen",
            ),
        ]
    )
