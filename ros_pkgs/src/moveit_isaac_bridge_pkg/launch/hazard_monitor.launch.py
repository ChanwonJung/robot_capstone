"""Launch the hazard monitor — cancels MoveGroup motion on Fast Brain detections.

Run alongside the YOLO hazard nodes (yolo_hazard_pkg) and the MoveIt pipeline
(panda_isaac_moveit / target_pose_executor). Tuning lives in node parameters
(detection_topics, trigger_class_ids, conf_threshold, cancel_cooldown_sec).
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="moveit_isaac_bridge_pkg",
                executable="hazard_monitor_node",
                name="hazard_monitor_node",
                output="screen",
            ),
        ]
    )
