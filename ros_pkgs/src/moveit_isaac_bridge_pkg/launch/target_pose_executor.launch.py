"""Launch the TargetPoseExecutor for Slow Brain motion validation.

Run alongside `panda_isaac_moveit.launch.py` (MoveIt + Isaac bridge) and the
perception stack (`grounded_sam_dual.launch.py` + mask_projection +
target_pose_bridge). The executor subscribes to ``/grasp_target_pose`` and
asks MoveIt to plan + execute the arm motion.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    target_topic_arg = DeclareLaunchArgument(
        "target_pose_topic", default_value="/grasp_target_pose"
    )
    auto_execute_arg = DeclareLaunchArgument(
        "auto_execute",
        default_value="true",
        description="If false, plan only (no execution on the arm)",
    )
    link_z_offset_arg = DeclareLaunchArgument(
        "link_target_z_offset",
        default_value="0.10",
        description="Z offset added to the incoming pose so the EE link sits "
                    "above the object centroid (hand fingers reach down).",
    )
    planning_time_arg = DeclareLaunchArgument(
        "allowed_planning_time", default_value="5.0"
    )
    pos_tol_arg = DeclareLaunchArgument(
        "position_tolerance", default_value="0.01"
    )
    ori_tol_arg = DeclareLaunchArgument(
        "orientation_tolerance", default_value="0.05"
    )
    one_shot_arg = DeclareLaunchArgument(
        "one_shot",
        default_value="true",
        description="If true, ignore further target poses after the first "
                    "successful plan+execute (Slow Brain pacing).",
    )

    executor_node = Node(
        package="moveit_isaac_bridge_pkg",
        executable="target_pose_executor_node",
        name="target_pose_executor_node",
        output="screen",
        parameters=[
            {
                "target_pose_topic": LaunchConfiguration("target_pose_topic"),
                "auto_execute": LaunchConfiguration("auto_execute"),
                "link_target_z_offset": LaunchConfiguration("link_target_z_offset"),
                "allowed_planning_time": LaunchConfiguration("allowed_planning_time"),
                "position_tolerance": LaunchConfiguration("position_tolerance"),
                "orientation_tolerance": LaunchConfiguration("orientation_tolerance"),
                "one_shot": LaunchConfiguration("one_shot"),
                "use_sim_time": True,
            }
        ],
    )

    return LaunchDescription(
        [
            target_topic_arg,
            auto_execute_arg,
            link_z_offset_arg,
            planning_time_arg,
            pos_tol_arg,
            ori_tol_arg,
            one_shot_arg,
            executor_node,
        ]
    )
