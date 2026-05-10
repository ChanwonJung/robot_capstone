from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_model_config = PathJoinSubstitution(
        [FindPackageShare("grounded_sam_pkg"), "config", "model_paths.yaml"]
    )
    grounded_sam_launch = PathJoinSubstitution(
        [FindPackageShare("grounded_sam_pkg"), "launch", "grounded_sam.launch.py"]
    )
    mask_projector_launch = PathJoinSubstitution(
        [FindPackageShare("mask_projection_pkg"), "launch", "mask_projector.launch.py"]
    )

    return LaunchDescription([
        DeclareLaunchArgument("launch_grounded_sam", default_value="true"),
        DeclareLaunchArgument("prompt", default_value="apple, glass, book"),
        DeclareLaunchArgument("model_config", default_value=default_model_config),
        DeclareLaunchArgument("image_topic", default_value="/ee_camera/image_raw"),
        DeclareLaunchArgument("process_every_n_frames", default_value="60"),
        DeclareLaunchArgument("min_process_interval_sec", default_value="2.0"),
        DeclareLaunchArgument("depth_topic", default_value="/ee_rgbd_camera/depth_image"),
        DeclareLaunchArgument("camera_info_topic", default_value="/ee_rgbd_camera/camera_info"),
        DeclareLaunchArgument("annotated_topic", default_value="/ee_view/grounded_sam/annotated_image"),
        DeclareLaunchArgument("mask_topic", default_value="/ee_view/grounded_sam/mask_image"),
        DeclareLaunchArgument("detections_topic", default_value="/ee_view/grounded_sam/detections_json"),
        DeclareLaunchArgument("output_cloud_topic", default_value="/ee_view/labeled_points"),
        DeclareLaunchArgument("output_result_topic", default_value="/ee_view/projection_result"),
        DeclareLaunchArgument("output_frame_id", default_value=""),
        DeclareLaunchArgument("min_depth", default_value="0.05"),
        DeclareLaunchArgument("max_depth", default_value="15.0"),
        DeclareLaunchArgument("initials", default_value="agb"),
        DeclareLaunchArgument("output_subdir", default_value="ee_view"),
        DeclareLaunchArgument("min_project_interval_sec", default_value="3.0"),
        DeclareLaunchArgument("save_ply", default_value="false"),
        DeclareLaunchArgument("max_ply_saves", default_value="3"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(grounded_sam_launch),
            condition=IfCondition(LaunchConfiguration("launch_grounded_sam")),
            launch_arguments={
                "model_config": LaunchConfiguration("model_config"),
                "prompt": LaunchConfiguration("prompt"),
                "image_topic": LaunchConfiguration("image_topic"),
                "annotated_topic": LaunchConfiguration("annotated_topic"),
                "mask_topic": LaunchConfiguration("mask_topic"),
                "detections_topic": LaunchConfiguration("detections_topic"),
                "output_subdir": LaunchConfiguration("output_subdir"),
                "process_every_n_frames": LaunchConfiguration("process_every_n_frames"),
                "min_process_interval_sec": LaunchConfiguration("min_process_interval_sec"),
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(mask_projector_launch),
            launch_arguments={
                "depth_topic": LaunchConfiguration("depth_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "mask_topic": LaunchConfiguration("mask_topic"),
                "detections_topic": LaunchConfiguration("detections_topic"),
                "output_cloud_topic": LaunchConfiguration("output_cloud_topic"),
                "output_result_topic": LaunchConfiguration("output_result_topic"),
                "output_frame_id": LaunchConfiguration("output_frame_id"),
                "min_depth": LaunchConfiguration("min_depth"),
                "max_depth": LaunchConfiguration("max_depth"),
                "initials": LaunchConfiguration("initials"),
                "output_subdir": LaunchConfiguration("output_subdir"),
                "min_project_interval_sec": LaunchConfiguration("min_project_interval_sec"),
                "save_ply": LaunchConfiguration("save_ply"),
                "max_ply_saves": LaunchConfiguration("max_ply_saves"),
            }.items(),
        ),
    ])
