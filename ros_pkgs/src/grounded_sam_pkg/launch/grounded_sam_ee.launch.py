from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    grounded_sam_launch = PathJoinSubstitution(
        [FindPackageShare("grounded_sam_pkg"), "launch", "grounded_sam.launch.py"]
    )

    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(grounded_sam_launch),
                launch_arguments={
                    "prompt": "glass cup, blue cube, book",
                    "image_topic": "/ee_camera/image_raw",
                    "annotated_topic": "/grounded_sam/annotated_image",
                    "mask_topic": "/grounded_sam/mask_image",
                    "detections_topic": "/grounded_sam/detections_json",
                    "output_subdir": "ee_view",
                    "process_every_n_frames": "60",
                    "min_process_interval_sec": "2.0",
                }.items(),
            ),
        ]
    )
