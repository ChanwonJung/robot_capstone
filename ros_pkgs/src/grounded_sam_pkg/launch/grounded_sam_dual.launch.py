"""Dual-view Grounded SAM launch.

Runs a single GroundedSAMNode instance that processes BOTH the EE eye-in-hand
camera and the Top static camera, sequentially, using one shared model.

Topic layout:
  EE   ──► /ee_camera/image_raw          (input)
       ──► /grounded_sam/annotated_image
       ──► /grounded_sam/mask_image
       ──► /grounded_sam/detections_json
  Top  ──► /camera/image_raw             (input — Isaac build_top_view_bridge default)
       ──► /top/grounded_sam/annotated_image
       ──► /top/grounded_sam/mask_image
       ──► /top/grounded_sam/detections_json

min_process_interval_sec defaults to 10.0 s (Slow Brain pacing — slower than
the single-view EE launch). Override at the command line if needed.
"""
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
                    "prompt": "blue cube",
                    # EE view (primary trigger)
                    "image_topic": "/ee_camera/image_raw",
                    "annotated_topic": "/grounded_sam/annotated_image",
                    "mask_topic": "/grounded_sam/mask_image",
                    "detections_topic": "/grounded_sam/detections_json",
                    # Top view (cached, processed on each EE cycle)
                    "top_image_topic": "/camera/image_raw",
                    # Multi-class noun list; table-as-X false positives filtered
                    # out downstream by max_bbox_area_ratio.
                    "top_prompt": "blue cube",
                    "top_annotated_topic": "/top/grounded_sam/annotated_image",
                    "top_mask_topic": "/top/grounded_sam/mask_image",
                    "top_detections_topic": "/top/grounded_sam/detections_json",
                    # Top depth gating disabled — observe the robot arm as-is.
                    # (Set top_depth_topic / top_min_depth / top_max_depth to
                    #  re-enable depth-based image masking when needed.)
                    "top_depth_topic": "",
                    "top_min_depth": "0.0",
                    "top_max_depth": "100.0",
                    # Slow Brain pacing
                    "process_every_n_frames": "60",
                    "min_process_interval_sec": "10.0",
                    "output_subdir": "dual_view",
                }.items(),
            ),
        ]
    )
