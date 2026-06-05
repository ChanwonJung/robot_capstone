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
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    grounded_sam_launch = PathJoinSubstitution(
        [FindPackageShare("grounded_sam_pkg"), "launch", "grounded_sam.launch.py"]
    )

    # EE-view 탐지 프롬프트. 투명 유리컵은 depth가 불안정해 centroid가 튀므로,
    # 불투명 물체(red ball / apple / book)로 덮어쓰기:  prompt:="red ball"
    prompt_arg = DeclareLaunchArgument("prompt", default_value="glass cup")
    # Slow Brain 페이싱. 정적 EE extrinsics 전제 — 한 번 관찰 후 재실행 시
    # EE 카메라가 이동해 extrinsics 가 더 이상 유효하지 않으면 world cloud 가
    # 엉뚱한 좌표로 투영됨. 기본 10초는 너무 짧아 pick-and-place 도중 재투영
    # → 책 좌표가 사라지는 증상 발생. CLI 로 충분히 크게 override 권장.
    interval_arg = DeclareLaunchArgument("min_process_interval_sec", default_value="10.0")
    frames_arg   = DeclareLaunchArgument("process_every_n_frames",   default_value="60")

    return LaunchDescription(
        [
            prompt_arg,
            interval_arg,
            frames_arg,
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(grounded_sam_launch),
                launch_arguments={
                    "prompt": LaunchConfiguration("prompt"),
                    # EE view (primary trigger)
                    "image_topic": "/ee_camera/image_raw",
                    "annotated_topic": "/grounded_sam/annotated_image",
                    "mask_topic": "/grounded_sam/mask_image",
                    "detections_topic": "/grounded_sam/detections_json",
                    # Top view (cached, processed on each EE cycle)
                    "top_image_topic": "/camera/image_raw",
                    # Multi-class noun list; table-as-X false positives filtered
                    # out downstream by max_bbox_area_ratio.
                    "top_prompt": "glass cup",
                    "top_annotated_topic": "/top/grounded_sam/annotated_image",
                    "top_mask_topic": "/top/grounded_sam/mask_image",
                    "top_detections_topic": "/top/grounded_sam/detections_json",
                    # Top depth gating disabled — observe the robot arm as-is.
                    # (Set top_depth_topic / top_min_depth / top_max_depth to
                    #  re-enable depth-based image masking when needed.)
                    "top_depth_topic": "",
                    "top_min_depth": "0.0",
                    "top_max_depth": "100.0",
                    # Slow Brain pacing (CLI override 가능)
                    "process_every_n_frames":   LaunchConfiguration("process_every_n_frames"),
                    "min_process_interval_sec": LaunchConfiguration("min_process_interval_sec"),
                    "output_subdir": "dual_view",
                }.items(),
            ),
        ]
    )
