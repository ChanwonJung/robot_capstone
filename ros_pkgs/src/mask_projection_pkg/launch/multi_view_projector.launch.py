"""
multi_view_projector.launch.py

Launches multi_view_projector_node — two-camera world-frame PointCloud2 builder.

────────────────────────────────────────────────────────────────
 USAGE (Gazebo / default)
────────────────────────────────────────────────────────────────
  ros2 launch mask_projection_pkg multi_view_projector.launch.py \
    prompt:="cup, table, object"

────────────────────────────────────────────────────────────────
 USAGE (Isaac Sim — teammate TODO)
────────────────────────────────────────────────────────────────
  Override topic names to match Isaac Sim's ROS 2 bridge output:

  ros2 launch mask_projection_pkg multi_view_projector.launch.py \
    top_depth_topic:=/isaac/top/depth_image \
    top_camera_info_topic:=/isaac/top/camera_info \
    ee_depth_topic:=/isaac/ee/depth_image \
    ee_camera_info_topic:=/isaac/ee/camera_info

  Also launch GSAM on the EE camera RGB stream:

  ros2 launch grounded_sam_pkg grounded_sam.launch.py \
    image_topic:=/ee_camera/image \
    prompt:="cup, table, object"

────────────────────────────────────────────────────────────────
 CAMERA → WORLD TRANSFORMS  (teammate TODO)
────────────────────────────────────────────────────────────────
  R and t are hardcoded in multi_view_projector_node.py:
    _R_TOP, _t_TOP  — top camera  (fixed mount)
    _R_EE,  _t_EE   — EE camera   (home/init pose snapshot)

  Convention:  p_world = R @ p_cam + t

  Measurement from Isaac Sim:
    1. Open the USD stage in Isaac Sim.
    2. Select the camera prim (e.g. /World/top_camera).
    3. Read the World Transform matrix from the Property panel.
    4. Extract the 3×3 rotation block (columns = camera X/Y/Z axes in world)
       and the translation column (camera origin in world, meters).
    5. Paste into _R_TOP/_t_TOP in multi_view_projector_node.py.
    6. Repeat for the EE camera at its home joint configuration.

  Example (replace with real values):
    _R_TOP = np.array([
        [ 1.0,  0.0,  0.0],   # top-camera X  in world
        [ 0.0,  0.0,  1.0],   # top-camera Y  in world
        [ 0.0, -1.0,  0.0],   # top-camera Z  in world
    ])
    _t_TOP = np.array([0.0, 0.0, 1.5])  # camera at 1.5 m above world origin

────────────────────────────────────────────────────────────────
 QWEN INTEGRATION (future TODO)
────────────────────────────────────────────────────────────────
  When Qwen VLM is ready, it will determine which GSAM mask ID is
  TARGET vs WORKSPACE.  Expected change: add a Qwen node that
  subscribes to /grounded_sam/detections_json + annotated image,
  publishes a refined label mapping, and multi_view_projector_node
  reads that mapping instead of using prompt-order assignment.
  No changes to this launch file are needed for that integration.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:

    # ── declare all arguments ─────────────────────────────────────────────────
    args = [
        # ── Top camera topics ─────────────────────────────────────────────────
        # Gazebo default: /top_camera/depth_image
        # Isaac Sim TODO: override to match your bridge output topic
        DeclareLaunchArgument('top_depth_topic',
                              default_value='/top_camera/depth_image'),
        DeclareLaunchArgument('top_camera_info_topic',
                              default_value='/top_camera/camera_info'),

        # ── EE camera topics ──────────────────────────────────────────────────
        # Gazebo default: /ee_camera/depth_image
        # Isaac Sim TODO: override to match your bridge output topic
        DeclareLaunchArgument('ee_depth_topic',
                              default_value='/ee_camera/depth_image'),
        DeclareLaunchArgument('ee_camera_info_topic',
                              default_value='/ee_camera/camera_info'),

        # ── Qwen / stub output topics ─────────────────────────────────────────
        # Default: qwen_stub_node (or real Qwen) output.
        # To bypass Qwen and connect directly to GSAM, override:
        #   mask_topic:=/grounded_sam/mask_image
        #   detections_topic:=/grounded_sam/detections_json
        DeclareLaunchArgument('mask_topic',
                              default_value='/qwen/mask_image'),
        DeclareLaunchArgument('detections_topic',
                              default_value='/qwen/labeled_detections'),

        # ── Output topics ─────────────────────────────────────────────────────
        DeclareLaunchArgument('output_cloud_topic',
                              default_value='/world_map'),
        DeclareLaunchArgument('output_result_topic',
                              default_value='/world_map_result'),

        # ── Depth filter ──────────────────────────────────────────────────────
        DeclareLaunchArgument('min_depth', default_value='0.05'),
        DeclareLaunchArgument('max_depth', default_value='15.0'),

        # ── Optional filename prefix for saved PLY files ──────────────────────
        DeclareLaunchArgument('initials', default_value=''),

        # ── Camera extrinsics YAML ────────────────────────────────────────────
        # Gazebo demo default: package config/camera_extrinsics.yaml
        # Isaac Sim: override with your own YAML containing R/t from USD stage.
        #   extrinsics_config:=/path/to/camera_extrinsics_isaac.yaml
        DeclareLaunchArgument('extrinsics_config', default_value=''),
    ]

    node = Node(
        package    = 'mask_projection_pkg',
        executable = 'multi_view_projector_node',
        name       = 'multi_view_projector_node',
        output     = 'screen',
        parameters = [{
            'top_depth_topic':       LaunchConfiguration('top_depth_topic'),
            'top_camera_info_topic': LaunchConfiguration('top_camera_info_topic'),
            'ee_depth_topic':        LaunchConfiguration('ee_depth_topic'),
            'ee_camera_info_topic':  LaunchConfiguration('ee_camera_info_topic'),
            'mask_topic':            LaunchConfiguration('mask_topic'),
            'detections_topic':      LaunchConfiguration('detections_topic'),
            'output_cloud_topic':    LaunchConfiguration('output_cloud_topic'),
            'output_result_topic':   LaunchConfiguration('output_result_topic'),
            'min_depth':             LaunchConfiguration('min_depth'),
            'max_depth':             LaunchConfiguration('max_depth'),
            'initials':              LaunchConfiguration('initials'),
            'extrinsics_config':     LaunchConfiguration('extrinsics_config'),
        }],
    )

    return LaunchDescription(args + [node])
