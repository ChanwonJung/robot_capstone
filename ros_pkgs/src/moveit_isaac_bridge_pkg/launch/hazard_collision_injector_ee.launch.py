"""Launch the hazard collision injector wired to the EE-view camera.

EE companion to hazard_collision_injector.launch.py — same node, different
topic + extrinsics wiring so the injector consumes /yolo_hazard/ee/* instead
of the top camera. Robot-self perspective: the arm only injects what its own
EE camera sees, not what the overhead "omniscient" top camera observes.

Extrinsics caveat: this uses the STATIC ee_camera entry in
camera_extrinsics.yaml, which is the panda_link0 <- ee_view_camera optical
transform at the robot's START pose. Accuracy degrades as the arm moves
away from that pose. Acceptable for the replan demo because the bottle parks
in front of the arm with minimal motion before detection. If a fully dynamic
EE transform is needed later, swap this for a TF-based mode.

Run alongside yolo_hazard_ee.launch.py, the Isaac camera bridge, and
move_group / hybrid planning manager.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="moveit_isaac_bridge_pkg",
                executable="hazard_collision_injector_node",
                name="hazard_collision_injector_ee_node",
                output="screen",
                parameters=[{
                    "detection_topic": "/yolo_hazard/ee/detections_json",
                    "depth_topic": "/ee_rgbd_camera/depth_image",
                    "camera_info_topic": "/ee_rgbd_camera/camera_info",
                    "extrinsics_key": "ee_camera",
                    # Give EE-injected obstacles a distinct id namespace so
                    # they coexist cleanly with the top injector if both run.
                    "object_id_prefix": "hazard_ee_",
                }],
            ),
        ]
    )
