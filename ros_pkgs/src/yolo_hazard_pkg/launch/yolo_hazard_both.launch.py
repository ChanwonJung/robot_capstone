"""Launch YOLO hazard detection on both top-view and eye-in-hand cameras simultaneously.

Tuning (conf, iou, class filter, etc.) lives in config/runtime.yaml.
Per-camera wiring (model weights, topics) lives in this file.

The ROS 2 entry-point script is hard-shebanged to the system python, but the
ultralytics stack is installed in the project-local .venv-yolo. We expose the
venv's site-packages via PYTHONPATH so the system python can import it.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node


def _venv_site_packages() -> str:
    share = get_package_share_directory("yolo_hazard_pkg")
    project_root = os.path.realpath(os.path.join(share, *([".."] * 5)))
    return os.path.join(project_root, ".venv-yolo", "lib", "python3.12", "site-packages")


def _robot_defaults() -> str:
    root = os.environ.get(
        "ROBOT_CAPSTONE_ROOT",
        os.path.realpath(os.path.join(
            get_package_share_directory("yolo_hazard_pkg"), *([".."] * 4))),
    )
    return os.path.join(root, "config", "robot_defaults.yaml")


def generate_launch_description():
    share = get_package_share_directory("yolo_hazard_pkg")
    model_config   = os.path.join(share, "config", "model_paths.yaml")
    runtime_config = os.path.join(share, "config", "runtime.yaml")
    defaults       = _robot_defaults()

    pythonpath = _venv_site_packages() + os.pathsep + os.environ.get("PYTHONPATH", "")

    top_node = Node(
        package="yolo_hazard_pkg",
        executable="yolo_hazard_node",
        name="yolo_hazard_top",
        output="screen",
        parameters=[
            defaults,
            runtime_config,
            {
                "model_config": model_config,
                "image_topic": "/camera/image_raw",
                "detections_topic": "/yolo_hazard/top/detections_json",
                "annotated_topic": "/yolo_hazard/top/annotated_image",
            },
        ],
    )

    ee_node = Node(
        package="yolo_hazard_pkg",
        executable="yolo_hazard_node",
        name="yolo_hazard_ee",
        output="screen",
        parameters=[
            defaults,
            runtime_config,
            {
                "model_config": model_config,
                "image_topic": "/ee_camera/image_raw",
                "detections_topic": "/yolo_hazard/ee/detections_json",
                "annotated_topic": "/yolo_hazard/ee/annotated_image",
            },
        ],
    )

    return LaunchDescription([
        SetEnvironmentVariable("PYTHONPATH", pythonpath),
        top_node,
        ee_node,
    ])
