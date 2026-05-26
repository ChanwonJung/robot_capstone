from setuptools import find_packages, setup


package_name = "moveit_isaac_bridge_pkg"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", [
            "config/moveit_controllers.yaml",
            "config/moveit.rviz",
        ]),
        ("share/" + package_name + "/config/hybrid", [
            "config/hybrid/global_planner.yaml",
            "config/hybrid/local_planner.yaml",
            "config/hybrid/hybrid_planning_manager.yaml",
        ]),
        ("share/" + package_name + "/launch", [
            "launch/joint_trajectory_bridge.launch.py",
            "launch/panda_isaac_moveit.launch.py",
            "launch/capstone_pick_pipeline.launch.py",
            "launch/target_pose_executor.launch.py",
            "launch/hazard_monitor.launch.py",
            "launch/hazard_collision_injector.launch.py",
            "launch/hazard_avoidance_viz.launch.py",
            "launch/hybrid_planning.launch.py",
        ]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="chanwonjung",
    maintainer_email="chanwonjung@todo.todo",
    description="MoveIt FollowJointTrajectory bridge for Isaac Sim joint commands",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "joint_trajectory_bridge_node = moveit_isaac_bridge_pkg.joint_trajectory_bridge_node:main",
            "target_pose_executor_node = moveit_isaac_bridge_pkg.target_pose_executor_node:main",
            "hazard_monitor_node = moveit_isaac_bridge_pkg.hazard_monitor_node:main",
            "hazard_collision_injector_node = moveit_isaac_bridge_pkg.hazard_collision_injector_node:main",
            "hybrid_command_bridge_node = moveit_isaac_bridge_pkg.hybrid_command_bridge_node:main",
            "hybrid_pose_client_node = moveit_isaac_bridge_pkg.hybrid_pose_client_node:main",
            "joint_state_restamp_node = moveit_isaac_bridge_pkg.joint_state_restamp_node:main",
            "gripper_action_server = moveit_isaac_bridge_pkg.gripper_action_server:main",
        ],
    },
)
