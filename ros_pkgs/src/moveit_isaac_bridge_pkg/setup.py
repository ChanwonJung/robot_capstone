from setuptools import find_packages, setup


package_name = "moveit_isaac_bridge_pkg"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/moveit_controllers.yaml"]),
        ("share/" + package_name + "/launch", [
            "launch/joint_trajectory_bridge.launch.py",
            "launch/panda_isaac_moveit.launch.py",
            "launch/capstone_pick_pipeline.launch.py",
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
        ],
    },
)
