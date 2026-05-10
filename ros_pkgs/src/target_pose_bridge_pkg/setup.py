from setuptools import find_packages, setup


package_name = "target_pose_bridge_pkg"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/target_pose_bridge.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="chanwonjung",
    maintainer_email="chanwonjung@todo.todo",
    description="Bridge target centroid JSON into MoveIt-compatible PoseStamped goals",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "target_pose_bridge_node = target_pose_bridge_pkg.target_pose_bridge_node:main",
        ],
    },
)
