from setuptools import find_packages, setup

package_name = 'yolo_hazard_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', [
            'config/model_paths.yaml',
            'config/runtime.yaml',
        ]),
        ('share/' + package_name + '/launch', [
            'launch/yolo_hazard_top.launch.py',
            'launch/yolo_hazard_ee.launch.py',
            'launch/yolo_hazard_both.launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='capstone',
    maintainer_email='hyper.robotics.khu@gmail.com',
    description='Fast Brain YOLO26-seg hazard detection node.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'yolo_hazard_node = yolo_hazard_pkg.ros_node:main',
        ],
    },
)
