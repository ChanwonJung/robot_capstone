from setuptools import find_packages, setup

package_name = 'graspgen_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/graspgen.launch.py',
            'launch/full_pipeline_graspgen.launch.py',
        ]),
        ('share/' + package_name + '/config', [
            'config/graspgen_params.yaml',
        ]),
        # Rescued from vgn_grasp_pkg when that package was retired. The
        # GraspMarkers display filters on namespace 'vgn_grasps', which
        # marker_publisher.py still publishes under — do not rename one
        # without the other.
        ('share/' + package_name + '/rviz', [
            'rviz/grasp_demo.rviz',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='parksanghyun',
    maintainer_email='parksanghyun@todo.todo',
    description='GraspGen remote inference client — TARGET cloud → ZMQ → /grasp_candidates',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'graspgen_node = graspgen_pkg.graspgen_node:main',
        ],
    },
)
