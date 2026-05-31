from setuptools import find_packages, setup

package_name = 'vgn_grasp_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/vgn_grasp.launch.py',
            'launch/vgn_grasp_4cam.launch.py',
            'launch/full_pipeline.launch.py',
            'launch/full_pipeline_4cam.launch.py',
            'launch/grasp_debug.launch.py',
        ]),
        ('share/' + package_name + '/config', [
            'config/vgn_params.yaml',
        ]),
        ('share/' + package_name + '/rviz', [
            'rviz/grasp_demo.rviz',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='parksanghyun',
    maintainer_email='parksanghyun@todo.todo',
    description='VGN grasp detection node — /world_map PointCloud2 → /grasp_candidates',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'vgn_grasp_node = vgn_grasp_pkg.vgn_grasp_node:main',
            'vgn_grasp_4cam_node = vgn_grasp_pkg.vgn_grasp_4cam_node:main',
        ],
    },
)
