from setuptools import find_packages, setup

package_name = 'mask_projection_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/mask_projector.launch.py',
            'launch/multi_view_projector.launch.py',
        ]),
        ('share/' + package_name + '/config', [
            'config/camera_extrinsics.yaml',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='parksanghyun',
    maintainer_email='parksanghyun@todo.todo',
    description='Mask projection: 2D segmentation mask → labeled PointCloud2',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'mask_projector_node = mask_projection_pkg.projector_node:main',
            'multi_view_projector_node = mask_projection_pkg.multi_view_projector_node:main',
        ],
    },
)
