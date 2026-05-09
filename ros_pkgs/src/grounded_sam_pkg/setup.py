from setuptools import find_packages, setup

package_name = 'grounded_sam_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/model_paths.yaml', 'config/runtime.yaml']),
        ('share/' + package_name + '/launch', [
            'launch/grounded_sam.launch.py',
            'launch/test_inference.launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='parksanghyun',
    maintainer_email='parksanghyun@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'grounded_sam_node = grounded_sam_pkg.ros_node:main',
            'test_image_pub = grounded_sam_pkg.test_image_pub:main',
            'qwen_stub_node = grounded_sam_pkg.qwen_stub_node:main',
        ],
    },
)
