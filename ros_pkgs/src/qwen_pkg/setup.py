from setuptools import find_packages, setup

package_name = 'qwen_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/inst_input_qwen.launch.py',
            'launch/TEMP_instruction_prompt.launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jaewon Heo',
    maintainer_email='jaewonheo1101@gmail.com',
    description='Qwen VLM nodes for the Slow Brain pipeline.',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'instruction_prompt_node = qwen_pkg.instruction_prompt_node:main',
            'qwen_bridge_node = qwen_pkg.qwen_bridge:main',
        ],
    },
)
