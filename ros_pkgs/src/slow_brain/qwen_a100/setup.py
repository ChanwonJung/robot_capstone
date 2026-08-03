from setuptools import find_packages, setup

package_name = 'qwen_a100'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/qwen_a100.launch.py',
            'launch/slow_brain.launch.py',
        ]),
        ('share/' + package_name + '/config', [
            'config/qwen_a100_params.yaml',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jaewon Heo',
    maintainer_email='jaewonheo1101@gmail.com',
    description='Slow Brain grounding — Qwen VLM on the A100 via vLLM.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'qwen_bridge_node = qwen_a100.qwen_bridge:main',
            'instruction_prompt_node = qwen_a100.instruction_prompt_node:main',
        ],
    },
)
