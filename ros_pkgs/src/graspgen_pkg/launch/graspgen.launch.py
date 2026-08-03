"""
graspgen.launch.py — graspgen_node 단독 실행 (기존 파이프라인 끝에 붙이기용)

전제: GSAM + Qwen stub + Projection 파이프라인이 이미 실행 중이어야 함.
전제: SSH tunnel이 열려 있어야 함.

Usage:
  ros2 launch graspgen_pkg graspgen.launch.py
  ros2 launch graspgen_pkg graspgen.launch.py zmq_host:=127.0.0.1 zmq_port:=5556
  ros2 launch graspgen_pkg graspgen.launch.py \
    zmq_host:=127.0.0.1 zmq_port:=5558 topk_num_grasps:=3

SSH tunnel:
  # 학내망
  ssh -N -L 5556:aurora-g5:5556 <user>@aurora.khu.ac.kr
  # 외부망
  ssh -p 30080 -N -L 5556:aurora-g5:5556 <user>@aurora.khu.ac.kr
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:

    pkg = get_package_share_directory('graspgen_pkg')

    # 기본 extrinsics 경로 — ROBOT_CAPSTONE_ROOT(launch_env.bash 설정) 기반.
    # graspgen_node 의 _DEFAULT_EXTRINSICS 는 옛 gsam_ws 경로라 신뢰 불가.
    _default_ext = os.path.join(
        os.environ.get('ROBOT_CAPSTONE_ROOT', ''),
        'ros_pkgs/src/mask_projection_pkg/config/camera_extrinsics.yaml',
    )

    args = [
        DeclareLaunchArgument(
            'zmq_host',
            default_value='127.0.0.1',
            description='GraspGen server host (SSH tunnel endpoint)',
        ),
        DeclareLaunchArgument(
            'zmq_port',
            default_value='5556',
            description='GraspGen server port (SSH tunnel local port)',
        ),
        DeclareLaunchArgument(
            'zmq_timeout_ms',
            default_value='30000',
            description='ZMQ recv timeout (ms) — scale with num_grasps batch size',
        ),
        DeclareLaunchArgument(
            'num_grasps',
            default_value='200',
            description='Diffusion sampling batch (paper Fig. 14 — scale with min_quality)',
        ),
        DeclareLaunchArgument(
            'topk_num_grasps',
            default_value='100',
            description='Top-K grasp candidates to publish (paper §5.1: top-100, NMS off)',
        ),
        DeclareLaunchArgument(
            'min_quality',
            default_value='0.5',
            description='Confidence threshold (paper §6.10 recommends ≥ 0.5)',
        ),
        DeclareLaunchArgument(
            'min_point_count',
            default_value='50',
            description='Minimum TARGET point count to trigger inference',
        ),
        DeclareLaunchArgument(
            'panda_link8_offset',
            default_value='0.103',
            description=(
                'Shift the grasp pose back along -approach by this many metres '
                'so panda_link8 lands behind the fingertips. Correct ONLY if '
                'GraspGen returns a fingertip/TCP-centred pose; set 0.0 if it '
                'already returns a link8-centred pose (see graspgen_node.py)'
            ),
        ),
        # ── Client-side filters (Option A) ────────────────────────────
        DeclareLaunchArgument(
            'top_down_filter_enabled',
            default_value='true',
            description='Drop side grasps; keep only top-down approaches',
        ),
        DeclareLaunchArgument(
            'top_down_angle_deg',
            default_value='45.0',
            description='Max deviation from straight-down for top_down filter',
        ),
        DeclareLaunchArgument(
            'ik_filter_enabled',
            default_value='true',
            description='Drop grasps unreachable via MoveIt /compute_ik',
        ),
        DeclareLaunchArgument(
            'max_published_grasps',
            default_value='10',
            description='Final cap on the published candidate pool size',
        ),
        # ── force top-down / grasp Z tuning ───────────────────────────
        DeclareLaunchArgument(
            'force_top_down_orientation',
            default_value='true',
            description='Replace GraspGen orientation with a clean vertical '
                        'top-down grasp (finger spread = PCA short axis). '
                        'Verified on the glass cup and the book; turn off to '
                        'use GraspGen orientations as-is.',
        ),
        DeclareLaunchArgument(
            'force_top_down_grasp_z_frac',
            default_value='0.65',
            description='force_top_down fingertip height along the restored '
                        'cloud height (0=table, 1=rim). 0.65 = upper body.',
        ),
        DeclareLaunchArgument(
            'force_top_down_grasp_z_offset',
            default_value='0.0',
            description='Extra +/- metres on top of the z_frac fingertip height.',
        ),
        DeclareLaunchArgument(
            'grasp_xy_anchor',
            default_value='median',
            description="force_top_down 의 XY 앵커. 'median'=복원 cloud 중앙값"
                        "(점 밀도가 높은 쪽으로 끌림), 'extent'=실루엣 폭의 중점"
                        '(p2+p98)/2 로 밀도와 무관. 구처럼 한쪽 면만 찍히는 '
                        '물체에서 median 이 밀리면 extent 로.',
        ),
        DeclareLaunchArgument(
            'override_xy_with_bbox_center',
            default_value='false',
            description='Snap each grasp XY to the TARGET bbox centre '
                        '(centres a vertical grasp on the object).',
        ),
        DeclareLaunchArgument(
            'extrinsics_config',
            default_value=_default_ext,
            description='Path to camera_extrinsics.yaml (기본: ROBOT_CAPSTONE_ROOT 기반)',
        ),
        DeclareLaunchArgument(
            'world_frame',
            default_value='world',
            description='World frame ID',
        ),
        DeclareLaunchArgument(
            'robot_frame',
            default_value='panda_link0',
            description='Robot base frame for output poses',
        ),
        DeclareLaunchArgument(
            'ee_depth_topic',
            default_value='/ee_rgbd_camera/depth_image',
            description='Isaac EE depth 토픽 (실제: /ee_rgbd_camera/*)',
        ),
        DeclareLaunchArgument(
            'ee_camera_info_topic',
            default_value='/ee_rgbd_camera/camera_info',
            description='Isaac EE camera_info 토픽',
        ),
        DeclareLaunchArgument(
            'mask_topic',
            default_value='/qwen/mask_image',
            description='GSAM/qwen 마스크 토픽 (TARGET 마스크)',
        ),
        DeclareLaunchArgument(
            'transparent_reconstruct_enabled',
            default_value='true',
            description=(
                '투명 TARGET 의 see-through depth 를 복원. 켜두고 쓰는 게 정상이다 — '
                'transparent_force 가 false 인 한 graspgen_node 의 '
                '_is_transparent_target() 이 TARGET 라벨을 transparent_labels '
                "(glass/cup/bottle/transparent/wine) 와 대조해서, 'red ball' 이나 "
                "'book' 같은 불투명 대상에서는 복원 단계를 통째로 건너뛴다."
            ),
        ),
        DeclareLaunchArgument(
            'transparent_force',
            default_value='false',
            description='라벨 무시하고 항상 투명 복원 (테스트용)',
        ),
        DeclareLaunchArgument(
            'transparent_cylinder_height',
            default_value='0.10',   # 유리 위쪽 depth 신호 없어 추정 불가 → 고정값(m)
            description='컵 높이(m). <0 이면 데이터에서 추정',
        ),
        # ── SwinDRNet (Stage 2) ─────────────────────────────────────────
        DeclareLaunchArgument(
            'swindrnet_enabled',
            default_value='true',
            description=(
                'SwinDRNet 깊이 복원 (A100 서버 + 5557 터널 필요). 투명 TARGET 으로 '
                '판정됐을 때만 호출되므로 켜두어도 불투명 대상에는 비용이 없다. '
                '서버가 없으면 analytic 원통 복원으로 폴백한다.'
            ),
        ),
        DeclareLaunchArgument(
            'swindrnet_host',
            default_value='127.0.0.1',
            description='SwinDRNet server host (SSH tunnel endpoint)',
        ),
        DeclareLaunchArgument(
            'swindrnet_port',
            default_value='5557',
            description='SwinDRNet server port (SSH tunnel local port)',
        ),
        DeclareLaunchArgument(
            'swindrnet_timeout_ms',
            default_value='30000',
            description='SwinDRNet ZMQ timeout (ms)',
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='false',
            description='Open RViz2 preloaded with graspgen displays '
                        '(/graspgen/target_cloud restored cup + /grasp_markers)',
        ),
    ]

    node = Node(
        package    = 'graspgen_pkg',
        executable = 'graspgen_node',
        name       = 'graspgen_node',
        output     = 'screen',
        parameters = [
            os.path.join(pkg, 'config', 'graspgen_params.yaml'),
            {
                'zmq_host':         LaunchConfiguration('zmq_host'),
                'zmq_port':         LaunchConfiguration('zmq_port'),
                'zmq_timeout_ms':   LaunchConfiguration('zmq_timeout_ms'),
                'num_grasps':              LaunchConfiguration('num_grasps'),
                'topk_num_grasps':         LaunchConfiguration('topk_num_grasps'),
                'min_quality':             LaunchConfiguration('min_quality'),
                'min_point_count':         LaunchConfiguration('min_point_count'),
                'panda_link8_offset':      LaunchConfiguration('panda_link8_offset'),
                'top_down_filter_enabled': LaunchConfiguration('top_down_filter_enabled'),
                'top_down_angle_deg':      LaunchConfiguration('top_down_angle_deg'),
                'ik_filter_enabled':       LaunchConfiguration('ik_filter_enabled'),
                'max_published_grasps':    LaunchConfiguration('max_published_grasps'),
                'force_top_down_orientation':    LaunchConfiguration('force_top_down_orientation'),
                'force_top_down_grasp_z_frac':   LaunchConfiguration('force_top_down_grasp_z_frac'),
                'force_top_down_grasp_z_offset': LaunchConfiguration('force_top_down_grasp_z_offset'),
                'grasp_xy_anchor':               LaunchConfiguration('grasp_xy_anchor'),
                'override_xy_with_bbox_center':  LaunchConfiguration('override_xy_with_bbox_center'),
                'extrinsics_config':LaunchConfiguration('extrinsics_config'),
                'world_frame':      LaunchConfiguration('world_frame'),
                'robot_frame':      LaunchConfiguration('robot_frame'),
                'ee_depth_topic':        LaunchConfiguration('ee_depth_topic'),
                'ee_camera_info_topic':  LaunchConfiguration('ee_camera_info_topic'),
                'mask_topic':            LaunchConfiguration('mask_topic'),
                'transparent_reconstruct_enabled': LaunchConfiguration('transparent_reconstruct_enabled'),
                'transparent_force':               LaunchConfiguration('transparent_force'),
                'transparent_cylinder_height':     LaunchConfiguration('transparent_cylinder_height'),
                'swindrnet_enabled':     LaunchConfiguration('swindrnet_enabled'),
                'swindrnet_host':        LaunchConfiguration('swindrnet_host'),
                'swindrnet_port':        LaunchConfiguration('swindrnet_port'),
                'swindrnet_timeout_ms':  LaunchConfiguration('swindrnet_timeout_ms'),
            },
        ],
    )

    rviz_node = Node(
        package    = 'rviz2',
        executable = 'rviz2',
        name       = 'graspgen_rviz',
        arguments  = ['-d', os.path.join(pkg, 'rviz', 'graspgen.rviz')],
        condition  = IfCondition(LaunchConfiguration('rviz')),
        output     = 'screen',
    )

    return LaunchDescription(args + [node, rviz_node])
