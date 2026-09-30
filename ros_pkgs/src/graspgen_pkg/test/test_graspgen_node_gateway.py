"""graspgen_node end to end against a stand-in NOVA gateway.

The real node (rclpy, real parameters, real extrinsics) is built with
gateway_url pointing at the fake gateway in conftest.py, handed one scan — a
see-through glass whose depth is missing — and triggered the way
/world_map_result triggers it. Checked along the way:

  1. the frame goes to /compare/restore_depth: full frame, RAW depth, TARGET mask
  2. the cloud sent to /graspgen/infer is built from the RESTORED depth
  3. /grasp_candidates carries panda_link8 poses the BT can parse

Needs a sourced ROS 2 (rclpy, sensor_msgs) and skips otherwise.
"""
import json
from pathlib import Path

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
pytest.importorskip('requests')
from std_msgs.msg import String  # noqa: E402

from conftest import _png_decode  # noqa: E402
from graspgen_pkg.cloud_extractor import extract_target_cloud  # noqa: E402
from graspgen_pkg.depth_utils import load_ee_extrinsics  # noqa: E402

EXTRINSICS = (Path(__file__).resolve().parents[2]
              / 'mask_projection_pkg' / 'config' / 'camera_extrinsics.yaml')
H, W = 120, 160
K = np.array([[140.0, 0.0, W / 2], [0.0, 140.0, H / 2], [0.0, 0.0, 1.0]])
GLASS = (slice(40, 80), slice(60, 100))       # 40 x 40 px glass in the frame


@pytest.fixture
def node(gateway):
    from graspgen_pkg.graspgen_node import GraspGenNode

    rclpy.init(args=[
        '--ros-args',
        '-p', f'gateway_url:={gateway.url}',
        '-p', f'extrinsics_config:={EXTRINSICS}',
        '-p', 'swindrnet_enabled:=true',             # depth restoration on
        '-p', 'transparent_reconstruct_enabled:=true',
        '-p', 'ik_filter_enabled:=false',            # no MoveIt here
        '-p', 'use_grasp_profiles:=false',
        '-p', 'z_floor_margin:=-1.0',
    ])
    n = GraspGenNode()
    n.published = []
    n._grasp_pub.publish = lambda msg: n.published.append(json.loads(msg.data))
    yield n
    n.destroy_node()
    rclpy.shutdown()


def _load_scan(n):
    depth = np.full((H, W), 0.60, np.float32)        # table behind everything
    depth[GLASS] = np.nan                            # see-through: no return
    mask = np.zeros((H, W), np.uint8)
    mask[GLASS] = 1                                  # TARGET is detection #1
    rgb = np.zeros((H, W, 3), np.uint8)

    n._ee_depth, n._ee_K, n._ee_rgb, n._mask = depth, K, rgb, mask
    n._labeled_dets = [{'label': 'glass cup', 'category': 'TARGET'},
                       {'label': 'table', 'category': 'DESTINATION'}]


def test_scan_goes_through_compare_and_graspgen(node, gateway):
    _load_scan(node)
    # A real-shaped /world_map_result. The node only needs the TARGET entry.
    node._result_cb(String(data=json.dumps({'target': {
        'label': 'glass cup', 'centroid': [0.5, 0.0, 0.05], 'point_count': 1600,
        'bbox_3d_world': {'min': [0.45, -0.05, 0.0], 'max': [0.55, 0.05, 0.1]}}})))

    paths = [r['path'] for r in gateway.requests]
    assert paths == ['/compare/restore_depth', '/graspgen/infer']

    # 1. full frame, raw depth (NaN → 0 mm, NOT hole-punched further), mask = glass
    form = gateway.requests[0]['form']
    assert _png_decode(form['depth']).shape == (H, W)
    sent_depth = _png_decode(form['depth'])
    assert sent_depth[0, 0] == 600 and sent_depth[60, 80] == 0
    expect_mask = np.zeros((H, W), bool)
    expect_mask[GLASS] = True
    np.testing.assert_array_equal(_png_decode(form['mask']) > 0, expect_mask)

    # 2. the cloud GraspGen saw is the glass at the RESTORED 0.45 m
    R, t = load_ee_extrinsics(str(EXTRINSICS))
    restored = np.full((H, W), 0.60, np.float32)
    restored[GLASS] = 0.45
    expected = extract_target_cloud(restored, K, expect_mask.astype(np.uint8), 1,
                                     R, t, 0.05, 15.0, 10_000)
    sent_cloud = np.asarray(gateway.requests[1]['json']['point_cloud'])
    assert len(sent_cloud) == len(expected) == 1600
    np.testing.assert_allclose(sent_cloud.mean(axis=0), expected.mean(axis=0), atol=1e-4)

    # 3. /grasp_candidates: parseable by bt_pkg, link8 behind the grasp centre
    assert len(node.published) == 1
    cands = node.published[0]['candidates']
    assert len(cands) == 3
    for c in cands:
        assert {'position', 'quaternion', 'width', 'quality', 'frame'} <= set(c)
        assert len(c['position']) == 3 and len(c['quaternion']) == 4
    top = max(cands, key=lambda c: c['quality'])
    assert top['quality'] == pytest.approx(0.9)
    # Top-down grasp at the cloud centre → link8 0.103 m above it.
    np.testing.assert_allclose(top['position'],
                               expected.mean(axis=0) + [0, 0, 0.103], atol=1e-3)


def test_opaque_target_skips_depth_restoration(node, gateway):
    _load_scan(node)
    node._labeled_dets[0]['label'] = 'book'
    node._ee_depth[GLASS] = 0.55                         # opaque: real depth
    node._result_cb(String(data=json.dumps({'target': {
        'label': 'book', 'centroid': [0.5, 0.0, 0.05], 'point_count': 1600}})))
    assert [r['path'] for r in gateway.requests] == ['/graspgen/infer']
    assert len(node.published) == 1
