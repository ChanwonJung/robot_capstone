"""Hand-off contracts between packages that no single package's tests cover.

Each stage is tested in its own package; these pin the JOINS, where a mismatch
fails silently rather than loudly:

  qwen_a100 detections ──► sam_a100 label map ──► graspgen target mask value
      a wrong index join makes graspgen send the DESTINATION's cloud to the
      server and plan a grasp on the basket.

  graspgen depth + mask ──► world-frame target cloud
      the cloud that actually crosses the wire to the GraspGen server.

  GraspGen 4x4 grasps ──► /grasp_candidates JSON ──► bt_executor_node
      a renamed key parses as a default (0,0,0) pose without an error, and a
      wrong TCP offset drives the fingertips 10 cm through the object.

Runs straight from the source tree, no colcon build needed. The graspgen parts
need a sourced ROS 2 install (for sensor_msgs / rclpy) and skip otherwise:

    source /opt/ros/jazzy/setup.bash
    python3 -m pytest tests/ -q
"""
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'ros_pkgs' / 'src'
for pkg in ('graspgen_pkg', 'slow_brain/sam_a100'):
    sys.path.insert(0, str(SRC / pkg))

pytest.importorskip('cv2')
from sam_a100.label_map import compose_label_map  # noqa: E402

pytest.importorskip('sensor_msgs.msg', reason='source a ROS 2 install first')
from graspgen_pkg.cloud_extractor import (  # noqa: E402
    extract_target_cloud,
    find_target_mask_val,
)
from graspgen_pkg.grasp_filter import (  # noqa: E402
    confidence_top_n,
    top_down_filter,
)

H, W = 60, 80
K = np.array([[100.0, 0.0, W / 2], [0.0, 100.0, H / 2], [0.0, 0.0, 1.0]])


# ── detections → label map → graspgen target value ──────────────────────────

@pytest.mark.parametrize('order', [
    ['TARGET', 'DESTINATION'],                 # qwen_a100's normal ordering
    ['DESTINATION', 'OBSTACLE', 'TARGET'],     # target not first
])
def test_graspgen_reads_the_target_pixels_sam_painted(order):
    dets = [{'label': c.lower(), 'category': c} for c in order]
    masks = np.zeros((len(dets), H, W), np.uint8)
    for i in range(len(dets)):
        masks[i, 5:15, 10 + 20 * i:25 + 20 * i] = 1

    label_map = compose_label_map(masks, list(range(len(dets))), (H, W))
    val = find_target_mask_val(dets)

    t = order.index('TARGET')
    np.testing.assert_array_equal(label_map == val, masks[t].astype(bool))


def test_target_survives_overlap_with_its_destination():
    # A book held over the basket: the target mask must stay intact.
    dets = [{'category': 'TARGET'}, {'category': 'DESTINATION'}]
    masks = np.zeros((2, H, W), np.uint8)
    masks[0, 10:20, 10:30] = 1
    masks[1, 0:40, 0:60] = 1
    label_map = compose_label_map(masks, [0, 1], (H, W), target_priority=True)
    assert np.all(label_map[masks[0] > 0] == find_target_mask_val(dets))


def test_label_map_follows_depth_resolution_not_rgb():
    # SAM runs on the RGB frame; graspgen indexes the mask with DEPTH pixels.
    masks = np.zeros((1, 2 * H, 2 * W), np.uint8)
    masks[0, 20:40, 20:60] = 1
    label_map = compose_label_map(masks, [0], (H, W))
    assert label_map.shape == (H, W)
    assert label_map[15, 20] == 1 and label_map[0, 0] == 0


# ── depth + mask → the cloud sent to the GraspGen server ────────────────────

def test_target_cloud_is_the_masked_patch_in_world_frame():
    depth = np.full((H, W), 0.5, np.float32)
    depth[0, 0] = np.nan                                  # invalid pixels drop
    mask = np.zeros((H, W), np.uint8)
    mask[20:30, 30:50] = 1
    mask[40:50, 0:10] = 2                                 # not the target

    R = np.diag([1.0, -1.0, -1.0])                        # camera looking down
    t = np.array([0.4, 0.0, 0.8])
    cloud = extract_target_cloud(depth, K, mask, 1, R, t,
                                 min_depth=0.05, max_depth=2.0, max_points=4096)

    assert cloud.dtype == np.float32 and cloud.shape == (200, 3)
    np.testing.assert_allclose(cloud[:, 2], 0.3, atol=1e-6)   # 0.8 - 0.5
    # Pixel (row 25, col 40) ↔ x = (40-40)*0.5/100, y = (25-30)*0.5/100.
    assert cloud[:, 0].min() == pytest.approx(0.4 - 0.05, abs=1e-6)
    assert cloud[:, 1].max() == pytest.approx(0.05, abs=1e-6)


def test_target_cloud_is_capped_and_empty_mask_returns_none():
    depth = np.full((H, W), 0.5, np.float32)
    mask = np.ones((H, W), np.uint8)
    R, t = np.eye(3), np.zeros(3)
    capped = extract_target_cloud(depth, K, mask, 1, R, t, 0.05, 2.0, 100)
    assert capped.shape == (100, 3)
    assert extract_target_cloud(depth, K, mask, 7, R, t, 0.05, 2.0, 100) is None


# ── GraspGen grasps → /grasp_candidates → bt_pkg ────────────────────────────

def _grasp(R, p):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3], T[:3, 3] = R, p
    return T


TOP_DOWN = np.diag([1.0, -1.0, -1.0])      # gripper +Z (approach) = world -Z
SIDE = np.array([[0.0, 0.0, 1.0],          # gripper +Z = world +X
                 [0.0, 1.0, 0.0],
                 [-1.0, 0.0, 0.0]])


@pytest.fixture
def build_candidates():
    rclpy = pytest.importorskip('rclpy')  # noqa: F841 — graspgen_node imports it
    from graspgen_pkg.graspgen_node import GraspGenNode

    def build(grasps, confs, tcp_offset=0.103):
        # Only the attributes _build_candidates reads, so no node is spun.
        stub = SimpleNamespace(_tcp_offset=tcp_offset, _flip_x=False,
                               _gripper_width=0.08)
        return GraspGenNode._build_candidates(
            stub, np.asarray(grasps), np.asarray(confs), None, 'panda_link0')
    return build


def test_link8_pose_sits_behind_the_grasp_centre(build_candidates):
    centre = [0.5, 0.1, 0.05]
    (c,) = build_candidates([_grasp(TOP_DOWN, centre)], [0.9])

    # Top-down: panda_link8 must be 0.103 m ABOVE the fingertip centre.
    np.testing.assert_allclose(c['position'], [0.5, 0.1, 0.153], atol=1e-6)
    # Orientation unchanged, and [x, y, z, w] — the order bt_pkg reads.
    np.testing.assert_allclose(np.abs(c['quaternion']), [1, 0, 0, 0], atol=1e-6)
    assert c['frame'] == 'panda_link0'


def test_top_down_filter_uses_the_same_approach_convention(build_candidates):
    cands = build_candidates([_grasp(TOP_DOWN, [0.5, 0, 0.05]),
                              _grasp(SIDE, [0.5, 0, 0.05])], [0.6, 0.9])
    kept = top_down_filter(cands, angle_threshold_deg=45.0)
    assert len(kept) == 1 and kept[0]['quality'] == pytest.approx(0.6)
    assert [c['quality'] for c in confidence_top_n(cands, 1)] == \
        [pytest.approx(0.9)]


def _bt_candidate_keys():
    """JSON keys parse_grasp_candidates() in bt_executor_node.cpp reads."""
    src = (SRC / 'bt_pkg' / 'src' / 'bt_executor_node.cpp').read_text()
    body = re.search(r'parse_grasp_candidates\(.*?\n}\n', src, re.S)
    assert body, 'parse_grasp_candidates() not found — update this test'
    body = body.group(0)
    return (set(re.findall(r'c\["(\w+)"\]', body)),
            set(re.findall(r'c\.value\("(\w+)"', body)),
            set(re.findall(r'j\.value\("(\w+)"', body)))


def test_published_json_carries_every_key_the_bt_reads(build_candidates):
    required, optional, top_level = _bt_candidate_keys()
    assert required == {'position', 'quaternion'}      # parser still as audited

    cands = build_candidates([_grasp(TOP_DOWN, [0.5, 0, 0.05])], [0.9])
    # Same envelope graspgen_node publishes, through a real JSON round trip.
    msg = json.loads(json.dumps({'candidates': cands,
                                 'target_centroid': [0.5, 0.0, 0.05],
                                 'stamp': 0.0}))

    assert top_level <= set(msg)
    for c in msg['candidates']:
        # A missing OPTIONAL key does not error in the BT — it silently takes a
        # default (frame → panda_link0, width → 0.08). Require all of them.
        assert (required | optional) <= set(c)
        assert len(c['position']) == 3 and len(c['quaternion']) == 4
        assert np.linalg.norm(c['quaternion']) == pytest.approx(1.0, abs=1e-5)
