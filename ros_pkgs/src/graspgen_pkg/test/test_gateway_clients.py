"""GraspGen + depth-restoration clients against a stand-in NOVA gateway.

A real HTTP server on 127.0.0.1 answers each route the way nova-server's
gateway_server.py / tdr_server.py do, so the clients run their real requests
path: headers, JSON and multipart bodies, 16-bit millimetre PNGs, and the error
statuses (401 wrong key, 502 backend down, connection refused = tunnel down).

The unit conversions are the high-risk part. The node works in float METRES
with NaN for no-return; the wire is uint16 MILLIMETRES with 0 for no-return.
Getting either side wrong does not error — it just hands GraspGen a cloud that
is 1000x off or full of garbage points.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')
pytest.importorskip('requests')

from graspgen_pkg.gateway_client import (  # noqa: E402
    DepthRestoreHttpClient,
    depth_m_to_mm_png,
    GraspGenHttpClient,
    mm_png_to_depth_m,
)

from conftest import _png_decode, KEY  # noqa: E402


# ── GraspGen ─────────────────────────────────────────────────────────────────

def test_graspgen_request_matches_gateway_protocol(gateway):
    cloud = np.random.default_rng(0).random((200, 3)).astype(np.float32)
    client = GraspGenHttpClient(gateway.url)
    grasps, confs = client.request(cloud, num_grasps=200, topk_num_grasps=100)
    client.close()

    req = gateway.requests[0]
    assert req['path'] == '/graspgen/infer'
    assert req['headers']['x-api-key'] == KEY
    sent = req['json']
    assert sent['num_grasps'] == 200 and sent['topk_num_grasps'] == 100
    np.testing.assert_allclose(np.asarray(sent['point_cloud']), cloud, rtol=1e-6)

    assert grasps.shape == (3, 4, 4) and grasps.dtype == np.float32
    np.testing.assert_allclose(grasps[:, :3, 3], np.tile(cloud.mean(axis=0), (3, 1)), rtol=1e-5)
    np.testing.assert_allclose(confs, [0.9, 0.8, 0.7], rtol=1e-6)


def test_graspgen_rejects_bad_input_and_bad_replies(gateway):
    client = GraspGenHttpClient(gateway.url)
    with pytest.raises(ValueError, match=r'\(N,3\)'):
        client.request(np.zeros((5, 4), np.float32), 10, 5)
    assert gateway.requests == []                     # rejected before sending

    gateway.grasp_reply = {'grasps': np.eye(4)[None].tolist(), 'confidences': [0.9, 0.8]}
    with pytest.raises(ValueError, match='1 grasps but 2 confidences'):
        client.request(np.zeros((5, 3), np.float32), 10, 5)

    gateway.grasp_reply = {'poses': []}
    with pytest.raises(ValueError, match='missing'):
        client.request(np.zeros((5, 3), np.float32), 10, 5)
    client.close()


def test_wrong_key_names_the_key(gateway, monkeypatch):
    monkeypatch.setenv('NOVA_API_KEY', 'wrong')
    client = GraspGenHttpClient(gateway.url)
    with pytest.raises(RuntimeError, match='NOVA_API_KEY'):
        client.request(np.zeros((5, 3), np.float32), 10, 5)
    client.close()


def test_backend_down_points_at_health(gateway):
    gateway.status = 502
    client = GraspGenHttpClient(gateway.url)
    with pytest.raises(RuntimeError, match='/health'):
        client.request(np.zeros((5, 3), np.float32), 10, 5)
    client.close()


def test_tunnel_down_says_so():
    # Nothing listens on port 1: the "tunnel is down" case.
    client = GraspGenHttpClient('http://127.0.0.1:1', api_key='k', timeout_s=2)
    with pytest.raises(RuntimeError, match='tunnel'):
        client.request(np.zeros((5, 3), np.float32), 10, 5)
    client.close()


# ── depth: unit conversion ───────────────────────────────────────────────────

def test_metres_to_millimetre_png_and_back():
    d = np.array([[0.4567, np.nan], [np.inf, -1.0], [70.0, 0.0]], np.float32)
    mm = _png_decode(depth_m_to_mm_png(d))
    assert mm.dtype == np.uint16
    np.testing.assert_array_equal(mm, [[457, 0], [0, 0], [65535, 0]])   # invalid → 0
    back = mm_png_to_depth_m(depth_m_to_mm_png(d))
    assert back.dtype == np.float32
    assert back[0, 0] == pytest.approx(0.457) and back[0, 1] == 0.0


# ── depth: compare route (the default) ───────────────────────────────────────

def _scene(h=48, w=64):
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[..., 0] = 200                                    # red in RGB order
    depth = np.full((h, w), 0.8, np.float32)
    depth[20:30, 20:40] = np.nan                         # see-through glass
    mask = np.zeros((h, w), bool)
    mask[20:30, 20:40] = True
    return rgb, depth, mask


def test_compare_sends_full_frame_raw_depth_and_mask(gateway):
    rgb, depth, mask = _scene()
    client = DepthRestoreHttpClient(gateway.url)          # default route
    assert client.route == 'compare'
    restored = client.restore(rgb, depth, mask)
    client.close()

    req = gateway.requests[0]
    assert req['path'] == '/compare/restore_depth'
    assert req['headers']['x-api-key'] == KEY
    form = req['form']
    assert set(form) == {'rgb', 'depth', 'mask'}

    sent_rgb = _png_decode(form['rgb'])                  # decoded as BGR by cv2
    assert sent_rgb.shape == (48, 64, 3) and sent_rgb[0, 0, 2] == 200
    sent_depth = _png_decode(form['depth'])
    assert sent_depth.dtype == np.uint16 and sent_depth.shape == (48, 64)
    assert sent_depth[0, 0] == 800 and sent_depth[25, 25] == 0     # raw, NaN → 0
    np.testing.assert_array_equal(_png_decode(form['mask']) > 0, mask)

    assert restored.dtype == np.float32 and restored.shape == depth.shape
    assert restored[25, 25] == pytest.approx(0.45)                   # filled
    assert restored[0, 0] == pytest.approx(0.8)                      # kept
    assert client.last_meta['mask_source'] == 'supplied'


def test_swindrnet_route_sends_no_mask(gateway):
    rgb, depth, mask = _scene()
    client = DepthRestoreHttpClient(gateway.url, route='swindrnet')
    client.restore(rgb, depth, np.zeros_like(mask))      # empty mask is fine here
    client.close()
    assert gateway.requests[0]['path'] == '/swindrnet/restore_depth'
    assert set(gateway.requests[0]['form']) == {'rgb', 'depth'}


def test_depth_input_errors_are_caught_before_sending(gateway):
    rgb, depth, mask = _scene()
    with pytest.raises(ValueError, match='not in'):
        DepthRestoreHttpClient(gateway.url, route='midas')
    client = DepthRestoreHttpClient(gateway.url)
    with pytest.raises(ValueError, match='non-empty'):
        client.restore(rgb, depth, np.zeros_like(mask))
    with pytest.raises(ValueError, match='same size'):
        client.restore(rgb, depth[:, :-1], mask)
    client.close()
    assert gateway.requests == []
