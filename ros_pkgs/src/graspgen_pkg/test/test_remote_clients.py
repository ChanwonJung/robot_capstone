"""GraspGen + SwinDRNet ZMQ clients against a stand-in server on loopback.

No GPU box and no tunnel: each test binds a real ZMQ REP socket on 127.0.0.1, so
the client code runs its real send/recv path and the wire format is checked
byte-for-byte — msgpack + msgpack_numpy, the same stack the remote servers use.

What matters most here is the timeout path. A ZMQ REQ socket that times out is
stuck mid-transaction and refuses every later send, so a client without its
reset-on-timeout recovery works exactly once and then fails forever after the
first slow inference.
"""
import threading
import time

import numpy as np
import pytest

zmq = pytest.importorskip('zmq')
msgpack = pytest.importorskip('msgpack')
msgpack_numpy = pytest.importorskip('msgpack_numpy')
msgpack_numpy.patch()

from graspgen_pkg.swindrnet_client import SwinDRNetClient  # noqa: E402
from graspgen_pkg.zmq_client import GraspGenClient  # noqa: E402


class FakeServer:
    """REP server on a random loopback port, one handler call per request.

    ``handler(request_dict) -> reply_dict``. Set ``delay_s`` to make the next
    reply late, simulating an inference slower than the client's timeout.
    """

    def __init__(self, handler):
        self.handler = handler
        self.requests = []
        self.delay_s = 0.0
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.setsockopt(zmq.LINGER, 0)
        self.port = self._sock.bind_to_random_port('tcp://127.0.0.1')
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            if not self._sock.poll(50):
                continue
            req = msgpack.unpackb(self._sock.recv(), raw=False)
            self.requests.append(req)
            delay, self.delay_s = self.delay_s, 0.0
            time.sleep(delay)
            self._sock.send(msgpack.packb(self.handler(req), use_bin_type=True))

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self._sock.close()
        self._ctx.term()


@pytest.fixture
def server_factory():
    servers = []

    def make(handler):
        s = FakeServer(handler)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.close()


# ── GraspGen ─────────────────────────────────────────────────────────────────

def _grasp_reply(n=3):
    grasps = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    grasps[:, 2, 3] = np.arange(n, dtype=np.float32)   # distinguishable z
    return {'grasps': grasps,
            'confidences': np.linspace(0.9, 0.5, n).astype(np.float32)}


def test_graspgen_request_matches_server_protocol(server_factory):
    srv = server_factory(lambda req: _grasp_reply())
    cloud = np.random.default_rng(0).random((500, 3)).astype(np.float32)

    client = GraspGenClient('127.0.0.1', srv.port, timeout_ms=2000)
    try:
        grasps, confs = client.request(cloud, num_grasps=200, topk_num_grasps=100)
    finally:
        client.close()

    req = srv.requests[0]
    assert req['action'] == 'infer'
    assert req['num_grasps'] == 200 and req['topk_num_grasps'] == 100
    sent = np.asarray(req['point_cloud'])
    assert sent.dtype == np.float32 and sent.shape == (500, 3)
    np.testing.assert_array_equal(sent, cloud)

    assert grasps.shape == (3, 4, 4) and grasps.dtype == np.float32
    np.testing.assert_allclose(grasps[:, 2, 3], [0, 1, 2])
    assert confs.shape == (3,)


def test_graspgen_flat_grasp_buffer_is_reshaped(server_factory):
    # A server that ships grasps as one flat (M*16,) buffer must still decode.
    reply = _grasp_reply(2)
    reply['grasps'] = reply['grasps'].reshape(-1)
    srv = server_factory(lambda req: reply)
    client = GraspGenClient('127.0.0.1', srv.port, timeout_ms=2000)
    try:
        grasps, _ = client.request(np.zeros((10, 3), np.float32), 10, 2)
    finally:
        client.close()
    assert grasps.shape == (2, 4, 4)


def test_graspgen_server_error_raises(server_factory):
    srv = server_factory(lambda req: {'error': 'CUDA out of memory'})
    client = GraspGenClient('127.0.0.1', srv.port, timeout_ms=2000)
    try:
        with pytest.raises(ValueError, match='CUDA out of memory'):
            client.request(np.zeros((10, 3), np.float32), 10, 2)
    finally:
        client.close()


def test_graspgen_missing_keys_raise(server_factory):
    srv = server_factory(lambda req: {'poses': []})
    client = GraspGenClient('127.0.0.1', srv.port, timeout_ms=2000)
    try:
        with pytest.raises(ValueError, match='missing'):
            client.request(np.zeros((10, 3), np.float32), 10, 2)
    finally:
        client.close()


def test_graspgen_rejects_non_xyz_cloud():
    client = GraspGenClient('127.0.0.1', 1, timeout_ms=100)
    try:
        with pytest.raises(ValueError, match=r'\(N,3\)'):
            client.request(np.zeros((10, 4), np.float32), 10, 2)
    finally:
        client.close()


def test_graspgen_recovers_after_timeout(server_factory):
    srv = server_factory(lambda req: _grasp_reply(1))
    client = GraspGenClient('127.0.0.1', srv.port, timeout_ms=300)
    cloud = np.zeros((10, 3), np.float32)
    try:
        srv.delay_s = 1.0
        with pytest.raises(RuntimeError, match='timeout'):
            client.request(cloud, 10, 1)
        time.sleep(1.1)   # let the server flush the late reply to the dead socket
        grasps, _ = client.request(cloud, 10, 1)
    finally:
        client.close()
    assert grasps.shape == (1, 4, 4)


# ── SwinDRNet ────────────────────────────────────────────────────────────────

def _frame(h=48, w=64):
    rgb = np.random.default_rng(1).integers(0, 255, (h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), 0.6, np.float32)
    depth[10:20, 10:20] = 0.0                     # the "transparent" hole
    K = np.array([[60.0, 0, w / 2], [0, 60.0, h / 2], [0, 0, 1]])
    return rgb, depth, K


def _restore(req):
    d = np.asarray(req['broken_depth']).copy()
    d[d == 0] = 0.55
    return {'status': 'ok', 'restored_depth': d}


def test_swindrnet_request_matches_server_protocol(server_factory):
    srv = server_factory(_restore)
    rgb, depth, K = _frame()
    client = SwinDRNetClient('127.0.0.1', srv.port, timeout_ms=2000)
    try:
        restored = client.restore(rgb, depth, K, use_pp=True)
    finally:
        client.close()

    req = srv.requests[0]
    assert set(req) == {'rgb', 'broken_depth', 'K', 'use_pp'}
    np.testing.assert_array_equal(np.asarray(req['rgb']), rgb)
    np.testing.assert_array_equal(np.asarray(req['broken_depth']), depth)
    assert np.asarray(req['K']).shape == (3, 3) and req['use_pp'] is True

    assert restored.dtype == np.float32 and restored.shape == depth.shape
    assert np.all(restored > 0)       # hole filled, metres preserved elsewhere
    assert restored[0, 0] == pytest.approx(0.6)


def test_swindrnet_server_error_raises(server_factory):
    srv = server_factory(lambda req: {'status': 'error', 'message': 'bad K'})
    rgb, depth, K = _frame()
    client = SwinDRNetClient('127.0.0.1', srv.port, timeout_ms=2000)
    try:
        with pytest.raises(RuntimeError, match='bad K'):
            client.restore(rgb, depth, K)
    finally:
        client.close()


def test_swindrnet_recovers_after_timeout(server_factory):
    srv = server_factory(_restore)
    rgb, depth, K = _frame()
    client = SwinDRNetClient('127.0.0.1', srv.port, timeout_ms=300)
    try:
        srv.delay_s = 1.0
        with pytest.raises(TimeoutError):
            client.restore(rgb, depth, K)
        time.sleep(1.1)
        restored = client.restore(rgb, depth, K)
    finally:
        client.close()
    assert restored.shape == depth.shape
