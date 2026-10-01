"""SamClient against a stand-in SAM 2.1 server on loopback.

A real ZMQ REP socket on 127.0.0.1 answers each request, so the client's real
send/recv path and msgpack_numpy encoding are exercised without the GPU box or
the tunnel.  Pins the wire protocol in sam_client.py's docstring, the reply
normalisation (squeezed single masks, raw logits), and reset-on-timeout
recovery — a REQ socket that times out otherwise refuses every later request.
"""
import threading
import time

import numpy as np
import pytest

zmq = pytest.importorskip("zmq")
msgpack = pytest.importorskip("msgpack")
msgpack_numpy = pytest.importorskip("msgpack_numpy")
msgpack_numpy.patch()

from sam_a100.sam_client import SamClient  # noqa: E402

H, W = 40, 60


class FakeSamServer:
    """REP server; ``handler(request) -> reply``, ``delay_s`` makes one reply late."""

    def __init__(self, handler):
        self.handler = handler
        self.requests = []
        self.delay_s = 0.0
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.setsockopt(zmq.LINGER, 0)
        self.port = self._sock.bind_to_random_port("tcp://127.0.0.1")
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
def serve():
    servers, clients = [], []

    def make(handler, timeout_ms=2000):
        srv = FakeSamServer(handler)
        servers.append(srv)
        client = SamClient("127.0.0.1", srv.port, timeout_ms=timeout_ms)
        clients.append(client)
        return srv, client

    yield make
    for c in clients:
        c.close()
    for s in servers:
        s.close()


def _box_masks(req):
    """One filled rectangle per box, as 0/1 uint8 — the documented reply."""
    boxes = np.asarray(req["boxes"])
    masks = np.zeros((len(boxes), H, W), np.uint8)
    for i, (x1, y1, x2, y2) in enumerate(boxes.astype(int)):
        masks[i, y1:y2, x1:x2] = 1
    return {"masks": masks, "scores": np.full(len(boxes), 0.9, np.float32)}


def _image():
    return np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)


def test_request_matches_server_protocol(serve):
    srv, client = serve(_box_masks)
    boxes = [[5, 5, 20, 15], [30, 10, 50, 35]]
    image = _image()

    masks, scores = client.segment(image, boxes, multimask=True)

    req = srv.requests[0]
    assert req["action"] == "segment" and req["multimask"] is True
    np.testing.assert_array_equal(np.asarray(req["image"]), image)
    sent = np.asarray(req["boxes"])
    assert sent.dtype == np.float32 and sent.shape == (2, 4)

    assert masks.shape == (2, H, W) and masks.dtype == np.uint8
    assert masks[0, 10, 10] == 1 and masks[0, 30, 40] == 0
    assert masks[1, 30, 40] == 1
    np.testing.assert_allclose(scores, [0.9, 0.9])


def test_single_squeezed_mask_is_restored_to_batch(serve):
    _, client = serve(lambda req: {"masks": _box_masks(req)["masks"][0]})
    masks, scores = client.segment(_image(), [[5, 5, 20, 15]])
    assert masks.shape == (1, H, W)
    assert scores.shape == (1,)          # defaulted when the server omits it


def test_logit_masks_are_thresholded_not_wrapped(serve):
    # Raw SAM logits: casting -8.0 straight to uint8 wraps to 248 and the
    # "mask" covers the whole frame.
    def logits(req):
        m = np.full((1, H, W), -8.0, np.float32)
        m[0, 5:15, 5:20] = 6.0
        return {"masks": m}

    _, client = serve(logits)
    masks, _ = client.segment(_image(), [[5, 5, 20, 15]])
    assert masks.sum() == 10 * 15


def test_mask_count_mismatch_raises(serve):
    _, client = serve(lambda req: {"masks": np.zeros((1, H, W), np.uint8)})
    with pytest.raises(RuntimeError, match="1 masks for 2 boxes"):
        client.segment(_image(), [[0, 0, 5, 5], [5, 5, 9, 9]])


def test_server_error_raises(serve):
    _, client = serve(lambda req: {"error": "image too large"})
    with pytest.raises(RuntimeError, match="image too large"):
        client.segment(_image(), [[0, 0, 5, 5]])


def test_input_validation_happens_before_sending(serve):
    srv, client = serve(_box_masks)
    with pytest.raises(ValueError):
        client.segment(np.zeros((H, W), np.uint8), [[0, 0, 5, 5]])
    with pytest.raises(ValueError):
        client.segment(_image(), np.zeros((0, 4)))
    assert srv.requests == []


def test_recovers_after_timeout(serve):
    srv, client = serve(_box_masks, timeout_ms=300)
    srv.delay_s = 1.0
    with pytest.raises(RuntimeError, match="timeout"):
        client.segment(_image(), [[5, 5, 20, 15]])
    time.sleep(1.1)   # let the server flush its late reply to the dead socket
    masks, _ = client.segment(_image(), [[5, 5, 20, 15]])
    assert masks.shape == (1, H, W)
