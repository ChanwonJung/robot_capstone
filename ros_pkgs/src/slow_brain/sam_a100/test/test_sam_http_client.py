"""SamHttpClient against a stand-in NOVA gateway /sam2/segment on loopback.

Mirrors nova-server's sam2_server.py: multipart ``image`` + ``box`` JSON +
``multimask_output``, reply ``{"masks": [base64 PNG 0/255], "scores": [...]}``
best first.  Pins what the gateway forces on the client: ONE box per request,
the best-of-N pick for multimask, the x-api-key header, and errors that name
their cause (tunnel down, key rejected, backend down).
"""
import base64
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("requests")

from sam_a100.sam_http_client import SamHttpClient  # noqa: E402

KEY = "test-key"
H, W = 40, 60


def _form(content_type, body):
    msg = BytesParser(policy=email_policy).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
    return {p.get_param("name", header="content-disposition"): p.get_payload(decode=True)
            for p in msg.iter_parts()}


def _mask_b64(mask):
    ok, buf = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    return base64.b64encode(buf.tobytes()).decode()


class FakeSam2:
    """sam2_server.py /segment behaviour; ``status`` forces an error reply."""

    def __init__(self):
        self.requests = []
        self.status = 200
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                form = _form(self.headers["Content-Type"], body)
                fake.requests.append({"path": self.path, "headers": dict(self.headers),
                                      "form": form})
                if self.headers.get("x-api-key") != KEY:
                    return self._send(401, {"detail": "invalid API key"})
                if fake.status != 200:
                    return self._send(fake.status, {"detail": "sam2 unreachable"})
                img = cv2.imdecode(np.frombuffer(form["image"], np.uint8), cv2.IMREAD_COLOR)
                x0, y0, x1, y1 = (int(v) for v in json.loads(form["box"]))
                best = np.zeros(img.shape[:2], bool)
                best[y0:y1, x0:x1] = True
                if form["multimask_output"] == b"true":
                    # three candidates, best first — as the server sorts them
                    masks = [best, np.ones_like(best), np.zeros_like(best)]
                    scores = [0.95, 0.5, 0.1]
                else:
                    masks, scores = [best], [0.9]
                return self._send(200, {"masks": [_mask_b64(m) for m in masks],
                                        "scores": scores})

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._srv.server_port}"
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    def close(self):
        self._srv.shutdown()
        self._srv.server_close()


@pytest.fixture
def sam(monkeypatch):
    monkeypatch.setenv("NOVA_API_KEY", KEY)
    fake = FakeSam2()
    client = SamHttpClient(fake.url)
    yield fake, client
    client.close()
    fake.close()


def _image():
    img = np.zeros((H, W, 3), np.uint8)
    img[..., 0] = 200                                   # red, in RGB order
    return img


def test_one_request_per_box_matching_the_server_protocol(sam):
    fake, client = sam
    boxes = [[5, 5, 20, 15], [30, 10, 50, 35]]
    masks, scores = client.segment(_image(), boxes)

    assert [r["path"] for r in fake.requests] == ["/sam2/segment"] * 2
    for req, box in zip(fake.requests, boxes):
        assert req["headers"]["x-api-key"] == KEY
        assert json.loads(req["form"]["box"]) == box
        assert req["form"]["multimask_output"] == b"false"
        sent = cv2.imdecode(np.frombuffer(req["form"]["image"], np.uint8), cv2.IMREAD_COLOR)
        assert sent.shape == (H, W, 3) and sent[0, 0, 2] == 200   # RGB→BGR for PNG

    assert masks.shape == (2, H, W) and masks.dtype == np.uint8
    assert set(np.unique(masks)) == {0, 1}                        # 0/255 → 0/1
    assert masks[0, 10, 10] == 1 and masks[0, 30, 40] == 0
    assert masks[1, 30, 40] == 1
    np.testing.assert_allclose(scores, [0.9, 0.9])


def test_multimask_keeps_the_best_candidate(sam):
    fake, client = sam
    masks, scores = client.segment(_image(), [[5, 5, 20, 15]], multimask=True)
    assert fake.requests[0]["form"]["multimask_output"] == b"true"
    assert masks[0].sum() == 15 * 10                   # the box, not the all-ones one
    assert scores[0] == pytest.approx(0.95)


def test_wrong_key_backend_down_and_tunnel_down_name_their_cause(sam, monkeypatch):
    fake, client = sam
    fake.status = 502
    with pytest.raises(RuntimeError, match="/health"):
        client.segment(_image(), [[5, 5, 20, 15]])

    monkeypatch.setenv("NOVA_API_KEY", "wrong")
    bad_key = SamHttpClient(fake.url)
    with pytest.raises(RuntimeError, match="NOVA_API_KEY"):
        bad_key.segment(_image(), [[5, 5, 20, 15]])
    bad_key.close()

    down = SamHttpClient("http://127.0.0.1:1", timeout_s=2)
    with pytest.raises(RuntimeError, match="tunnel"):
        down.segment(_image(), [[5, 5, 20, 15]])
    down.close()


def test_input_validation_happens_before_sending(sam):
    fake, client = sam
    with pytest.raises(ValueError):
        client.segment(np.zeros((H, W), np.uint8), [[0, 0, 5, 5]])
    with pytest.raises(ValueError):
        client.segment(_image(), np.zeros((0, 4)))
    assert fake.requests == []
