"""qwen_call.ground() against a stand-in NOVA gateway /qwen/v1/chat/completions.

The gateway authenticates with an ``x-api-key`` header, which the OpenAI SDK
never sends on its own (it sends ``Authorization: Bearer``).  Without the
default header every grounding call is a 401, so that — plus the model id, the
endpoint derived from $NOVA_GATEWAY_URL, and the reasoning-model failure mode —
is what this pins.  The real openai client does the HTTP.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("openai")
pytest.importorskip("pydantic")

from qwen_a100.qwen_call import DEFAULT_MODEL, default_endpoint, ground  # noqa: E402

KEY = "test-key"

REPLY = {
    "objects": [
        {"label": "glass cup", "category": "TARGET",
         "bbox_xyxy": [100, 200, 300, 600], "confidence": 0.9},
        {"label": "basket", "category": "DESTINATION",
         "bbox_xyxy": [500, 400, 900, 950], "confidence": 0.8},
    ],
    "target_label": "glass cup",
    "destination": {"reference_label": "basket", "type": "container"},
    "confidence": 0.9,
    "needs_clarification": False,
}


class FakeQwen:
    """OpenAI-compatible chat completions, SGLang reasoning-parser shaped."""

    def __init__(self):
        self.requests = []
        self.content = json.dumps(REPLY)
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append({"path": self.path, "headers": dict(self.headers),
                                      "json": body})
                if self.headers.get("x-api-key") != KEY:
                    return self._send(401, {"detail": "invalid API key"})
                done = fake.content is not None
                return self._send(200, {
                    "id": "chatcmpl-1", "object": "chat.completion", "created": 0,
                    "model": body["model"],
                    "choices": [{"index": 0, "finish_reason": "stop" if done else "length",
                                 "message": {"role": "assistant", "content": fake.content,
                                             "reasoning_content": "thinking..."}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                              "total_tokens": 15},
                })

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
def qwen(monkeypatch):
    fake = FakeQwen()
    monkeypatch.setenv("NOVA_API_KEY", KEY)
    monkeypatch.setenv("NOVA_GATEWAY_URL", fake.url)
    yield fake
    fake.close()


def _ground(**kw):
    frame = np.zeros((480, 640, 3), np.uint8)
    return ground(frame, "put the glass cup in the basket",
                  endpoint_url=default_endpoint(), model=DEFAULT_MODEL,
                  bbox_convention="normalized_1000", timeout_sec=10, **kw)


def test_endpoint_and_model_follow_the_gateway(qwen):
    assert default_endpoint() == f"{qwen.url}/qwen/v1"
    assert DEFAULT_MODEL == "qwen3.5-27b"


def test_ground_sends_key_model_and_schema_and_parses(qwen):
    objects, grounding, meta = _ground()

    req = qwen.requests[0]
    assert req["path"] == "/qwen/v1/chat/completions"
    assert req["headers"]["x-api-key"] == KEY
    assert req["json"]["model"] == "qwen3.5-27b"
    assert req["json"]["response_format"]["type"] == "json_schema"
    image_part = req["json"]["messages"][1]["content"][0]
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")

    assert [d.category for d in objects] == ["TARGET", "DESTINATION"]
    # normalized_1000 → pixels on the 640x480 frame
    assert objects[0].bbox_xyxy == pytest.approx([64.0, 96.0, 192.0, 288.0])
    assert grounding.target_label == "glass cup"
    assert grounding.destination.type == "container"
    assert meta["has_destination"] and not meta["ambiguous"]


def test_missing_key_is_a_401_not_a_silent_hang(qwen, monkeypatch):
    monkeypatch.delenv("NOVA_API_KEY")
    with pytest.raises(Exception, match="401|invalid API key"):
        _ground()
    assert "x-api-key" not in {k.lower() for k in qwen.requests[0]["headers"]}


def test_reasoning_that_eats_the_budget_fails_loudly(qwen):
    # Qwen3.5 on SGLang: thinking tokens count against max_tokens. Run out and
    # content comes back null with finish_reason "length".
    qwen.content = None
    with pytest.raises(RuntimeError, match="null content"):
        _ground(max_tokens=64)
