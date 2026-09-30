"""Shared stand-in for the NOVA HTTP gateway (nova-server gateway_server.py).

A real HTTP server on 127.0.0.1 that answers /graspgen/infer and
/<route>/restore_depth the way the cluster does, with x-api-key auth. Used by
the client tests and by the node-level test.
"""
import base64
import json
import threading
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

KEY = 'test-key'


def _multipart(content_type: str, body: bytes) -> dict:
    msg = BytesParser(policy=email_policy).parsebytes(
        b'Content-Type: ' + content_type.encode() + b'\r\n\r\n' + body)
    return {part.get_param('name', header='content-disposition'):
            part.get_payload(decode=True) for part in msg.iter_parts()}


def _png_decode(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)


class FakeGateway:
    """Routes: /graspgen/infer and /<route>/restore_depth, with x-api-key auth.

    ``status`` forces the next reply's HTTP status (e.g. 502) with a detail body.
    """

    def __init__(self):
        self.requests = []
        self.status = 200
        self.grasp_reply = None
        gw = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers['Content-Length']))
                req = {'path': self.path, 'headers': dict(self.headers)}
                gw.requests.append(req)
                if self.headers.get('x-api-key') != KEY:
                    return self._send(401, {'detail': 'invalid API key'})
                if gw.status != 200:
                    return self._send(gw.status, {'detail': 'backend unreachable'})
                if self.path == '/graspgen/infer':
                    req['json'] = json.loads(body)
                    return self._send(200, gw.grasp_reply or gw.default_grasps(req['json']))
                if self.path.endswith('/restore_depth'):
                    req['form'] = _multipart(self.headers['Content-Type'], body)
                    return self._send(200, gw.restore(req['form']))
                return self._send(404, {'detail': 'Not Found'})

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self._srv = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self._srv.server_port}'
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    @staticmethod
    def default_grasps(payload):
        # Three top-down grasps (gripper +Z = approach = world -Z) centred on the
        # cloud it was sent — so a caller can check WHICH cloud arrived.
        centre = np.asarray(payload['point_cloud'], dtype=np.float64).mean(axis=0)
        n = 3
        g = np.tile(np.eye(4), (n, 1, 1))
        g[:, :3, :3] = np.diag([1.0, -1.0, -1.0])
        g[:, :3, 3] = centre
        return {'grasps': g.tolist(), 'confidences': [0.9, 0.8, 0.7],
                'num_grasps': n, 'timing': {'infer': 0.1}}

    @staticmethod
    def restore(form):
        # tdr/remake semantics: fill the masked (transparent) pixels, keep raw
        # elsewhere, answer at the input resolution in 16-bit millimetres.
        depth_mm = _png_decode(form['depth']).copy()
        if 'mask' in form:
            depth_mm[_png_decode(form['mask']) > 0] = 450
        ok, buf = cv2.imencode('.png', depth_mm)
        return {'depth_png_base64': base64.b64encode(buf.tobytes()).decode(),
                'mask_source': 'supplied' if 'mask' in form else None,
                'size': [depth_mm.shape[1], depth_mm.shape[0]]}

    def close(self):
        self._srv.shutdown()
        self._srv.server_close()


@pytest.fixture
def gateway(monkeypatch):
    monkeypatch.setenv('NOVA_API_KEY', KEY)
    gw = FakeGateway()
    yield gw
    gw.close()
