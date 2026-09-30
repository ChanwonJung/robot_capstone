"""
gateway_client.py — GraspGen + depth restoration over the NOVA HTTP gateway.

The current inference server (KHU cluster, nova-server stack) exposes every model
through ONE HTTP gateway, reached at $NOVA_GATEWAY_URL once launch_env_seraph.sh
has opened the tunnel. These clients replace the ZMQ ones for that server, and
keep their call shapes so graspgen_node only chooses which to construct:

  GraspGenHttpClient.request(cloud, num_grasps, topk)  ≙ zmq_client.GraspGenClient
  DepthRestoreHttpClient.restore(rgb, depth_m, mask)   — full frame, see below

Wire protocol (nova-server scripts/gateway_server.py, ACCESS.md):

  POST /graspgen/infer          JSON {"point_cloud": [[x,y,z],...] (m), "num_grasps",
                                      "topk_num_grasps"}
                                -> {"grasps": (M,4,4) nested lists, "confidences": (M,),
                                    "num_grasps", "timing"}
  POST /<route>/restore_depth   multipart: rgb (png), depth (16-bit PNG, MILLIMETRES,
                                same size as rgb), mask (png, nonzero = transparent)
                                -> {"depth_png_base64": 16-bit PNG mm at input size,
                                    "mask_source", "sam2_score", "size"}
     route = compare  (tdr CA-Dual, ours — the default)
           | remake   (ReMake, official release)
           | swindrnet (TransCG fine-tune; ignores the mask)

Every request carries `x-api-key`, read from $NOVA_API_KEY. The key is never a ROS
parameter: parameters are world-readable through `ros2 param get`.

DEPTH RESTORATION IS FULL-FRAME. The gateway models are trained on TransCG full
frames with the RAW depth plus a transparent-object mask, and blend
C * prediction + (1 - C) * raw. That is NOT the retired A100 SwinDRNet
(cup_ft_v3), which needed a hole-punched 252 px cup crop with a shifted principal
point — do not port that preprocessing here.
"""
from __future__ import annotations

import base64
import os

import numpy as np

try:
    import cv2
    import requests
    _DEPS_OK = True
    _DEPS_ERROR = ''
except ImportError as _e:  # pragma: no cover - environment dependent
    _DEPS_OK = False
    _DEPS_ERROR = str(_e)

DEPTH_ROUTES = ('compare', 'remake', 'swindrnet')
_MASKLESS_ROUTES = ('swindrnet',)
_DEFAULT_GATEWAY_URL = 'http://127.0.0.1:9000'


def default_gateway_url() -> str:
    """$NOVA_GATEWAY_URL (exported by launch_env_seraph.sh), else the default tunnel."""
    return os.environ.get('NOVA_GATEWAY_URL', '') or _DEFAULT_GATEWAY_URL


class _GatewaySession:
    """requests.Session + x-api-key + error messages that name the actual cause.

    Every failure becomes RuntimeError, which graspgen_node already catches for
    the ZMQ clients. The message says which of the three usual suspects it is —
    tunnel down, wrong key, or gateway up with the backend dead — because they
    need three different fixes.
    """

    def __init__(self, base_url: str, api_key: str | None, timeout_s: float) -> None:
        if not _DEPS_OK:
            raise ImportError(
                f'gateway client requires requests and opencv: {_DEPS_ERROR}\n'
                'Run: gsam_venv/bin/pip install -r ros_pkgs/src/graspgen_pkg/requirements.txt')
        self.base_url = base_url.rstrip('/')
        key = os.environ.get('NOVA_API_KEY', '') if api_key is None else api_key
        self._timeout = float(timeout_s)
        self._session = requests.Session()
        if key:
            self._session.headers['x-api-key'] = key

    def post(self, path: str, **kwargs) -> dict:
        url = f'{self.base_url}{path}'
        try:
            resp = self._session.post(url, timeout=self._timeout, **kwargs)
        except requests.exceptions.Timeout:
            raise RuntimeError(f'gateway timeout after {self._timeout:.0f} s ({url})')
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f'gateway unreachable at {self.base_url} — is the tunnel open? '
                f'(source launch_env_seraph.sh)  [{e.__class__.__name__}]')

        if resp.status_code != 200:
            try:
                detail = resp.json().get('detail', resp.text)
            except ValueError:
                detail = resp.text
            detail = str(detail)[:300]
            if resp.status_code == 401:
                raise RuntimeError(
                    'gateway rejected the API key (HTTP 401) — export NOVA_API_KEY '
                    'or put it in .env, then re-source launch_env_seraph.sh')
            if resp.status_code == 502:
                raise RuntimeError(
                    f'gateway is up but the backend is not (HTTP 502): {detail}. '
                    f'Check {self.base_url}/health')
            raise RuntimeError(f'gateway HTTP {resp.status_code} on {path}: {detail}')

        try:
            return resp.json()
        except ValueError:
            raise RuntimeError(f'gateway returned non-JSON on {path}: {resp.text[:200]!r}')

    def close(self) -> None:
        self._session.close()


# ── GraspGen ─────────────────────────────────────────────────────────────────

class GraspGenHttpClient:
    """Drop-in for zmq_client.GraspGenClient against POST /graspgen/infer."""

    def __init__(self, base_url: str, api_key: str | None = None,
                 timeout_s: float = 120.0) -> None:
        # 120 s matches the gateway's own ZMQ timeout to GraspGen: diffusion
        # inference time grows with num_grasps, and a shorter client timeout
        # would abandon requests the server is still going to answer.
        self._http = _GatewaySession(base_url, api_key, timeout_s)

    def request(
        self,
        point_cloud: np.ndarray,    # (N, 3) float32, world/base frame (metres)
        num_grasps: int,
        topk_num_grasps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Same contract as GraspGenClient.request: ((M,4,4) float32, (M,) float32)."""
        cloud = np.asarray(point_cloud, dtype=np.float32)
        if cloud.ndim != 2 or cloud.shape[1] != 3:
            raise ValueError(f'point_cloud must be (N,3), got {cloud.shape}')

        body = self._http.post('/graspgen/infer', json={
            'point_cloud':     cloud.tolist(),
            'num_grasps':      int(num_grasps),
            'topk_num_grasps': int(topk_num_grasps),
        })
        if 'grasps' not in body or 'confidences' not in body:
            raise ValueError(
                f'Response missing "grasps"/"confidences". Got keys: {list(body)}')

        grasps = np.asarray(body['grasps'], dtype=np.float32).reshape(-1, 4, 4)
        confs = np.asarray(body['confidences'], dtype=np.float32).reshape(-1)
        if len(grasps) != len(confs):
            raise ValueError(f'{len(grasps)} grasps but {len(confs)} confidences')
        return grasps, confs

    def close(self) -> None:
        self._http.close()


# ── Depth restoration ────────────────────────────────────────────────────────

def _png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode('.png', img)
    if not ok:
        raise ValueError(f'PNG encode failed for {img.dtype} {img.shape}')
    return buf.tobytes()


def depth_m_to_mm_png(depth_m: np.ndarray) -> bytes:
    """Float metres -> 16-bit millimetre PNG. NaN/inf/<=0 become 0 = no depth."""
    d = np.nan_to_num(np.asarray(depth_m, dtype=np.float64),
                      nan=0.0, posinf=0.0, neginf=0.0)
    mm = np.clip(np.rint(d * 1000.0), 0, 65535).astype(np.uint16)
    return _png(mm)


def mm_png_to_depth_m(data: bytes) -> np.ndarray:
    """16-bit millimetre PNG -> float32 metres. 0 stays 0 (invalid)."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.dtype != np.uint16 or img.ndim != 2:
        raise ValueError('restored depth is not a single-channel 16-bit PNG')
    return img.astype(np.float32) / 1000.0


class DepthRestoreHttpClient:
    """Transparent-object depth restoration via POST /<route>/restore_depth."""

    def __init__(self, base_url: str, route: str = 'compare',
                 api_key: str | None = None, timeout_s: float = 60.0) -> None:
        if route not in DEPTH_ROUTES:
            raise ValueError(f'depth route {route!r} not in {DEPTH_ROUTES}')
        self.route = route
        self.last_meta: dict = {}
        self._http = _GatewaySession(base_url, api_key, timeout_s)

    def restore(self, rgb: np.ndarray, depth_m: np.ndarray,
                mask: np.ndarray) -> np.ndarray:
        """Restore the full frame.

        rgb     (H, W, 3) uint8, RGB order
        depth_m (H, W) float metres — RAW, not hole-punched; the model blends it
        mask    (H, W) bool/int, nonzero = transparent object (the TARGET)

        Returns (H, W) float32 metres; 0 = no depth, which extract_target_cloud
        already drops via min_depth. ``last_meta`` holds mask_source/sam2_score.
        """
        rgb = np.asarray(rgb)
        depth_m = np.asarray(depth_m)
        mask = np.asarray(mask)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f'rgb must be (H,W,3), got {rgb.shape}')
        if depth_m.shape != rgb.shape[:2] or mask.shape != rgb.shape[:2]:
            # The server requires depth aligned to the colour frame; resizing
            # here would silently misregister the mask against the depth.
            raise ValueError(
                f'rgb {rgb.shape[:2]}, depth {depth_m.shape}, mask {mask.shape} '
                'must all be the same size')
        needs_mask = self.route not in _MASKLESS_ROUTES
        if needs_mask and not np.any(mask):
            raise ValueError(f'{self.route} needs a non-empty transparent-object mask')

        files = {
            'rgb':   ('rgb.png', _png(cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)),
                      'image/png'),
            'depth': ('depth.png', depth_m_to_mm_png(depth_m), 'image/png'),
        }
        if needs_mask:
            files['mask'] = ('mask.png', _png((mask > 0).astype(np.uint8) * 255), 'image/png')

        body = self._http.post(f'/{self.route}/restore_depth', files=files)
        if 'depth_png_base64' not in body:
            raise ValueError(f'Response missing "depth_png_base64". Got keys: {list(body)}')

        restored = mm_png_to_depth_m(base64.b64decode(body['depth_png_base64']))
        if restored.shape != depth_m.shape:
            raise ValueError(
                f'restored depth {restored.shape} != input {depth_m.shape}')
        self.last_meta = {k: body[k] for k in ('mask_source', 'sam2_score', 'size')
                          if k in body}
        return restored

    def close(self) -> None:
        self._http.close()
