"""SAM 2.1 client for the NOVA HTTP gateway (the current server).

ROS-free.  Drop-in for sam_client.SamClient: same ``segment(image_rgb, boxes,
multimask)`` returning ``(masks (N,H,W) uint8 0/1, scores (N,) float32)``, so
sam_mask_node only chooses which one to construct.

Wire protocol (nova-server scripts/sam2_server.py, proxied at /sam2/*)
----------------------------------------------------------------------
request  POST $NOVA_GATEWAY_URL/sam2/segment, multipart form:
           image            PNG (the server converts to RGB)
           box              JSON "[x0, y0, x1, y1]" in image pixels
           multimask_output "true" | "false"
         header x-api-key: $NOVA_API_KEY
response {"masks": [base64 PNG, ...], "scores": [float, ...]}  best first

ONE BOX PER REQUEST.  The server hands ``box`` to SAM2ImagePredictor.predict as
a single prompt and indexes the result as one object's candidates, so a batch of
boxes cannot be sent in one call the way the old ZMQ server took them.  Each
call re-encodes the image server-side; with two boxes per scan (TARGET,
DESTINATION) that is two encoder passes, not one.

multimask=True asks for 3 candidates and keeps the best — the same semantics
as the ZMQ client, done by taking the first of the server's best-first list.
"""
from __future__ import annotations

import base64
import json
import os

import numpy as np

from .sam_client import _binarise

try:
    import cv2
    import requests
    _DEPS_OK = True
    _DEPS_ERR = ""
except ImportError as exc:  # pragma: no cover - environment dependent
    _DEPS_OK = False
    _DEPS_ERR = str(exc)

_DEFAULT_GATEWAY_URL = "http://127.0.0.1:9000"


def default_gateway_url() -> str:
    """$NOVA_GATEWAY_URL (exported by launch_env_seraph.sh), else the default tunnel."""
    return os.environ.get("NOVA_GATEWAY_URL", "") or _DEFAULT_GATEWAY_URL


class SamHttpClient:
    """POST /sam2/segment, one request per box."""

    def __init__(self, base_url: str, api_key: str | None = None,
                 timeout_s: float = 30.0) -> None:
        if not _DEPS_OK:
            raise RuntimeError(
                f"SAM gateway client dependencies unavailable ({_DEPS_ERR}). "
                "Install into gsam_venv:\n    gsam_venv/bin/pip install -r "
                "ros_pkgs/src/slow_brain/requirements.txt")
        self._base = base_url.rstrip("/")
        self._url = f"{self._base}/sam2/segment"
        self._timeout = float(timeout_s)
        self._session = requests.Session()
        key = os.environ.get("NOVA_API_KEY", "") if api_key is None else api_key
        if key:
            self._session.headers["x-api-key"] = key

    def segment(
        self,
        image_rgb: np.ndarray,
        boxes: np.ndarray,
        multimask: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Segment each box. Returns (masks (N,H,W) uint8, scores (N,))."""
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"image must be (H, W, 3) RGB, got {image_rgb.shape}")

        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            raise ValueError("no boxes to segment")

        h, w = image_rgb.shape[:2]
        ok, png = cv2.imencode(
            ".png", cv2.cvtColor(np.ascontiguousarray(image_rgb, np.uint8),
                                 cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError("PNG encode of the source frame failed")
        png = png.tobytes()

        masks, scores = [], []
        for box in boxes:
            body = self._post(png, [float(v) for v in box], multimask)
            if not body.get("masks"):
                raise RuntimeError(f"SAM returned no mask for box {box.tolist()}")
            m = cv2.imdecode(np.frombuffer(base64.b64decode(body["masks"][0]), np.uint8),
                             cv2.IMREAD_GRAYSCALE)
            if m is None or m.shape != (h, w):
                raise RuntimeError(
                    f"SAM mask shape {None if m is None else m.shape} != frame {(h, w)}")
            masks.append(m)
            s = body.get("scores") or [1.0]
            scores.append(float(s[0]))

        return _binarise(np.stack(masks)), np.asarray(scores, dtype=np.float32)

    def _post(self, png: bytes, box: list, multimask: bool) -> dict:
        try:
            resp = self._session.post(
                self._url,
                files={"image": ("frame.png", png, "image/png")},
                data={"box": json.dumps(box),
                      "multimask_output": "true" if multimask else "false"},
                timeout=self._timeout)
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"SAM gateway timeout after {self._timeout:.0f} s at {self._url}") from exc
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"SAM gateway unreachable at {self._base} — is the tunnel open? "
                "(source launch_env_seraph.sh)") from exc

        if resp.status_code == 401:
            raise RuntimeError(
                "gateway rejected the API key (HTTP 401) — export NOVA_API_KEY or put "
                "it in .env, then re-source launch_env_seraph.sh")
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            hint = f" — check {self._base}/health" if resp.status_code == 502 else ""
            raise RuntimeError(
                f"SAM server error: HTTP {resp.status_code}: {str(detail)[:300]}{hint}")
        try:
            return resp.json()
        except ValueError as exc:
            raise RuntimeError(f"SAM gateway returned non-JSON: {resp.text[:200]!r}") from exc

    def close(self) -> None:
        self._session.close()
