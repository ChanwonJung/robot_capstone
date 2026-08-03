"""SAM 2.1 client for the A100 segmentation server.

ROS-free.  ZMQ REQ/REP + msgpack, matching graspgen_pkg/zmq_client.py and
swindrnet_client.py — same transport, same recovery behaviour, different port.

Port 5558 (5556 = GraspGen, 5557 = SwinDRNet, 8000 = Qwen).  The server is
loopback-only on the A100, so the SSH tunnel from launch_env.bash is mandatory.

Wire protocol
-------------
request  {"action": "segment",
          "image":  (H, W, 3) uint8, **RGB** order,
          "boxes":  (N, 4) float32, [x1, y1, x2, y2] in image pixels,
          "multimask": bool}
response {"masks":  (N, H, W) uint8 0/1,
          "scores": (N,) float32}
         or {"error": "..."}

Boxes are prompt geometry, NOT drawn on the image — SAM 2.1 has a native box
prompt.  Painting rectangles into the pixels would corrupt the encoder input at
exactly the object boundary the mask decoder needs.
"""
from __future__ import annotations

import numpy as np

try:
    import msgpack
    import msgpack_numpy
    import zmq
    msgpack_numpy.patch()
    _DEPS_OK = True
    _DEPS_ERR = ""
except ImportError as exc:  # pragma: no cover - environment dependent
    _DEPS_OK = False
    _DEPS_ERR = str(exc)


def check_deps() -> None:
    """Raise with an actionable message if the ZMQ stack is missing."""
    if not _DEPS_OK:
        raise RuntimeError(
            f"SAM client dependencies unavailable ({_DEPS_ERR}). Install into "
            "gsam_venv:\n    gsam_venv/bin/pip install pyzmq msgpack msgpack-numpy"
        )


class SamClient:
    """REQ/REP client with reset-on-timeout recovery.

    A ZMQ REQ socket that times out is left in an unusable state — it will not
    accept another send until the strict send/recv alternation is restored.  The
    only clean fix is to tear the socket down and rebuild it, which is what
    every timeout path here does.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5558,
                 timeout_ms: int = 30000) -> None:
        check_deps()
        self._host = host
        self._port = port
        self._timeout_ms = timeout_ms
        self._ctx = zmq.Context.instance()
        self._sock = None
        self._connect()

    def _connect(self) -> None:
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(f"tcp://{self._host}:{self._port}")

    def _reset(self) -> None:
        if self._sock is not None:
            self._sock.close()
        self._connect()

    def segment(
        self,
        image_rgb: np.ndarray,
        boxes: np.ndarray,
        multimask: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Segment each box. Returns (masks (N,H,W) uint8, scores (N,)).

        One call per frame: the server runs the expensive image encoder once via
        set_image() and then a cheap decoder pass per box, so batching the boxes
        here is materially faster than calling once per object.
        """
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"image must be (H, W, 3) RGB, got {image_rgb.shape}")

        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            raise ValueError("no boxes to segment")

        req = {
            "action": "segment",
            "image": np.ascontiguousarray(image_rgb, dtype=np.uint8),
            "boxes": boxes,
            "multimask": bool(multimask),
        }

        try:
            self._sock.send(msgpack.packb(req, use_bin_type=True))
            reply = msgpack.unpackb(self._sock.recv(), raw=False)
        except zmq.Again as exc:
            self._reset()
            raise RuntimeError(
                f"SAM server timeout after {self._timeout_ms} ms at "
                f"{self._host}:{self._port} — is the tunnel open and the server up?"
            ) from exc
        except zmq.ZMQError as exc:
            self._reset()
            raise RuntimeError(f"SAM server transport error: {exc}") from exc

        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(f"SAM server error: {reply['error']}")

        masks = np.asarray(reply["masks"])
        scores = np.asarray(reply.get("scores", np.ones(len(masks), np.float32)))

        if masks.ndim == 2:                      # single box, server squeezed it
            masks = masks[None, ...]
        if len(masks) != len(boxes):
            raise RuntimeError(
                f"server returned {len(masks)} masks for {len(boxes)} boxes")

        return _binarise(masks), scores.astype(np.float32)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None


def _binarise(masks: np.ndarray) -> np.ndarray:
    """Normalise whatever the server sent into a clean 0/1 uint8 array.

    Do NOT reach for astype(np.uint8) here.  SAM2ImagePredictor returns raw
    LOGITS unless the server thresholds them, and casting a negative float to
    uint8 is undefined in numpy — it wraps, so -8.0 becomes 248.  Every pixel
    then reads as "inside the mask", the label map covers the whole frame, and
    the overlay comes out as a flat wash with no structure.  Threshold first.

    Accepted inputs:
      bool                      -> as-is
      float (logits, SAM's cutoff is 0.0) -> > 0
      float in [0, 1]           -> >= 0.5
      integer 0/1 or 0/255      -> > 0
    """
    m = np.asarray(masks)

    if m.dtype == bool:
        return m.astype(np.uint8)

    if np.issubdtype(m.dtype, np.floating):
        finite = m[np.isfinite(m)]
        # Probabilities live in [0,1]; logits span well beyond it.
        if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
            return (m >= 0.5).astype(np.uint8)
        return (m > 0.0).astype(np.uint8)

    # Integer of some width — 0/1, 0/255, or a label value.
    return (m > 0).astype(np.uint8)
