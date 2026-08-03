#!/usr/bin/env python3
"""
SwinDRNet Client — Local ROS node connector.

Used by graspgen_node to restore transparent object depth via ZMQ.
Connects to A100 server (via SSH tunnel on port 5557).

Usage in graspgen_node:
  client = SwinDRNetClient(host='127.0.0.1', port=5557, timeout_ms=30000)
  restored = client.restore(rgb_image, broken_depth_image)
"""

import logging
import time
from typing import Optional, Tuple

import msgpack
import msgpack_numpy as m
import numpy as np
import zmq

logger = logging.getLogger(__name__)


class SwinDRNetClient:
    """Client for SwinDRNet ZMQ inference server."""

    def __init__(self, host: str = '127.0.0.1', port: int = 5557, timeout_ms: int = 30000):
        """
        Initialize SwinDRNet client.

        Parameters
        ----------
        host : str
            Server host (default: 127.0.0.1 for local SSH tunnel)
        port : int
            Server port (default: 5557)
        timeout_ms : int
            ZMQ timeout in milliseconds (default: 30000)
        """
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.context = None
        self.socket = None
        self._connect()

    def _connect(self):
        """Establish ZMQ connection."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)

        addr = f"tcp://{self.host}:{self.port}"
        try:
            self.socket.connect(addr)
            logger.info(f"Connected to SwinDRNet server at {addr}")
        except Exception as e:
            logger.error(f"Failed to connect to {addr}: {e}")
            raise

    def restore(self, rgb: np.ndarray, broken_depth: np.ndarray,
                K: np.ndarray, use_pp: bool = False) -> np.ndarray:
        """
        Restore depth for transparent object.

        Parameters
        ----------
        rgb : (H, W, 3) uint8
            RGB image, **RGB channel order** (DREDS loads via PIL, not cv2).
        broken_depth : (H, W) float32
            Broken depth in METRES. 0 = invalid. Do not normalise — the network
            blends the input Z channel directly into its output.
        K : (3, 3)
            Camera intrinsics. The server needs fx/fy to build the XYZ point map
            that forms SwinDRNet's second input branch.
        use_pp : bool
            If True, the server uses the off-centre principal point from K
            (cx/cy scaled to 224) instead of pinning it to the image centre.
            Required when the input is a cup-centred crop whose pp is off-centre
            — this matches the fine-tune preprocessing. Default False preserves
            the centred-pp behaviour for full-frame DREDS-style inputs.

        Returns
        -------
        restored_depth : (H, W) float32
            Restored depth in metres.

        Raises
        ------
        RuntimeError
            If server is unreachable or returns error
        TimeoutError
            If request times out
        """
        # Validate inputs
        assert rgb.ndim == 3 and rgb.shape[2] == 3, f"RGB must be (H,W,3), got {rgb.shape}"
        assert broken_depth.ndim == 2, f"Depth must be (H,W), got {broken_depth.shape}"
        assert rgb.shape[:2] == broken_depth.shape, "RGB and depth shape mismatch"

        # Prepare request
        request = {
            'rgb': rgb.astype(np.uint8),
            'broken_depth': broken_depth.astype(np.float32),  # Server expects 'broken_depth'
            'K': np.asarray(K, dtype=np.float64).reshape(3, 3),
            'use_pp': bool(use_pp),
        }

        t0 = time.time()

        try:
            # Send request
            message = msgpack.packb(request, default=m.encode)
            self.socket.send(message)

            # Receive response
            message = self.socket.recv()
            response = msgpack.unpackb(message, object_hook=m.decode)

            elapsed = time.time() - t0
            logger.debug(f"Server latency: {elapsed*1000:.1f} ms")

            # Check status
            if response['status'] != 'ok':
                raise RuntimeError(f"Server error: {response.get('message', 'unknown')}")

            restored_depth = response['restored_depth'].astype(np.float32)

            logger.debug(f"Restored depth: {restored_depth.shape}, "
                        f"range=[{restored_depth.min():.3f}, {restored_depth.max():.3f}]")

            return restored_depth

        except zmq.error.Again:
            # A timed-out REQ socket is stuck mid-transaction (recv pending) —
            # the next send() would raise "Operation cannot be accomplished in
            # current state". Reset the socket so the next call starts clean.
            self._reset_socket()
            raise TimeoutError(f"Server timeout after {self.timeout_ms} ms")
        except Exception as e:
            logger.error(f"Request failed: {e}")
            self._reset_socket()
            raise RuntimeError(f"Depth restoration failed: {e}")

    def _reset_socket(self):
        """Tear down and re-create the REQ socket after a failed transaction."""
        try:
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.close()
            self.context.term()
        except Exception:
            pass
        self._connect()

    def close(self):
        """Close connection."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == '__main__':
    # Quick test
    import sys

    logging.basicConfig(level=logging.INFO)

    # Generate test data
    H, W = 480, 640
    rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    depth = np.random.uniform(0.5, 1.5, (H, W)).astype(np.float32)
    depth[100:200, 150:250] = 0  # Simulate transparent hole

    print(f"Test RGB: {rgb.shape}, Depth: {depth.shape}")
    print(f"Depth range: {depth.min():.3f} - {depth.max():.3f} m")

    try:
        with SwinDRNetClient(host='127.0.0.1', port=5557) as client:
            restored = client.restore(rgb, depth)
            print(f"Restored: {restored.shape}, "
                  f"range={restored.min():.3f}-{restored.max():.3f} m")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
