"""
zmq_client.py — ZMQ REQ-REP client wrapper for GraspGen inference server.

Server protocol (aurora-g5, graspgen_franka_panda.yml):
  - Transport : ZMQ REQ/REP over TCP
  - Request   : msgpack-encoded dict
      {
          "point_cloud":     bytes   — (N,3) float32, row-major, world/base frame (metres)
          "num_grasps":      int     — total candidates server should generate
          "topk_num_grasps": int     — top-K the server returns (sorted by confidence)
      }
  - Response  : msgpack-encoded dict
      {
          "grasps":      bytes — (M,4,4) float32, homogeneous transforms, same frame as input
          "confidences": bytes — (M,)   float32, grasp scores (descending)
      }

TODO (server-side, paper §5.1): the GraspGen paper reports better empirical
results *without* non-maximal suppression — "inference without non-maximal
suppression was better, most likely since our motion planner is proficient
with goal set targets." The current ZMQ protocol does not expose an NMS
toggle. Coordinate with the server maintainer to either (a) disable NMS
internally by default, or (b) add a `use_nms: bool` field here. Once added,
also extend `_pack_request` and surface it as a graspgen_node parameter.

NOTE: If the server schema changes, update _pack_request / _unpack_response only.
      graspgen_node.py does not need to change.

SSH tunnel prerequisite (run before starting this node):
  # on-campus
  ssh -N -L <local_port>:<server_host>:<server_port> <user>@aurora.khu.ac.kr
  # off-campus
  ssh -p 30080 -N -L <local_port>:<server_host>:<server_port> <user>@aurora.khu.ac.kr
"""
from __future__ import annotations

import numpy as np

try:
    import zmq
    import msgpack
    import msgpack_numpy
    msgpack_numpy.patch()
    _DEPS_OK = True
    _DEPS_ERROR: str = ''
except ImportError as _e:
    _DEPS_OK = False
    _DEPS_ERROR = str(_e)


class GraspGenClient:
    """ZMQ REQ-REP client for a remote GraspGen inference server.

    Parameters
    ----------
    host        : str  — server host after SSH tunnel (usually '127.0.0.1')
    port        : int  — server port after SSH tunnel (e.g. 5556)
    timeout_ms  : int  — recv timeout in milliseconds; raises RuntimeError on expiry
    """

    def __init__(self, host: str, port: int, timeout_ms: int = 5000) -> None:
        if not _DEPS_OK:
            raise ImportError(
                f'graspgen_pkg requires pyzmq and msgpack: {_DEPS_ERROR}\n'
                'Run: pip install pyzmq msgpack'
            )
        self._addr       = f'tcp://{host}:{port}'
        self._timeout_ms = timeout_ms

        self._ctx    = zmq.Context()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        # Linger=0: don't block on close if peer is dead
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self._addr)

    # ── Public API ────────────────────────────────────────────────────────────

    def request(
        self,
        point_cloud: np.ndarray,    # (N, 3) float32, world/base frame (metres)
        num_grasps: int,
        topk_num_grasps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Send point cloud to server and return grasp poses + scores.

        Returns
        -------
        grasps      : (M, 4, 4) float32 — homogeneous transform matrices
        confidences : (M,)      float32 — grasp scores, descending order

        Raises
        ------
        RuntimeError  — ZMQ timeout or connection error
        ValueError    — unexpected response format
        """
        cloud = np.asarray(point_cloud, dtype=np.float32)
        if cloud.ndim != 2 or cloud.shape[1] != 3:
            raise ValueError(f'point_cloud must be (N,3), got {cloud.shape}')

        payload = self._pack_request(cloud, num_grasps, topk_num_grasps)

        # ZMQ REQ sockets enforce a strict send-recv state machine. A timeout
        # or error mid-cycle leaves the socket stuck in an invalid state and
        # every subsequent request raises "Operation cannot be accomplished in
        # current state". Reset the socket on any failure so the next request
        # gets a clean REQ.
        try:
            self._socket.send(payload)
            raw = self._socket.recv()
        except zmq.error.Again:
            self._reset_socket()
            raise RuntimeError(
                f'GraspGen server timeout after {self._timeout_ms} ms '
                f'(addr={self._addr}). Is the SSH tunnel open? '
                f'Larger num_grasps may need a higher zmq_timeout_ms.'
            )
        except zmq.ZMQError as e:
            self._reset_socket()
            raise RuntimeError(f'ZMQ error: {e}  (addr={self._addr})')

        return self._unpack_response(raw)

    def _reset_socket(self) -> None:
        """Recreate the REQ socket after a failure.

        REQ sockets cannot recover in-place from a timeout: send() can only
        follow recv(), and after a failed recv() the next send() raises EFSM
        ("Operation cannot be accomplished in current state"). Close and
        reopen to return to a clean SEND state.
        """
        try:
            self._socket.close(linger=0)
        except Exception:
            pass
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self._addr)

    def close(self) -> None:
        self._socket.close()
        self._ctx.term()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── Serialisation (change here if server protocol changes) ────────────────

    @staticmethod
    def _pack_request(
        cloud: np.ndarray,
        num_grasps: int,
        topk_num_grasps: int,
    ) -> bytes:
        return msgpack.packb(
            {
                'action':          'infer',
                'point_cloud':     cloud,
                'num_grasps':      int(num_grasps),
                'topk_num_grasps': int(topk_num_grasps),
            },
            use_bin_type=True,
        )

    @staticmethod
    def _unpack_response(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
        resp = msgpack.unpackb(raw, raw=False)

        error_msg = resp.get('error') if 'error' in resp else resp.get(b'error')
        if error_msg is not None:
            raise ValueError(f'Server error: {error_msg}')

        grasps = resp.get('grasps') if 'grasps' in resp else resp.get(b'grasps')
        confs  = resp.get('confidences') if 'confidences' in resp else resp.get(b'confidences')

        if grasps is None or confs is None:
            raise ValueError(
                f'Response missing "grasps"/"confidences". '
                f'Got keys: {list(resp.keys())}'
            )

        grasps = np.asarray(grasps, dtype=np.float32).reshape(-1, 4, 4)
        confs  = np.asarray(confs,  dtype=np.float32)
        return grasps, confs


# ── Dependency check helper ───────────────────────────────────────────────────

def check_deps() -> tuple[bool, str]:
    """Return (ok, error_message). Use at node startup."""
    return _DEPS_OK, _DEPS_ERROR
