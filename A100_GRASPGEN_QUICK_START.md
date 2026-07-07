# GraspGen on A100 — Quick Start Guide

**Status**: ✅ Server deployed on `tta@123.37.28.208` (2026-07-07)  
**Server Port**: 5556 (ZMQ, loopback only)  
**Latency**: ~184 ms round-trip (via SSH tunnel)

---

## Prerequisites

### 1. Local SSH setup (one-time)
Ensure you can connect without a password:

```bash
ssh-copy-id tta@123.37.28.208
# Test: ssh tta@123.37.28.208 'echo OK'
```

### 2. Python dependencies

In your **gsam_venv** (or global env):
```bash
pip install pyzmq msgpack msgpack-numpy numpy
```

### 3. A100 Server running

Check if the server is already running:
```bash
ssh tta@123.37.28.208 'pgrep -f graspgen_server.py && echo RUNNING || echo DOWN'
```

**To start it** (if down):
```bash
ssh tta@123.37.28.208 'nohup bash /data/tta/graspgen/run_server.sh > /data/tta/graspgen/server.log 2>&1 &'
# Check status after ~5 seconds:
ssh tta@123.37.28.208 'tail /data/tta/graspgen/server.log'
```

---

## One-time: Open SSH Tunnel

Each session, run this in a dedicated terminal (or background it):

```bash
ssh -N -L 5556:127.0.0.1:5556 tta@123.37.28.208 &
sleep 1
# Test tunnel: nc -zv 127.0.0.1 5556
```

Or add to `launch_env.bash` to automate it.

---

## Quick Test: Synthetic Cube

```bash
cd ros_pkgs/src/graspgen_pkg

# Activate venv (if not in launch_env.bash already)
source ../../../gsam_venv/bin/activate

# Run CLI with synthetic cube
python3 graspgen_pkg/graspgen_cli.py \
  --host 127.0.0.1 --port 5556 \
  --shape cube \
  --num-grasps 100 --topk 10
```

Expected output:
```
==============================================================
GraspGen Inference Result
==============================================================
Total grasps returned: 10

Top 10 grasps by confidence:

  [1] conf=0.9876
      pos=(+0.0234, -0.0102, +0.1050) m
      rotation=
[[-0.123  0.456  0.881]
 [ 0.321 -0.789  0.524]
 [ 0.940  0.213 -0.258]]
  ...
```

---

## Advanced: Load Your Own Point Cloud

### From NumPy file:
```bash
python3 graspgen_pkg/graspgen_cli.py \
  --host 127.0.0.1 --port 5556 \
  --from-file /path/to/my_cloud.npy \
  --num-grasps 100 --topk 10
```

### From JSON file:
```bash
python3 graspgen_pkg/graspgen_cli.py \
  --host 127.0.0.1 --port 5556 \
  --from-file detections.json \
  --num-grasps 100 --topk 10
```

**JSON format** (two options):
```json
// Option 1: array of [x, y, z]
[[0.1, 0.2, 0.15], [0.12, 0.21, 0.16], ...]

// Option 2: dict with "points" key
{"points": [[0.1, 0.2, 0.15], [0.12, 0.21, 0.16], ...], ...}
```

---

## Save Results

### Save computed grasps:
```bash
python3 graspgen_pkg/graspgen_cli.py \
  --host 127.0.0.1 --port 5556 \
  --shape sphere \
  --num-grasps 100 --topk 10 \
  --save-grasps /tmp/grasps.npz
```

Inspect:
```python
import numpy as np
npz = np.load('/tmp/grasps.npz')
grasps = npz['grasps']        # (10, 4, 4)
confs = npz['confidences']    # (10,)
```

### Save input cloud (for debugging):
```bash
python3 graspgen_pkg/graspgen_cli.py \
  --host 127.0.0.1 --port 5556 \
  --shape cube \
  --num-grasps 100 --topk 10 \
  --save-cloud /tmp/cloud.npy
```

---

## Full CLI Reference

```bash
python3 graspgen_pkg/graspgen_cli.py --help
```

### Connection options:
- `--host` : Server host (default: `127.0.0.1`)
- `--port` : Server port (default: `5556`)
- `--timeout-ms` : ZMQ timeout (default: `10000`)

### Cloud input (pick one):
- `--shape {cube,sphere}` : Generate synthetic
  - `--num-points` : Points in cloud (default: 1000)
  - `--size` : Cube size or sphere radius (default: 0.1 m)
  - `--center X Y Z` : Cloud center (default: 0 0 0.1)
- `--from-file PATH` : Load from JSON/NPY/NPZ

### Inference:
- `--num-grasps` : Total candidates to generate (default: 100)
- `--topk` : Top-K to return (default: 10)
- `--show` : Display top-N grasps (default: 5)

### Output:
- `--save-grasps PATH` : Save to `.npz` (grasps, confidences)
- `--save-cloud PATH` : Save input to `.npy`

---

## Integration with ROS2 Pipeline

The server is **already parameterized** in `graspgen_params.yaml`:
```yaml
zmq_host: 127.0.0.1
zmq_port: 5556
zmq_timeout_ms: 10000
```

Just launch as usual:
```bash
ros2 launch graspgen_pkg graspgen.launch.py
```

The node will automatically use the A100 server if tunnel is open.

---

## Troubleshooting

### "Connection refused"
```bash
# Check tunnel is open
ss -tlnp | grep 5556

# If not, open it:
ssh -N -L 5556:127.0.0.1:5556 tta@123.37.28.208 &
```

### "Server timeout after 10000 ms"
- Check A100 server is running: `ssh tta@123.37.28.208 'pgrep -f graspgen_server.py'`
- Increase `--timeout-ms` for large `num_grasps`
- Check logs: `ssh tta@123.37.28.208 'tail -50 /data/tta/graspgen/server.log'`

### "No module named 'msgpack_numpy'"
```bash
pip install msgpack-numpy
```

### "Point cloud must be (N, 3)"
Ensure JSON/NPY shape is `(num_points, 3)` in XYZ order, `float32`.

---

## Server Maintenance (A100 admin only)

### View server log:
```bash
ssh tta@123.37.28.208 'tail -100 /data/tta/graspgen/server.log'
```

### Restart server:
```bash
ssh tta@123.37.28.208 'pkill -f graspgen_server.py; sleep 1'
ssh tta@123.37.28.208 'nohup bash /data/tta/graspgen/run_server.sh > /dev/null 2>&1 &'
```

### Check hardware:
```bash
ssh tta@123.37.28.208 'nvidia-smi'  # A100 status
ssh tta@123.37.28.208 'free -h'     # Memory
```

---

## Example: ROS2 + CLI Comparison

### Via ROS2 (real-time grasp grounding):
```bash
# Terminal 1: Launch graspgen (uses ZMQ server)
ros2 launch graspgen_pkg graspgen.launch.py

# Terminal 2: Trigger GSAM + watch /grasp_candidates
ros2 launch grounded_sam_pkg grounded_sam_dual.launch.py
```

### Via CLI (offline test):
```bash
# One-liner to test a saved point cloud
python3 graspgen_pkg/graspgen_cli.py \
  --from-file /tmp/my_scene.npy \
  --topk 5 --show 5
```

---

## Notes

- **Loopback only**: Server listens on `127.0.0.1:5556`, not public. SSH tunnel required.
- **No NMS toggle**: Current server uses fixed NMS. See `zmq_client.py` header for TODO.
- **Shared cluster**: Qwen (port 8000) coexists on same A100. Do not modify.
