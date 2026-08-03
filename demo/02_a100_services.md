# 02 — A100 Service Smoke Tests

**Shows:** all three remote inference services are alive and answering.

**Needs:** SSH access to `tta@123.37.28.208`.

Run this before any demo that touches the A100. It takes under a minute and
turns "the pipeline is silent" into a specific diagnosis.

---

## Tunnels

`launch_env.bash` opens all four. To check:

```bash
ss -tln | grep -E ':(8000|555[678])'
```

| Port | Service |
|---|---|
| 8000 | Qwen3.5-27B, served as `qwen35-local` (HTTP, OpenAI-compatible) |
| 5556 | GraspGen (ZMQ + msgpack) |
| 5557 | SwinDRNet (ZMQ + msgpack) |
| 5558 | SAM 2.1 (ZMQ + msgpack) |

All four are **loopback-only** on the server, so the tunnel is mandatory —
there is no direct connection to fall back on.

To open just one by hand:

```bash
ssh -fN -L 8000:127.0.0.1:8000 tta@123.37.28.208
```

---

## Qwen3.5-27B — port 8000

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

The returned `id` should be **`qwen35-local`**, and must match `model_name` in
`ros_pkgs/src/slow_brain/qwen_a100/config/qwen_a100_params.yaml`. A mismatch
gives a 404 at inference time, not at startup.

Full round trip: see [demo 01](01_qwen_grounding.md).

---

## GraspGen — port 5556

Is it running?

```bash
ssh tta@123.37.28.208 'pgrep -f graspgen_server.py && echo RUNNING || echo DOWN'
```

Start it if down:

```bash
ssh tta@123.37.28.208 'nohup bash /data/tta/graspgen/run_server.sh > /data/tta/graspgen/server.log 2>&1 &'
```

Synthetic-cube inference, no ROS:

```bash
cd ros_pkgs/src/graspgen_pkg && python3 graspgen_pkg/graspgen_cli.py --host 127.0.0.1 --port 5556 --shape cube --num-grasps 100 --topk 10
```

Expect ten grasps with confidences and 4×4 poses, ~184 ms round trip.

> Requires `msgpack-numpy` in `gsam_venv`. It is **not** declared in any
> `package.xml` or requirements file, so a fresh clone will not have it:
> ```bash
> gsam_venv/bin/pip install msgpack-numpy
> ```

---

## SwinDRNet — port 5557

```bash
ssh tta@123.37.28.208 'pgrep -f swindrnet_server && echo RUNNING || echo DOWN'
```
```bash
ssh tta@123.37.28.208 'nohup bash /data/tta/swindrnet/run_server.sh > /data/tta/swindrnet/server.log 2>&1 &'
```

There is no standalone CLI for this one — it is exercised through
`graspgen.launch.py` with `swindrnet_enabled:=true`. See
[demo 04](04_full_pick.md).

---

## SAM 2.1 — port 5558

```bash
ssh tta@123.37.28.208 'pgrep -af "[s]erver.zmq_server" && echo RUNNING || echo DOWN'
```

> **Match the module, not a filename.** The process is launched as
> `python -m server.zmq_server`, so its command line contains no `.py` and no
> string `sam_server`. Patterns like `pgrep -f sam_server` or
> `pgrep -f zmq_server.py` report **DOWN on a healthy server** — a false alarm
> that sends you chasing a tunnel that was never broken.
>
> **The `[s]` is load-bearing.** `pgrep -f` scans full command lines, including
> the `bash -c` that ssh spawns to run the check — which contains the pattern.
> Without the bracket the check matches itself and prints `RUNNING`
> unconditionally, so a dead server still looks healthy. The bracket makes the
> regex match `server.zmq_server` while the literal `[s]erver.zmq_server` in
> the wrapper's own command line does not.

It serves **two** actions on 5558. `sam_a100` uses `segment`; `perceive` exists
for a different client architecture that runs Qwen server-side. An
`unknown action` error means the deployed server predates the `segment` handler:

| action | Used by | Request → response |
|---|---|---|
| `segment` | `sam_a100/sam_client.py` — **our path** | `{image, boxes (N,4) px, multimask}` → `{masks (N,H,W), scores}` |
| `perceive` | grounding + segmentation in one call | `{image, prompt}` → `{mask, detections, target_id, ...}` |

Round trip without ROS, writing an overlay you can eyeball:

```bash
cd ros_pkgs/src/slow_brain/sam_a100 && python3 -m sam_a100.sam_cli --image ../../../../demo/resources/ee_raw.png --box 210 150 330 260 --overlay /tmp/sam.png
```

The printed pixel count per mask is the quick health check: a few hundred to a
few thousand is a real object. Tens means the box was wrong; a count near the
full frame area means the server is returning **unthresholded logits**.

---

## Reading failures

| Signature | Meaning |
|---|---|
| Connection refused / error in **< 1 s** | Tunnel is down |
| Timeout after the full budget | Server down, loaded, or the request is too large |
| `No module named 'msgpack_numpy'` | See the note above |
| `Point cloud must be (N, 3)` | Malformed input, not a service fault |

Server logs:

```bash
ssh tta@123.37.28.208 'tail -50 /data/tta/graspgen/server.log'
```

## Note on contention

All three services share **one A100**. A Qwen call and a grasp request issued
together will contend for the GPU — Qwen alone is budgeted around 68 GB, with
GraspGen at ~4 GB and SwinDRNet at ~1.5 GB. If latency spikes during a full
pipeline run, this is the first thing to suspect.
