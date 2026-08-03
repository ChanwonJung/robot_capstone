# Demos

Runnable demonstrations of each subsystem, ordered from "no hardware needed" to
"full pipeline". Each file is self-contained: prerequisites, commands, what you
should see, and how to tell what broke.

| # | Demo | Needs | Status |
|---|---|---|---|
| [01](01_qwen_grounding.md) | Qwen grounding — instruction + image → target, destination, relation | A100 tunnel only | ✅ works |
| [02](02_a100_services.md) | A100 service smoke tests — Qwen, GraspGen, SwinDRNet, SAM 2.1 | A100 tunnels | ✅ works |
| [03](03_hazard_avoidance.md) | Fast Brain — hazard stop/resume and replan | Isaac Sim | ✅ works |
| [04](04_full_pick.md) | End-to-end pick via the current (GSAM) Slow Brain | Isaac Sim + A100 | ✅ works |
| [05](05_slow_brain_v2.md) | New Slow Brain — Qwen → SAM 2.1 | tunnels 8000 + 5558 | ✅ works |

## Shared prerequisites

Every demo assumes these. Run from the repo root.

**1. Build:**
```bash
cd ros_pkgs && colcon build --symlink-install && source install/setup.bash
```

**2. Environment — in every terminal, first:**
```bash
source launch_env.bash
```
Sources ROS 2 + the workspace overlay, injects `gsam_venv` into `PYTHONPATH`,
exports `ROBOT_CAPSTONE_ROOT`, and opens the four A100 SSH tunnels (8000 Qwen, 5556 GraspGen, 5557 SwinDRNet, 5558 SAM 2.1).

Off-network, skip the tunnels so they don't hang:
```bash
source launch_env.bash --no-tunnel
```

**3. Verify the tunnels are up:**
```bash
ss -tln | grep -E ':(8000|555[678])'
```

You should see all four. If a port is missing, the corresponding demo will fail
with a **connection error in under a second** — that signature means the tunnel,
not the model. A slow failure means the service itself.

**4. After any rebuild of `grounded_sam_pkg`** (demo 04 only), patch the entry
script shebangs back to the venv Python:
```bash
VENV_PY="$PWD/gsam_venv/bin/python"; for f in $(find ros_pkgs/install/grounded_sam_pkg/lib -maxdepth 3 -type f -executable); do head -1 "$f" | grep -q "^#!/usr/bin/python3$" && sed -i "1s|^#!/usr/bin/python3$|#!${VENV_PY}|" "$f"; done
```

## Resources

`resources/` holds still frames captured from Isaac Sim, for demos that run
without the simulator:

| File | Size | View |
|---|---|---|
| `ee_raw.png` | 631×401 | end-effector camera |
| `ee_raw2.png` | 1054×584 | end-effector camera, alternate scene |
| `top_raw.png` | 959×628 | top-down camera |

These are screenshot crops, not true camera-resolution frames. Fine for
grounding demos; **not** suitable for anything that pairs colour with depth,
since there is no matching depth image and the intrinsics won't correspond.
