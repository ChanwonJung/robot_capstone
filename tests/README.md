# Tests

This file indexes every test in the repo: when each one was created, what it
protects, and how to run it. The tests in this directory check how packages
fit together. Each package also has its own `test/` directory for tests of
that package alone.

**Dates**: for committed files, the date is the commit that first added the
file. The 2026-09-29 files belong to the NOVA-gateway migration and are **not
committed yet**; their date is the file creation time.

## Quick start

Run everything from the repo root, in the environment described in
`CLAUDE.md` (ROS 2 Jazzy, Python 3.12, `gsam_venv`):

```bash
source launch_env_seraph.sh --no-tunnel   # ROS 2 + gsam_venv on PYTHONPATH, no SSH

# Cross-package tests. No build, no network needed.
python3 -m pytest tests/ -q

# Package unit tests. Run them from the package directory so the package can be imported.
(cd ros_pkgs/src/graspgen_pkg        && python3 -m pytest test/ -q)
(cd ros_pkgs/src/slow_brain/sam_a100  && python3 -m pytest test/ -q -m 'not linter')
(cd ros_pkgs/src/slow_brain/qwen_a100 && python3 -m pytest test/ -q -m 'not linter')
(cd ros_pkgs/src/mask_projection_pkg && python3 -m pytest test/ -q)

# bt_pkg C++ gtest (needs a colcon build)
cd ros_pkgs && colcon build --packages-select bt_pkg \
  && colcon test --packages-select bt_pkg && colcon test-result --verbose
```

Use `python3 -m pytest`, not plain `pytest`. The `-m` form puts the current
directory on `sys.path`, and that is how the package tests find their package.

None of these tests reach the real cluster or the A100. Remote servers are
replaced by fake HTTP or ZMQ servers on `127.0.0.1`, and a missing dependency
causes a **skip**, not a failure. A green run with many skips proves very
little, so add `-rs` to see what was skipped and why.

| Dependency | Needed by | Source |
|---|---|---|
| sourced ROS 2 (`launch`, `launch_ros`, `rclpy`, `sensor_msgs`) | `test_launch_gateway_defaults`, `test_pipeline_contracts`, `test_graspgen_node_gateway` | `/opt/ros/jazzy` |
| `cv2`, `requests` | gateway client tests | `graspgen_pkg/requirements.txt` |
| `zmq`, `msgpack`, `msgpack_numpy` | ZMQ fallback client tests; also imported by `graspgen_node` | `graspgen_pkg/requirements.txt` |
| `openai`, `pydantic` | `test_qwen_gateway` | `slow_brain/requirements.txt` |

---

## `tests/` — cross-package contracts

### `test_remote_endpoints.py` — 2026-09-29 (uncommitted) — 5 tests
**Purpose:** checks that the ports opened by the tunnel scripts match the
ports the launch files connect to. If they disagree, nothing fails at startup:
the tunnel reports ✓ and the node starts, but every request times out.
- The deprecated A100 script `launch_env.bash` opens one tunnel per service.
  Every ZMQ fallback port default in the launch files (`sam_port`, `zmq_port`,
  `swindrnet_port`) matches its tunnel.
- `launch_env_seraph.sh` opens exactly one forward
  (`${NOVA_GATEWAY_PORT}:${NOVA_NODE}:9000`) through `aurora-master:30080`,
  and exports a matching `NOVA_GATEWAY_URL`.
- `NOVA_API_KEY` is never hardcoded in the script. It is only loaded from `.env`.

**Usage:** `python3 -m pytest tests/test_remote_endpoints.py -q`. Parses source
files only, so it needs no ROS and no network.

### `test_launch_gateway_defaults.py` — 2026-09-29 (uncommitted) — 3 tests (×2 env cases)
**Purpose:** checks that every node that calls a remote model gets the
gateway by default. It builds the real `LaunchDescription`s, including nodes
nested in `TimerAction` or `GroupAction`, and checks the parameters each node
would receive. Each test runs twice: with `NOVA_GATEWAY_URL` set, and with it
unset (falls back to `http://127.0.0.1:9000`).
- `slow_brain.launch.py`: Qwen uses `<gw>/qwen/v1` with model `qwen3.5-27b`,
  and **both** SAM views use the gateway.
- `qwen_a100.launch.py` and `sam_a100.launch.py` use the gateway when launched on their own.
- `graspgen.launch.py` uses the gateway with `depth_restore_route=compare`.

**Usage:** `python3 -m pytest tests/test_launch_gateway_defaults.py -q`. Needs a
sourced ROS 2 install. No colcon build is needed, because share directories
point at the source tree.

### `test_pipeline_contracts.py` — 2026-09-29 (uncommitted) — 8 tests
**Purpose:** checks the hand-offs between packages, where a mismatch goes wrong
without raising any error:
- **qwen → SAM label map → graspgen**: the pixel value graspgen reads as the
  TARGET is the mask SAM actually drew, for any detection order and when the
  target overlaps its destination. The label map uses the depth resolution,
  not the RGB resolution.
- **depth + mask → cloud**: the world-frame target cloud sent to GraspGen is
  correct (NaN pixels dropped, point count capped, `None` for an empty mask).
- **GraspGen → `/grasp_candidates` → bt_pkg**: the `panda_link8` pose sits
  0.103 m behind the grasp centre, the quaternion is ordered `[x,y,z,w]`, the
  top-down filter uses the same approach axis, and the published JSON has every
  key that `parse_grasp_candidates()` in `bt_executor_node.cpp` reads. Those
  keys are taken from the C++ source itself.

**Usage:** `python3 -m pytest tests/test_pipeline_contracts.py -q`. The label-map
tests need only `cv2`. The graspgen and BT tests also need a sourced ROS 2 plus `zmq`/`msgpack`.

---

## `graspgen_pkg/test/`

### `conftest.py` — 2026-09-29 (uncommitted)
A fake NOVA gateway (`FakeGateway`), a real HTTP server on 127.0.0.1. It
serves `/graspgen/infer` and `/<route>/restore_depth` with `x-api-key` auth
(key `test-key`). Set `status` to force an error reply such as 502. Used by the
two gateway test files below.

### `test_gateway_clients.py` — 2026-09-29 (uncommitted) — 9 tests
**Purpose:** tests `GraspGenHttpClient` and `DepthRestoreHttpClient` against the fake gateway.
- Request format: JSON body for GraspGen, and a multipart body of rgb +
  16-bit millimetre depth + mask for depth restoration.
- Unit conversion: the node uses metres with NaN for missing depth, and the
  request uses uint16 millimetres with 0 for missing. A bug here does not
  raise an error; the cloud just ends up 1000× wrong.
- Routes: `compare` (the default) sends the full frame with raw depth and the
  mask, `swindrnet` sends no mask, and an unknown route is rejected.
- Error messages name their cause: 401 names `NOVA_API_KEY`, 502 points at
  `/health`, and a refused connection says the tunnel is down. Bad input is
  rejected before anything is sent.

**Usage:** `cd ros_pkgs/src/graspgen_pkg && python3 -m pytest test/test_gateway_clients.py -q`

### `test_graspgen_node_gateway.py` — 2026-09-29 (uncommitted) — 2 tests
**Purpose:** end-to-end test of the real `GraspGenNode` against the fake
gateway. It feeds one scan of a see-through glass and checks:
1. the full frame, raw depth and TARGET mask go to `/compare/restore_depth`;
2. the cloud sent to `/graspgen/infer` is built from the **restored** depth;
3. `/grasp_candidates` holds `panda_link8` poses that bt_pkg can parse.

It also checks that an opaque target skips depth restoration entirely.

**Usage:** `cd ros_pkgs/src/graspgen_pkg && python3 -m pytest test/test_graspgen_node_gateway.py -q`.
Needs a sourced ROS 2 plus `zmq`/`msgpack` (the node imports the ZMQ clients).

### `test_remote_clients.py` — 2026-09-29 (uncommitted) — 9 tests
**Purpose:** tests the ZMQ **fallback** clients (`GraspGenClient`,
`SwinDRNetClient`) against a fake ZMQ REP server on 127.0.0.1. It checks the
msgpack message format byte for byte, flat grasp buffers, server errors,
missing keys, and above all **recovery after a timeout**. Without the reset
logic, a REQ socket that times out stays stuck and every later request fails.

**Usage:** `cd ros_pkgs/src/graspgen_pkg && python3 -m pytest test/test_remote_clients.py -q`

---

## `slow_brain/sam_a100/test/`

### `test_sam_http_client.py` — 2026-09-29 (uncommitted) — 4 tests
**Purpose:** tests `SamHttpClient` against a fake `/sam2/segment` server. It
checks one HTTP request per box (the gateway takes only one box per request),
that multimask keeps the best-scoring candidate, that the `x-api-key` header
is sent, and that a bad key, a down backend or a down tunnel each produce an
error naming that cause. Bad input is rejected before anything is sent.

**Usage:** `cd ros_pkgs/src/slow_brain/sam_a100 && python3 -m pytest test/test_sam_http_client.py -q`

### `test_sam_client.py` — 2026-09-29 (uncommitted) — 7 tests
**Purpose:** tests the ZMQ **fallback** `SamClient` against a fake REP server.
It checks the message format, restoring a squeezed single mask to a batch,
thresholding raw logit masks, a mask/box count mismatch, server errors,
validating input before sending, and recovery after a timeout.

**Usage:** `cd ros_pkgs/src/slow_brain/sam_a100 && python3 -m pytest test/test_sam_client.py -q`

## `slow_brain/qwen_a100/test/`

### `test_qwen_gateway.py` — 2026-09-29 (uncommitted) — 4 tests
**Purpose:** tests `qwen_call.ground()` against a fake OpenAI-compatible
`/qwen/v1/chat/completions` server, using the real `openai` client. It checks:
- the endpoint is built from `$NOVA_GATEWAY_URL`, and the model is `qwen3.5-27b`;
- the `x-api-key` header is sent. The OpenAI SDK only sends
  `Authorization: Bearer`, so without this header every call gets a 401;
- the JSON-schema response format, the image data URL, and the conversion
  from `normalized_1000` boxes to pixels;
- a missing key raises a clear 401 instead of hanging;
- when reasoning tokens use up `max_tokens` and `content` comes back null, the call **raises**.

**Usage:** `cd ros_pkgs/src/slow_brain/qwen_a100 && python3 -m pytest test/test_qwen_gateway.py -q`

---

## `mask_projection_pkg/test/`

### `test_projection_engine.py` — 2026-08-11 (ChanwonJung) — 12 tests
**Purpose:** tests tabletop obstacle extraction. Objects on a surface are
separated by **height**, not by category, because a "table" mask covers
everything on the table. The tests check that it finds objects by height,
ignores the surface itself, drops the robot arm, applies the surface z offset,
covers the object footprint, ignores points outside the workspace, and treats
UNKNOWN the same as DESTINATION without counting TARGET points as clutter. They
also check that boxes project onto the support plane correctly
(`obstacles_from_boxes`). Pure numpy, no ROS.

**Usage:** `cd ros_pkgs/src/mask_projection_pkg && python3 -m pytest test/ -q`

## `bt_pkg/test/`

### `test_destination_calculator.cpp` — 2026-08-06 (ChanwonJung) — 34 gtest cases
**Purpose:** tests the placement geometry in `compute_place_pose()` using real
Isaac Sim numbers (basket rim at z 0.18, placed at (0.480, −0.420); a book
about 20 mm thick). The two failures it guards against are "arm drives into
the basket wall" and "arm dives at its own base".

| Suite | Cases |
|---|---|
| `ComputePlacePose` | 25 |
| `NudgeToFreeSpace` | 6 |
| `CarryLift` | 1 |
| `RetractAlongApproach` | 1 |
| `MeasuredTableSurface` (parameterised) | 1 |

**Usage** (the only test that needs a build):
```bash
cd ros_pkgs
colcon build --packages-select bt_pkg
colcon test --packages-select bt_pkg --ctest-args -R test_destination_calculator
colcon test-result --verbose
```

---

## Lint tests (ament boilerplate)

| File | Packages | Added | Notes |
|---|---|---|---|
| `test_flake8.py` | grounded_sam_pkg, qwen_a100, sam_a100 | 2026-04-10 (tydfuyhf) | qwen_a100 and sam_a100 pass their own `test/ament_flake8.ini` through `--config`. Without that flag, ament_flake8 silently ignores the file. |
| `test_pep257.py` | same | 2026-04-10 (tydfuyhf) | Docstring style |
| `test_copyright.py` | same | 2026-04-10 (tydfuyhf) | **Skipped** in grounded_sam_pkg and qwen_a100 (no copyright headers yet). Active in sam_a100. |

**Usage:** run as part of `colcon test --packages-select <pkg>`. With pytest,
run `python3 -m pytest test/ -m linter` from the package directory. Pass
`-m 'not linter'` to leave them out.

## Not tests (manual helpers)

- `grounded_sam_pkg/grounded_sam_pkg/test_image_pub.py` and
  `grounded_sam_pkg/launch/test_inference.launch.py`, 2026-04-10 (tydfuyhf).
  They publish a still image and run the **deprecated** local Grounded-SAM on
  it for a manual check:
  `ros2 launch grounded_sam_pkg test_inference.launch.py prompt:="glass, chair"`.
  Set the image path in `IMAGE_PATH` at the top of `test_image_pub.py`. It
  currently points at another user's home directory.
