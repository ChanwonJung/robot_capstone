"""Remote-inference endpoints: do the launch files dial the ports the tunnels open?

Every model runs on a remote GPU box reached through loopback SSH tunnels that an
env script opens. Two shapes exist:

  launch_env.bash       → the A100 (DEPRECATED, no longer in use): one tunnel
                          PER MODEL, raw ZMQ/HTTP ports. Still checked because
                          the launch-file port defaults still point at it.
  launch_env_seraph.sh  → the KHU nova-server stack: ONE tunnel to its HTTP
                          gateway (:9000), which proxies every model.

The ROS side never sees the host — only ``127.0.0.1:<port>``. So if a tunnel
script and the launch-file defaults disagree on a port, nothing looks
misconfigured anywhere: the tunnel reports ✓, the node starts, and the first
request dies with a timeout or "connection refused".

This test parses both sides from source, so it runs with no ROS install and no
network. Run from the repo root:

    python3 -m pytest tests/ -q
"""
import re
from pathlib import Path
from urllib.parse import urlparse

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'ros_pkgs' / 'src'

# Per-model tunnel scripts. The launch defaults below must match these.
ENV_SCRIPTS = ['launch_env.bash']
GATEWAY_SCRIPT = 'launch_env_seraph.sh'

# (launch file, launch argument) pairs whose DEFAULT is what a bare
# `ros2 launch ...` from CLAUDE.md actually connects to. Launch arguments are
# passed as node parameters, so they override both the package YAML and the
# node's declare_parameter default — these are the values that matter.
#
# These are the ZMQ FALLBACK ports, used only with gateway_url:= (empty). Qwen
# has no fallback — its default is the gateway — so it is checked in
# test_launch_gateway_defaults.py instead.
LAUNCH_DEFAULTS = {
    'sam': [
        ('slow_brain/qwen_a100/launch/slow_brain.launch.py', 'sam_port'),
        ('slow_brain/sam_a100/launch/sam_a100.launch.py', 'zmq_port'),
    ],
    'graspgen': [
        ('graspgen_pkg/launch/graspgen.launch.py', 'zmq_port'),
        ('graspgen_pkg/launch/full_pipeline_graspgen.launch.py', 'zmq_port'),
    ],
    'swindrnet': [
        ('graspgen_pkg/launch/graspgen.launch.py', 'swindrnet_port'),
    ],
}


def _service_of(label: str) -> str:
    """Map an _open_tunnel label ("Qwen (SGLang)", "SAM 2.1", ...) to a key."""
    low = label.lower()
    for key in ('qwen', 'swindrnet', 'graspgen', 'sam'):
        if key in low:
            return key
    raise ValueError(f'unrecognised tunnel label {label!r}')


def tunnel_ports(script: str) -> dict:
    """Ports opened by the ACTIVE (uncommented) _open_tunnel calls."""
    ports = {}
    for line in (ROOT / script).read_text().splitlines():
        m = re.match(r'\s*_open_tunnel\s+(\d+)\s+"([^"]+)"', line)
        if m:
            ports[_service_of(m.group(2))] = int(m.group(1))
    return ports


def launch_default(rel_path: str, arg: str) -> int:
    """Port encoded in a DeclareLaunchArgument default (a port or a URL)."""
    text = (SRC / rel_path).read_text()
    m = re.search(
        r'DeclareLaunchArgument\(\s*[\'"]' + re.escape(arg) +
        r'[\'"]\s*,\s*default_value\s*=\s*[\'"]([^\'"]*)[\'"]', text)
    assert m, f'{rel_path}: no DeclareLaunchArgument({arg!r}, default_value=...)'
    value = m.group(1)
    return urlparse(value).port if '://' in value else int(value)


@pytest.mark.parametrize('script', ENV_SCRIPTS)
def test_script_tunnels_every_service(script):
    assert set(tunnel_ports(script)) == set(LAUNCH_DEFAULTS) | {'qwen'}, (
        f'{script} does not open one tunnel per remote service')


@pytest.mark.parametrize('script', ENV_SCRIPTS)
@pytest.mark.parametrize('service', sorted(LAUNCH_DEFAULTS))
def test_launch_defaults_match_tunnel(script, service):
    tunnelled = tunnel_ports(script).get(service)
    wrong = {
        f'{path}::{arg}': port
        for path, arg in LAUNCH_DEFAULTS[service]
        if (port := launch_default(path, arg)) != tunnelled
    }
    assert not wrong, (
        f'{script} tunnels {service} on localhost:{tunnelled}, but these launch '
        f'defaults dial elsewhere: {wrong}. After `source {script}` the node '
        'starts cleanly and every request times out.')


# ── gateway script ───────────────────────────────────────────────────────────

def _gateway_text():
    return (ROOT / GATEWAY_SCRIPT).read_text()


def _default(text, var):
    m = re.search(r'\$\{' + var + r':-([^}]*)\}', text)
    assert m, f'{GATEWAY_SCRIPT}: no ${{{var}:-default}}'
    return m.group(1)


def test_gateway_tunnel_and_exported_url_agree():
    text = _gateway_text()
    port = _default(text, 'NOVA_GATEWAY_PORT')
    # One forward, local port → the gateway's fixed :9000 on the compute node.
    live = '\n'.join(l for l in text.splitlines() if not l.lstrip().startswith('#'))
    forwards = re.findall(r'-L\s+"?([^"\s]+)"?', live)
    assert forwards == ['${NOVA_GATEWAY_PORT}:${NOVA_NODE}:9000']
    assert 'export NOVA_GATEWAY_URL="http://127.0.0.1:${NOVA_GATEWAY_PORT}"' in text
    assert port == '9000'


def test_gateway_script_goes_through_the_login_node():
    text = _gateway_text()
    # Compute-node ports are loopback-only and there is no SSH onto the nodes,
    # so the forward must terminate on aurora-master (SSH port 30080).
    assert re.search(r'_NOVA_LOGIN="\w+@163\.180\.160\.105"', text)
    assert '_NOVA_SSH_PORT=30080' in text
    assert _default(text, 'NOVA_NODE') == 'aurora-g6'


def test_gateway_key_is_never_hardcoded():
    assignments = re.findall(r'^\s*(?:export\s+)?NOVA_API_KEY=(.*)$',
                             _gateway_text(), re.M)
    assert assignments, 'expected the .env fallback to assign NOVA_API_KEY'
    # Only the .env plumbing may assign it, never a literal secret.
    for value in assignments:
        assert value.startswith('"$('), f'literal key: NOVA_API_KEY={value}'
