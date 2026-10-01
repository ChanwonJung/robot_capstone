"""Every remote-model node defaults to the NOVA gateway — checked by running the launch files.

launch_env_seraph.sh exports NOVA_GATEWAY_URL; the launch files are supposed to
hand it to every node that calls a remote model, and to keep the A100 ZMQ path
as an opt-in fallback only. Reading the source cannot show that: the defaults
are launch substitutions (EnvironmentVariable + string lists), resolved at
launch time, and several nodes sit inside TimerAction / GroupAction wrappers.

So this evaluates the real LaunchDescriptions with launch_ros — declared
arguments, substitutions, parameter dicts — and asserts on the parameters each
node would actually receive. Package share directories are pointed at the
source tree, so no colcon build is needed; a sourced ROS 2 is.

    source /opt/ros/<distro>/setup.bash && python3 -m pytest tests/ -q
"""
import importlib.util
from pathlib import Path

import pytest

launch = pytest.importorskip('launch', reason='source a ROS 2 install first')
launch_ros = pytest.importorskip('launch_ros')
import ament_index_python.packages as ament_packages  # noqa: E402
from launch.actions import DeclareLaunchArgument  # noqa: E402
from launch_ros.actions import Node  # noqa: E402
from launch_ros.utilities import evaluate_parameters, normalize_parameters  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'ros_pkgs' / 'src'
SHARE = {
    'graspgen_pkg': SRC / 'graspgen_pkg',
    'qwen_a100': SRC / 'slow_brain' / 'qwen_a100',
    'sam_a100': SRC / 'slow_brain' / 'sam_a100',
    'mask_projection_pkg': SRC / 'mask_projection_pkg',
}
GATEWAY_KEYS = ('gateway_url', 'vllm_endpoint_url', 'model_name', 'depth_restore_route')


def _walk(entities):
    for e in entities:
        yield e
        for attr in ('actions', '_GroupAction__actions', '_TimerAction__actions'):
            sub = getattr(e, attr, None)
            if isinstance(sub, (list, tuple)):
                yield from _walk(sub)


def node_params(rel_path):
    """{node_name: {key: resolved value}} for GATEWAY_KEYS, as launch would pass them."""
    spec = importlib.util.spec_from_file_location('launch_under_test', SRC / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ld = mod.generate_launch_description()
    ctx = launch.LaunchContext()
    for e in ld.entities:
        if isinstance(e, DeclareLaunchArgument):
            e.visit(ctx)
    out = {}
    for e in _walk(ld.entities):
        if not isinstance(e, Node):
            continue
        params = evaluate_parameters(ctx, normalize_parameters(e._Node__parameters))
        for p in params:
            if isinstance(p, dict):
                for k in GATEWAY_KEYS:
                    if k in p:
                        out.setdefault(e._Node__node_name, {})[k] = p[k]
    return out


@pytest.fixture(params=['http://127.0.0.1:9123', None], ids=['env-set', 'env-unset'])
def gateway(request, monkeypatch):
    monkeypatch.setattr(ament_packages, 'get_package_share_directory',
                        lambda name: str(SHARE[name]))
    monkeypatch.setenv('ROBOT_CAPSTONE_ROOT', str(ROOT))
    if request.param:
        monkeypatch.setenv('NOVA_GATEWAY_URL', request.param)
        return request.param
    monkeypatch.delenv('NOVA_GATEWAY_URL', raising=False)
    return 'http://127.0.0.1:9000'          # the tunnel launch_env_seraph.sh opens


def test_slow_brain_qwen_and_both_sam_views_use_the_gateway(gateway):
    params = node_params('slow_brain/qwen_a100/launch/slow_brain.launch.py')
    assert params['qwen_bridge_node'] == {
        'vllm_endpoint_url': f'{gateway}/qwen/v1', 'model_name': 'qwen3.5-27b'}
    # dual view: the overhead SAM instance must not be left on the A100 port
    assert params['sam_mask_node']['gateway_url'] == gateway
    assert params['sam_mask_top_node']['gateway_url'] == gateway


def test_standalone_qwen_and_sam_launches_use_the_gateway(gateway):
    assert node_params('slow_brain/qwen_a100/launch/qwen_a100.launch.py')[
        'qwen_bridge_node']['vllm_endpoint_url'] == f'{gateway}/qwen/v1'
    assert node_params('slow_brain/sam_a100/launch/sam_a100.launch.py')[
        'sam_mask_node']['gateway_url'] == gateway


def test_graspgen_uses_the_gateway_with_compare_depth(gateway):
    params = node_params('graspgen_pkg/launch/graspgen.launch.py')['graspgen_node']
    assert params == {'gateway_url': gateway, 'depth_restore_route': 'compare'}
