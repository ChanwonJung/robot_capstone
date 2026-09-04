#!/usr/bin/env bash
# ROS2 + venv 통합 환경 설정
# 사용법: source launch_env.bash
#
# 전제조건:
#   1. python3 -m venv ${VENV_NAME}
#   2. source ${VENV_NAME}/bin/activate
#   3. pip install -r ros_pkgs/src/grounded_sam_pkg/requirements.txt --no-build-isolation
    
# resolve script dir in both bash (BASH_SOURCE) and zsh (ZSH_SCRIPT / $0)
if [ -n "${BASH_SOURCE[0]}" ]; then
    WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
elif [ -n "${ZSH_SCRIPT}" ]; then
    WS="$(cd "$(dirname "${ZSH_SCRIPT}")" && pwd)"
else
    WS="$(cd "$(dirname "$0")" && pwd)"
fi

VENV_NAME="gsam_venv"
PYTHON_VERSION="3.12"

source /opt/ros/jazzy/setup.bash
source "${WS}/ros_pkgs/install/setup.bash" 2>/dev/null || true

# venv site-packages (torch, groundingdino, segment_anything 등 pip install 된 패키지)
VENV_SITE="${WS}/${VENV_NAME}/lib/python${PYTHON_VERSION}/site-packages"

if [ ! -d "${VENV_SITE}" ]; then
      echo "[launch_env] ERROR: venv site-packages not found at ${VENV_SITE}"
      return 1
fi

export PYTHONPATH="${VENV_SITE}:${PYTHONPATH}"
export ROBOT_CAPSTONE_ROOT="${WS}"

echo "[launch_env] ROS2 Jazzy + venv PYTHONPATH set"
echo "  venv : ${VENV_SITE}"

# ── SSH tunnels ───────────────────────────────────────────────────────────────
# Active target: aurora-g6 (KHU cluster), via the campus jump host.
# Four direct per-service tunnels — no gateway; each model owns a port.
# 1. Qwen3.5-27B (full bf16, SGLang, TP=4)  → localhost:10000
# 2. SAM 2.1 (point/box segmentation)       → localhost:20000
# 3. SwinDRNet (depth restoration)          → localhost:30000
# 4. GraspGen (ZMQ REP, 6-DOF grasping)     → localhost:40000

_SKIP_TUNNEL="${SKIP_A100_TUNNEL:-0}"
for _arg in "$@"; do
    [ "${_arg}" = "--no-tunnel" ] && _SKIP_TUNNEL=1
done

_KHU_JUMP="allen516@aurora.khu.ac.kr:30080"
_AURORA_HOST="allen516@aurora-g6"

# Is a LISTENER bound to this port?
#
# The LISTEN scope is load-bearing. A bare `lsof -i tcp:PORT` also matches
# OUTBOUND client sockets, so a crashed node leaving a connection in CLOSE_WAIT
# makes the port look occupied, the tunnel gets skipped, and every client then
# fails with "connection refused" while the script reports success.
_port_listening() {
    lsof -ti "tcp:$1" -sTCP:LISTEN &>/dev/null
}

# Open one loopback tunnel and REPORT HONESTLY.
#
# `ssh -f` backgrounds itself, so a zero exit does not prove the forward came
# up — a refused bind still detaches cleanly. Poll for the listener instead of
# trusting the exit code. Takes the destination-side ssh args (the `-L ...`
# spec plus whatever host/jump gets you there) as trailing arguments, so the
# same honest-reporting logic works for a direct host or a jump-host chain.
_open_tunnel() {
    local port="$1" label="$2"; shift 2
    local rc i
    if _port_listening "${port}"; then
        echo "[launch_env] ✓ ${label} tunnel already open (localhost:${port})"
        return 0
    fi

    # ConnectTimeout bounds the off-network case, which otherwise hangs the
    # whole shell on a dead route. Use --no-tunnel to skip these entirely.
    ssh -fN -o ConnectTimeout=10 "$@"
    rc=$?
    if [ ${rc} -ne 0 ]; then
        echo "[launch_env] ✗ ${label} tunnel FAILED — ssh exited ${rc} (port ${port})"
        return 1
    fi

    for i in 1 2 3 4 5 6; do
        if _port_listening "${port}"; then
            echo "[launch_env] ✓ ${label} tunnel → localhost:${port}"
            return 0
        fi
        sleep 0.5
    done
    echo "[launch_env] ✗ ${label} tunnel: ssh returned 0 but nothing is listening"
    echo "[launch_env]   on ${port}. Usually the port is taken, or the remote"
    echo "[launch_env]   service is not up. Check: ss -tlnp | grep ${port}"
    return 1
}

if [ "${_SKIP_TUNNEL}" = "1" ]; then
    echo "[launch_env] aurora-g6 SSH tunnels skipped (--no-tunnel / SKIP_A100_TUNNEL=1)"
else
    _TUNNEL_FAILS=0
    _open_tunnel 10000 "Qwen (SGLang)"  -L 10000:127.0.0.1:10000 -J "${_KHU_JUMP}" "${_AURORA_HOST}" || _TUNNEL_FAILS=$((_TUNNEL_FAILS + 1))
    _open_tunnel 20000 "SAM 2.1"        -L 20000:127.0.0.1:20000 -J "${_KHU_JUMP}" "${_AURORA_HOST}" || _TUNNEL_FAILS=$((_TUNNEL_FAILS + 1))
    _open_tunnel 30000 "SwinDRNet"      -L 30000:127.0.0.1:30000 -J "${_KHU_JUMP}" "${_AURORA_HOST}" || _TUNNEL_FAILS=$((_TUNNEL_FAILS + 1))
    _open_tunnel 40000 "GraspGen"       -L 40000:127.0.0.1:40000 -J "${_KHU_JUMP}" "${_AURORA_HOST}" || _TUNNEL_FAILS=$((_TUNNEL_FAILS + 1))

    if [ "${_TUNNEL_FAILS}" -ne 0 ]; then
        echo "[launch_env] ⚠ ${_TUNNEL_FAILS} tunnel(s) unavailable — the"
        echo "[launch_env]   corresponding services will fail with a connection"
        echo "[launch_env]   error in under a second."
    fi
    unset _TUNNEL_FAILS

    # ── OLD: all four services on the A100 (tta@123.37.28.208) ───────────────
    # Kept in case the model(s) move back there. Direct single-hop, no jump
    # host needed. Swap the four lines above for this block:
    #
    # _A100_HOST="tta@123.37.28.208"
    # _open_tunnel 8000 "Qwen vLLM"  -L 8000:127.0.0.1:8000 "${_A100_HOST}"
    # _open_tunnel 5556 "GraspGen"   -L 5556:127.0.0.1:5556 "${_A100_HOST}"
    # _open_tunnel 5557 "SwinDRNet"  -L 5557:127.0.0.1:5557 "${_A100_HOST}"
    # _open_tunnel 5558 "SAM 2.1"    -L 5558:127.0.0.1:5558 "${_A100_HOST}"
    # ─────────────────────────────────────────────────────────────────────────
fi
# This file is SOURCED, so the helpers would otherwise linger in the caller's
# shell. The functions go; _KHU_JUMP/_AURORA_HOST are kept — handy for manual ssh.
unset -f _open_tunnel _port_listening
