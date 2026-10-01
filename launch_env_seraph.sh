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


# ── NOVA gateway tunnel ───────────────────────────────────────────────────────
# Target: the nova-server stack (Slurm job "NOVA_depth3", 4 GPUs) on the KHU
# cluster, reached through its HTTP gateway. Start it on aurora-master with
#   NOVA_GATEWAY=1 NOVA_GATEWAY_HOST=0.0.0.0 NOVA_GATEWAY_KEY='<key>' \
#       sbatch services/all_in_one_depth.sh
# See ACCESS.md in /data/jaewonheo1101/nova-server (HJ1-1101/nova-server,
# branch all-in-one-depth).
#
# ONE tunnel, not one per model. Every model server binds 127.0.0.1 on the
# compute node and there is no SSH onto compute nodes, so the per-model ports
# (qwen 10000, sam2 20000, swindrnet 30000, remake 31000, tdr CA-Dual 32000,
# graspgen 40000) are unreachable from here. The gateway on :9000 proxies all
# of them; the forward terminates on aurora-master, which reaches the node:
#
#   127.0.0.1:9000 ──ssh -L──▶ aurora-master:30080 ──▶ ${NOVA_NODE}:9000
#
#   GET  /health                     200 even with a backend down — read the body
#   POST /qwen/v1/chat/completions   OpenAI-compatible, model "qwen3.5-27b"
#   POST /sam2/segment               multipart: image + box / points
#   POST /{swindrnet,remake,compare}/restore_depth   multipart, 16-bit mm PNGs
#   POST /graspgen/infer             JSON {"point_cloud": [[x,y,z],...], ...}
#
# Every request needs the header  x-api-key: $NOVA_API_KEY
#
# Overrides (set before sourcing):
#   NOVA_NODE          compute node the job landed on (default aurora-g6).
#                      It changes between jobs — check `squeue -u jaewonheo1101`.
#   NOVA_GATEWAY_PORT  local port for the forward (default 9000)
#   NOVA_API_KEY       gateway key; otherwise read from ${WS}/.env (gitignored).
#                      Never commit it.
#
# Campus network or VPN only. Off-network, skip the tunnel or it hangs:
#   source launch_env_seraph.sh --no-tunnel   (or SKIP_A100_TUNNEL=1)

_SKIP_TUNNEL="${SKIP_A100_TUNNEL:-0}"
for _arg in "$@"; do
    [ "${_arg}" = "--no-tunnel" ] && _SKIP_TUNNEL=1
done

_NOVA_LOGIN="jaewonheo1101@163.180.160.105"   # aurora-master
_NOVA_SSH_PORT=30080
NOVA_NODE="${NOVA_NODE:-aurora-g6}"
NOVA_GATEWAY_PORT="${NOVA_GATEWAY_PORT:-9000}"
export NOVA_GATEWAY_URL="http://127.0.0.1:${NOVA_GATEWAY_PORT}"

# The key comes from the environment first, then from .env. Only the one line is
# read — sourcing .env wholesale would run whatever else is in it.
if [ -z "${NOVA_API_KEY:-}" ] && [ -f "${WS}/.env" ]; then
    NOVA_API_KEY="$(sed -n 's/^[[:space:]]*\(export[[:space:]]\+\)\?NOVA_API_KEY=//p' "${WS}/.env" \
        | tail -n 1 | sed -e "s/^[\"']//" -e "s/[\"']\$//")"
fi
if [ -n "${NOVA_API_KEY:-}" ]; then
    export NOVA_API_KEY
else
    echo "[launch_env] ⚠ NOVA_API_KEY is not set — every gateway request will"
    echo "[launch_env]   return 401. export NOVA_API_KEY=... or add it to ${WS}/.env"
fi

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
# trusting the exit code.
_open_tunnel() {
    local port="$1" label="$2"; shift 2
    local rc i
    if _port_listening "${port}"; then
        echo "[launch_env] ✓ ${label} tunnel already open (localhost:${port})"
        return 0
    fi

    # ConnectTimeout bounds the off-network case, which otherwise hangs the
    # whole shell on a dead route. Use --no-tunnel to skip these entirely.
    # ServerAliveInterval drops a dead forward instead of leaving it half-open.
    ssh -fN -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes "$@"
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

# Per-backend status from /health. The route answers 200 even when a backend is
# down, so the status code proves nothing — only the body does. A tunnel that
# comes up cleanly while the Slurm job is dead (or on another node) is the
# common failure, and it looks exactly like success without this.
_gateway_health() {
    local body
    [ -n "${NOVA_API_KEY:-}" ] || return 0
    body="$(curl -s --max-time 8 -H "x-api-key: ${NOVA_API_KEY}" \
        "${NOVA_GATEWAY_URL}/health")" || {
        echo "[launch_env] ✗ gateway not answering at ${NOVA_GATEWAY_URL} — is job"
        echo "[launch_env]   NOVA_depth3 running on ${NOVA_NODE}? (squeue -u jaewonheo1101)"
        return 1
    }
    printf '%s' "${body}" | python3 -c '
import json, sys
raw = sys.stdin.read()
try:
    health = json.loads(raw)
except ValueError:
    print(f"[launch_env] ✗ gateway replied with non-JSON: {raw[:120]!r}")
    sys.exit(1)
if "detail" in health:            # FastAPI error body, e.g. a 401
    detail = health["detail"]
    print(f"[launch_env] ✗ gateway: {detail}")
    sys.exit(1)
for name, st in health.items():
    status = st.get("status", "?") if isinstance(st, dict) else st
    mark = "✓" if status == "ok" else "✗"
    print(f"[launch_env]   {mark} {name:10s} {status}")
'
}

if [ "${_SKIP_TUNNEL}" = "1" ]; then
    echo "[launch_env] NOVA gateway tunnel skipped (--no-tunnel / SKIP_A100_TUNNEL=1)"
else
    if _open_tunnel "${NOVA_GATEWAY_PORT}" "NOVA gateway (${NOVA_NODE})" \
            -p "${_NOVA_SSH_PORT}" \
            -L "${NOVA_GATEWAY_PORT}:${NOVA_NODE}:9000" "${_NOVA_LOGIN}"; then
        _gateway_health
    else
        echo "[launch_env] ⚠ gateway unavailable — every remote model call will fail."
    fi

    # ── DEPRECATED: all four services on the A100 (tta@123.37.28.208) ────────
    # The A100 is no longer in use. Kept for reference only: direct single-hop,
    # one ZMQ/HTTP port per model — what launch_env.bash (also deprecated)
    # opens, and what the launch-file port defaults still point at.
    #
    # _A100_HOST="tta@123.37.28.208"
    # _open_tunnel 8000 "Qwen vLLM"  -L 8000:127.0.0.1:8000 "${_A100_HOST}"
    # _open_tunnel 5556 "GraspGen"   -L 5556:127.0.0.1:5556 "${_A100_HOST}"
    # _open_tunnel 5557 "SwinDRNet"  -L 5557:127.0.0.1:5557 "${_A100_HOST}"
    # _open_tunnel 5558 "SAM 2.1"    -L 5558:127.0.0.1:5558 "${_A100_HOST}"
    # ─────────────────────────────────────────────────────────────────────────
fi
# This file is SOURCED, so the helpers would otherwise linger in the caller's
# shell. NOVA_GATEWAY_URL / NOVA_API_KEY / NOVA_NODE stay exported for clients.
unset -f _open_tunnel _port_listening _gateway_health
unset _SKIP_TUNNEL _arg _NOVA_LOGIN _NOVA_SSH_PORT
