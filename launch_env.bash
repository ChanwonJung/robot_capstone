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

# ── SSH tunnel: aurora-g6 vLLM → localhost:8000 ───────────────────────────────
_TUNNEL_USER="jaewonheo1101"
_JUMP_HOST="aurora.khu.ac.kr"
_TARGET_HOST="aurora-g6"
_LOCAL_PORT=8000

if lsof -ti tcp:${_LOCAL_PORT} &>/dev/null; then
    echo "[launch_env] Seraph ssh tunnel to localhost already established"
else
    ssh -fN -L ${_LOCAL_PORT}:${_TARGET_HOST}:${_LOCAL_PORT} \
        -J ${_TUNNEL_USER}@${_JUMP_HOST}:30080 \
        ${_TUNNEL_USER}@${_TARGET_HOST}
    echo "[launch_env] Seraph ssh tunnel to localhost established"
fi