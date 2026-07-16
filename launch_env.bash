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
# 1. Qwen vLLM (aurora-g6) → localhost:8000
# 2. GraspGen ZMQ (A100) → localhost:5556
# 3. SwinDRNet ZMQ (A100) → localhost:5557

_SKIP_TUNNEL="${SKIP_A100_TUNNEL:-0}"
for _arg in "$@"; do
    [ "${_arg}" = "--no-tunnel" ] && _SKIP_TUNNEL=1
done

if [ "${_SKIP_TUNNEL}" = "1" ]; then
    echo "[launch_env] A100 SSH tunnels skipped (--no-tunnel / SKIP_A100_TUNNEL=1)"
else
    # Qwen vLLM tunnel (port 8000)
    if lsof -ti tcp:8000 &>/dev/null; then
        echo "[launch_env] ✓ Qwen vLLM tunnel already open"
    else
        ssh -fN -L 8000:aurora-g6:8000 \
            -J jaewonheo1101@aurora.khu.ac.kr:30080 \
            jaewonheo1101@aurora-g6
        echo "[launch_env] ✓ Qwen vLLM tunnel → localhost:8000"
    fi

    # A100 GraspGen server tunnel (port 5556)
    if lsof -ti tcp:5556 &>/dev/null; then
        echo "[launch_env] ✓ GraspGen server tunnel already open"
    else
        ssh -fN -L 5556:127.0.0.1:5556 tta@123.37.28.208
        echo "[launch_env] ✓ GraspGen server tunnel → localhost:5556"
    fi

    # A100 SwinDRNet server tunnel (port 5557)
    if lsof -ti tcp:5557 &>/dev/null; then
        echo "[launch_env] ✓ SwinDRNet server tunnel already open"
    else
        ssh -fN -L 5557:127.0.0.1:5557 tta@123.37.28.208
        echo "[launch_env] ✓ SwinDRNet server tunnel → localhost:5557"
    fi
fi