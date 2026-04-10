#!/usr/bin/env bash
# ROS2 + venv 통합 환경 설정
# 사용법: source launch_env.bash
#
# 전제조건:
#   1. python3 -m venv gsam_venv
#   2. source gsam_venv/bin/activate
#   3. pip install -r ros_pkgs/src/grounded_sam_pkg/requirements.txt

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/jazzy/setup.bash
source "${WS}/ros_pkgs/install/setup.bash" 2>/dev/null || true

# venv site-packages (torch, groundingdino, segment_anything 등 pip install 된 패키지)
VENV_SITE="${WS}/gsam_venv/lib/python3.12/site-packages"

export PYTHONPATH="${VENV_SITE}:${PYTHONPATH}"

echo "[launch_env] ROS2 Jazzy + venv PYTHONPATH set"
echo "  venv : ${VENV_SITE}"
