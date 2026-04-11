#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACSIM_DIR="${PROJECT_ROOT}/isaacsim"

if [[ ! -x "${ISAACSIM_DIR}/python.sh" ]]; then
  echo "Isaac Sim python launcher not found: ${ISAACSIM_DIR}/python.sh" >&2
  exit 1
fi

cd "${ISAACSIM_DIR}"
./python.sh "${SCRIPT_DIR}/import_downloaded_assets.py"
