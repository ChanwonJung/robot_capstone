#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/isaacsim"
./python.sh "${SCRIPT_DIR}/import_downloaded_assets.py"
