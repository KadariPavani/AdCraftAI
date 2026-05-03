#!/usr/bin/env bash
set -euo pipefail

export PORT="${PORT:-7860}"
export MADVERSE_MODE="${MADVERSE_MODE:-deploy}"

echo "Starting MAdVerse on port ${PORT} (mode=${MADVERSE_MODE})..."
exec python run.py --mode "${MADVERSE_MODE}" --port "${PORT}"
