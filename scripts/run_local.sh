#!/usr/bin/env bash
#
# Both processes, one command.
#
#   ./scripts/run_local.sh          # API on 8000, UI on 8501
#   ./scripts/run_local.sh 9000 9501
#
# Ctrl-C stops both.

set -euo pipefail
cd "$(dirname "$0")/.."

API_PORT="${1:-8000}"
UI_PORT="${2:-8501}"

cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "api  -> http://127.0.0.1:$API_PORT"
uv run uvicorn src.api.main:app --port "$API_PORT" --log-level warning &

until curl -sS "http://127.0.0.1:$API_PORT/healthz" >/dev/null 2>&1; do sleep 1; done
echo "ui   -> http://127.0.0.1:$UI_PORT"

uv run streamlit run ui/app.py \
  --server.port "$UI_PORT" \
  --server.headless true \
  --browser.gatherUsageStats false &

wait
