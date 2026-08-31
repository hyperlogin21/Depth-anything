#!/usr/bin/env bash
# Entry point for AutoDL (https://www.autodl.com) GPU instances: points
# storage and the Hugging Face model cache at the persistent data disk,
# optionally uses the HF mirror for faster downloads from mainland China,
# runs setup, and launches the API in the background.
set -euo pipefail
cd "$(dirname "$0")"

# AutoDL's system disk is small (~30GB) and instance-local; the large data
# disk mounted at /root/autodl-tmp persists across "关机/开机" (stop/start)
# cycles, so weights and job files should live there instead.
if [ -d "/root/autodl-tmp" ]; then
  export STORAGE_DIR="${STORAGE_DIR:-/root/autodl-tmp/depth-anything-storage}"
  export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
fi

# huggingface.co can be slow/unreachable from some AutoDL regions; hf-mirror.com
# is a common drop-in mirror. Override or unset HF_ENDPOINT if you don't need it.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

./setup.sh

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

mkdir -p logs
echo "Starting API on ${HOST}:${PORT} ..."
nohup uvicorn api.main:app --host "$HOST" --port "$PORT" > logs/api.log 2>&1 &
echo $! > logs/api.pid

sleep 2
if kill -0 "$(cat logs/api.pid)" 2>/dev/null; then
  echo "API started (pid $(cat logs/api.pid)). Logs: logs/api.log"
  echo
  echo "To reach it from outside the container, map port ${PORT} via AutoDL's"
  echo "'Custom Service' (自定义服务) port proxy for this instance, or an SSH"
  echo "tunnel: ssh -L ${PORT}:localhost:${PORT} <autodl-ssh-target>"
else
  echo "API failed to start; see logs/api.log" >&2
  tail -n 50 logs/api.log >&2 || true
  exit 1
fi
