#!/usr/bin/env bash
# One-shot environment setup: installs torch (only if missing), the rest of
# the Python dependencies, and creates the local storage directories.
set -euo pipefail
cd "$(dirname "$0")"

echo "== Depth-Anything Video API setup =="

if python3 -c "import torch" >/dev/null 2>&1; then
  python3 - <<'PY'
import torch
print(f"Found torch {torch.__version__} (cuda available: {torch.cuda.is_available()})")
PY
else
  echo "torch not found; installing a CUDA 12.1 build from pytorch.org..."
  echo "(If this machine has a different CUDA version, install torch yourself first"
  echo " following https://pytorch.org/get-started/locally/ then re-run this script.)"
  pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

echo "Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Note: system ffmpeg not found on PATH; imageio-ffmpeg will download its own"
  echo "static binary automatically on first use."
fi

mkdir -p storage/uploads storage/outputs

echo
echo "Setup complete. Start the API with:"
echo "  uvicorn api.main:app --host 0.0.0.0 --port 8000"
echo "or, on AutoDL, just run: ./autodl_start.sh"
