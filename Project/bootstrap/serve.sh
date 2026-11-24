#!/usr/bin/env bash
set -euo pipefail
project=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project"
test "$(git -C .. branch --show-current)" = feat/project-video-deme || { echo '分支不符'; exit 2; }
export PATH="$project/.venv/bin:$project/.node/bin:$project/tools/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_MODULES_CACHE="$project/storage/hf_modules"
mkdir -p "$project/storage/tmp"
export TMPDIR="$project/storage/tmp"
export DEMO_ENABLE_MODEL="${DEMO_ENABLE_MODEL:-0}"
if [ "$DEMO_ENABLE_MODEL" != 1 ]; then export CUDA_VISIBLE_DEVICES=""; fi
# Default preview never enables or imports model runtime libraries.
exec .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port "${DEMO_PORT:-7860}" --workers 1
