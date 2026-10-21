#!/usr/bin/env bash
# Patch-level same-K soccer ablation (prefix / random / queryip).
# Agent does NOT run this — you execute locally.
# Usage:
#   bash VQA/Explore-Then-Select/scripts/smoke_soccer_samek.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

VENV_BIN="${VENV_BIN:-/root/autodl-tmp/VLMEvalKit/vlmeval_env/bin}"
PYTHON="${VENV_BIN}/python"
export PATH="${VENV_BIN}:${PATH}"
export PYTHONPATH="${REPO_ROOT}/VQA/mage_vl/models:${REPO_ROOT}/VQA/Explore-Then-Select:${PYTHONPATH:-}"

MODEL="${MODEL:-${REPO_ROOT}/VQA/mage_vl/models}"
VIDEO="${VIDEO:-${REPO_ROOT}/VQA/mage_vl/models/examples/soccer-broadcast.mp4}"
KEEP_RATIO="${KEEP_RATIO:-0.75}"
NUM_FRAMES="${NUM_FRAMES:-32}"
MAX_PIXELS="${MAX_PIXELS:-150000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
SEED="${SEED:-0}"

echo "[samek] repo=${REPO_ROOT}"
if ! "${PYTHON}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  echo "[error] CUDA not available" >&2
  exit 1
fi

exec "${PYTHON}" "${REPO_ROOT}/VQA/Explore-Then-Select/mage_query_ip/smoke_soccer_samek.py" \
  --model "${MODEL}" \
  --video "${VIDEO}" \
  --keep-ratio "${KEEP_RATIO}" \
  --num-frames "${NUM_FRAMES}" \
  --max-pixels "${MAX_PIXELS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --seed "${SEED}" \
  "$@"
