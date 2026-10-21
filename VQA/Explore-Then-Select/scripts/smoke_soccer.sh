#!/usr/bin/env bash
# Mage Query-IP soccer smoke (static vs dynamic under fixed budget K).
# Usage (from anywhere):
#   bash VQA/Explore-Then-Select/scripts/smoke_soccer.sh
# Or after chmod +x:
#   ./VQA/Explore-Then-Select/scripts/smoke_soccer.sh

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

echo "[smoke] repo=${REPO_ROOT}"
echo "[smoke] python=${PYTHON}"
if ! command -v cv-preinfer >/dev/null 2>&1; then
  echo "[warn] cv-preinfer not on PATH (expected under ${VENV_BIN})" >&2
fi

if ! "${PYTHON}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  echo "[error] CUDA not available (torch.cuda.is_available()=False). Wait for a GPU instance, then re-run." >&2
  exit 1
fi

exec "${PYTHON}" "${REPO_ROOT}/VQA/Explore-Then-Select/mage_query_ip/smoke_soccer.py" \
  --model "${MODEL}" \
  --video "${VIDEO}" \
  --budget-mode match_default \
  --keep-ratio "${KEEP_RATIO}" \
  --num-frames "${NUM_FRAMES}" \
  --max-pixels "${MAX_PIXELS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  "$@"
