#!/usr/bin/env bash
# Soccer MV/residual alignment. Writes per-patch scores. No Mage weights unless GENERATE=1.
# Agent does NOT run this.
# Usage:
#   ./VQA/Explore-Then-Select/scripts/align_soccer_mvres.sh
#   GENERATE=1 ./VQA/Explore-Then-Select/scripts/align_soccer_mvres.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

VENV_BIN="${VENV_BIN:-/root/autodl-tmp/VLMEvalKit/vlmeval_env/bin}"
PYTHON="${VENV_BIN}/python"
export PATH="${VENV_BIN}:${PATH}"
export PYTHONPATH="${REPO_ROOT}/VQA/mage_vl/models:${REPO_ROOT}/VQA/Explore-Then-Select:${PYTHONPATH:-}"

VIDEO="${VIDEO:-${REPO_ROOT}/VQA/mage_vl/models/examples/soccer-broadcast.mp4}"
MODEL="${MODEL:-${REPO_ROOT}/VQA/mage_vl/models}"
KEEP_RATIO="${KEEP_RATIO:-0.75}"
NUM_FRAMES="${NUM_FRAMES:-32}"
MAX_PIXELS="${MAX_PIXELS:-150000}"
SEED="${SEED:-0}"
GENERATE="${GENERATE:-0}"

echo "[mvres] repo=${REPO_ROOT}"
args=(
  --video "${VIDEO}"
  --model "${MODEL}"
  --keep-ratio "${KEEP_RATIO}"
  --num-frames "${NUM_FRAMES}"
  --max-pixels "${MAX_PIXELS}"
  --seed "${SEED}"
)
if [[ "${GENERATE}" == "1" ]]; then
  if ! "${PYTHON}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    echo "[error] CUDA not available" >&2
    exit 1
  fi
  args+=(--generate)
fi

exec "${PYTHON}" "${REPO_ROOT}/VQA/Explore-Then-Select/mage_query_ip/align_soccer_mvres.py" \
  "${args[@]}" \
  "$@"
