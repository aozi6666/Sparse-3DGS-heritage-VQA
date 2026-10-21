#!/usr/bin/env bash
# Video-MME short first-60 same-K Acc. Run AFTER soccer SAMEK_OK.
# Agent does NOT run this.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

VENV_BIN="${VENV_BIN:-/root/autodl-tmp/VLMEvalKit/vlmeval_env/bin}"
PYTHON="${VENV_BIN}/python"
export PATH="${VENV_BIN}:${PATH}"
export PYTHONPATH="${REPO_ROOT}/VQA/mage_vl/models:${REPO_ROOT}/VQA/Explore-Then-Select:${PYTHONPATH:-}"
export MAGE_VIDEOMME_ROOT="${MAGE_VIDEOMME_ROOT:-${REPO_ROOT}/VQA/mage_vl/data/VideoMME}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

N="${N:-60}"
KEEP_RATIO="${KEEP_RATIO:-0.75}"
SEED="${SEED:-0}"

if ! "${PYTHON}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  echo "[error] CUDA not available" >&2
  exit 1
fi

exec "${PYTHON}" "${REPO_ROOT}/VQA/Explore-Then-Select/mage_query_ip/eval_short60_samek.py" \
  --n "${N}" \
  --keep-ratio "${KEEP_RATIO}" \
  --seed "${SEED}" \
  "$@"
