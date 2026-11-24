#!/usr/bin/env bash
# Opt-in workflow; the existing lineB scripts retain their defaults.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${GO}/../.." && pwd)"
VENV="${VENV:-${REPO_ROOT}/.venv-ddgs-vggt/bin/activate}"
source "${VENV}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LD_LIBRARY_PATH="$(python -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
cd "${GO}"
exec python tools/run_full_scene.py "$@"
