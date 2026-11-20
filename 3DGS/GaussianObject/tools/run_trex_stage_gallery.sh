#!/usr/bin/env bash
set -euo pipefail
GO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$GO/../.." && pwd)"
source "$ROOT/.venv-ddgs-vggt/bin/activate"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$(python -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"
cd "$ROOT"
exec python -u "$GO/tools/trex_stage_gallery.py" --root "$ROOT" --out "$GO/output/${RUN_TAG:-trex_stage_gallery_v1}" "$@"
