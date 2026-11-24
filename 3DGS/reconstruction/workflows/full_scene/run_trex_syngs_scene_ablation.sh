#!/usr/bin/env bash
set -euo pipefail
GO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GO_DIR"
PY="$(command -v python)"
TORCH_LIB="$($PY -c 'import os,torch; print(os.path.join(os.path.dirname(torch.__file__),"lib"))')"
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
exec "$PY" tools/run_syngs_scene_ablation.py "$@"
