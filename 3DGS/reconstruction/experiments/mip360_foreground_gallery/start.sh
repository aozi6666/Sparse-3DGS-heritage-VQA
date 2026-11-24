#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GO="$(cd "$HERE/../.." && pwd)"
ROOT="$(cd "$GO/../.." && pwd)"
source "$ROOT/.venv-ddgs-vggt/bin/activate"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$(python -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("torch").origin).parent/"lib")'):${LD_LIBRARY_PATH:-}"
exec python -u "$HERE/run.py" "$@"
