#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$PROJECT/.." && pwd)"
source "$REPO/.venv-ddgs-vggt/bin/activate"
export PYTHONPATH="$PROJECT:$REPO/3DGS/reconstruction/third_party/vggt:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENCV_IO_ENABLE_OPENEXR=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"
exec python -u -m vggt_object.cli "$@"
