#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
project=/root/autodl-tmp/Sparse-3DGS-heritage-VQA
python_bin=/root/autodl-tmp/Mage_ViT/mage/bin/python
cd "$project"
test "$(git branch --show-current)" = feat/air-visual-evidence || { echo '分支不符，停止；不会自动切分支'; exit 2; }
exec nice -n 15 "$python_bin" VQA/mage_vl/codec_evidence_v1/experiment.py "$@"
