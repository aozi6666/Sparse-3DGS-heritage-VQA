#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[[ $(git branch --show-current) == vlm ]] || { echo '需要在 vlm 分支运行'; exit 1; }
bash run_cpu.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
exec /root/autodl-tmp/VLMEvalKit/vlmeval_env/bin/python -Bu evaluate.py
