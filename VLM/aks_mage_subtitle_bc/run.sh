#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
exec /root/autodl-tmp/VLMEvalKit/vlmeval_env/bin/python -u bc.py "$@"
