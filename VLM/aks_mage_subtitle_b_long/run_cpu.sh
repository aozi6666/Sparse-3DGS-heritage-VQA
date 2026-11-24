#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES='' TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec flock -n .prepare.lock nice -n 19 /root/autodl-tmp/VLMEvalKit/vlmeval_env/bin/python -B prepare.py
