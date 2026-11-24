#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
venv="${MAGE_ENV:-/root/autodl-tmp/VLMEvalKit/vlmeval_env}"
export PATH="$venv/bin:/root/autodl-tmp/VLMEvalKit/ffmpeg_linux:$PATH"
export CV_PREINFER_BIN="$venv/bin/cv-preinfer" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
exec "$venv/bin/python" -u audit.py "$@"
