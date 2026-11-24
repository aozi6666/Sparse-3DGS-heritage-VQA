#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
venv="${MAGE_ENV:-/root/autodl-tmp/VLMEvalKit/vlmeval_env}"
export PATH="$venv/bin:/root/autodl-tmp/VLMEvalKit/ffmpeg_linux:$PATH"
export CV_PREINFER_BIN="$venv/bin/cv-preinfer"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
exec "$venv/bin/python" -u evaluate.py "$@"
