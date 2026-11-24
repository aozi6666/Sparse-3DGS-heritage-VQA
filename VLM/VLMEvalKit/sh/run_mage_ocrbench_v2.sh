#!/usr/bin/env bash
set -euo pipefail

# Mage-VL × OCRBench_v2 全量 baseline（图像；Agent 不代跑）
# 用法: bash VLM/VLMEvalKit/sh/run_mage_ocrbench_v2.sh

REPO="${REPO:-/root/autodl-tmp/Sparse-3DGS-heritage-VQA}"
KIT="${KIT:-$REPO/VLM/VLMEvalKit}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/root/autodl-tmp/VLMEvalKit/vlmeval_env/bin/activate}"

# shellcheck source=/dev/null
source "$ENV_ACTIVATE"
unset OMP_NUM_THREADS || true

mkdir -p "$REPO/VLM/datasets/lmu" \
  "$REPO/VLM/evaluation/results/baseline_mage_ocrbench_v2" \
  "$REPO/VLM/evaluation/results/logs"

TSV_SRC="$REPO/VLM/datasets/OCRBench_v2/OCRBench_v2.tsv"
if [[ ! -f "$TSV_SRC" ]]; then
  echo "[error] missing $TSV_SRC" >&2
  exit 2
fi
cache="$REPO/VLM/datasets/lmu/OCRBench_v2.tsv"
if ! [[ "$TSV_SRC" -ef "$cache" ]]; then
  cp --reflink=auto --remove-destination "$TSV_SRC" "$cache"
fi

export LMUData="$REPO/VLM/datasets/lmu"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="$KIT:${PYTHONPATH:-}"

cd "$KIT"
echo "[ocrbench_v2] LMUData=$LMUData"
echo "[ocrbench_v2] work-dir=$REPO/VLM/evaluation/results/baseline_mage_ocrbench_v2"
echo "[ocrbench_v2] model=Mage-VL (image). Do NOT use Mage-VL-QueryIP."

exec python run.py \
  --model Mage-VL \
  --data OCRBench_v2 \
  --work-dir "$REPO/VLM/evaluation/results/baseline_mage_ocrbench_v2" \
  --verbose
