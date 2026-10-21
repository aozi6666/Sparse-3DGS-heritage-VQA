#!/usr/bin/env bash
set -euo pipefail

# Mage-VL-QueryIP x Video-MME（行 B：问题类型重加权 + keep_ratio 预算）
# 用法: bash VQA/VLMEvalKit/sh/run_mage_queryip_videomme.sh

REPO="${REPO:-/root/autodl-tmp/Sparse-3DGS-heritage-VQA}"
ROOT="${VLM_ROOT:-/root/autodl-tmp/VLMEvalKit}"
VENV_BIN="${VENV_BIN:-$ROOT/vlmeval_env/bin}"

cd "$ROOT"
# shellcheck disable=SC1091
source "$VENV_BIN/activate"
export PATH="$VENV_BIN:$PATH"
unset OMP_NUM_THREADS

export PYTHONPATH="${REPO}/VQA/mage_vl/models:${REPO}/VQA/Explore-Then-Select:${PYTHONPATH:-}"
# 离线评测：用本仓 Video-MME 子集，勿回落 HF 下载
export MAGE_VIDEOMME_ROOT="${MAGE_VIDEOMME_ROOT:-${REPO}/VQA/mage_vl/data/VideoMME}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# nframe=32 与 Mage 注册的 target_canvas 对齐
python run.py \
  --data Video-MME-32frame \
  --data-config '{"Video-MME-32frame": {"class": "VideoMME", "dataset": "Video-MME", "nframe": 32}}' \
  --model Mage-VL-QueryIP
