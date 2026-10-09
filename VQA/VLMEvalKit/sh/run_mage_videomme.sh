#!/usr/bin/env bash
set -e

# Mage-VL x Video-MME 本地子集评测（238 视频 / 714 题）
# 用法: bash run_mage_videomme.sh

ROOT=/root/autodl-tmp/VLMEvalKit
cd "$ROOT"
source "$ROOT/vlmeval_env/bin/activate"

# 消掉 libgomp 的无效线程数报错（OMP_NUM_THREADS=0 不合法）
unset OMP_NUM_THREADS

# nframe=32 与 Mage-VL 注册的 target_canvas 对齐，避免 fps/nframe 校验被跳过
python run.py \
  --data Video-MME-32frame \
  --data-config '{"Video-MME-32frame": {"class": "VideoMME", "dataset": "Video-MME", "nframe": 32}}' \
  --model Mage-VL
