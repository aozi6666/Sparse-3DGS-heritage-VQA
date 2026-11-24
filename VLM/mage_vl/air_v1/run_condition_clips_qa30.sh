#!/usr/bin/env bash
# 独立v2；不修改旧A.I.R./v1，不运行最终QA，不下载/安装。
set -euo pipefail
source /root/autodl-tmp/Mage_ViT/mage/bin/activate
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
[[ "$(git branch --show-current)" == feat/air-visual-evidence ]] || { echo '分支不符，保留现场并停止';exit 1; }
df -h /root/autodl-tmp
du -sh VLM/evaluation/results/practical_mage_v1/18_air_frames_v1 /root/autodl-tmp/Mage_ViT/mage
ROOT="$PWD/VLM/evaluation/results/practical_mage_v1/18_air_frames_v1/condition_clips_v2_qa30"
DIR="$PWD/VLM/mage_vl/air_v1"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$ROOT/hf_home" HF_MODULES_CACHE="$ROOT/hf_modules" TORCH_HOME="$ROOT/torch_home" CUDA_CACHE_PATH="$ROOT/cuda_cache" TRITON_CACHE_DIR="$ROOT/triton_cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ $# == 1 ]] || { echo '用法: init/preflight/test-cpu/run/summarize';exit 2; }
case "$1" in
 init|preflight|summarize) exec python -B "$DIR/condition_clips_qa30.py" "$1" ;;
 test-cpu) cd "$DIR";exec python -B -m unittest test_condition_clips_qa30 test_air_native_processor ;;
 run)
  [[ "${PRACTICAL_STAGE_CONFIRMED:-0}" == 1 ]] || { echo '用户确认后设置PRACTICAL_STAGE_CONFIRMED=1';exit 1; }
  mkdir -p "$ROOT/logs"
  LOG="$(mktemp "$ROOT/logs/qa30_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
  python -B "$DIR/condition_clips_qa30.py" run 2>&1 | tee "$LOG" ;;
 *) echo '未知动作';exit 2 ;;
esac
