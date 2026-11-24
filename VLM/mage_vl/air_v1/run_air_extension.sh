#!/usr/bin/env bash
# 冻结规则的新增90题；旧脚本/候选/缓存保持原位。
set -euo pipefail
source /root/autodl-tmp/Mage_ViT/mage/bin/activate
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
[[ "$(git branch --show-current)" == feat/air-visual-evidence ]] || { echo '分支不符，保留改动并停止';exit 1; }
df -h /root/autodl-tmp
du -sh VLM/evaluation/results/practical_mage_v1/18_air_frames_v1 /root/autodl-tmp/Mage_ViT/mage
ROOT="$PWD/VLM/evaluation/results/practical_mage_v1/18_air_frames_v1/extension90_v1"
DIR="$PWD/VLM/mage_vl/air_v1"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$ROOT/hf_home" HF_MODULES_CACHE="$ROOT/hf_modules" TORCH_HOME="$ROOT/torch_home" CUDA_CACHE_PATH="$ROOT/cuda_cache" TRITON_CACHE_DIR="$ROOT/triton_cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ $# == 1 ]] || exit 2
case "$1" in
 init|preflight|summarize) exec python -B "$DIR/air_extension.py" "$1" ;;
 test-cpu) cd "$DIR";exec python -B -m unittest test_air_extension test_air_hybrid test_air_native_processor ;;
 run)
  [[ "${PRACTICAL_STAGE_CONFIRMED:-0}" == 1 ]] || { echo '用户设置PRACTICAL_STAGE_CONFIRMED=1后启动';exit 1; }
  mkdir -p "$ROOT/logs"
  LOG="$(mktemp "$ROOT/logs/additional90_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
  python -B "$DIR/air_extension.py" run 2>&1 | tee "$LOG" ;;
 *) echo '用法: init/preflight/test-cpu/run/summarize';exit 2 ;;
esac
