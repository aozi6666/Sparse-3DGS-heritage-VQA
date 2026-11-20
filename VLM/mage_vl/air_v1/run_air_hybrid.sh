#!/usr/bin/env bash
# 独立入口；回退时使用旧run_air_frames.sh，不删除任何结果。
set -euo pipefail
source /root/autodl-tmp/Mage_ViT/mage/bin/activate
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
[[ "$(git branch --show-current)" == feat/air-visual-evidence ]] || { echo '分支不符，保留改动并停止';exit 1; }
df -h /root/autodl-tmp
du -sh VQA/eval_results/practical_mage_v1/18_air_frames_v1 /root/autodl-tmp/Mage_ViT/mage
ROOT="$PWD/VQA/eval_results/practical_mage_v1/18_air_frames_v1/hybrid_g16l16_v1"
DIR="$PWD/VQA/mage_vl/air_v1"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME="$ROOT/hf_home" HF_MODULES_CACHE="$ROOT/hf_modules" TORCH_HOME="$ROOT/torch_home" CUDA_CACHE_PATH="$ROOT/cuda_cache" TRITON_CACHE_DIR="$ROOT/triton_cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ $# == 1 ]] || { echo '用法: init/preflight/test-cpu/run/summarize';exit 2; }
case "$1" in
 init|preflight|summarize) exec python -B "$DIR/air_hybrid.py" "$1" ;;
 test-cpu) cd "$DIR"; exec python -B -m unittest test_air_hybrid test_air_native_processor ;;
 run)
  [[ "${PRACTICAL_STAGE_CONFIRMED:-0}" == 1 ]] || { echo '用户设置PRACTICAL_STAGE_CONFIRMED=1后运行';exit 1; }
  python -B "$DIR/air_hybrid.py" preflight
  mkdir -p "$ROOT/logs"
  LOG="$(mktemp "$ROOT/logs/dev30_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
  python -B "$DIR/air_hybrid.py" run 2>&1 | tee "$LOG" ;;
 *) echo '未知操作';exit 2 ;;
esac
