#!/usr/bin/env bash
# 人工证据定位诊断独立入口；不会改写AIR/混合候选。
set -euo pipefail
source /root/autodl-tmp/Mage_ViT/mage/bin/activate
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
[[ "$(git branch --show-current)" == feat/air-visual-evidence ]] || { echo '分支不符，保留改动并停止';exit 1; }
df -h /root/autodl-tmp
du -sh VQA/eval_results/practical_mage_v1/18_air_frames_v1 /root/autodl-tmp/Mage_ViT/mage
ROOT="$PWD/VQA/eval_results/practical_mage_v1/18_air_frames_v1/oracle_evidence_diag_v1"
DIR="$PWD/VQA/mage_vl/air_v1"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$ROOT/hf_home" HF_MODULES_CACHE="$ROOT/hf_modules" TORCH_HOME="$ROOT/torch_home" CUDA_CACHE_PATH="$ROOT/cuda_cache" TRITON_CACHE_DIR="$ROOT/triton_cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ $# == 1 ]] || exit 2
case "$1" in
 init|preflight|summarize) exec python -B "$DIR/oracle_diag.py" "$1" ;;
 test-cpu) cd "$DIR";exec python -B -m unittest test_oracle_diag test_air_native_processor ;;
 run)
  [[ "${PRACTICAL_STAGE_CONFIRMED:-0}" == 1 ]] || { echo '用户设置PRACTICAL_STAGE_CONFIRMED=1后启动';exit 1; }
  mkdir -p "$ROOT/logs"
  LOG="$(mktemp "$ROOT/logs/diag12_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
  python -B "$DIR/oracle_diag.py" run 2>&1 | tee "$LOG" ;;
 *) echo '用法:init/preflight/test-cpu/run/summarize';exit 2 ;;
esac
