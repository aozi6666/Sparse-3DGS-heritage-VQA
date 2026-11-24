#!/usr/bin/env bash
# 独立条件解析/定位/QA；不下载、不安装，不改旧实验。
set -euo pipefail
source /root/autodl-tmp/Mage_ViT/mage/bin/activate
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
[[ "$(git branch --show-current)" == feat/air-visual-evidence ]] || { echo '分支不符，保留现场';exit 1; }
df -h /root/autodl-tmp
du -sh VLM/evaluation/results/practical_mage_v1/18_air_frames_v1 /root/autodl-tmp/Mage_ViT/mage
ROOT="$PWD/VLM/evaluation/results/practical_mage_v1/18_air_frames_v1/condition_grounding_v1"
DIR="$PWD/VLM/mage_vl/air_v1"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$ROOT/hf_home" HF_MODULES_CACHE="$ROOT/hf_modules" TORCH_HOME="$ROOT/torch_home" CUDA_CACHE_PATH="$ROOT/cuda_cache" TRITON_CACHE_DIR="$ROOT/triton_cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ $# == 1 ]] || { echo '用法:init/preflight/test-cpu/parse5/summarize5/parse30/summarize30/ground/ground-report/qa/summarize-qa';exit 2; }
if [[ "$1" == test-cpu ]];then cd "$DIR";exec python -B -m unittest test_condition_grounding test_air_native_processor;fi
case "$1" in
 init|preflight|summarize5|summarize30|ground-report|summarize-qa) exec python -B "$DIR/condition_grounding_run.py" "$1" ;;
 parse5|parse30|ground|qa)
  [[ "${PRACTICAL_STAGE_CONFIRMED:-0}" == 1 ]] || { echo '用户确认阶段后设置PRACTICAL_STAGE_CONFIRMED=1';exit 1; }
  mkdir -p "$ROOT/logs";LOG="$(mktemp "$ROOT/logs/${1}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
  python -B "$DIR/condition_grounding_run.py" "$1" 2>&1 | tee "$LOG" ;;
 *) echo '未知动作';exit 2;;
esac
