#!/usr/bin/env bash
# 查询→画面诊断：只运行本地CLIP文本编码和CPU预览，无Mage/LLM/API/QA入口。
set -euo pipefail
source /root/autodl-tmp/Mage_ViT/mage/bin/activate
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
[[ "$(git branch --show-current)" == feat/air-visual-evidence ]] || { echo '分支不符，保留现场'; exit 1; }
df -h /root/autodl-tmp
du -sh VQA/eval_results/practical_mage_v1/18_air_frames_v1 /root/autodl-tmp/Mage_ViT/mage
DIR="$PWD/VQA/mage_vl/air_v1"
RESULT="$PWD/VQA/eval_results/practical_mage_v1/18_air_frames_v1/condition_leaf_localization_probe_v1"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$RESULT/hf_home" HF_MODULES_CACHE="$RESULT/hf_modules" TORCH_HOME="$RESULT/torch_home" CUDA_CACHE_PATH="$RESULT/cuda_cache" TRITON_CACHE_DIR="$RESULT/triton_cache"
export XDG_CACHE_HOME="$RESULT/xdg_cache" TORCHINDUCTOR_CACHE_DIR="$RESULT/torch_inductor" PYTORCH_KERNEL_CACHE_PATH="$RESULT/torch_kernel_cache"
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
mkdir -p "$RESULT/tmp"
export TMPDIR="$RESULT/tmp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ $# == 1 ]] || { echo '用法:test-cpu/init/preflight/encode/locate/render/summarize'; exit 2; }
case "$1" in
 test-cpu) cd "$DIR"; exec python -B -m unittest test_leaf_probe ;;
 init|preflight|summarize) exec python -B "$DIR/leaf_probe_run.py" "$1" ;;
 encode|locate|render)
  if [[ "$1" == encode && "${PRACTICAL_STAGE_CONFIRMED:-0}" != 1 ]];then echo '用户确认后设置PRACTICAL_STAGE_CONFIRMED=1';exit 1;fi
  mkdir -p "$RESULT/logs"
  LOG="$(mktemp "$RESULT/logs/${1}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
  python -B "$DIR/leaf_probe_run.py" "$1" 2>&1 | tee "$LOG" ;;
 *) echo '未知动作，无QA或API入口'; exit 2 ;;
esac
