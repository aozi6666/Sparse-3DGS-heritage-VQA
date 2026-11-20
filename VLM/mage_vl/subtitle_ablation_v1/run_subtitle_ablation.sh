#!/usr/bin/env bash
# 所有结果在独立目录；不安装依赖、不下载、不清理，不自动切换分支。
set -euo pipefail
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
test "$(git branch --show-current)" = feat/air-visual-evidence
df -h /root/autodl-tmp
du -sh VQA/eval_results/practical_mage_v1 /root/autodl-tmp/Mage_ViT/mage
python_bin=/root/autodl-tmp/Mage_ViT/mage/bin/python
out="$PWD/VQA/eval_results/practical_mage_v1/20_subtitle_ablation_v1/frozen_r1"
mkdir -p "$out/logs" "$out/cache/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_MODULES_CACHE="$out/cache/hf_modules" CUDA_CACHE_PATH="$out/cache/cuda"
export TRITON_CACHE_DIR="$out/cache/triton" TORCH_HOME="$out/cache/torch" TMPDIR="$out/cache/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
case "${1:-}" in
  test-cpu)
    exec env CUDA_VISIBLE_DEVICES="" "$python_bin" -B -m unittest discover -s VQA/mage_vl/subtitle_ablation_v1 -p 'test_*.py' -v
    ;;
  init|preflight|run|summarize) ;;
  *) echo 'Usage: bash run_subtitle_ablation.sh {test-cpu|init|preflight|run|summarize} [smoke3|full714]' >&2; exit 2 ;;
esac
log="$(mktemp "$out/logs/${1}_${2:-smoke3}_XXXXXXXX.log")"
echo "完整日志: $log"
"$python_bin" -B -u VQA/mage_vl/subtitle_ablation_v1/subtitle_ablation.py "$@" 2>&1 | tee "$log"
