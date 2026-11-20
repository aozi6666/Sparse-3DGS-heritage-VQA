#!/usr/bin/env bash
# 用户执行缓存构建和GPU阶段；不删除任何旧缓存，不自动安装/下载。
set -euo pipefail
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
test "$(git branch --show-current)" = feat/air-visual-evidence
df -h /root/autodl-tmp
du -sh VLM/eval_results/practical_mage_v1 VLM/mage_vl/outputs/r2v_codec_cache /root/autodl-tmp/Mage_ViT/mage
out="$PWD/VLM/eval_results/practical_mage_v1/23_codec_full714_v1"
mkdir -p "$out/logs" "$out/cache/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_MODULES_CACHE="$out/cache/hf_modules" CUDA_CACHE_PATH="$out/cache/cuda" TRITON_CACHE_DIR="$out/cache/triton" TORCH_HOME="$out/cache/torch" TMPDIR="$out/cache/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PATH="/root/autodl-tmp/VLMEvalKit/ffmpeg_linux:$PATH"
python=/root/autodl-tmp/Mage_ViT/mage/bin/python
if [[ "${1:-}" == test-cpu ]]; then
 exec env CUDA_VISIBLE_DEVICES="" "$python" -B -m unittest discover -s VLM/mage_vl/codec_full714_v1 -p 'test_*.py' -v
fi
case "${1:-}" in init|prepare-cache|preflight|audit-inputs|seed-dev|probe|run-val|run-full|summarize-val|summarize-full) ;; *) echo '入口：test-cpu | init | prepare-cache | preflight | audit-inputs | seed-dev | probe | run-val | summarize-val | run-full | summarize-full'; exit 2;; esac
log="$(mktemp "$out/logs/${1}_XXXXXXXX.log")"
echo "完整日志: $log"
case "$1" in
 init|prepare-cache|preflight|audit-inputs|seed-dev)
  CUDA_VISIBLE_DEVICES="" "$python" -B -u VLM/mage_vl/codec_full714_v1/codec_full714.py "$1" 2>&1 | tee "$log" ;;
 *) "$python" -B -u VLM/mage_vl/codec_full714_v1/codec_full714.py "$1" 2>&1 | tee "$log" ;;
esac
