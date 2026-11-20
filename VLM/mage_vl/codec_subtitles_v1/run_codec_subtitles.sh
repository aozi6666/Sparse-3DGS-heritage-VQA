#!/usr/bin/env bash
# 本入口只开放短中长9题，不自动扩开发/全量，不安装或下载，不删除缓存。
set -euo pipefail
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
test "$(git branch --show-current)" = feat/air-visual-evidence
df -h /root/autodl-tmp
du -sh VQA/eval_results/practical_mage_v1 /root/autodl-tmp/Mage_ViT/mage
out="$PWD/VQA/eval_results/practical_mage_v1/21_codec_subtitles_v1"
mkdir -p "$out/logs" "$out/cache/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_MODULES_CACHE="$out/cache/hf_modules" CUDA_CACHE_PATH="$out/cache/cuda" TRITON_CACHE_DIR="$out/cache/triton" TORCH_HOME="$out/cache/torch" TMPDIR="$out/cache/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python=/root/autodl-tmp/Mage_ViT/mage/bin/python
if [[ "${1:-}" == test-cpu ]]; then
 exec env CUDA_VISIBLE_DEVICES="" "$python" -B -m unittest discover -s VQA/mage_vl/codec_subtitles_v1 -p 'test_*.py' -v
fi
case "${1:-}" in init|preflight|audit-inputs|run|summarize) ;; *) echo '入口：test-cpu | init | preflight | audit-inputs | run | summarize'; exit 2;; esac
log="$(mktemp "$out/logs/${1}_XXXXXXXX.log")"
echo "完整日志: $log"
"$python" -B -u VQA/mage_vl/codec_subtitles_v1/codec_subtitles.py "$1" 2>&1 | tee "$log"
