#!/usr/bin/env bash
# 路线 B 逐阶段入口；失败即停，同一命令重新运行可复用已完整落盘的单元。
set -euo pipefail
PROJECT=/root/autodl-tmp/Sparse-3DGS-heritage-VQA
cd "$PROJECT"
[[ "$(git branch --show-current)" == vlm ]] || { echo '必须在 vlm；保留现有改动，不自动切换'; exit 1; }
git status --short --branch
df -h /root/autodl-tmp
du -sh "$PROJECT/VQA/eval_results/practical_mage_v1" "$PROJECT/VQA/mage_vl/practical" /root/autodl-tmp/Mage_ViT/mage
source /root/autodl-tmp/Mage_ViT/mage/bin/activate
export PATH="${R2V_FFMPEG_DIR:-/root/autodl-tmp/VLMEvalKit/ffmpeg_linux}:$PATH"
export CV_PREINFER_BIN="${CV_PREINFER_BIN:-/root/autodl-tmp/Mage_ViT/mage/bin/cv-preinfer}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ACTION="${1:-preflight}"
FRAMES="${2:-48}"
STAGE="${3:-smoke}"
case "$ACTION" in
    archive|preflight|run|summarize) ;;
    *) echo '用法: bash run_practical_coverage.sh archive|preflight|run|summarize 48|64 smoke|dev30|dev60'; exit 2 ;;
esac
exec python -B "$PROJECT/VQA/mage_vl/practical/practical_coverage.py" "$ACTION" \
    --frames "$FRAMES" --stage "$STAGE" \
    --min-free "${PRACTICAL_MIN_FREE_GB:-10}" --budget "${PRACTICAL_BUDGET_GB:-1}"
