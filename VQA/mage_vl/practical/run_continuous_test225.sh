#!/usr/bin/env bash
# 冻结2048的test225最终评测；用户手动启动，可逐答案恢复。
set -euo pipefail
source /root/autodl-tmp/Mage_ViT/mage/bin/activate
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
[[ "$(git branch --show-current)" == vlm ]] || { echo '必须在vlm；保留改动并停止'; exit 1; }
df -h /root/autodl-tmp
du -sh VQA/eval_results/practical_mage_v1 VQA/mage_vl/practical VQA/mage_vl/data/VideoMME/subtitle /root/autodl-tmp/Mage_ViT/mage
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
[[ $# -le 1 ]] || { echo '仅接受init/preflight/run/summarize，不接受其他范围'; exit 2; }
ACTION="${1:-preflight}"
SCRIPT=VQA/mage_vl/practical/continuous_test225.py
case "$ACTION" in
  init|preflight|summarize) exec python -B "$SCRIPT" "$ACTION" ;;
  run)
    [[ "${PRACTICAL_STAGE_CONFIRMED:-0}" == 1 ]] || { echo '用户设置PRACTICAL_STAGE_CONFIRMED=1后启动'; exit 1; }
    python -B "$SCRIPT" preflight
    LOGDIR=VQA/eval_results/practical_mage_v1/16_continuous_test225/logs
    mkdir -p "$LOGDIR"
    LOGFILE="$(mktemp "$LOGDIR/test225_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
    echo "完整日志: $LOGFILE"
    python -B "$SCRIPT" run 2>&1 | tee "$LOGFILE"
    ;;
  *) echo '仅接受init/preflight/run/summarize'; exit 2 ;;
esac
