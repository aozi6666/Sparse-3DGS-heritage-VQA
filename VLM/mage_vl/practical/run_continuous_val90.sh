#!/usr/bin/env bash
# 固定2048的val90补充验证；不开放其他数据范围，不调用API。
set -euo pipefail
source /root/autodl-tmp/Mage_ViT/mage/bin/activate
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
[[ "$(git branch --show-current)" == vlm ]] || { echo '必须在vlm；保留改动并停止'; exit 1; }
df -h /root/autodl-tmp
du -sh VQA/eval_results/practical_mage_v1 VQA/mage_vl/practical /root/autodl-tmp/Mage_ViT/mage
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
ACTION="${1:-preflight}"
[[ $# -le 1 ]] || { echo '入口只接受init/preflight/run/summarize，不接受其他范围'; exit 2; }
SCRIPT=VQA/mage_vl/practical/continuous_val90.py
case "$ACTION" in
  init|preflight|summarize) exec python -B "$SCRIPT" "$ACTION" ;;
  run)
    [[ "${PRACTICAL_STAGE_CONFIRMED:-0}" == 1 ]] || { echo '用户设置PRACTICAL_STAGE_CONFIRMED=1后启动'; exit 1; }
    python -B "$SCRIPT" preflight
    LOGDIR=VQA/eval_results/practical_mage_v1/15_continuous_val90/logs
    mkdir -p "$LOGDIR"
    LOGFILE="$(mktemp "$LOGDIR/val90_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
    echo "完整日志: $LOGFILE"
    python -B "$SCRIPT" run 2>&1 | tee "$LOGFILE"
    ;;
  *) echo '仅接受init/preflight/run/summarize'; exit 2 ;;
esac
