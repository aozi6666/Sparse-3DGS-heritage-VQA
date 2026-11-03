#!/usr/bin/env bash
# 字幕基线收尾：不包含Agent/API，不自动选择预算，不开放test/full。
set -euo pipefail
source /root/autodl-tmp/Mage_ViT/mage/bin/activate
PROJECT=/root/autodl-tmp/Sparse-3DGS-heritage-VQA
ROOT="$PROJECT/VQA/eval_results/practical_mage_v1"
DIR="$PROJECT/VQA/mage_vl/practical"
cd "$PROJECT"
[[ "$(git branch --show-current)" == vlm ]] || { echo '必须在vlm；保留所有改动并停止'; exit 1; }
git status --short --branch
df -h /root/autodl-tmp
du -sh "$ROOT" "$DIR" "$PROJECT/VQA/mage_vl/data/VideoMME/subtitle" /root/autodl-tmp/Mage_ViT/mage
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
ACTION="${1:-preflight}"; SCOPE="${2:-budget}"; shift "$(( $# > 1 ? 2 : $# ))"
COMMON=(--scope "$SCOPE" --min-free "${PRACTICAL_MIN_FREE_GB:-10}" --budget "${PRACTICAL_BUDGET_GB:-1}")
case "$ACTION" in
    init|preflight|prepare-short) exec python -B "$DIR/subtitle_completion_run.py" "$ACTION" "${COMMON[@]}" "$@" ;;
    summarize|freeze-short) exec python -B "$DIR/subtitle_completion_analyze.py" "$ACTION" "${COMMON[@]}" "$@" ;;
    test-cpu) exec python -B "$DIR/test_subtitle_completion.py" ;;
    run)
        [[ "${PRACTICAL_STAGE_CONFIRMED:-0}" == 1 ]] || { echo '由用户设置PRACTICAL_STAGE_CONFIRMED=1后启动'; exit 1; }
        python -B "$DIR/subtitle_completion_run.py" preflight "${COMMON[@]}"
        LOGDIR="$ROOT/14_subtitle_completion/logs"; mkdir -p "$LOGDIR"
        LOGFILE="$(mktemp "$LOGDIR/${SCOPE}_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
        echo "完整日志: $LOGFILE"
        python -B "$DIR/subtitle_completion_run.py" run "${COMMON[@]}" 2>&1 | tee "$LOGFILE"
        ;;
    *) echo '用法: init/preflight/run/summarize budget|short；prepare-short；freeze-short budget --tokens 2048|3072；test-cpu'; exit 2 ;;
esac
