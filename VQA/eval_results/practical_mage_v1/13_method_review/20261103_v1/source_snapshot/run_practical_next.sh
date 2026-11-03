#!/usr/bin/env bash
# 单阶段执行，不自动扩到pilot/dev60/test/full；完整stderr/tqdm日志统一归档。
set -euo pipefail
source /root/autodl-tmp/Mage_ViT/mage/bin/activate
PROJECT=/root/autodl-tmp/Sparse-3DGS-heritage-VQA
ROOT="$PROJECT/VQA/eval_results/practical_mage_v1"
DIR="$PROJECT/VQA/mage_vl/practical"
cd "$PROJECT"
[[ "$(git branch --show-current)" == vlm ]] || { echo '必须在vlm，保留所有改动并停止'; exit 1; }
git status --short --branch
df -h /root/autodl-tmp
du -sh "$ROOT" "$DIR" /root/autodl-tmp/Mage_ViT/mage
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
ACTION="${1:-preflight}"; STAGE="${2:-smoke}"; shift "$(( $# > 1 ? 2 : $# ))"
COMMON=(--stage "$STAGE" --min-free "${PRACTICAL_MIN_FREE_GB:-10}" --budget "${PRACTICAL_BUDGET_GB:-1}")
MODE=mage
case "$ACTION" in api-*) MODE=api; ACTION="${ACTION#api-}" ;; esac
if [[ "$MODE" == api ]]; then SCRIPT=deepseek_review.py; ANALYZE=deepseek_analyze.py; LOGBASE=12_deepseek_review
else SCRIPT=continuous_mage.py; ANALYZE=continuous_analyze.py; LOGBASE=11_continuous_subtitles; fi
case "$ACTION" in
    init|preflight) exec python -B "$DIR/$SCRIPT" "$ACTION" "${COMMON[@]}" "$@" ;;
    summarize) exec python -B "$DIR/$ANALYZE" "${COMMON[@]}" ;;
    test-cpu) exec python -B "$DIR/test_practical_next.py" ;;
    run)
        if [[ "$MODE" == api ]]; then
            [[ "${DEEPSEEK_API_CONFIRMED:-0}" == 1 ]] || { echo '请由用户设置DEEPSEEK_API_CONFIRMED=1'; exit 1; }
        else
            [[ "${PRACTICAL_STAGE_CONFIRMED:-0}" == 1 ]] || { echo '请由用户设置PRACTICAL_STAGE_CONFIRMED=1'; exit 1; }
        fi
        python -B "$DIR/$SCRIPT" preflight "${COMMON[@]}" "$@"
        LOGDIR="$ROOT/$LOGBASE/logs"; mkdir -p "$LOGDIR"
        LOGFILE="$(mktemp "$LOGDIR/${STAGE}_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
        echo "日志: $LOGFILE"
        python -B "$DIR/$SCRIPT" run "${COMMON[@]}" "$@" 2>&1 | tee "$LOGFILE"
        ;;
    *) echo '用法: init/preflight/run/summarize [smoke|dev30|dev60]；api-init/api-preflight/api-run/api-summarize [smoke|pilot|dev60]；test-cpu'; exit 2 ;;
esac
