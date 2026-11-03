#!/usr/bin/env bash
# val90 固定两路配对；不自动运行测试/全量，不删除文件。
set -euo pipefail
source /root/autodl-tmp/Mage_ViT/mage/bin/activate
PROJECT=/root/autodl-tmp/Sparse-3DGS-heritage-VQA
ROOT="$PROJECT/VQA/eval_results/practical_mage_v1"
DIR="$PROJECT/VQA/mage_vl/practical"
cd "$PROJECT"
[[ "$(git branch --show-current)" == vlm ]] || { echo '必须在vlm，保留现有改动并停止'; exit 1; }
git status --short --branch
df -h /root/autodl-tmp
du -sh "$ROOT" "$DIR" "$PROJECT/VQA/mage_vl/data/VideoMME/subtitle" /root/autodl-tmp/Mage_ViT/mage
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
ACTION="${1:-preflight}"
STAGE="${2:-val}"
[[ "$STAGE" == val ]] || { echo '本入口仅支持val；测试/全量尚未开放'; exit 2; }
COMMON=(--stage val --min-free "${PRACTICAL_MIN_FREE_GB:-10}" --budget "${PRACTICAL_BUDGET_GB:-1}")
case "$ACTION" in
    init|preflight) exec python -B "$DIR/subtitle_holdout.py" "$ACTION" "${COMMON[@]}" ;;
    test-cpu) exec python -B "$DIR/test_subtitle_holdout.py" ;;
    summarize) exec python -B "$DIR/subtitle_holdout_analyze.py" "${COMMON[@]}" ;;
    run)
        [[ "${PRACTICAL_STAGE_CONFIRMED:-0}" == 1 ]] || { echo '由用户设置PRACTICAL_STAGE_CONFIRMED=1后运行'; exit 1; }
        python -B "$DIR/subtitle_holdout.py" preflight "${COMMON[@]}"
        LOGDIR="$ROOT/10_subtitle_holdout/logs"
        mkdir -p "$LOGDIR"
        LOGFILE="$(mktemp "$LOGDIR/val_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
        echo "完整stdout/stderr日志: $LOGFILE"
        python -B "$DIR/subtitle_holdout.py" run "${COMMON[@]}" 2>&1 | tee "$LOGFILE"
        ;;
    *) echo '用法: init | preflight | test-cpu | run | summarize [val]'; exit 2 ;;
esac
