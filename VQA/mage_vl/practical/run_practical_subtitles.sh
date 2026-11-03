#!/usr/bin/env bash
# 一个阶段一个命令；不自动进入dev60/验证/测试/全量。失败即停，旧记录不清理。
set -euo pipefail
PROJECT=/root/autodl-tmp/Sparse-3DGS-heritage-VQA
ROOT="$PROJECT/VQA/eval_results/practical_mage_v1"
DIR="$PROJECT/VQA/mage_vl/practical"
cd "$PROJECT"
[[ "$(git branch --show-current)" == vlm ]] || { echo '必须在vlm；不处理现有改动'; exit 1; }
git status --short --branch
df -h /root/autodl-tmp
du -sh "$ROOT" "$DIR" "$PROJECT/VQA/mage_vl/data/VideoMME/subtitle" /root/autodl-tmp/Mage_ViT/mage
source /root/autodl-tmp/Mage_ViT/mage/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
ACTION="${1:-preflight}"
STAGE="${2:-smoke}"
COMMON=(--stage "$STAGE" --min-free "${PRACTICAL_MIN_FREE_GB:-10}" --budget "${PRACTICAL_BUDGET_GB:-1}")
case "$ACTION" in
    init|preflight) exec python -B "$DIR/practical_subtitles.py" "$ACTION" "${COMMON[@]}" ;;
    test-cpu) exec python -B "$DIR/test_practical_subtitles.py" ;;
    summarize) exec python -B "$DIR/subtitle_analyze.py" "${COMMON[@]}" ;;
    run)
        # 先做无写入预检，再建立唯一日志。tee同时捕获C解码器stderr与Python/tqdm。
        [[ "${PRACTICAL_STAGE_CONFIRMED:-0}" == 1 ]] || { echo '由用户设置PRACTICAL_STAGE_CONFIRMED=1后运行'; exit 1; }
        python -B "$DIR/practical_subtitles.py" preflight "${COMMON[@]}"
        LOGDIR="$ROOT/09_subtitles/logs"
        mkdir -p "$LOGDIR"
        LOGFILE="$(mktemp "$LOGDIR/${STAGE}_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
        echo "本次完整stdout/stderr日志: $LOGFILE"
        python -B "$DIR/practical_subtitles.py" run "${COMMON[@]}" 2>&1 | tee "$LOGFILE"
        ;;
    *) echo '用法: init | test-cpu | preflight [smoke/dev30/dev60] | run [stage] | summarize [stage]'; exit 2 ;;
esac
