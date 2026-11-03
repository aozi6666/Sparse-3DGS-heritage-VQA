#!/usr/bin/env bash
# 一个命令只做一个阶段；不自动从 smoke 一直跑到 714 题。
set -euo pipefail
PROJECT=/root/autodl-tmp/Sparse-3DGS-heritage-VQA
ENV_DIR=/root/autodl-tmp/Mage_ViT/mage
cd "$PROJECT"
[[ "$(git branch --show-current)" == vlm ]] || { echo '必须在 vlm；不自动切分支或处理现有改动'; exit 1; }
git status --short --branch
df -h /root/autodl-tmp
du -sh "$PROJECT" "$ENV_DIR"
source "$ENV_DIR/bin/activate"
FFMPEG_DIR="${R2V_FFMPEG_DIR:-/root/autodl-tmp/VLMEvalKit/ffmpeg_linux}"
export PATH="$FFMPEG_DIR:$PATH"
export CV_PREINFER_BIN="${CV_PREINFER_BIN:-$ENV_DIR/bin/cv-preinfer}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
DIR="$PROJECT/VQA/mage_vl/practical"
ACTION="${1:-preflight}"
STAGE="${2:-smoke}"
COMMON=(--frames "${PRACTICAL_FRAMES:-32}" --min-free "${PRACTICAL_MIN_FREE_GB:-10}" --budget "${PRACTICAL_BUDGET_GB:-1}")
[[ "${PRACTICAL_CODEC_VOTE:-0}" != 1 ]] || COMMON+=(--codec-vote)
case "$ACTION" in
    init) exec python -B "$DIR/practical_vote.py" init "${COMMON[@]}" ;;
    test-cpu) exec python -B "$DIR/test_practical_mage.py" ;;
    preflight) exec python -B "$DIR/practical_vote.py" preflight --stage "$STAGE" "${COMMON[@]}" ;;
    smoke|dev30|dev60|val|test|full)
        exec python -B "$DIR/practical_vote.py" run --stage "$ACTION" "${COMMON[@]}" ;;
    summarize) exec python -B "$DIR/practical_analyze.py" summarize --stage "$STAGE" "${COMMON[@]}" ;;
    freeze) exec python -B "$DIR/practical_analyze.py" freeze --stage val --method "${2:-frames_vote}" "${COMMON[@]}" ;;
    *) echo '入口: init | test-cpu | preflight [stage] | smoke | dev30 | dev60 | val | freeze [method] | test | full | summarize [stage]'; exit 2 ;;
esac
