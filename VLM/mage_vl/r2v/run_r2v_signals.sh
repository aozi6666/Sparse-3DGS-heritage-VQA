#!/usr/bin/env bash
# 阶段 2 独立入口：预检 -> 6题打分冒烟 -> 汇总审计 -> 219题打分 -> 汇总。
set -euo pipefail
PROJECT=/root/autodl-tmp/Sparse-3DGS-heritage-VQA
ENV_DIR=/root/autodl-tmp/Mage_ViT/mage
cd "$PROJECT"
[[ "$(git branch --show-current)" == vlm ]] || { echo '必须在 vlm 分支'; exit 1; }
git status --short --branch
df -h /root/autodl-tmp
du -sh "$PROJECT" "$ENV_DIR"
source "$ENV_DIR/bin/activate"
FFMPEG_DIR="${R2V_FFMPEG_DIR:-/root/autodl-tmp/VLMEvalKit/ffmpeg_linux}"
export PATH="$FFMPEG_DIR:$PATH"
export CV_PREINFER_BIN="${CV_PREINFER_BIN:-$ENV_DIR/bin/cv-preinfer}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ACTION="${1:-preflight}"
SCOPE="${2:-smoke}"
case "$ACTION" in preflight|smoke|short|summarize|test) ;; *) echo '入口: preflight [smoke|short] | test | smoke | short | summarize [smoke|short]'; exit 2;; esac
[[ "$ACTION" != smoke && "$ACTION" != short ]] || SCOPE="$ACTION"
[[ "$SCOPE" == smoke || "$SCOPE" == short ]] || { echo 'scope 必须为 smoke/short'; exit 2; }
TAG="${R2V_TAG:-px150000_f32}"
SIGNAL_TAG="${R2V_SIGNAL_TAG:-phase2_signals_v1}"
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ && "$SIGNAL_TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo '标签非法'; exit 2; }
DIR="$PROJECT/VQA/mage_vl/r2v"
BASE="$PROJECT/VQA/eval_results/r2v_mage/$TAG"
OUTPUT="$BASE/$SIGNAL_TAG/$SCOPE"
if [[ "$ACTION" == test ]]; then
    # 纯 CPU 合成数据测试，不导入 Torch/模型、不访问 GPU、不写结果。
    exec python -B "$DIR/test_r2v_signals.py"
fi
if [[ "$ACTION" == summarize ]]; then
    exec python -B "$DIR/r2v_signal_analyze.py" "$OUTPUT"
fi
MODE=preflight
[[ "$ACTION" == preflight ]] || MODE=score
if [[ "$ACTION" == short && "${R2V_SIGNAL_SMOKE_CONFIRMED:-0}" != 1 ]]; then
    echo '先 smoke 并 summarize smoke，审查接口后设置 R2V_SIGNAL_SMOKE_CONFIRMED=1'; exit 1
fi
exec python -B "$DIR/r2v_signal_score.py" "$MODE" --scope "$SCOPE" \
    --source "$BASE/short" --output "$OUTPUT" --min-free-gb "${R2V_MIN_FREE_GB:-10}"
