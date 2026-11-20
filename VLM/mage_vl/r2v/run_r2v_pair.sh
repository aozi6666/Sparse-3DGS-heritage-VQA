#!/usr/bin/env bash
# 阶段 0/1 手动入口；一个命令只做一个阶段，绝不自动从冒烟进入全量。
set -euo pipefail
PROJECT=/root/autodl-tmp/Sparse-3DGS-heritage-VQA
ENV_DIR=/root/autodl-tmp/Mage_ViT/mage
cd "$PROJECT"
[[ "$(git branch --show-current)" == vlm ]] || { echo '请先安全切到 vlm 分支'; exit 1; }
git status --short --branch
# 每次准备运行前检查空间和目录大小；不清理任何已有文件或缓存。
df -h /root/autodl-tmp
du -sh "$PROJECT" "$ENV_DIR"
source "$ENV_DIR/bin/activate"
# 使用服务器现有的真实二进制；仓内 _archive 下同名文件是 LFS 指针。
FFMPEG_DIR="${R2V_FFMPEG_DIR:-/root/autodl-tmp/VLMEvalKit/ffmpeg_linux}"
[[ -x "$FFMPEG_DIR/ffmpeg" && -x "$FFMPEG_DIR/ffprobe" ]] || { echo '现有 ffmpeg/ffprobe 不可执行；请设置 R2V_FFMPEG_DIR'; exit 1; }
export PATH="$FFMPEG_DIR:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export CV_PREINFER_BIN="${CV_PREINFER_BIN:-$ENV_DIR/bin/cv-preinfer}"
ACTION="${1:-preflight}"
PHASE="${2:-smoke}"
case "$ACTION" in preflight|smoke|short|summarize) ;; *) echo '入口: preflight [smoke|short] | smoke | short | summarize [smoke|short]'; exit 2;; esac
[[ "$ACTION" != smoke && "$ACTION" != short ]] || PHASE="$ACTION"
[[ "$PHASE" == smoke || "$PHASE" == short ]] || { echo 'phase 只能为 smoke/short'; exit 2; }
MAX_PIXELS="${R2V_MAX_PIXELS:-150000}"
FRAMES="${R2V_FRAMES:-32}"
MIN_FREE="${R2V_MIN_FREE_GB:-10}"
TAG="${R2V_TAG:-px${MAX_PIXELS}_f${FRAMES}}"
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo '实验标签只允许字母数字下划线横线'; exit 2; }
BASE="$PROJECT/VQA/eval_results/r2v_mage/$TAG"
OUTPUT="$BASE/$PHASE"
SCRIPT_DIR="$PROJECT/VQA/mage_vl/r2v"
if [[ "$ACTION" == summarize ]]; then
    exec python "$SCRIPT_DIR/r2v_pair_summarize.py" "$OUTPUT"
fi
MODE=preflight
[[ "$ACTION" == preflight ]] || MODE=infer
if [[ "$ACTION" == short ]]; then
    # 必须先手工阅读同配置冒烟结果，明确确认后才允许运行阶段 1。
    [[ "${R2V_SMOKE_CONFIRMED:-0}" == 1 && -f "$BASE/smoke/summary_short.json" ]] || {
        echo '先运行并汇总 smoke，核查显存/像素/token 后设置 R2V_SMOKE_CONFIRMED=1'; exit 1;
    }
fi
exec python "$SCRIPT_DIR/r2v_pair_infer.py" "$MODE" --phase "$PHASE" \
    --model "$PROJECT/VQA/mage_vl/models" \
    --data-root "${MAGE_VIDEOMME_ROOT:-$PROJECT/VQA/mage_vl/data/VideoMME}" \
    --output "$OUTPUT" --cache-root "$PROJECT/VQA/mage_vl/outputs/r2v_codec_cache" \
    --max-pixels "$MAX_PIXELS" --frames "$FRAMES" --min-free-gb "$MIN_FREE"
