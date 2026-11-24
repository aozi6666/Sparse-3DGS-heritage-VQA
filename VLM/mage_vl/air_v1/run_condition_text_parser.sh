#!/usr/bin/env bash
# A.I.R.文本查询编译独立版本；到parse5即停。无视频/模型推理/安装/下载。
set -euo pipefail
source /root/autodl-tmp/Mage_ViT/mage/bin/activate
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
[[ "$(git branch --show-current)" == feat/air-visual-evidence ]] || { echo '分支不符，保留现场'; exit 1; }
df -h /root/autodl-tmp
du -sh VLM/evaluation/results/practical_mage_v1/18_air_frames_v1 /root/autodl-tmp/Mage_ViT/mage
EXPERIMENT="$PWD/VLM/evaluation/results/practical_mage_v1/18_air_frames_v1/condition_grounding_deepseek_v2"
DIR="$PWD/VLM/mage_vl/air_v1"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ $# -ge 1 && $# -le 2 ]] || { echo '用法:test-cpu/init/preflight/parse5/summarize5 [--retry-uncertain]'; exit 2; }
ACTION="$1"; shift
if [[ "$ACTION" == test-cpu ]]; then
  [[ $# == 0 ]] || exit 2
  cd "$DIR"
  exec python -B -m unittest test_condition_text_parser test_condition_text_api
fi
case "$ACTION" in
  init|preflight|summarize5) exec python -B "$DIR/condition_text_run.py" "$ACTION" "$@" ;;
  parse5)
    [[ "${DEEPSEEK_API_CONFIRMED:-0}" == 1 ]] || { echo '由用户在运行前设置DEEPSEEK_API_CONFIRMED=1'; exit 1; }
    mkdir -p "$EXPERIMENT/logs"
    LOG="$(mktemp "$EXPERIMENT/logs/parse5_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
    # 不source env、不set -x；由Python在首次实际请求时加载DEEPSEEK_API_KEY。
    python -B "$DIR/condition_text_run.py" "$ACTION" "$@" 2>&1 | tee "$LOG" ;;
  *) echo '未知动作；本版只开放5题文本编译，不进入定位/QA'; exit 2 ;;
esac
