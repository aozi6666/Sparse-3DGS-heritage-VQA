#!/usr/bin/env bash
# CPU离线重审，不需要DEEPSEEK_API_CONFIRMED，不加载任何真实env或模型。
set -euo pipefail
source /root/autodl-tmp/Mage_ViT/mage/bin/activate
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
[[ "$(git branch --show-current)" == feat/air-visual-evidence ]] || { echo '分支不符，保留现场'; exit 1; }
df -h /root/autodl-tmp
du -sh VQA/eval_results/practical_mage_v1/18_air_frames_v1 /root/autodl-tmp/Mage_ViT/mage
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
DIR="$PWD/VQA/mage_vl/air_v1"
RESULT="$PWD/VQA/eval_results/practical_mage_v1/18_air_frames_v1/condition_text30_offline_reaudit_v1"
[[ $# == 1 ]] || { echo '用法:test-cpu/init/preflight/run/summarize'; exit 2; }
case "$1" in
 test-cpu) cd "$DIR"; exec python -B -m unittest test_condition_text30_reaudit ;;
 init|preflight|summarize) exec python -B "$DIR/condition_text30_reaudit_run.py" "$1" ;;
 run)
  mkdir -p "$RESULT/logs"
  LOG="$(mktemp "$RESULT/logs/offline_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
  python -B "$DIR/condition_text30_reaudit_run.py" run 2>&1 | tee "$LOG" ;;
 *) echo '未知动作；无API、定位或QA动作'; exit 2 ;;
esac
