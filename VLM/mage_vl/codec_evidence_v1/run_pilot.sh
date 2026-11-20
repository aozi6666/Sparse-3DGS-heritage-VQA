#!/usr/bin/env bash
# User-started GPU sequence only; no waiting loop, no task control, stop on error.
set -euo pipefail
project=/root/autodl-tmp/Sparse-3DGS-heritage-VQA
cd "$project"
test "$(git branch --show-current)" = feat/air-visual-evidence || { echo '分支不符，停止'; exit 2; }
test "${CODEC_EVIDENCE_GPU_CONFIRMED:-}" = 1 || { echo '请由用户显式设置CODEC_EVIDENCE_GPU_CONFIRMED=1'; exit 2; }
mkdir -p VLM/eval_results/practical_mage_v1/codec_evidence_v1/logs
log=VLM/eval_results/practical_mage_v1/codec_evidence_v1/logs/pilot_$(date +%Y%m%d_%H%M%S)_$$.log
exec > >(tee -a "$log") 2>&1
bash VLM/mage_vl/codec_evidence_v1/run.sh probe-gpu
bash VLM/mage_vl/codec_evidence_v1/run.sh score-gpu --scope pilot60
bash VLM/mage_vl/codec_evidence_v1/run.sh run-gpu --scope pilot60
bash VLM/mage_vl/codec_evidence_v1/run.sh summarize --scope pilot60
echo "pilot60已完成，结果和日志保留；没有启动remaining120或714。日志：$log"
