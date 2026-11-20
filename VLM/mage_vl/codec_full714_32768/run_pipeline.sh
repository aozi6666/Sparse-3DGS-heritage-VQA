#!/usr/bin/env bash
# 所有阶段顺序固定；val90只做数据审计，不按得分决定是否扩全量。
set -euo pipefail
source /root/autodl-tmp/Mage_ViT/mage/bin/activate
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
test "$(git branch --show-current)" = feat/air-visual-evidence
entry=VLM/mage_vl/codec_full714_32768/run_codec_full714.sh
out="$PWD/VLM/eval_results/practical_mage_v1/24_codec_full714_32768_v1"
mkdir -p "$out/logs"
times="$(mktemp "$out/logs/pipeline_walltime_XXXXXXXX.tsv")"
printf 'action\tstart_epoch\tend_epoch\tstatus\n' > "$times"
export PRACTICAL_STAGE_CONFIRMED=1
run_step() {
    local started ended status
    started="$(date +%s)"
    set +e
    bash "$entry" "$1"
    status=$?
    set -e
    ended="$(date +%s)"
    printf '%s\t%s\t%s\t%s\n' "$1" "$started" "$ended" "$status" >> "$times"
    return "$status"
}
run_step preflight
run_step prepare-cache
run_step audit-inputs
run_step seed-dev
run_step probe
run_step run-val
run_step summarize-val
run_step run-full
run_step summarize-full
printf '\n全量结果：%s/results/full714/summary.json\n阶段耗时：%s\n' "$out" "$times"
