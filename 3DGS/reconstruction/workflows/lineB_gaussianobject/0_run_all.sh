#!/bin/bash
# 线 B：VGGT hull → 粗GS(±Drop) → 评测 → LOO → LoRA → repair → 最终评测
# 用法:
#   USE_DROP=1 bash 0_run_all.sh        # 全跑（默认开 Drop，跳过路径视频）
#   USE_DROP=1 bash 0_run_all.sh 5      # 从第5步起（粗训已完成时用）
#   SKIP_PATH=0 bash 0_run_all.sh       # 额外跑路径视频 4/10
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"
START="${1:-1}"

STEPS=(
  "1_vggt_hull.sh"
  "2_train_gs.sh"
  "3_render.sh"
  "4_render_path.sh"
  "5_loo_stage1.sh"
  "6_loo_stage2.sh"
  "7_train_lora.sh"
  "8_train_repair.sh"
  "9_render_last.sh"
  "10_render_path_final.sh"
)

echo "=========================================="
echo "线 B: GaussianObject  (GPU=${CUDA_VISIBLE_DEVICES})"
lineb_print_paths
echo "=========================================="

for i in "${!STEPS[@]}"; do
  step_num=$((i + 1))
  # 默认跳过路径视频（不影响 results.json 评测）
  if [[ "${SKIP_PATH}" == "1" ]] && { [[ "${step_num}" -eq 4 ]] || [[ "${step_num}" -eq 10 ]]; }; then
    echo ">>> 跳过步骤 ${step_num}: ${STEPS[$i]}  (SKIP_PATH=1)"
    continue
  fi
  if [[ "${START}" -le "${step_num}" ]]; then
    echo ">>> 运行步骤 ${step_num}: ${STEPS[$i]}"
    bash "${SCRIPT_DIR}/${STEPS[$i]}"
  else
    echo ">>> 跳过步骤 ${step_num}: ${STEPS[$i]}  (START=${START})"
  fi
done

echo "=========================================="
echo "线 B 全部完成"
echo "最终 ply: ${GO}/${FINAL_PLY}"
echo "指标: ${GO}/${GS_DIR}/test/*/results.json 与 repair 后 render 目录"
echo "=========================================="
