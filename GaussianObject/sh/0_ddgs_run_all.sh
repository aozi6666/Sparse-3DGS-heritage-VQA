#!/bin/bash
# D²GS 全流程：1 点云 → 2 训练 → 3 渲染
# 用法:
#   bash sh/0_ddgs_run_all.sh          # 跑 1+2+3
#   bash sh/0_ddgs_run_all.sh 2        # 从第 2 步开始
#   bash sh/0_ddgs_run_all.sh 2 3      # 只跑 2 和 3
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START="${1:-1}"
END="${2:-3}"

echo "##########################################"
echo " D²GS pipeline  steps ${START} → ${END}"
echo "##########################################"

run_step() {
  local n="$1"
  local f="$2"
  if (( n >= START && n <= END )); then
    echo ""
    echo ">>> 开始 step ${n}: ${f}"
    bash "${SCRIPT_DIR}/${f}"
    echo ">>> 完成 step ${n}"
  else
    echo ">>> 跳过 step ${n}: ${f}"
  fi
}

run_step 1 "1_ddgs_vggt_hull.sh"
run_step 2 "2_ddgs_train.sh"
run_step 3 "3_ddgs_render.sh"

echo ""
echo "##########################################"
echo " 全部完成 (steps ${START}→${END})"
echo "##########################################"
