#!/usr/bin/env bash
# 仅增加进度显示；原推理入口负责分支/磁盘预检、环境激活及断点续跑。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
exec /root/autodl-tmp/Mage_ViT/mage/bin/python -B "$SCRIPT_DIR/r2v_progress.py" "${1:-short}"
