#!/bin/bash
# =============================================================================
# DISABLED — 当前 D²GS 流程不用（原 GaussianObject 粗训链路）
# 现行脚本: 0_ddgs_run_all.sh / 1_ddgs_vggt_hull.sh / 2_ddgs_train.sh / 3_ddgs_render.sh
# =============================================================================
#
# 原用途: 可视化 hull → data/.../visual_hull_4.ply
#
# CUDA_VISIBLE_DEVICES=2  python visual_hull_test.py \
#     --sparse_id 9 \
#     --data_dir /data/zhangao_data/3DGS/GaussianObject/data/mip360/kitchen \
#     --reso 2 --not_vis
#
echo "[DISABLED] 1_visual_hull.sh — 请用 1_ddgs_vggt_hull.sh" >&2
exit 1
