#!/bin/bash
# Depthft G2: hull (base model.pt + depth_head overlay) → coarse GS → test render.
# Writes a NEW gs_init dir; does NOT touch output/gs_init/kitchen_drop (G1 baseline).
# Stops before LOO / LoRA / repair. Compare ours_10000 vs kitchen_drop.
#
# Usage (from anywhere):
#   bash sh/lineB_gaussianobject/run_depthft_coarse_eval.sh
# Optional:
#   GS_NAME=kitchen_drop_depthft INIT_PCD=visual_hull_vggt_depthft_4 \
#   VGGT_DEPTH_HEAD=/path/to/depth_head_ft_kitchen.pt \
#   bash sh/lineB_gaussianobject/run_depthft_coarse_eval.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults for depthft branch (export so child scripts keep them after re-sourcing _common)
export USE_DROP="${USE_DROP:-1}"
export SKIP_PATH="${SKIP_PATH:-1}"
export SPARSE_ID="${SPARSE_ID:-4}"
export GS_NAME="${GS_NAME:-kitchen_drop_depthft}"
export INIT_PCD="${INIT_PCD:-visual_hull_vggt_depthft_${SPARSE_ID}}"
# Force GS_DIR from GS_NAME (ignore stale GS_DIR from caller)
export GS_DIR="output/gs_init/${GS_NAME}"
export LOO_DIR="output/gs_init/${GS_NAME}_loo"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"
# Re-assert after source (defense against stale env)
export GS_NAME GS_DIR LOO_DIR INIT_PCD

if [[ "${GS_NAME}" == "kitchen_drop" || "${GS_NAME}" == "kitchen" ]]; then
  echo "[ERR] GS_NAME=${GS_NAME} would overwrite the baseline coarse run. Use kitchen_drop_depthft."
  exit 1
fi
if [[ "${GS_DIR}" == "output/gs_init/kitchen_drop" || "${GS_DIR}" == "output/gs_init/kitchen" ]]; then
  echo "[ERR] GS_DIR=${GS_DIR} is the baseline directory. Refusing."
  exit 1
fi
if [[ "${INIT_PCD}" == "visual_hull_vggt_enhanced_${SPARSE_ID}" ]]; then
  echo "[ERR] INIT_PCD must be the depthft ply name (e.g. visual_hull_vggt_depthft_${SPARSE_ID}), not baseline enhanced."
  exit 1
fi

echo "=========================================="
echo "depthft coarse eval (steps 1→2→3 only)"
lineb_print_paths
echo "  baseline (untouched): output/gs_init/kitchen_drop"
echo "=========================================="

bash "${SCRIPT_DIR}/1_vggt_hull_depthft.sh"
bash "${SCRIPT_DIR}/2_train_gs.sh"
bash "${SCRIPT_DIR}/3_render.sh"

BASE_JSON="${GO}/output/gs_init/kitchen_drop/test/ours_10000/results.json"
NEW_JSON="${GO}/${GS_DIR}/test/ours_10000/results.json"

echo "=========================================="
echo "depthft coarse eval done"
echo "  G2 ply/init: ${DATA}/${INIT_PCD}.ply"
echo "  G2 gs_init:  ${GO}/${GS_DIR}"
echo "  G2 metrics:  ${NEW_JSON}"
echo "  G1 metrics:  ${BASE_JSON}"
echo "Compare test PSNR/SSIM/LPIPS (ours_10000)."
echo "=========================================="
if [[ -f "${BASE_JSON}" && -f "${NEW_JSON}" ]]; then
  echo "--- G1 (kitchen_drop) ---"
  cat "${BASE_JSON}"
  echo "--- G2 (${GS_NAME}) ---"
  cat "${NEW_JSON}"
fi
