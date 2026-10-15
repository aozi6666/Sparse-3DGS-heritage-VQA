# Shared env for line B. Sourced by step scripts.
# USE_DROP=1 → kitchen_drop 主线；USE_DROP=0 → kitchen
#
# Layout: <REPO_ROOT>/{3DGS,VQA}/  with scripts under
#   3DGS/GaussianObject/sh/lineB_gaussianobject/

_LINEB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# lineB → sh → GaussianObject → 3DGS → REPO_ROOT
REPO_ROOT="$(cd "${_LINEB_DIR}/../../../.." && pwd)"
ROOT="${ROOT:-${REPO_ROOT}}"
GO="${GO:-${REPO_ROOT}/3DGS/GaussianObject}"
VGGT="${VGGT:-${REPO_ROOT}/3DGS/vggt}"
DATA="${DATA:-${GO}/data/mip360/kitchen}"
VENV="${VENV:-${REPO_ROOT}/.venv-ddgs-vggt/bin/activate}"
VGGT_MODEL="${VGGT_MODEL:-${VGGT}/models/model.pt}"

SPARSE_ID="${SPARSE_ID:-4}"
SPARSE_VIEW_NUM="${SPARSE_VIEW_NUM:-9}"
USE_DROP="${USE_DROP:-1}"
DROP_RATE="${DROP_RATE:-0.2}"
DROP_SCHEDULE="${DROP_SCHEDULE:-linear}"
ITERATIONS="${ITERATIONS:-10000}"
# 路径视频不参与指标；默认跳过 4/10
SKIP_PATH="${SKIP_PATH:-1}"
# LoRA 显存：rank 默认 32（原 64）；测评协议不变，略省显存
LORA_RANK="${LORA_RANK:-32}"
LORA_IMAGE_SIZE="${LORA_IMAGE_SIZE:-512}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset OMP_NUM_THREADS || true

# Allow override: GS_NAME=kitchen_drop_depthft (depthft coarse eval; keep kitchen_drop baseline)
if [[ -z "${GS_NAME:-}" ]]; then
  if [[ "${USE_DROP}" == "1" ]]; then
    GS_NAME="kitchen_drop"
  else
    GS_NAME="kitchen"
  fi
fi

GS_DIR="${GS_DIR:-output/gs_init/${GS_NAME}}"
LOO_DIR="${LOO_DIR:-output/gs_init/${GS_NAME}_loo}"
# 主线：AbsoluteReality v1.6 + 同底模 LoRA；repair 默认 CADR（*_abs16_conf）
# 回退 RV51：SD_CKPT=./models/Realistic_Vision_V5.1.safetensors LORA_EXP=.../_*_rv51 REPAIR_TAG=...
SD_CKPT="${SD_CKPT:-./models/absolutereality_v16.safetensors}"
LORA_EXP="${LORA_EXP:-controlnet_finetune/${GS_NAME}_abs16}"
BASE_TAG="${BASE_TAG:-${GS_NAME}_abs16}"
# CADR 默认开；关：USE_CONF_WEIGHT=0（REPAIR_TAG 回落到 BASE_TAG）
USE_CONF_WEIGHT="${USE_CONF_WEIGHT:-1}"
if [[ -z "${REPAIR_TAG:-}" ]]; then
  if [[ "${USE_CONF_WEIGHT}" == "1" || "${USE_CONF_WEIGHT}" == "True" || "${USE_CONF_WEIGHT}" == "true" ]]; then
    REPAIR_TAG="${BASE_TAG}_conf"
  else
    REPAIR_TAG="${BASE_TAG}"
  fi
fi
FINAL_PLY="output/gaussian_object/${REPAIR_TAG}/save/last.ply"
# Allow override: INIT_PCD=visual_hull_vggt_depthft_4 (do not overwrite baseline enhanced ply name)
INIT_PCD="${INIT_PCD:-visual_hull_vggt_enhanced_${SPARSE_ID}}"

lineb_activate() {
  # shellcheck disable=SC1090
  source "${VENV}"
  export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH:-}"
  cd "${GO}"
}

lineb_print_paths() {
  echo "REPO_ROOT=${REPO_ROOT}"
  echo "GO=${GO}  VGGT=${VGGT}  VENV=${VENV}"
  echo "USE_DROP=${USE_DROP}  GS_NAME=${GS_NAME}  GS_DIR=${GS_DIR}  LOO_DIR=${LOO_DIR}"
  echo "INIT_PCD=${INIT_PCD}"
  echo "SD_CKPT=${SD_CKPT}"
  echo "LORA_EXP=${LORA_EXP}  REPAIR_TAG=${REPAIR_TAG}  USE_CONF_WEIGHT=${USE_CONF_WEIGHT}"
  echo "FINAL_PLY=${FINAL_PLY}  SKIP_PATH=${SKIP_PATH}"
}
