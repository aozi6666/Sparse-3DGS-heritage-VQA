# Shared env for line B. Sourced by step scripts.
# USE_DROP=1 → kitchen_drop 主线；USE_DROP=0 → kitchen

ROOT="${ROOT:-/root/autodl-tmp/3DGS}"
GO="${GO:-${ROOT}/GaussianObject}"
DATA="${DATA:-${GO}/data/mip360/kitchen}"
VENV="${VENV:-${ROOT}/.venv-ddgs-vggt/bin/activate}"

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

if [[ "${USE_DROP}" == "1" ]]; then
  GS_NAME="kitchen_drop"
else
  GS_NAME="kitchen"
fi

GS_DIR="output/gs_init/${GS_NAME}"
LOO_DIR="output/gs_init/${GS_NAME}_loo"
LORA_EXP="controlnet_finetune/${GS_NAME}"
REPAIR_TAG="${GS_NAME}"
FINAL_PLY="output/gaussian_object/${REPAIR_TAG}/save/last.ply"
INIT_PCD="visual_hull_vggt_enhanced_${SPARSE_ID}"

lineb_activate() {
  # shellcheck disable=SC1090
  source "${VENV}"
  export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH:-}"
  cd "${GO}"
}

lineb_print_paths() {
  echo "USE_DROP=${USE_DROP}  GS_DIR=${GS_DIR}  LOO_DIR=${LOO_DIR}"
  echo "LORA_EXP=${LORA_EXP}  FINAL_PLY=${FINAL_PLY}  SKIP_PATH=${SKIP_PATH}"
}
