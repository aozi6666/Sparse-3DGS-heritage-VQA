#!/usr/bin/env bash
# 用户手动启动；prepare仅准备三个CPU依赖，run才会加载GPU模型。
set -euo pipefail
source /root/autodl-tmp/Mage_ViT/mage/bin/activate
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
[[ "$(git branch --show-current)" == feat/air-visual-evidence ]] || { echo '必须在feat/air-visual-evidence，保留改动并停止'; exit 1; }
df -h /root/autodl-tmp
du -sh VLM/eval_results/practical_mage_v1 VLM/models/CLIP-ViT-Large-patch14 /root/autodl-tmp/Mage_ViT/mage
DIR="$PWD/VLM/mage_vl/air_v1"
ROOT="$PWD/VLM/eval_results/practical_mage_v1/18_air_frames_v1"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$ROOT/hf_home" HF_MODULES_CACHE="$ROOT/hf_modules" TORCH_HOME="$ROOT/torch_home" CUDA_CACHE_PATH="$ROOT/cuda_cache" TRITON_CACHE_DIR="$ROOT/triton_cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ACTION="${1:-preflight}";STAGE="${2:-smoke3}"
[[ $# -le 2 && ( "$STAGE" == smoke3 || "$STAGE" == dev30 ) ]] || { echo '仅smoke3/dev30；不开放val/test/full'; exit 2; }
case "$ACTION" in
 prepare)
   PYTHONPATH="$DIR" python -B -c 'from air_protocol import disk_guard;print(disk_guard())'
   # 补齐完整CPU依赖闭包；禁止升级原Mage的Torch/Transformers/NumPy/SciPy。
   mkdir -p "$ROOT/tmp"
   if ! python -B -m pip --version >/dev/null 2>&1; then
     TMPDIR="$ROOT/tmp" python -B -m ensurepip --upgrade
   fi
   if ! python -B -c 'from sklearn.mixture import GaussianMixture; import joblib,threadpoolctl,narwhals,cloudpickle' >/dev/null 2>&1; then
     TMPDIR="$ROOT/tmp" python -m pip install --no-deps --no-cache-dir scikit-learn==1.9.1 joblib==1.6.0 threadpoolctl==3.7.0 narwhals==2.26.0 cloudpickle==3.1.2
   fi
   python -B -c 'import sklearn,joblib,threadpoolctl,narwhals,cloudpickle;from sklearn.mixture import GaussianMixture;print("CPU依赖就绪",sklearn.__version__,joblib.__version__,threadpoolctl.__version__,narwhals.__version__,cloudpickle.__version__)'
   ;;
 init|preflight|summarize) exec python -B "$DIR/air_run.py" "$ACTION" --stage "$STAGE" ;;
 test-cpu)
   cd "$DIR"
   exec python -B -m unittest discover -p 'test_air_*.py' ;;
 run)
   [[ "${PRACTICAL_STAGE_CONFIRMED:-0}" == 1 ]] || { echo '用户设置PRACTICAL_STAGE_CONFIRMED=1后启动'; exit 1; }
   python -B "$DIR/air_run.py" preflight --stage "$STAGE"
   mkdir -p "$ROOT/logs"
   LOG="$(mktemp "$ROOT/logs/${STAGE}_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
   echo "完整日志: $LOG"
   python -B "$DIR/air_run.py" run --stage "$STAGE" 2>&1 | tee "$LOG"
   ;;
 *) echo '用法: prepare/init/preflight/test-cpu/run/summarize [smoke3|dev30]';exit 2 ;;
esac
