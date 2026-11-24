#!/usr/bin/env bash
# User-executed environment setup ONLY. No model downloads, GPU inference or cleanup.
set -euo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
repo=$(dirname "$project")
cd "$project"
branch=$(git -C "$repo" branch --show-current)
test "$branch" = feat/project-video-demo || { echo "请先切到feat/project-video-demo；当前：$branch"; exit 2; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export UV_CACHE_DIR="$project/.uv-cache" UV_LINK_MODE=hardlink UV_PYTHON_DOWNLOADS=never
export PIP_CACHE_DIR="$project/.pip-cache"
export PATH="$project/.node/bin:$project/.bootstrap/bin:$project/.venv/bin:$project/tools/bin:$PATH"
mkdir -p storage/logs storage/audits
log="storage/logs/setup_$(date +%Y%m%d_%H%M%S)_$$.log"
exec > >(tee -a "$log") 2>&1
stage=${1:-all}
case "$stage" in tools|backend|frontend|check|all) ;; *) echo '用法：setup_env.sh tools|backend|frontend|check|all'; exit 2;; esac

disk_guard() {
 /root/miniconda3/bin/python -c 'import shutil; f=shutil.disk_usage(".").free; print("空闲GiB",round(f/1024**3,2)); assert f>=10*1024**3,"磁盘不足10GiB，停止，不清理"'
}

tools() {
 disk_guard
 if [ ! -x .bootstrap/bin/python ]; then
  test ! -e .bootstrap || { echo '.bootstrap残缺，保留现场，请检查'; exit 2; }
  /root/miniconda3/bin/python -m venv .bootstrap
 fi
 .bootstrap/bin/python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple 'uv==0.13.7' 'nodeenv>=1.9,<2'
 .bootstrap/bin/uv --version
 .bootstrap/bin/python -m pip freeze > "storage/audits/bootstrap_packages_$(date +%Y%m%d_%H%M%S)_$$.txt"
}

backend() {
 disk_guard
 test -x .bootstrap/bin/uv || { echo '先运行setup_env.sh tools'; exit 2; }
 python311=/root/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11
 test -x "$python311"
 if [ ! -x .venv/bin/python ]; then
  test ! -e .venv || { echo '.venv残缺，保留现场，请检查'; exit 2; }
  .bootstrap/bin/uv venv --python "$python311" .venv
 fi
 .venv/bin/python -c 'import sys; assert sys.version_info[:2]==(3,11),"已有.venv不是Python3.11，保留现场，请检查"'
 # Sources in pyproject route only torch/torchvision to official CUDA12.8 wheels.
 if [ "${PROJECT_REUSE_LOCAL_TORCH:-}" = 1 ]; then
  .venv/bin/python bootstrap/reuse_local_torch.py
  .bootstrap/bin/uv sync --python "$python311" --no-install-package torch
 else
  .bootstrap/bin/uv sync --python "$python311"
 fi
 .bootstrap/bin/uv pip check --python .venv/bin/python
 CUDA_VISIBLE_DEVICES="" .venv/bin/python -c 'import torch; print("torch",torch.__version__,"CPU导入通过；CUDA未初始化",not torch.cuda.is_initialized())'
 .venv/bin/python bootstrap/local_assets.py --link
 .bootstrap/bin/uv pip freeze --python .venv/bin/python > "storage/audits/backend_packages_$(date +%Y%m%d_%H%M%S)_$$.txt"
}

frontend() {
 disk_guard
 test -x .bootstrap/bin/nodeenv || { echo '先运行setup_env.sh tools'; exit 2; }
 if [ ! -x .node/bin/node ]; then
  test ! -e .node || { echo '.node残缺，保留现场，请检查；不要加--force覆盖'; exit 2; }
  .bootstrap/bin/nodeenv --node=22.22.0 --prebuilt --mirror=https://npmmirror.com/mirrors/node .node
 fi
 .node/bin/node --version
 .node/bin/npm --version
 (cd frontend && npm install --registry=https://registry.npmmirror.com --cache="$project/.npm-cache" --no-fund --no-audit)
 frontend/node_modules/.bin/tsc --version
 frontend/node_modules/.bin/vite --version
}

check() {
 test -x .venv/bin/python || { echo '先运行setup_env.sh backend'; exit 2; }
 .venv/bin/python bootstrap/check_env.py
}

if [ "$stage" = all ]; then tools; backend; frontend; check; else "$stage"; fi
echo "完成阶段：$stage；日志：$project/$log。尚未启动任何服务或模型推理。"
