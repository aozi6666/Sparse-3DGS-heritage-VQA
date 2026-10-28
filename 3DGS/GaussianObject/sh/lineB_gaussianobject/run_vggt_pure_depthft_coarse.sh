#!/bin/bash
# V2 独立任务：纯 VGGT（无 classic hull）+ depth_head 微调 → 粗训+评测
# 计划：docs/PLAN_vggt_pure_depthft.md
#
#   bash run_vggt_pure_depthft_coarse.sh       # stage 0..3
#   bash run_vggt_pure_depthft_coarse.sh 2     # 从粗训
#   FORCE_HULL=1 / FORCE_TRAIN=1 / ALLOW_OTHER_BRANCH=1
#
# 隔离目录：output/gs_init/kitchen_drop_vggt_pure_depthft
# 禁止覆盖 kitchen_drop / kitchen / kitchen_drop_classic / kitchen_classic
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export USE_DROP="${USE_DROP:-1}"
export SKIP_PATH="${SKIP_PATH:-1}"
export SPARSE_ID="${SPARSE_ID:-4}"
export GS_NAME="${GS_NAME:-kitchen_drop_vggt_pure_depthft}"
export INIT_PCD="${INIT_PCD:-visual_hull_vggt_pure_depthft_${SPARSE_ID}}"
export GS_DIR="output/gs_init/${GS_NAME}"
export LOO_DIR="output/gs_init/${GS_NAME}_loo"
export VGGT_BASE="${VGGT_BASE:-}"
export VGGT_DEPTH_HEAD="${VGGT_DEPTH_HEAD:-}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"
export GS_NAME GS_DIR LOO_DIR INIT_PCD

EXPECTED_BRANCH="feat/sd15-stronger-base-lora"
ALLOW_OTHER_BRANCH="${ALLOW_OTHER_BRANCH:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_HULL="${FORCE_HULL:-0}"
START_STAGE="${1:-0}"

EVAL_CSV="${REPO_ROOT}/3DGS/eval_results/lineB_kitchen_mip360.csv"
PROTECTED=(kitchen_drop kitchen kitchen_drop_classic kitchen_classic)

die() { echo "[ERR] $*" >&2; exit 1; }

require_branch() {
  local br
  br="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "git_branch=${br}"
  if [[ "${br}" != "${EXPECTED_BRANCH}" && "${ALLOW_OTHER_BRANCH}" != "1" ]]; then
    die "expect ${EXPECTED_BRANCH} (now ${br}); ALLOW_OTHER_BRANCH=1 to override"
  fi
}

for p in "${PROTECTED[@]}"; do
  [[ "${GS_NAME}" != "${p}" ]] || die "GS_NAME=${GS_NAME} is protected"
done
if [[ "${INIT_PCD}" == "visual_hull_vggt_enhanced_${SPARSE_ID}" || "${INIT_PCD}" == "visual_hull_${SPARSE_ID}" ]]; then
  die "INIT_PCD=${INIT_PCD} would collide with 2x2 / baseline names"
fi

results_json() {
  echo "${GO}/output/gs_init/${GS_NAME}/test/ours_10000/results.json"
}

stage0_cleanup() {
  echo "[stage 0] light D1 cleanup (repair logs only)"
  local root sub
  for root in \
    "${GO}/output/gaussian_object/kitchen_drop_rv51" \
    "${GO}/output/gaussian_object/kitchen_drop_rv51_conf" \
    "${GO}/output/gaussian_object/kitchen_drop_abs16_conf"
  do
    [[ -d "${root}" ]] || continue
    for sub in tb_logs csv_logs save/controlnet_out; do
      [[ -e "${root}/${sub}" ]] && rm -rf "${root}/${sub}" && echo "  rm ${root}/${sub}" || true
    done
  done
}

stage1_hull() {
  echo "[stage 1] pure VGGT + depthft hull"
  FORCE_HULL="${FORCE_HULL}" bash "${SCRIPT_DIR}/1_vggt_pure_hull_depthft.sh"
}

stage2_train_render() {
  echo "[stage 2] coarse train+render → ${GS_DIR}"
  local rj
  rj="$(results_json)"
  if [[ -f "${rj}" && "${FORCE_TRAIN}" != "1" ]]; then
    echo "[skip] ${rj}"
    return 0
  fi
  lineb_print_paths
  bash "${SCRIPT_DIR}/2_train_gs.sh"
  bash "${SCRIPT_DIR}/3_render.sh"
  rm -rf "${GO}/${GS_DIR}/train" "${GO}/${GS_DIR}/test/ours_10000/renders" 2>/dev/null || true
  [[ -f "${rj}" ]] || die "missing ${rj}"
}

stage3_csv() {
  echo "[stage 3] append CSV row (if metrics exist)"
  local rj base_rj
  rj="$(results_json)"
  base_rj="${GO}/output/gs_init/kitchen_drop/test/ours_10000/results.json"
  [[ -f "${rj}" ]] || die "missing ${rj}"
  python3 - "${EVAL_CSV}" "${GO}" "${GS_NAME}" "${INIT_PCD}" "${SPARSE_ID}" "${SPARSE_VIEW_NUM}" "${ITERATIONS}" "${USE_DROP}" <<'PY'
import csv, json, sys
from datetime import date
from pathlib import Path

csv_path, go, gs_name, init_pcd, sparse_id, sparse_view_num, iterations, use_drop = sys.argv[1:]
go = Path(go)
today = date.today().isoformat()
sparse_id, sparse_view_num, iterations = map(int, (sparse_id, sparse_view_num, iterations))

HEADER = [
    "recorded_date","row_type","scene","pipeline","stage","ckpt_tag","split","init_pcd",
    "use_drop","drop_rate","sparse_id","sparse_view_num","resolution","sh_degree",
    "iterations_coarse","lora_rank","sd_base","controlnet",
    "SSIM","PSNR","LPIPS","dSSIM","dPSNR","dLPIPS","vs_stage",
    "gs_dir","lora_exp","final_ply","source","notes",
]

def load(gs, split):
    p = go / "output" / "gs_init" / gs / split / "ours_10000" / "results.json"
    with open(p) as f:
        d = json.load(f)
    return float(d["SSIM"]), float(d["PSNR"]), float(d["LPIPS"]), str(p.relative_to(go))

path = Path(csv_path)
rows = []
if path.exists():
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("stage") == "coarse_gs_vggt_pure_depthft" and r.get("gs_dir", "").endswith(gs_name):
                continue
            if r.get("stage") == "pure_depthft_vs_vggt_drop" and gs_name in r.get("ckpt_tag", ""):
                continue
            rows.append(r)

for split in ("test", "all"):
    ssim, psnr, lpips, src = load(gs_name, split)
    rows.append({
        "recorded_date": today, "row_type": "absolute", "scene": "mip360/kitchen",
        "pipeline": "lineB_gaussianobject", "stage": "coarse_gs_vggt_pure_depthft",
        "ckpt_tag": "ours_10000", "split": split, "init_pcd": init_pcd,
        "use_drop": str(use_drop), "drop_rate": "0.2" if str(use_drop) == "1" else "",
        "sparse_id": str(sparse_id), "sparse_view_num": str(sparse_view_num),
        "resolution": "4", "sh_degree": "2", "iterations_coarse": str(iterations),
        "lora_rank": "32", "sd_base": "", "controlnet": "",
        "SSIM": f"{ssim:.10f}", "PSNR": f"{psnr:.10f}", "LPIPS": f"{lpips:.10f}",
        "dSSIM": "", "dPSNR": "", "dLPIPS": "", "vs_stage": "",
        "gs_dir": f"output/gs_init/{gs_name}", "lora_exp": "", "final_ply": "",
        "source": src, "notes": "V2 pure VGGT + depth_head_ft coarse; isolated dir",
    })
    base_p = go / "output" / "gs_init" / "kitchen_drop" / split / "ours_10000" / "results.json"
    if base_p.exists():
        bs, bp, bl, _ = load("kitchen_drop", split)
        rows.append({
            "recorded_date": today, "row_type": "delta", "scene": "mip360/kitchen",
            "pipeline": "lineB_gaussianobject", "stage": "pure_depthft_vs_vggt_drop",
            "ckpt_tag": f"{gs_name}_vs_kitchen_drop", "split": split, "init_pcd": init_pcd,
            "use_drop": str(use_drop), "drop_rate": "0.2",
            "sparse_id": str(sparse_id), "sparse_view_num": str(sparse_view_num),
            "resolution": "4", "sh_degree": "2", "iterations_coarse": str(iterations),
            "lora_rank": "32", "sd_base": "", "controlnet": "",
            "SSIM": "", "PSNR": "", "LPIPS": "",
            "dSSIM": f"{ssim-bs:.10f}", "dPSNR": f"{psnr-bp:.10f}", "dLPIPS": f"{lpips-bl:.10f}",
            "vs_stage": "coarse_gs_vggt_drop", "gs_dir": f"output/gs_init/{gs_name}",
            "lora_exp": "", "final_ply": "",
            "source": f"derived {gs_name} - kitchen_drop",
            "notes": "pure+depthft vs enhanced+Drop baseline",
        })

with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=HEADER, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in HEADER})
print(f"wrote {path}")
ssim, psnr, lpips, _ = load(gs_name, "test")
print(f"--- V2 test PSNR={psnr:.4f} SSIM={ssim:.6f} LPIPS={lpips:.6f} ---")
PY
}

echo "=========================================="
echo "V2 pure VGGT + depthft coarse"
require_branch
# defaults for depth head if unset before source
export VGGT_BASE="${VGGT_BASE:-${VGGT}/models/model.pt}"
export VGGT_DEPTH_HEAD="${VGGT_DEPTH_HEAD:-${VGGT}/models/depth_head_ft_mip360.pt}"
lineb_activate >/dev/null
lineb_print_paths
echo "START_STAGE=${START_STAGE}"
echo "=========================================="

[[ "${START_STAGE}" =~ ^[0-3]$ ]] || die "stage must be 0..3"

if (( START_STAGE <= 0 )); then stage0_cleanup; fi
if (( START_STAGE <= 1 )); then stage1_hull; fi
if (( START_STAGE <= 2 )); then stage2_train_render; fi
if (( START_STAGE <= 3 )); then stage3_csv; fi

echo "=========================================="
echo "V2 done → $(results_json)"
echo "  ply: ${DATA}/${INIT_PCD}.ply"
echo "  gs:  ${GO}/${GS_DIR}"
echo "=========================================="
