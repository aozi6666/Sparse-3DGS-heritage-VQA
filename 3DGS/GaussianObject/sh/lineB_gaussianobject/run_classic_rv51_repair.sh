#!/bin/bash
# P2：kitchen_drop_classic（classic+Drop）复用 RV51 LoRA 做无 CADR repair
# 计划：docs/PLAN_p2_classic_repair_garden.md
#
#   bash run_classic_rv51_repair.sh
#   bash run_classic_rv51_repair.sh 1
#   FORCE_REPAIR=1 / ALLOW_OTHER_BRANCH=1 / SKIP_CLEANUP=1
#
# 对照：ours_None__kitchen_drop_rv51（VGGT 线 A0）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

EXPECTED_BRANCH="feat/sd15-stronger-base-lora"
ALLOW_OTHER_BRANCH="${ALLOW_OTHER_BRANCH:-0}"
FORCE_REPAIR="${FORCE_REPAIR:-0}"
SKIP_CLEANUP="${SKIP_CLEANUP:-0}"
START_STAGE="${1:-0}"

GS_NAME_FIXED="kitchen_drop_classic"
GS_DIR_FIXED="output/gs_init/${GS_NAME_FIXED}"
INIT_FIXED="visual_hull_${SPARSE_ID}"
TAG="kitchen_drop_classic_rv51"
RV51_CKPT="./models/Realistic_Vision_V5.1.safetensors"
RV51_LORA="controlnet_finetune/kitchen_drop_rv51"
EVAL_CSV="${REPO_ROOT}/3DGS/eval_results/lineB_kitchen_mip360.csv"
BASE_VGGT_TAG="kitchen_drop_rv51"
BASE_CADR_TAG="kitchen_drop_rv51_conf"
BASE_GS="kitchen_drop"

die() { echo "[ERR] $*" >&2; exit 1; }

require_branch() {
  local br
  br="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "git_branch=${br}"
  if [[ "${br}" != "${EXPECTED_BRANCH}" && "${ALLOW_OTHER_BRANCH}" != "1" ]]; then
    die "expect ${EXPECTED_BRANCH} (now ${br}); ALLOW_OTHER_BRANCH=1 to override"
  fi
}

results_tag() {
  # $1 = gs_name (under gs_init), $2 = repair tag suffix after ours_None__
  echo "${GO}/output/gs_init/$1/test/ours_None__$2/results.json"
}

strip_repair_logs() {
  local tag="$1"
  local root="${GO}/output/gaussian_object/${tag}"
  [[ -d "${root}" ]] || return 0
  local sub
  for sub in tb_logs csv_logs save/controlnet_out; do
    [[ -e "${root}/${sub}" ]] && rm -rf "${root}/${sub}" && echo "  rm -rf ${root}/${sub}" || true
  done
  rm -f "${root}"/events.out.tfevents.* 2>/dev/null || true
  [[ -f "${root}/save/last.ply" ]] && echo "  keep ${root}/save/last.ply" || true
}

stage0_cleanup() {
  if [[ "${SKIP_CLEANUP}" == "1" ]]; then
    echo "[stage 0] skip cleanup"
    return 0
  fi
  echo "[stage 0] D1 strip repair logs"
  local t
  for t in kitchen_drop_rv51 kitchen_drop_rv51_conf kitchen_drop_classic_rv51 \
           kitchen_drop_rv51_trust kitchen_drop_rv51_cov kitchen_drop_rv51_aden \
           kitchen_drop_abs16 kitchen_drop_abs16_conf; do
    strip_repair_logs "${t}"
  done
}

stage1_repair() {
  echo "=========================================="
  echo "[stage 1] classic→RV51 repair tag=${TAG}"
  echo "=========================================="
  local rj
  rj="$(results_tag "${GS_NAME_FIXED}" "${TAG}")"
  [[ -d "${GO}/${GS_DIR_FIXED}" ]] || die "missing coarse ${GO}/${GS_DIR_FIXED}"
  [[ -d "${GO}/output/${RV51_LORA}" ]] || die "missing LoRA ${GO}/output/${RV51_LORA}"

  if [[ -f "${rj}" && "${FORCE_REPAIR}" != "1" ]]; then
    echo "[skip] ${rj}"
    strip_repair_logs "${TAG}"
    return 0
  fi

  unset REPAIR_TAG BASE_TAG LORA_EXP SD_CKPT USE_CONF_WEIGHT \
        CADR_TRUST_ONLY USE_COV_IN_NEED USE_ADAPTIVE_DENOISE \
        GS_NAME GS_DIR INIT_PCD FINAL_PLY || true

  export GS_NAME="${GS_NAME_FIXED}"
  export GS_DIR="${GS_DIR_FIXED}"
  export INIT_PCD="${INIT_FIXED}"
  export REPAIR_TAG="${TAG}"
  export BASE_TAG="${TAG}"
  export FINAL_PLY="output/gaussian_object/${TAG}/save/last.ply"
  export SD_CKPT="${RV51_CKPT}"
  export LORA_EXP="${RV51_LORA}"
  export USE_CONF_WEIGHT=0
  export CADR_TRUST_ONLY=0
  export USE_COV_IN_NEED=0
  export USE_ADAPTIVE_DENOISE=0
  export SAVE_CONTROLNET_PREVIEW=0
  export SKIP_PATH="${SKIP_PATH:-1}"
  # kitchen data for render (same scene)
  export DATA="${GO}/data/mip360/kitchen"

  lineb_print_paths
  bash "${SCRIPT_DIR}/8_train_repair.sh"
  [[ -f "${GO}/${FINAL_PLY}" ]] || die "missing ${GO}/${FINAL_PLY}"
  bash "${SCRIPT_DIR}/9_render_last.sh"
  [[ -f "${rj}" ]] || die "missing ${rj}"
  strip_repair_logs "${TAG}"
  echo "[stage 1] done → ${rj}"
}

stage2_csv() {
  echo "=========================================="
  echo "[stage 2] CSV → ${EVAL_CSV}"
  echo "=========================================="
  local classic_rj vggt_rj cadr_rj
  classic_rj="$(results_tag "${GS_NAME_FIXED}" "${TAG}")"
  vggt_rj="$(results_tag "${BASE_GS}" "${BASE_VGGT_TAG}")"
  cadr_rj="$(results_tag "${BASE_GS}" "${BASE_CADR_TAG}")"
  [[ -f "${classic_rj}" ]] || die "missing ${classic_rj}"
  [[ -f "${vggt_rj}" ]] || die "missing ${vggt_rj}"

  python3 - "${EVAL_CSV}" "${GO}" "${GS_NAME_FIXED}" "${TAG}" \
    "${BASE_GS}" "${BASE_VGGT_TAG}" "${BASE_CADR_TAG}" \
    "${INIT_FIXED}" "${SPARSE_ID}" "${SPARSE_VIEW_NUM}" "${ITERATIONS}" <<'PY'
import csv, json, sys
from datetime import date
from pathlib import Path

(
    csv_path, go, classic_gs, classic_tag,
    base_gs, vggt_tag, cadr_tag,
    init_pcd, sparse_id, sparse_view_num, iterations,
) = sys.argv[1:]
go = Path(go)
sparse_id, sparse_view_num, iterations = map(int, (sparse_id, sparse_view_num, iterations))
today = date.today().isoformat()

HEADER = [
    "recorded_date","row_type","scene","pipeline","stage","ckpt_tag","split","init_pcd",
    "use_drop","drop_rate","sparse_id","sparse_view_num","resolution","sh_degree",
    "iterations_coarse","lora_rank","sd_base","controlnet",
    "SSIM","PSNR","LPIPS","dSSIM","dPSNR","dLPIPS","vs_stage",
    "gs_dir","lora_exp","final_ply","source","notes",
]
REPLACE = {
    "lora_repair_classic_rv51",
    "classic_rv51_vs_vggt_rv51",
    "classic_rv51_vs_vggt_cadr",
}

def load(gs, tag, split):
    p = go / "output" / "gs_init" / gs / split / f"ours_None__{tag}" / "results.json"
    with open(p) as f:
        d = json.load(f)
    return float(d["SSIM"]), float(d["PSNR"]), float(d["LPIPS"]), str(p.relative_to(go))

path = Path(csv_path)
rows = []
if path.exists():
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("stage") in REPLACE:
                continue
            rows.append(r)

new_rows = []
for split in ("test", "all"):
    s, p, l, src = load(classic_gs, classic_tag, split)
    new_rows.append({
        "recorded_date": today, "row_type": "absolute", "scene": "mip360/kitchen",
        "pipeline": "lineB_gaussianobject", "stage": "lora_repair_classic_rv51",
        "ckpt_tag": f"ours_None__{classic_tag}", "split": split, "init_pcd": init_pcd,
        "use_drop": "1", "drop_rate": "0.2",
        "sparse_id": str(sparse_id), "sparse_view_num": str(sparse_view_num),
        "resolution": "4", "sh_degree": "2", "iterations_coarse": str(iterations),
        "lora_rank": "32", "sd_base": "Realistic_Vision_V5.1.safetensors",
        "controlnet": "control_v11f1e_sd15_tile",
        "SSIM": f"{s:.10f}", "PSNR": f"{p:.10f}", "LPIPS": f"{l:.10f}",
        "dSSIM": "", "dPSNR": "", "dLPIPS": "", "vs_stage": "",
        "gs_dir": f"output/gs_init/{classic_gs}",
        "lora_exp": "output/controlnet_finetune/kitchen_drop_rv51",
        "final_ply": f"output/gaussian_object/{classic_tag}/save/last.ply",
        "source": src,
        "notes": "P2 classic+Drop init + reused RV51 LoRA; no CADR",
    })
    a, b = load(classic_gs, classic_tag, split), load(base_gs, vggt_tag, split)
    new_rows.append({
        "recorded_date": today, "row_type": "delta", "scene": "mip360/kitchen",
        "pipeline": "lineB_gaussianobject", "stage": "classic_rv51_vs_vggt_rv51",
        "ckpt_tag": f"{classic_tag}_vs_{vggt_tag}", "split": split, "init_pcd": init_pcd,
        "use_drop": "1", "drop_rate": "0.2",
        "sparse_id": str(sparse_id), "sparse_view_num": str(sparse_view_num),
        "resolution": "4", "sh_degree": "2", "iterations_coarse": str(iterations),
        "lora_rank": "32", "sd_base": "", "controlnet": "",
        "SSIM": "", "PSNR": "", "LPIPS": "",
        "dSSIM": f"{a[0]-b[0]:.10f}", "dPSNR": f"{a[1]-b[1]:.10f}", "dLPIPS": f"{a[2]-b[2]:.10f}",
        "vs_stage": "lora_repair_rv51", "gs_dir": f"output/gs_init/{classic_gs}",
        "lora_exp": "", "final_ply": f"output/gaussian_object/{classic_tag}/save/last.ply",
        "source": f"derived {classic_tag} - {vggt_tag}",
        "notes": "classic init repair vs VGGT init repair (both RV51 no CADR)",
    })
    cadr_p = go / "output" / "gs_init" / base_gs / split / f"ours_None__{cadr_tag}" / "results.json"
    if cadr_p.exists():
        c = load(base_gs, cadr_tag, split)
        new_rows.append({
            "recorded_date": today, "row_type": "delta", "scene": "mip360/kitchen",
            "pipeline": "lineB_gaussianobject", "stage": "classic_rv51_vs_vggt_cadr",
            "ckpt_tag": f"{classic_tag}_vs_{cadr_tag}", "split": split, "init_pcd": init_pcd,
            "use_drop": "1", "drop_rate": "0.2",
            "sparse_id": str(sparse_id), "sparse_view_num": str(sparse_view_num),
            "resolution": "4", "sh_degree": "2", "iterations_coarse": str(iterations),
            "lora_rank": "32", "sd_base": "", "controlnet": "",
            "SSIM": "", "PSNR": "", "LPIPS": "",
            "dSSIM": f"{a[0]-c[0]:.10f}", "dPSNR": f"{a[1]-c[1]:.10f}", "dLPIPS": f"{a[2]-c[2]:.10f}",
            "vs_stage": "lora_repair_cadr", "gs_dir": f"output/gs_init/{classic_gs}",
            "lora_exp": "", "final_ply": f"output/gaussian_object/{classic_tag}/save/last.ply",
            "source": f"derived {classic_tag} - {cadr_tag}",
            "notes": "classic init repair vs VGGT+CADR",
        })

rows.extend(new_rows)
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=HEADER, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in HEADER})

s, p, l, _ = load(classic_gs, classic_tag, "test")
vs = load(base_gs, vggt_tag, "test")
print("--- SUMMARY test ---")
print(f"classic_rv51: PSNR={p:.4f} SSIM={s:.6f} LPIPS={l:.6f}")
print(f"vggt_rv51:    PSNR={vs[1]:.4f}")
print(f"dPSNR classic-vggt: {p-vs[1]:+.4f}")
print(f"wrote {path}")
PY
  echo "[stage 2] done"
}

echo "=========================================="
echo "P2 classic→RV51 repair"
require_branch
lineb_activate >/dev/null
echo "START_STAGE=${START_STAGE}"
echo "=========================================="

[[ "${START_STAGE}" =~ ^[0-2]$ ]] || die "stage must be 0..2"
if (( START_STAGE <= 0 )); then stage0_cleanup; fi
if (( START_STAGE <= 1 )); then stage1_repair; fi
if (( START_STAGE <= 2 )); then stage2_csv; fi

echo "Done → $(results_tag "${GS_NAME_FIXED}" "${TAG}")"
