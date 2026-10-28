#!/bin/bash
# P3：kitchen_drop_depthft_mip360 → RV51 无 CADR + 完整 CADR（复用 RV51 LoRA）
# 计划：docs/PLAN_p3_iter_sweep_depthft_expand.md
#
#   bash run_depthft_mip360_repair_cadr.sh
#   bash run_depthft_mip360_repair_cadr.sh 1
#   FORCE_REPAIR=1 / ALLOW_OTHER_BRANCH=1 / SKIP_CLEANUP=1
#
# 不重训粗训 / LoRA；不覆盖 kitchen_drop_rv51 / kitchen_drop_rv51_conf。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

EXPECTED_BRANCH="feat/sd15-stronger-base-lora"
ALLOW_OTHER_BRANCH="${ALLOW_OTHER_BRANCH:-0}"
FORCE_REPAIR="${FORCE_REPAIR:-0}"
SKIP_CLEANUP="${SKIP_CLEANUP:-0}"
START_STAGE="${1:-0}"

GS_NAME_FIXED="kitchen_drop_depthft_mip360"
GS_DIR_FIXED="output/gs_init/${GS_NAME_FIXED}"
INIT_FIXED="visual_hull_vggt_depthft_mip360_${SPARSE_ID}"
TAG_A0="kitchen_drop_depthft_mip360_rv51"
TAG_A3="kitchen_drop_depthft_mip360_rv51_conf"
RV51_CKPT="./models/Realistic_Vision_V5.1.safetensors"
RV51_LORA="controlnet_finetune/kitchen_drop_rv51"
EVAL_CSV="${REPO_ROOT}/3DGS/eval_results/lineB_kitchen_mip360.csv"
BASE_GS="kitchen_drop"
BASE_A0="kitchen_drop_rv51"
BASE_A3="kitchen_drop_rv51_conf"

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

run_one_repair() {
  local tag="$1" use_conf="$2"
  local rj
  rj="$(results_tag "${GS_NAME_FIXED}" "${tag}")"
  if [[ -f "${rj}" && "${FORCE_REPAIR}" != "1" ]]; then
    echo "[skip] ${rj}"
    strip_repair_logs "${tag}"
    return 0
  fi

  unset REPAIR_TAG BASE_TAG LORA_EXP SD_CKPT USE_CONF_WEIGHT \
        CADR_TRUST_ONLY USE_COV_IN_NEED USE_ADAPTIVE_DENOISE \
        GS_NAME GS_DIR INIT_PCD FINAL_PLY || true

  export GS_NAME="${GS_NAME_FIXED}"
  export GS_DIR="${GS_DIR_FIXED}"
  export INIT_PCD="${INIT_FIXED}"
  export REPAIR_TAG="${tag}"
  export BASE_TAG="${tag}"
  export FINAL_PLY="output/gaussian_object/${tag}/save/last.ply"
  export SD_CKPT="${RV51_CKPT}"
  export LORA_EXP="${RV51_LORA}"
  export USE_CONF_WEIGHT="${use_conf}"
  export CADR_TRUST_ONLY=0
  export USE_COV_IN_NEED=0
  export USE_ADAPTIVE_DENOISE=0
  export SAVE_CONTROLNET_PREVIEW=0
  export SKIP_PATH="${SKIP_PATH:-1}"
  export DATA="${GO}/data/mip360/kitchen"

  lineb_print_paths
  bash "${SCRIPT_DIR}/8_train_repair.sh"
  [[ -f "${GO}/${FINAL_PLY}" ]] || die "missing ${GO}/${FINAL_PLY}"
  bash "${SCRIPT_DIR}/9_render_last.sh"
  [[ -f "${rj}" ]] || die "missing ${rj}"
  strip_repair_logs "${tag}"
  echo "[repair] done → ${rj}"
}

stage0_cleanup() {
  if [[ "${SKIP_CLEANUP}" == "1" ]]; then
    echo "[stage 0] skip cleanup"
    return 0
  fi
  echo "[stage 0] strip depthft repair logs only"
  strip_repair_logs "${TAG_A0}"
  strip_repair_logs "${TAG_A3}"
}

stage1_check() {
  echo "[stage 1] check coarse + LoRA"
  [[ -d "${GO}/${GS_DIR_FIXED}" ]] || die "missing coarse ${GO}/${GS_DIR_FIXED} (run run_depthft_coarse_eval_mip360.sh first)"
  local coarse_rj="${GO}/output/gs_init/${GS_NAME_FIXED}/test/ours_10000/results.json"
  [[ -f "${coarse_rj}" ]] || die "missing ${coarse_rj}"
  [[ -d "${GO}/output/${RV51_LORA}" ]] || die "missing LoRA ${GO}/output/${RV51_LORA}"
  [[ -f "$(results_tag "${BASE_GS}" "${BASE_A0}")" ]] || die "missing A0 baseline $(results_tag "${BASE_GS}" "${BASE_A0}")"
  [[ -f "$(results_tag "${BASE_GS}" "${BASE_A3}")" ]] || die "missing A3 baseline $(results_tag "${BASE_GS}" "${BASE_A3}")"
  echo "  OK coarse=$(python3 -c "import json;print(round(json.load(open('${coarse_rj}'))['PSNR'],3))")"
}

stage2_repair_a0() {
  echo "[stage 2] depthft → RV51 no CADR  tag=${TAG_A0}"
  run_one_repair "${TAG_A0}" 0
}

stage3_repair_a3() {
  echo "[stage 3] depthft → RV51 + CADR  tag=${TAG_A3}"
  run_one_repair "${TAG_A3}" 1
}

stage4_csv() {
  echo "[stage 4] CSV → ${EVAL_CSV}"
  local a0_rj a3_rj
  a0_rj="$(results_tag "${GS_NAME_FIXED}" "${TAG_A0}")"
  a3_rj="$(results_tag "${GS_NAME_FIXED}" "${TAG_A3}")"
  [[ -f "${a0_rj}" ]] || die "missing ${a0_rj}"
  [[ -f "${a3_rj}" ]] || die "missing ${a3_rj}"

  python3 - "${EVAL_CSV}" "${GO}" "${GS_NAME_FIXED}" "${TAG_A0}" "${TAG_A3}" \
    "${BASE_GS}" "${BASE_A0}" "${BASE_A3}" \
    "${INIT_FIXED}" "${SPARSE_ID}" "${SPARSE_VIEW_NUM}" "${ITERATIONS}" <<'PY'
import csv, json, sys
from datetime import date
from pathlib import Path

(
    csv_path, go, depth_gs, tag_a0, tag_a3,
    base_gs, base_a0, base_a3,
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
    "lora_repair_depthft_mip360_rv51",
    "lora_repair_depthft_mip360_cadr",
    "depthft_rv51_vs_vggt_rv51",
    "depthft_cadr_vs_vggt_cadr",
    "depthft_cadr_vs_depthft_rv51",
    "depthft_rv51_vs_depthft_coarse",
}

def load_repair(gs, tag, split):
    p = go / "output" / "gs_init" / gs / split / f"ours_None__{tag}" / "results.json"
    with open(p) as f:
        d = json.load(f)
    return float(d["SSIM"]), float(d["PSNR"]), float(d["LPIPS"]), str(p.relative_to(go))

def load_coarse(gs, split):
    p = go / "output" / "gs_init" / gs / split / "ours_10000" / "results.json"
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
    for tag, stage, note in (
        (tag_a0, "lora_repair_depthft_mip360_rv51", "P3 depthft_mip360 init + RV51 no CADR"),
        (tag_a3, "lora_repair_depthft_mip360_cadr", "P3 depthft_mip360 init + RV51 full CADR"),
    ):
        s, p, l, src = load_repair(depth_gs, tag, split)
        new_rows.append({
            "recorded_date": today, "row_type": "absolute", "scene": "mip360/kitchen",
            "pipeline": "lineB_gaussianobject", "stage": stage,
            "ckpt_tag": f"ours_None__{tag}", "split": split, "init_pcd": init_pcd,
            "use_drop": "1", "drop_rate": "0.2",
            "sparse_id": str(sparse_id), "sparse_view_num": str(sparse_view_num),
            "resolution": "4", "sh_degree": "2", "iterations_coarse": str(iterations),
            "lora_rank": "32", "sd_base": "Realistic_Vision_V5.1.safetensors",
            "controlnet": "control_v11f1e_sd15_tile",
            "SSIM": f"{s:.10f}", "PSNR": f"{p:.10f}", "LPIPS": f"{l:.10f}",
            "dSSIM": "", "dPSNR": "", "dLPIPS": "", "vs_stage": "",
            "gs_dir": f"output/gs_init/{depth_gs}",
            "lora_exp": "output/controlnet_finetune/kitchen_drop_rv51",
            "final_ply": f"output/gaussian_object/{tag}/save/last.ply",
            "source": src, "notes": note,
        })

    d0 = load_repair(depth_gs, tag_a0, split)
    d3 = load_repair(depth_gs, tag_a3, split)
    b0 = load_repair(base_gs, base_a0, split)
    b3 = load_repair(base_gs, base_a3, split)
    coarse = load_coarse(depth_gs, split)

    def delta(stage, ckpt, a, b, vs_stage, notes):
        new_rows.append({
            "recorded_date": today, "row_type": "delta", "scene": "mip360/kitchen",
            "pipeline": "lineB_gaussianobject", "stage": stage, "ckpt_tag": ckpt,
            "split": split, "init_pcd": init_pcd, "use_drop": "1", "drop_rate": "0.2",
            "sparse_id": str(sparse_id), "sparse_view_num": str(sparse_view_num),
            "resolution": "4", "sh_degree": "2", "iterations_coarse": str(iterations),
            "lora_rank": "32", "sd_base": "", "controlnet": "",
            "SSIM": "", "PSNR": "", "LPIPS": "",
            "dSSIM": f"{a[0]-b[0]:.10f}", "dPSNR": f"{a[1]-b[1]:.10f}", "dLPIPS": f"{a[2]-b[2]:.10f}",
            "vs_stage": vs_stage, "gs_dir": f"output/gs_init/{depth_gs}",
            "lora_exp": "", "final_ply": "",
            "source": f"derived {ckpt}", "notes": notes,
        })

    delta("depthft_rv51_vs_vggt_rv51", f"{tag_a0}_vs_{base_a0}", d0, b0,
          "lora_repair_rv51", "depthft init repair vs orig VGGT A0")
    delta("depthft_cadr_vs_vggt_cadr", f"{tag_a3}_vs_{base_a3}", d3, b3,
          "lora_repair_cadr", "depthft init CADR vs orig VGGT A3")
    delta("depthft_cadr_vs_depthft_rv51", f"{tag_a3}_vs_{tag_a0}", d3, d0,
          "lora_repair_depthft_mip360_rv51", "CADR gain on depthft init")
    delta("depthft_rv51_vs_depthft_coarse", f"{tag_a0}_vs_ours_10000", d0, coarse,
          "coarse_gs_vggt_depthft_mip360", "repair vs depthft coarse")

rows.extend(new_rows)
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=HEADER, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in HEADER})

d0t = load_repair(depth_gs, tag_a0, "test")
d3t = load_repair(depth_gs, tag_a3, "test")
b0t = load_repair(base_gs, base_a0, "test")
b3t = load_repair(base_gs, base_a3, "test")
ct = load_coarse(depth_gs, "test")
print("--- SUMMARY test ---")
print(f"depthft coarse:     PSNR={ct[1]:.4f}")
print(f"depthft RV51:       PSNR={d0t[1]:.4f}  vs A0 {d0t[1]-b0t[1]:+.4f}  vs coarse {d0t[1]-ct[1]:+.4f}")
print(f"depthft RV51+CADR:  PSNR={d3t[1]:.4f}  vs A3 {d3t[1]-b3t[1]:+.4f}  vs depthft-A0 {d3t[1]-d0t[1]:+.4f}")
print(f"wrote {path}")
PY
}

echo "=========================================="
echo "P3 depthft_mip360 → RV51 ± CADR"
require_branch
lineb_activate >/dev/null
echo "START_STAGE=${START_STAGE}"
echo "=========================================="

[[ "${START_STAGE}" =~ ^[0-4]$ ]] || die "stage must be 0..4"
if (( START_STAGE <= 0 )); then stage0_cleanup; fi
if (( START_STAGE <= 1 )); then stage1_check; fi
if (( START_STAGE <= 2 )); then stage2_repair_a0; fi
if (( START_STAGE <= 3 )); then stage3_repair_a3; fi
if (( START_STAGE <= 4 )); then stage4_csv; fi

echo "Done → $(results_tag "${GS_NAME_FIXED}" "${TAG_A0}")"
echo "     → $(results_tag "${GS_NAME_FIXED}" "${TAG_A3}")"
