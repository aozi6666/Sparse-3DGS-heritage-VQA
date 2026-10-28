#!/bin/bash
# P3：garden classic（复用）vs depthft_mip360 VGGT 粗训
# 计划：docs/PLAN_p3_iter_sweep_depthft_expand.md
#
#   bash run_garden_classic_vs_depthft_coarse.sh
#   bash run_garden_classic_vs_depthft_coarse.sh 2
#   FORCE_TRAIN=1 / FORCE_HULL=1 / ALLOW_OTHER_BRANCH=1
#
# 不覆盖 garden_drop（原版 enhanced）与 kitchen_*。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

EXPECTED_BRANCH="feat/sd15-stronger-base-lora"
ALLOW_OTHER_BRANCH="${ALLOW_OTHER_BRANCH:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_HULL="${FORCE_HULL:-0}"
START_STAGE="${1:-0}"

SCENE="garden"
DATA_SCENE="${GO}/data/mip360/${SCENE}"
CLASSIC_GS="garden_drop_classic"
DEPTHFT_GS="garden_drop_depthft_mip360"
ORIG_VGGT_GS="garden_drop"
CLASSIC_INIT="visual_hull_${SPARSE_ID}"
DEPTHFT_INIT="visual_hull_vggt_depthft_mip360_${SPARSE_ID}"
EVAL_CSV="${REPO_ROOT}/3DGS/eval_results/lineB_garden_mip360.csv"
VGGT_DEPTH_HEAD_DEFAULT="${VGGT}/models/depth_head_ft_mip360.pt"

die() { echo "[ERR] $*" >&2; exit 1; }

require_branch() {
  local br
  br="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "git_branch=${br}"
  if [[ "${br}" != "${EXPECTED_BRANCH}" && "${ALLOW_OTHER_BRANCH}" != "1" ]]; then
    die "expect ${EXPECTED_BRANCH} (now ${br}); ALLOW_OTHER_BRANCH=1 to override"
  fi
}

results_json() {
  echo "${GO}/output/gs_init/$1/test/ours_10000/results.json"
}

stage0_check() {
  echo "[stage 0] check ${DATA_SCENE}"
  [[ -d "${DATA_SCENE}" ]] || die "missing ${DATA_SCENE}"
  [[ -f "${DATA_SCENE}/sparse_${SPARSE_ID}.txt" ]] || die "missing sparse_${SPARSE_ID}.txt"
  [[ -f "${VGGT_DEPTH_HEAD:-${VGGT_DEPTH_HEAD_DEFAULT}}" ]] || die "missing depth_head"
  [[ -f "$(results_json "${CLASSIC_GS}")" ]] || die "missing classic $(results_json "${CLASSIC_GS}") — run garden classic first"
  echo "  SCENE=${SCENE} classic OK"
}

stage1_depthft_hull() {
  echo "[stage 1] depthft hull → ${DATA_SCENE}/${DEPTHFT_INIT}.ply"
  local ply="${DATA_SCENE}/${DEPTHFT_INIT}.ply"
  if [[ -f "${ply}" && "${FORCE_HULL}" != "1" ]]; then
    echo "[skip] ${ply}"
    return 0
  fi

  # Protect kitchen enhanced + kitchen depthft plys
  local kit_enh="${GO}/data/mip360/kitchen/visual_hull_vggt_enhanced_${SPARSE_ID}.ply"
  local kit_df="${GO}/data/mip360/kitchen/${DEPTHFT_INIT}.ply"
  local kit_enh_m="" kit_df_m=""
  [[ -f "${kit_enh}" ]] && kit_enh_m="$(stat -c %Y "${kit_enh}" 2>/dev/null || true)"
  [[ -f "${kit_df}" ]] && kit_df_m="$(stat -c %Y "${kit_df}" 2>/dev/null || true)"
  local gard_enh="${DATA_SCENE}/visual_hull_vggt_enhanced_${SPARSE_ID}.ply"
  local gard_enh_m=""
  [[ -f "${gard_enh}" ]] && gard_enh_m="$(stat -c %Y "${gard_enh}" 2>/dev/null || true)"

  export DATA="${DATA_SCENE}"
  export INIT_PCD="${DEPTHFT_INIT}"
  export VGGT_DEPTH_HEAD="${VGGT_DEPTH_HEAD:-${VGGT_DEPTH_HEAD_DEFAULT}}"
  export VGGT_RESO="${VGGT_RESO:-4}"
  echo "  VGGT_RESO=${VGGT_RESO} VGGT_DEPTH_HEAD=${VGGT_DEPTH_HEAD}"
  bash "${SCRIPT_DIR}/1_vggt_hull_depthft.sh" || die "depthft hull failed"

  [[ -f "${ply}" ]] || die "missing ${ply}"
  if [[ -n "${kit_enh_m}" && -f "${kit_enh}" ]]; then
    [[ "$(stat -c %Y "${kit_enh}")" == "${kit_enh_m}" ]] || die "kitchen enhanced ply mtime changed"
  fi
  if [[ -n "${kit_df_m}" && -f "${kit_df}" ]]; then
    [[ "$(stat -c %Y "${kit_df}")" == "${kit_df_m}" ]] || die "kitchen depthft ply mtime changed"
  fi
  if [[ -n "${gard_enh_m}" && -f "${gard_enh}" ]]; then
    [[ "$(stat -c %Y "${gard_enh}")" == "${gard_enh_m}" ]] || die "garden enhanced ply mtime changed (baseline must stay)"
  fi
  echo "[hull] done → ${ply}"
}

stage2_classic_reuse() {
  echo "[stage 2] reuse classic $(results_json "${CLASSIC_GS}")"
  [[ -f "$(results_json "${CLASSIC_GS}")" ]] || die "missing classic results"
}

stage3_depthft_train() {
  echo "[stage 3] coarse depthft+Drop → ${DEPTHFT_GS}"
  local rj ply
  rj="$(results_json "${DEPTHFT_GS}")"
  ply="${DATA_SCENE}/${DEPTHFT_INIT}.ply"
  [[ -f "${ply}" ]] || die "missing ${ply}"
  if [[ -f "${rj}" && "${FORCE_TRAIN}" != "1" ]]; then
    echo "[skip] ${rj}"
    return 0
  fi

  unset GS_NAME GS_DIR LOO_DIR INIT_PCD MODEL_DIR ITERATION SAVE_ITERS TEST_ITERS || true
  export DATA="${DATA_SCENE}"
  export USE_DROP=1
  export GS_NAME="${DEPTHFT_GS}"
  export GS_DIR="output/gs_init/${DEPTHFT_GS}"
  export LOO_DIR="output/gs_init/${DEPTHFT_GS}_loo"
  export INIT_PCD="${DEPTHFT_INIT}"
  export SKIP_PATH="${SKIP_PATH:-1}"
  # ensure default single-iter save
  unset SAVE_ITERS TEST_ITERS || true

  lineb_print_paths
  bash "${SCRIPT_DIR}/2_train_gs.sh"
  bash "${SCRIPT_DIR}/3_render.sh"
  rm -rf "${GO}/${GS_DIR}/train" "${GO}/${GS_DIR}/test/ours_10000/renders" 2>/dev/null || true
  [[ -f "${rj}" ]] || die "missing ${rj}"
  echo "[train] done → ${rj}"
}

stage4_csv() {
  echo "[stage 4] write ${EVAL_CSV}"
  local c_rj d_rj o_rj
  c_rj="$(results_json "${CLASSIC_GS}")"
  d_rj="$(results_json "${DEPTHFT_GS}")"
  o_rj="$(results_json "${ORIG_VGGT_GS}")"
  [[ -f "${c_rj}" ]] || die "missing ${c_rj}"
  [[ -f "${d_rj}" ]] || die "missing ${d_rj}"
  [[ -f "${o_rj}" ]] || echo "[WARN] missing orig VGGT ${o_rj}; delta vs garden_drop skipped"

  python3 - "${EVAL_CSV}" "${GO}" "${SCENE}" \
    "${CLASSIC_GS}" "${DEPTHFT_GS}" "${ORIG_VGGT_GS}" \
    "${CLASSIC_INIT}" "${DEPTHFT_INIT}" \
    "${SPARSE_ID}" "${SPARSE_VIEW_NUM}" "${ITERATIONS}" <<'PY'
import csv, json, sys
from datetime import date
from pathlib import Path

(
    csv_path, go, scene, classic_gs, depthft_gs, orig_gs,
    classic_init, depthft_init, sparse_id, sparse_view_num, iterations,
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
# Keep prior P2 rows; replace only P3 depthft-related stages
REPLACE = {
    "coarse_gs_vggt_depthft_mip360",
    "depthft_vs_classic_hull",
    "depthft_vs_orig_vggt_hull",
}

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
            if r.get("stage") in REPLACE:
                continue
            rows.append(r)

new_rows = []
for split in ("test", "all"):
    s, p, l, src = load(depthft_gs, split)
    new_rows.append({
        "recorded_date": today, "row_type": "absolute", "scene": f"mip360/{scene}",
        "pipeline": "lineB_gaussianobject", "stage": "coarse_gs_vggt_depthft_mip360",
        "ckpt_tag": "ours_10000", "split": split, "init_pcd": depthft_init,
        "use_drop": "1", "drop_rate": "0.2",
        "sparse_id": str(sparse_id), "sparse_view_num": str(sparse_view_num),
        "resolution": "4", "sh_degree": "2", "iterations_coarse": str(iterations),
        "lora_rank": "32", "sd_base": "", "controlnet": "",
        "SSIM": f"{s:.10f}", "PSNR": f"{p:.10f}", "LPIPS": f"{l:.10f}",
        "dSSIM": "", "dPSNR": "", "dLPIPS": "", "vs_stage": "",
        "gs_dir": f"output/gs_init/{depthft_gs}", "lora_exp": "", "final_ply": "",
        "source": src, "notes": "P3 garden depthft_mip360+Drop",
    })
    d, c = load(depthft_gs, split), load(classic_gs, split)
    new_rows.append({
        "recorded_date": today, "row_type": "delta", "scene": f"mip360/{scene}",
        "pipeline": "lineB_gaussianobject", "stage": "depthft_vs_classic_hull",
        "ckpt_tag": f"{depthft_gs}_vs_{classic_gs}", "split": split, "init_pcd": depthft_init,
        "use_drop": "1", "drop_rate": "0.2",
        "sparse_id": str(sparse_id), "sparse_view_num": str(sparse_view_num),
        "resolution": "4", "sh_degree": "2", "iterations_coarse": str(iterations),
        "lora_rank": "32", "sd_base": "", "controlnet": "",
        "SSIM": "", "PSNR": "", "LPIPS": "",
        "dSSIM": f"{d[0]-c[0]:.10f}", "dPSNR": f"{d[1]-c[1]:.10f}", "dLPIPS": f"{d[2]-c[2]:.10f}",
        "vs_stage": "coarse_gs_classic_drop", "gs_dir": f"output/gs_init/{depthft_gs}",
        "lora_exp": "", "final_ply": "",
        "source": f"derived {depthft_gs} - {classic_gs}",
        "notes": "P3: depthft-VGGT vs classic",
    })
    orig_p = go / "output" / "gs_init" / orig_gs / split / "ours_10000" / "results.json"
    if orig_p.exists():
        o = load(orig_gs, split)
        new_rows.append({
            "recorded_date": today, "row_type": "delta", "scene": f"mip360/{scene}",
            "pipeline": "lineB_gaussianobject", "stage": "depthft_vs_orig_vggt_hull",
            "ckpt_tag": f"{depthft_gs}_vs_{orig_gs}", "split": split, "init_pcd": depthft_init,
            "use_drop": "1", "drop_rate": "0.2",
            "sparse_id": str(sparse_id), "sparse_view_num": str(sparse_view_num),
            "resolution": "4", "sh_degree": "2", "iterations_coarse": str(iterations),
            "lora_rank": "32", "sd_base": "", "controlnet": "",
            "SSIM": "", "PSNR": "", "LPIPS": "",
            "dSSIM": f"{d[0]-o[0]:.10f}", "dPSNR": f"{d[1]-o[1]:.10f}", "dLPIPS": f"{d[2]-o[2]:.10f}",
            "vs_stage": "coarse_gs_vggt_drop", "gs_dir": f"output/gs_init/{depthft_gs}",
            "lora_exp": "", "final_ply": "",
            "source": f"derived {depthft_gs} - {orig_gs}",
            "notes": "P3: depthft vs orig VGGT-enhanced",
        })

rows.extend(new_rows)
path.parent.mkdir(parents=True, exist_ok=True)
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=HEADER, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in HEADER})

ct, dt = load(classic_gs, "test"), load(depthft_gs, "test")
print("--- SUMMARY test ---")
print(f"classic+Drop     PSNR={ct[1]:.4f}")
print(f"depthft+Drop     PSNR={dt[1]:.4f}")
print(f"depthft-classic  dPSNR={dt[1]-ct[1]:+.4f}")
op = go / "output" / "gs_init" / orig_gs / "test" / "ours_10000" / "results.json"
if op.exists():
    ot = load(orig_gs, "test")
    print(f"orig VGGT+Drop   PSNR={ot[1]:.4f}")
    print(f"depthft-orig     dPSNR={dt[1]-ot[1]:+.4f}")
print(f"wrote {path}")
PY
}

echo "=========================================="
echo "P3 garden classic vs depthft_mip360"
require_branch
lineb_activate >/dev/null
echo "START_STAGE=${START_STAGE}"
echo "=========================================="

[[ "${START_STAGE}" =~ ^[0-4]$ ]] || die "stage must be 0..4"
if (( START_STAGE <= 0 )); then stage0_check; fi
if (( START_STAGE <= 1 )); then stage1_depthft_hull; fi
if (( START_STAGE <= 2 )); then stage2_classic_reuse; fi
if (( START_STAGE <= 3 )); then stage3_depthft_train; fi
if (( START_STAGE <= 4 )); then stage4_csv; fi

echo "Done."
echo "  classic: $(results_json "${CLASSIC_GS}")"
echo "  depthft: $(results_json "${DEPTHFT_GS}")"
echo "  CSV:     ${EVAL_CSV}"
