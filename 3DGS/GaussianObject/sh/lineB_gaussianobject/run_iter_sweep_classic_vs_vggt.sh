#!/bin/bash
# P3：kitchen classic vs VGGT 粗训 PSNR–iter 轻量 sweep（3k/5k/7k/10k）
# 计划：docs/PLAN_p3_iter_sweep_depthft_expand.md
#
#   bash run_iter_sweep_classic_vs_vggt.sh
#   bash run_iter_sweep_classic_vs_vggt.sh 1
#   FORCE_SWEEP=1 / ALLOW_OTHER_BRANCH=1
#
# 独立目录，不覆盖 kitchen_drop / kitchen_drop_classic 基线。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

EXPECTED_BRANCH="feat/sd15-stronger-base-lora"
ALLOW_OTHER_BRANCH="${ALLOW_OTHER_BRANCH:-0}"
FORCE_SWEEP="${FORCE_SWEEP:-0}"
START_STAGE="${1:-0}"

DATA_KITCHEN="${GO}/data/mip360/kitchen"
CLASSIC_GS="kitchen_drop_classic_sweep"
VGGT_GS="kitchen_drop_sweep"
CLASSIC_INIT="visual_hull_${SPARSE_ID}"
VGGT_INIT="visual_hull_vggt_enhanced_${SPARSE_ID}"
SWEEP_ITERS=(3000 5000 7000 10000)
EVAL_CSV="${REPO_ROOT}/3DGS/eval_results/lineB_kitchen_iter_sweep.csv"

die() { echo "[ERR] $*" >&2; exit 1; }

require_branch() {
  local br
  br="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "git_branch=${br}"
  if [[ "${br}" != "${EXPECTED_BRANCH}" && "${ALLOW_OTHER_BRANCH}" != "1" ]]; then
    die "expect ${EXPECTED_BRANCH} (now ${br}); ALLOW_OTHER_BRANCH=1 to override"
  fi
}

results_iter() {
  echo "${GO}/output/gs_init/$1/test/ours_$2/results.json"
}

all_iters_done() {
  local gs="$1" it
  for it in "${SWEEP_ITERS[@]}"; do
    [[ -f "$(results_iter "${gs}" "${it}")" ]] || return 1
  done
  return 0
}

strip_mid_renders() {
  local gs="$1" it
  local root="${GO}/output/gs_init/${gs}"
  for it in "${SWEEP_ITERS[@]}"; do
    rm -rf "${root}/test/ours_${it}/renders" "${root}/test/ours_${it}/gt" 2>/dev/null || true
    rm -rf "${root}/all/ours_${it}/renders" "${root}/all/ours_${it}/gt" 2>/dev/null || true
    rm -rf "${root}/train" 2>/dev/null || true
  done
  # keep only last ply under point_cloud if present
  local pc="${root}/point_cloud"
  if [[ -d "${pc}" ]]; then
    local d
    for d in "${pc}"/iteration_*; do
      [[ -d "${d}" ]] || continue
      case "$(basename "${d}")" in
        iteration_10000) ;;
        *) rm -rf "${d}" 2>/dev/null || true ;;
      esac
    done
  fi
}

train_and_render_sweep() {
  local gs_name="$1" init_pcd="$2"
  local ply="${DATA_KITCHEN}/${init_pcd}.ply"
  [[ -f "${ply}" ]] || die "missing ${ply}"

  if all_iters_done "${gs_name}" && [[ "${FORCE_SWEEP}" != "1" ]]; then
    echo "[skip] ${gs_name} all iters present"
    strip_mid_renders "${gs_name}"
    return 0
  fi

  unset GS_NAME GS_DIR LOO_DIR INIT_PCD MODEL_DIR ITERATION || true
  export DATA="${DATA_KITCHEN}"
  export USE_DROP=1
  export GS_NAME="${gs_name}"
  export GS_DIR="output/gs_init/${gs_name}"
  export LOO_DIR="output/gs_init/${gs_name}_loo"
  export INIT_PCD="${init_pcd}"
  export ITERATIONS=10000
  export SAVE_ITERS="3000 5000 7000 10000"
  export TEST_ITERS="3000 5000 7000 10000"
  export SKIP_PATH="${SKIP_PATH:-1}"

  if [[ "${FORCE_SWEEP}" == "1" ]]; then
    rm -rf "${GO}/${GS_DIR}" 2>/dev/null || true
  fi

  lineb_print_paths
  # Train once if any mid ply missing
  local need_train=0
  local it
  for it in "${SWEEP_ITERS[@]}"; do
    if [[ ! -d "${GO}/${GS_DIR}/point_cloud/iteration_${it}" && ! -f "$(results_iter "${gs_name}" "${it}")" ]]; then
      need_train=1
      break
    fi
  done
  if [[ "${need_train}" == "1" || "${FORCE_SWEEP}" == "1" ]]; then
    bash "${SCRIPT_DIR}/2_train_gs.sh"
  else
    echo "[skip] train ${gs_name} (plys present)"
  fi

  for it in "${SWEEP_ITERS[@]}"; do
    local rj
    rj="$(results_iter "${gs_name}" "${it}")"
    if [[ -f "${rj}" && "${FORCE_SWEEP}" != "1" ]]; then
      echo "[skip] render ${rj}"
      continue
    fi
    export ITERATION="${it}"
    bash "${SCRIPT_DIR}/3_render.sh"
    [[ -f "${rj}" ]] || die "missing ${rj}"
  done
  unset ITERATION SAVE_ITERS TEST_ITERS || true
  strip_mid_renders "${gs_name}"
  echo "[train] done → ${gs_name}"
}

stage0_check() {
  echo "[stage 0] check init plys"
  [[ -f "${DATA_KITCHEN}/${CLASSIC_INIT}.ply" ]] || die "missing classic ply"
  [[ -f "${DATA_KITCHEN}/${VGGT_INIT}.ply" ]] || die "missing VGGT enhanced ply"
  echo "  OK"
}

stage1_classic() {
  echo "[stage 1] classic sweep → ${CLASSIC_GS}"
  train_and_render_sweep "${CLASSIC_GS}" "${CLASSIC_INIT}"
}

stage2_vggt() {
  echo "[stage 2] VGGT sweep → ${VGGT_GS}"
  train_and_render_sweep "${VGGT_GS}" "${VGGT_INIT}"
}

stage3_csv() {
  echo "[stage 3] write ${EVAL_CSV}"
  local gs it
  for gs in "${CLASSIC_GS}" "${VGGT_GS}"; do
    for it in "${SWEEP_ITERS[@]}"; do
      [[ -f "$(results_iter "${gs}" "${it}")" ]] || die "missing $(results_iter "${gs}" "${it}")"
    done
  done

  python3 - "${EVAL_CSV}" "${GO}" "${CLASSIC_GS}" "${VGGT_GS}" \
    "${CLASSIC_INIT}" "${VGGT_INIT}" "${SPARSE_ID}" "${SPARSE_VIEW_NUM}" <<'PY'
import csv, json, sys
from datetime import date
from pathlib import Path

csv_path, go, classic_gs, vggt_gs, classic_init, vggt_init, sparse_id, sparse_view_num = sys.argv[1:]
go = Path(go)
sparse_id, sparse_view_num = int(sparse_id), int(sparse_view_num)
today = date.today().isoformat()
ITERS = [3000, 5000, 7000, 10000]

HEADER = [
    "recorded_date","row_type","scene","pipeline","stage","ckpt_tag","split","init_pcd",
    "use_drop","drop_rate","sparse_id","sparse_view_num","resolution","sh_degree",
    "iterations_coarse","lora_rank","sd_base","controlnet",
    "SSIM","PSNR","LPIPS","dSSIM","dPSNR","dLPIPS","vs_stage",
    "gs_dir","lora_exp","final_ply","source","notes",
]

def load(gs, it, split):
    p = go / "output" / "gs_init" / gs / split / f"ours_{it}" / "results.json"
    with open(p) as f:
        d = json.load(f)
    return float(d["SSIM"]), float(d["PSNR"]), float(d["LPIPS"]), str(p.relative_to(go))

rows = []
for split in ("test", "all"):
    for it in ITERS:
        for gs, init, stage, note in (
            (classic_gs, classic_init, "iter_sweep_classic_drop", "P3 PSNR-vs-iter classic+Drop"),
            (vggt_gs, vggt_init, "iter_sweep_vggt_drop", "P3 PSNR-vs-iter VGGT-enhanced+Drop"),
        ):
            s, p, l, src = load(gs, it, split)
            rows.append({
                "recorded_date": today, "row_type": "absolute", "scene": "mip360/kitchen",
                "pipeline": "lineB_gaussianobject", "stage": stage,
                "ckpt_tag": f"ours_{it}", "split": split, "init_pcd": init,
                "use_drop": "1", "drop_rate": "0.2",
                "sparse_id": str(sparse_id), "sparse_view_num": str(sparse_view_num),
                "resolution": "4", "sh_degree": "2", "iterations_coarse": str(it),
                "lora_rank": "32", "sd_base": "", "controlnet": "",
                "SSIM": f"{s:.10f}", "PSNR": f"{p:.10f}", "LPIPS": f"{l:.10f}",
                "dSSIM": "", "dPSNR": "", "dLPIPS": "", "vs_stage": "",
                "gs_dir": f"output/gs_init/{gs}", "lora_exp": "", "final_ply": "",
                "source": src, "notes": note,
            })
        v, c = load(vggt_gs, it, split), load(classic_gs, it, split)
        rows.append({
            "recorded_date": today, "row_type": "delta", "scene": "mip360/kitchen",
            "pipeline": "lineB_gaussianobject", "stage": "iter_sweep_vggt_vs_classic",
            "ckpt_tag": f"ours_{it}_vggt_vs_classic", "split": split, "init_pcd": vggt_init,
            "use_drop": "1", "drop_rate": "0.2",
            "sparse_id": str(sparse_id), "sparse_view_num": str(sparse_view_num),
            "resolution": "4", "sh_degree": "2", "iterations_coarse": str(it),
            "lora_rank": "32", "sd_base": "", "controlnet": "",
            "SSIM": "", "PSNR": "", "LPIPS": "",
            "dSSIM": f"{v[0]-c[0]:.10f}", "dPSNR": f"{v[1]-c[1]:.10f}", "dLPIPS": f"{v[2]-c[2]:.10f}",
            "vs_stage": "iter_sweep_classic_drop", "gs_dir": f"output/gs_init/{vggt_gs}",
            "lora_exp": "", "final_ply": "",
            "source": f"derived {vggt_gs} - {classic_gs} @ {it}",
            "notes": "P3: VGGT-classic dPSNR at each iter",
        })

path = Path(csv_path)
path.parent.mkdir(parents=True, exist_ok=True)
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=HEADER)
    w.writeheader()
    w.writerows(rows)

print("--- SUMMARY test PSNR ---")
print(f"{'iter':>6}  {'classic':>8}  {'vggt':>8}  {'dPSNR':>8}")
for it in ITERS:
    c, v = load(classic_gs, it, "test"), load(vggt_gs, it, "test")
    print(f"{it:6d}  {c[1]:8.4f}  {v[1]:8.4f}  {v[1]-c[1]:+8.4f}")
print(f"wrote {path}")
PY
}

echo "=========================================="
echo "P3 iter sweep classic vs VGGT"
require_branch
lineb_activate >/dev/null
echo "START_STAGE=${START_STAGE}"
echo "=========================================="

[[ "${START_STAGE}" =~ ^[0-3]$ ]] || die "stage must be 0..3"
if (( START_STAGE <= 0 )); then stage0_check; fi
if (( START_STAGE <= 1 )); then stage1_classic; fi
if (( START_STAGE <= 2 )); then stage2_vggt; fi
if (( START_STAGE <= 3 )); then stage3_csv; fi

echo "Done → ${EVAL_CSV}"
