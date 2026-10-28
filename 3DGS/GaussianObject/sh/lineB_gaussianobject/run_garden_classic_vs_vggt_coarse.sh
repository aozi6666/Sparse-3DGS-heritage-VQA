#!/bin/bash
# P2：garden 上 classic+Drop vs VGGT+Drop 粗训两格（确认 kitchen −1 dB 是否特例）
# 计划：docs/PLAN_p2_classic_repair_garden.md
#
#   bash run_garden_classic_vs_vggt_coarse.sh
#   bash run_garden_classic_vs_vggt_coarse.sh 3
#   FORCE_TRAIN=1 / FORCE_HULL=1 / ALLOW_OTHER_BRANCH=1
#
# 禁止写入 kitchen_*。失败时打印 bonsai 改跑提示（不自动切换）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

EXPECTED_BRANCH="feat/sd15-stronger-base-lora"
ALLOW_OTHER_BRANCH="${ALLOW_OTHER_BRANCH:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_HULL="${FORCE_HULL:-0}"
START_STAGE="${1:-0}"

SCENE="${SCENE:-garden}"
DATA_SCENE="${GO}/data/mip360/${SCENE}"
CLASSIC_GS="garden_drop_classic"
VGGT_GS="garden_drop"
CLASSIC_INIT="visual_hull_${SPARSE_ID}"
VGGT_INIT="visual_hull_vggt_enhanced_${SPARSE_ID}"
EVAL_CSV="${REPO_ROOT}/3DGS/eval_results/lineB_garden_mip360.csv"

die() { echo "[ERR] $*" >&2; exit 1; }

bonsai_hint() {
  echo "[HINT] garden failed or missing data. Retry with:"
  echo "  SCENE=bonsai bash ${SCRIPT_DIR}/run_garden_classic_vs_vggt_coarse.sh"
}

require_branch() {
  local br
  br="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "git_branch=${br}"
  if [[ "${br}" != "${EXPECTED_BRANCH}" && "${ALLOW_OTHER_BRANCH}" != "1" ]]; then
    die "expect ${EXPECTED_BRANCH} (now ${br}); ALLOW_OTHER_BRANCH=1 to override"
  fi
}

assert_not_kitchen() {
  local name="$1"
  [[ "${name}" != kitchen* ]] || die "refusing kitchen GS_NAME=${name}"
}

results_json() {
  echo "${GO}/output/gs_init/$1/test/ours_10000/results.json"
}

skip_if_trained() {
  local gs="$1" rj
  rj="$(results_json "${gs}")"
  if [[ -f "${rj}" && "${FORCE_TRAIN}" != "1" ]]; then
    echo "[skip] ${rj}"
    return 0
  fi
  return 1
}

train_render() {
  local gs_name="$1" init_pcd="$2"
  assert_not_kitchen "${gs_name}"
  local ply="${DATA_SCENE}/${init_pcd}.ply"
  [[ -f "${ply}" ]] || { echo "[ERR] missing ${ply}"; bonsai_hint; exit 1; }

  if skip_if_trained "${gs_name}"; then
    return 0
  fi

  unset GS_NAME GS_DIR LOO_DIR INIT_PCD MODEL_DIR || true
  export DATA="${DATA_SCENE}"
  export USE_DROP=1
  export GS_NAME="${gs_name}"
  export GS_DIR="output/gs_init/${gs_name}"
  export LOO_DIR="output/gs_init/${gs_name}_loo"
  export INIT_PCD="${init_pcd}"
  export SKIP_PATH="${SKIP_PATH:-1}"

  lineb_print_paths
  bash "${SCRIPT_DIR}/2_train_gs.sh" || { bonsai_hint; exit 1; }
  bash "${SCRIPT_DIR}/3_render.sh" || { bonsai_hint; exit 1; }
  rm -rf "${GO}/${GS_DIR}/train" "${GO}/${GS_DIR}/test/ours_10000/renders" 2>/dev/null || true
  [[ -f "$(results_json "${gs_name}")" ]] || die "missing $(results_json "${gs_name}")"
  echo "[train] done → $(results_json "${gs_name}")"
}

stage0_check() {
  echo "[stage 0] check ${DATA_SCENE}"
  [[ -d "${DATA_SCENE}" ]] || { echo "[ERR] missing ${DATA_SCENE}"; bonsai_hint; exit 1; }
  [[ -f "${DATA_SCENE}/sparse_${SPARSE_ID}.txt" ]] || {
    echo "[ERR] missing sparse_${SPARSE_ID}.txt"; bonsai_hint; exit 1
  }
  echo "  SCENE=${SCENE} OK"
}

stage1_classic_hull() {
  echo "[stage 1] classic hull → ${DATA_SCENE}/${CLASSIC_INIT}.ply"
  export DATA="${DATA_SCENE}"
  FORCE_HULL="${FORCE_HULL}" bash "${SCRIPT_DIR}/1b_classic_hull.sh" || {
    bonsai_hint; exit 1
  }
}

stage2_vggt_hull() {
  echo "[stage 2] VGGT hull → ${DATA_SCENE}/${VGGT_INIT}.ply"
  local ply="${DATA_SCENE}/${VGGT_INIT}.ply"
  if [[ -f "${ply}" && "${FORCE_HULL}" != "1" ]]; then
    echo "[skip] ${ply}"
    return 0
  fi
  # Never touch kitchen enhanced ply
  local kit="${GO}/data/mip360/kitchen/${VGGT_INIT}.ply"
  local kit_mtime=""
  [[ -f "${kit}" ]] && kit_mtime="$(stat -c %Y "${kit}" 2>/dev/null || true)"

  # garden images_2 过大易 OOM；默认 VGGT_RESO=4（仍可用 env 覆盖）。bonsai≈kitchen 尺寸可保持 2。
  if [[ -z "${VGGT_RESO:-}" ]]; then
    if [[ "${SCENE}" == "garden" ]]; then
      export VGGT_RESO=4
    else
      export VGGT_RESO=2
    fi
  fi
  echo "  VGGT_RESO=${VGGT_RESO} (garden OOM 兜底：用 images_${VGGT_RESO})"

  unset GS_NAME GS_DIR INIT_PCD || true
  export DATA="${DATA_SCENE}"
  export INIT_PCD="${VGGT_INIT}"
  export GS_NAME="${VGGT_GS}"
  export GS_DIR="output/gs_init/${VGGT_GS}"
  bash "${SCRIPT_DIR}/1_vggt_hull.sh" || { bonsai_hint; exit 1; }

  if [[ -n "${kit_mtime}" && -f "${kit}" ]]; then
    local now
    now="$(stat -c %Y "${kit}" 2>/dev/null || true)"
    [[ "${now}" == "${kit_mtime}" ]] || die "kitchen ${VGGT_INIT}.ply mtime changed — abort"
  fi
  [[ -f "${ply}" ]] || die "missing ${ply}"
}

stage3_classic_train() {
  echo "[stage 3] coarse classic+Drop → ${CLASSIC_GS}"
  train_render "${CLASSIC_GS}" "${CLASSIC_INIT}"
}

stage4_vggt_train() {
  echo "[stage 4] coarse VGGT+Drop → ${VGGT_GS}"
  train_render "${VGGT_GS}" "${VGGT_INIT}"
}

stage5_csv() {
  echo "[stage 5] write ${EVAL_CSV}"
  local c_rj v_rj
  c_rj="$(results_json "${CLASSIC_GS}")"
  v_rj="$(results_json "${VGGT_GS}")"
  [[ -f "${c_rj}" ]] || die "missing ${c_rj}"
  [[ -f "${v_rj}" ]] || die "missing ${v_rj}"

  python3 - "${EVAL_CSV}" "${GO}" "${SCENE}" "${CLASSIC_GS}" "${VGGT_GS}" \
    "${CLASSIC_INIT}" "${VGGT_INIT}" "${SPARSE_ID}" "${SPARSE_VIEW_NUM}" "${ITERATIONS}" <<'PY'
import csv, json, sys
from datetime import date
from pathlib import Path

csv_path, go, scene, classic_gs, vggt_gs, classic_init, vggt_init, sparse_id, sparse_view_num, iterations = sys.argv[1:]
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

def load(gs, split):
    p = go / "output" / "gs_init" / gs / split / "ours_10000" / "results.json"
    with open(p) as f:
        d = json.load(f)
    return float(d["SSIM"]), float(d["PSNR"]), float(d["LPIPS"]), str(p.relative_to(go))

rows = []
for split in ("test", "all"):
    for gs, init, stage, note in (
        (classic_gs, classic_init, "coarse_gs_classic_drop", "garden classic+Drop"),
        (vggt_gs, vggt_init, "coarse_gs_vggt_drop", "garden VGGT-enhanced+Drop"),
    ):
        s, p, l, src = load(gs, split)
        rows.append({
            "recorded_date": today, "row_type": "absolute", "scene": f"mip360/{scene}",
            "pipeline": "lineB_gaussianobject", "stage": stage, "ckpt_tag": "ours_10000",
            "split": split, "init_pcd": init, "use_drop": "1", "drop_rate": "0.2",
            "sparse_id": str(sparse_id), "sparse_view_num": str(sparse_view_num),
            "resolution": "4", "sh_degree": "2", "iterations_coarse": str(iterations),
            "lora_rank": "32", "sd_base": "", "controlnet": "",
            "SSIM": f"{s:.10f}", "PSNR": f"{p:.10f}", "LPIPS": f"{l:.10f}",
            "dSSIM": "", "dPSNR": "", "dLPIPS": "", "vs_stage": "",
            "gs_dir": f"output/gs_init/{gs}", "lora_exp": "", "final_ply": "",
            "source": src, "notes": note,
        })
    v, c = load(vggt_gs, split), load(classic_gs, split)
    rows.append({
        "recorded_date": today, "row_type": "delta", "scene": f"mip360/{scene}",
        "pipeline": "lineB_gaussianobject", "stage": "vggt_vs_classic_hull",
        "ckpt_tag": f"{vggt_gs}_vs_{classic_gs}", "split": split, "init_pcd": vggt_init,
        "use_drop": "1", "drop_rate": "0.2",
        "sparse_id": str(sparse_id), "sparse_view_num": str(sparse_view_num),
        "resolution": "4", "sh_degree": "2", "iterations_coarse": str(iterations),
        "lora_rank": "32", "sd_base": "", "controlnet": "",
        "SSIM": "", "PSNR": "", "LPIPS": "",
        "dSSIM": f"{v[0]-c[0]:.10f}", "dPSNR": f"{v[1]-c[1]:.10f}", "dLPIPS": f"{v[2]-c[2]:.10f}",
        "vs_stage": "coarse_gs_classic_drop", "gs_dir": f"output/gs_init/{vggt_gs}",
        "lora_exp": "", "final_ply": "",
        "source": f"derived {vggt_gs} - {classic_gs}",
        "notes": "P2: VGGT-enhanced vs classic; Drop=1; compare to kitchen dPSNR≈-0.96",
    })

path = Path(csv_path)
path.parent.mkdir(parents=True, exist_ok=True)
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=HEADER)
    w.writeheader()
    w.writerows(rows)

ct, vt = load(classic_gs, "test"), load(vggt_gs, "test")
print("--- SUMMARY test ---")
print(f"classic+Drop PSNR={ct[1]:.4f}")
print(f"VGGT+Drop    PSNR={vt[1]:.4f}")
print(f"VGGT-classic dPSNR={vt[1]-ct[1]:+.4f}")
print(f"wrote {path}")
PY
}

echo "=========================================="
echo "P2 garden classic vs VGGT coarse"
require_branch
# Rebind scene names if SCENE=bonsai
if [[ "${SCENE}" != "garden" ]]; then
  CLASSIC_GS="${SCENE}_drop_classic"
  VGGT_GS="${SCENE}_drop"
  DATA_SCENE="${GO}/data/mip360/${SCENE}"
  EVAL_CSV="${REPO_ROOT}/3DGS/eval_results/lineB_${SCENE}_mip360.csv"
fi
lineb_activate >/dev/null
echo "SCENE=${SCENE} START_STAGE=${START_STAGE}"
echo "=========================================="

[[ "${START_STAGE}" =~ ^[0-5]$ ]] || die "stage must be 0..5"
if (( START_STAGE <= 0 )); then stage0_check; fi
if (( START_STAGE <= 1 )); then stage1_classic_hull; fi
if (( START_STAGE <= 2 )); then stage2_vggt_hull; fi
if (( START_STAGE <= 3 )); then stage3_classic_train; fi
if (( START_STAGE <= 4 )); then stage4_vggt_train; fi
if (( START_STAGE <= 5 )); then stage5_csv; fi

echo "Done."
echo "  classic: $(results_json "${CLASSIC_GS}")"
echo "  vggt:    $(results_json "${VGGT_GS}")"
echo "  CSV:     ${EVAL_CSV}"
