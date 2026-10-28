#!/bin/bash
# 凑齐 init×Drop 2×2：补 classic+无Drop（kitchen_classic），写满 CSV + ply 统计。
# 计划：docs/PLAN_g0_drop_vggt_ablation.md（R2 格）/ 根 README 叙事
#
#   bash run_ablation_2x2_fill_classic_nodrop.sh      # stage 0..4
#   bash run_ablation_2x2_fill_classic_nodrop.sh 2    # 从 stage 2
#   FORCE_TRAIN=1 / FORCE_HULL=1 / ALLOW_OTHER_BRANCH=1 / SKIP_CLEANUP=1
#
# 已有三格（产物在则 skip）：kitchen_drop_classic / kitchen / kitchen_drop
# 新训：kitchen_classic = classic hull + USE_DROP=0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

EXPECTED_BRANCH="feat/sd15-stronger-base-lora"
ALLOW_OTHER_BRANCH="${ALLOW_OTHER_BRANCH:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_HULL="${FORCE_HULL:-0}"
SKIP_CLEANUP="${SKIP_CLEANUP:-0}"
START_STAGE="${1:-0}"

CLASSIC_DROP_NAME="kitchen_drop_classic"
CLASSIC_NODROP_NAME="kitchen_classic"
VGGT_DROP_NAME="kitchen_drop"
VGGT_NODROP_NAME="kitchen"
CLASSIC_INIT="visual_hull_${SPARSE_ID}"
VGGT_INIT="visual_hull_vggt_enhanced_${SPARSE_ID}"

EVAL_CSV="${REPO_ROOT}/3DGS/eval_results/lineB_kitchen_mip360.csv"
PLY_CSV="${REPO_ROOT}/3DGS/eval_results/ply_init_stats_kitchen.csv"
EVAL_README="${REPO_ROOT}/3DGS/eval_results/README.md"

PROTECTED=(kitchen_drop)

die() { echo "[ERR] $*" >&2; exit 1; }

require_branch() {
  local br
  br="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "git_branch=${br}"
  if [[ "${br}" != "${EXPECTED_BRANCH}" && "${ALLOW_OTHER_BRANCH}" != "1" ]]; then
    die "expect branch ${EXPECTED_BRANCH} (now: ${br}). Set ALLOW_OTHER_BRANCH=1 to override."
  fi
}

assert_safe_train_name() {
  local name="$1" p
  for p in "${PROTECTED[@]}"; do
    [[ "${name}" != "${p}" ]] || die "refusing to train into protected GS_NAME=${p}"
  done
}

results_json() {
  echo "${GO}/output/gs_init/$1/test/ours_10000/results.json"
}

skip_if_trained() {
  local rj
  rj="$(results_json "$1")"
  if [[ -f "${rj}" && "${FORCE_TRAIN}" != "1" ]]; then
    echo "[skip] $1 already has ${rj}"
    return 0
  fi
  return 1
}

train_render() {
  local gs_name="$1" init_pcd="$2" use_drop="$3"
  assert_safe_train_name "${gs_name}"
  local ply="${DATA}/${init_pcd}.ply"
  [[ -f "${ply}" ]] || die "missing ${ply}"

  if skip_if_trained "${gs_name}"; then
    return 0
  fi

  unset GS_NAME GS_DIR LOO_DIR INIT_PCD MODEL_DIR || true
  export USE_DROP="${use_drop}"
  export GS_NAME="${gs_name}"
  export GS_DIR="output/gs_init/${gs_name}"
  export LOO_DIR="output/gs_init/${gs_name}_loo"
  export INIT_PCD="${init_pcd}"
  export SKIP_PATH="${SKIP_PATH:-1}"

  lineb_print_paths
  bash "${SCRIPT_DIR}/2_train_gs.sh"
  bash "${SCRIPT_DIR}/3_render.sh"
  rm -rf "${GO}/${GS_DIR}/train" "${GO}/${GS_DIR}/test/ours_10000/renders" 2>/dev/null || true
  [[ -f "$(results_json "${gs_name}")" ]] || die "missing results for ${gs_name}"
  echo "[train] done → $(results_json "${gs_name}")"
}

# ---------- stage 0 ----------
stage0_cleanup() {
  if [[ "${SKIP_CLEANUP}" == "1" ]]; then
    echo "[stage 0] skip cleanup (SKIP_CLEANUP=1)"
    return 0
  fi
  echo "=========================================="
  echo "[stage 0] D1 cleanup repair logs"
  echo "=========================================="
  local repair_roots=(
    "${GO}/output/gaussian_object/kitchen_drop_rv51"
    "${GO}/output/gaussian_object/kitchen_drop_rv51_conf"
    "${GO}/output/gaussian_object/kitchen_drop_abs16_conf"
    "${GO}/output/gaussian_object/kitchen_drop"
  )
  local root sub
  for root in "${repair_roots[@]}"; do
    [[ -d "${root}" ]] || continue
    for sub in tb_logs csv_logs save/controlnet_out; do
      if [[ -e "${root}/${sub}" ]]; then
        echo "  rm -rf ${root}/${sub}"
        rm -rf "${root}/${sub}"
      fi
    done
  done
  [[ -d "${GO}/output/gs_init/${VGGT_DROP_NAME}" ]] || die "missing baseline gs_init/${VGGT_DROP_NAME}"
  echo "[stage 0] done"
}

# ---------- stage 1 ----------
stage1_hull() {
  echo "=========================================="
  echo "[stage 1] ensure classic hull ${CLASSIC_INIT}.ply"
  echo "=========================================="
  FORCE_HULL="${FORCE_HULL}" bash "${SCRIPT_DIR}/1b_classic_hull.sh"
  [[ -f "${DATA}/${VGGT_INIT}.ply" ]] || die "missing VGGT ply ${DATA}/${VGGT_INIT}.ply"
}

# ---------- stage 2: kitchen_classic (NEW) ----------
stage2_classic_nodrop() {
  echo "=========================================="
  echo "[stage 2] classic+noDrop → ${CLASSIC_NODROP_NAME}"
  echo "=========================================="
  train_render "${CLASSIC_NODROP_NAME}" "${CLASSIC_INIT}" 0
}

# ---------- stage 3: ensure other three (skip if done) ----------
stage3_ensure_others() {
  echo "=========================================="
  echo "[stage 3] ensure other 2×2 cells (skip if present)"
  echo "=========================================="
  train_render "${CLASSIC_DROP_NAME}" "${CLASSIC_INIT}" 1
  train_render "${VGGT_NODROP_NAME}" "${VGGT_INIT}" 0
  # kitchen_drop: never retrain
  [[ -f "$(results_json "${VGGT_DROP_NAME}")" ]] || die "missing baseline $(results_json "${VGGT_DROP_NAME}")"
  echo "[stage 3] baseline ${VGGT_DROP_NAME} OK (read-only)"
}

# ---------- stage 4: ply stats ----------
stage4_ply_stats() {
  echo "=========================================="
  echo "[stage 4] ply stats → ${PLY_CSV}"
  echo "=========================================="
  python3 - "${PLY_CSV}" \
    "${DATA}/${CLASSIC_INIT}.ply" \
    "${DATA}/${VGGT_INIT}.ply" \
    "${CLASSIC_INIT}" "${VGGT_INIT}" <<'PY'
import csv, sys
from datetime import date
from pathlib import Path

out_csv, classic_path, vggt_path, classic_name, vggt_name = sys.argv[1:]

try:
    import open3d as o3d
    import numpy as np
except ImportError as e:
    raise SystemExit(f"need open3d+numpy: {e}")

def stats(path, tag):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"missing ply: {p}")
    pcd = o3d.io.read_point_cloud(str(p))
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros((0, 3))
    n = int(pts.shape[0])
    if n == 0:
        return {
            "recorded_date": date.today().isoformat(),
            "init_pcd": tag,
            "path": str(p),
            "n_points": 0,
            "bbox_dx": "", "bbox_dy": "", "bbox_dz": "",
            "color_var_mean": "",
            "notes": "empty point cloud",
        }
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    extent = mx - mn
    color_var = float(cols.var(axis=0).mean()) if cols.size else ""
    return {
        "recorded_date": date.today().isoformat(),
        "init_pcd": tag,
        "path": str(p),
        "n_points": n,
        "bbox_dx": f"{extent[0]:.6f}",
        "bbox_dy": f"{extent[1]:.6f}",
        "bbox_dz": f"{extent[2]:.6f}",
        "color_var_mean": f"{color_var:.8f}" if color_var != "" else "",
        "notes": "init ply qualitative; no render dumps",
    }

header = ["recorded_date","init_pcd","path","n_points","bbox_dx","bbox_dy","bbox_dz","color_var_mean","notes"]
rows = [stats(classic_path, classic_name), stats(vggt_path, vggt_name)]
Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=header)
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(f"  {r['init_pcd']}: n_points={r['n_points']} bbox=({r['bbox_dx']},{r['bbox_dy']},{r['bbox_dz']}) color_var={r['color_var_mean']}")
print(f"wrote {out_csv}")
PY
  echo "[stage 4] done"
}

# ---------- stage 5: full 2×2 CSV ----------
stage5_write_csv() {
  echo "=========================================="
  echo "[stage 5] persist 2×2 metrics → ${EVAL_CSV}"
  echo "=========================================="
  for n in "${CLASSIC_NODROP_NAME}" "${CLASSIC_DROP_NAME}" "${VGGT_NODROP_NAME}" "${VGGT_DROP_NAME}"; do
    [[ -f "$(results_json "${n}")" ]] || die "missing $(results_json "${n}")"
  done

  python3 - "${EVAL_CSV}" "${GO}" \
    "${CLASSIC_NODROP_NAME}" "${CLASSIC_DROP_NAME}" \
    "${VGGT_NODROP_NAME}" "${VGGT_DROP_NAME}" \
    "${CLASSIC_INIT}" "${VGGT_INIT}" \
    "${SPARSE_ID}" "${SPARSE_VIEW_NUM}" "${ITERATIONS}" <<'PY'
import csv, json, sys
from datetime import date
from pathlib import Path

(
    csv_path, go,
    classic_nd, classic_d, vggt_nd, vggt_d,
    classic_init, vggt_init,
    sparse_id, sparse_view_num, iterations,
) = sys.argv[1:]
go = Path(go)
sparse_id, sparse_view_num, iterations = map(int, (sparse_id, sparse_view_num, iterations))
today = date.today().isoformat()

def load_split(gs_name, split):
    p = go / "output" / "gs_init" / gs_name / split / "ours_10000" / "results.json"
    with open(p) as f:
        d = json.load(f)
    return float(d["SSIM"]), float(d["PSNR"]), float(d["LPIPS"]), str(p.relative_to(go))

HEADER = [
    "recorded_date","row_type","scene","pipeline","stage","ckpt_tag","split","init_pcd",
    "use_drop","drop_rate","sparse_id","sparse_view_num","resolution","sh_degree",
    "iterations_coarse","lora_rank","sd_base","controlnet",
    "SSIM","PSNR","LPIPS","dSSIM","dPSNR","dLPIPS","vs_stage",
    "gs_dir","lora_exp","final_ply","source","notes",
]

REPLACE_STAGES = {
    "coarse_gs_classic_drop",
    "coarse_gs_classic_nodrop",
    "coarse_gs_vggt_nodrop",
    "vggt_vs_classic_hull",
    "vggt_vs_classic_nodrop",
    "drop_vs_nodrop",
    "drop_vs_nodrop_classic",
    "drop_vs_nodrop_vggt",
    "init_drop_interaction",
}

path = Path(csv_path)
rows = []
if path.exists():
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("stage") in REPLACE_STAGES:
                continue
            rows.append(r)

def abs_row(gs_name, init_pcd, use_drop, stage, split, note):
    ssim, psnr, lpips, src = load_split(gs_name, split)
    return {
        "recorded_date": today, "row_type": "absolute", "scene": "mip360/kitchen",
        "pipeline": "lineB_gaussianobject", "stage": stage, "ckpt_tag": "ours_10000",
        "split": split, "init_pcd": init_pcd, "use_drop": str(use_drop),
        "drop_rate": "0.2" if int(use_drop) else "",
        "sparse_id": str(sparse_id), "sparse_view_num": str(sparse_view_num),
        "resolution": "4", "sh_degree": "2", "iterations_coarse": str(iterations),
        "lora_rank": "32", "sd_base": "", "controlnet": "",
        "SSIM": f"{ssim:.10f}", "PSNR": f"{psnr:.10f}", "LPIPS": f"{lpips:.10f}",
        "dSSIM": "", "dPSNR": "", "dLPIPS": "", "vs_stage": "",
        "gs_dir": f"output/gs_init/{gs_name}", "lora_exp": "", "final_ply": "",
        "source": src, "notes": note,
    }

def delta_row(stage, split, a_name, b_name, a_init, use_drop_field, vs_stage, note):
    a, b = load_split(a_name, split), load_split(b_name, split)
    return {
        "recorded_date": today, "row_type": "delta", "scene": "mip360/kitchen",
        "pipeline": "lineB_gaussianobject", "stage": stage,
        "ckpt_tag": f"{a_name}_vs_{b_name}", "split": split, "init_pcd": a_init,
        "use_drop": use_drop_field, "drop_rate": "0.2",
        "sparse_id": str(sparse_id), "sparse_view_num": str(sparse_view_num),
        "resolution": "4", "sh_degree": "2", "iterations_coarse": str(iterations),
        "lora_rank": "32", "sd_base": "", "controlnet": "",
        "SSIM": "", "PSNR": "", "LPIPS": "",
        "dSSIM": f"{a[0]-b[0]:.10f}", "dPSNR": f"{a[1]-b[1]:.10f}", "dLPIPS": f"{a[2]-b[2]:.10f}",
        "vs_stage": vs_stage, "gs_dir": f"output/gs_init/{a_name}",
        "lora_exp": "", "final_ply": "",
        "source": f"derived {a_name} - {b_name} ({split}/ours_10000)", "notes": note,
    }

new_rows = []
for split in ("test", "all"):
    new_rows += [
        abs_row(classic_nd, classic_init, 0, "coarse_gs_classic_nodrop", split,
                "classic hull + no Drop; 2x2 cell"),
        abs_row(classic_d, classic_init, 1, "coarse_gs_classic_drop", split,
                "G0 classic hull + Drop"),
        abs_row(vggt_nd, vggt_init, 0, "coarse_gs_vggt_nodrop", split,
                "VGGT enhanced + no Drop"),
        delta_row("vggt_vs_classic_hull", split, vggt_d, classic_d, vggt_init, "1",
                  "coarse_gs_classic_drop",
                  "VGGT-enhanced vs classic; Drop fixed=1"),
        delta_row("vggt_vs_classic_nodrop", split, vggt_nd, classic_nd, vggt_init, "0",
                  "coarse_gs_classic_nodrop",
                  "VGGT-enhanced vs classic; Drop fixed=0"),
        delta_row("drop_vs_nodrop_vggt", split, vggt_d, vggt_nd, vggt_init, "",
                  "coarse_gs_vggt_nodrop",
                  "Drop vs noDrop; VGGT init fixed"),
        delta_row("drop_vs_nodrop_classic", split, classic_d, classic_nd, classic_init, "",
                  "coarse_gs_classic_nodrop",
                  "Drop vs noDrop; classic init fixed"),
        delta_row("drop_vs_nodrop", split, vggt_d, vggt_nd, vggt_init, "",
                  "coarse_gs_vggt_nodrop",
                  "alias of drop_vs_nodrop_vggt (compat)"),
    ]
    # interaction: (VGGT-classic)|Drop1 - (VGGT-classic)|Drop0
    d1 = load_split(vggt_d, split)
    c1 = load_split(classic_d, split)
    d0 = load_split(vggt_nd, split)
    c0 = load_split(classic_nd, split)
    inter_psnr = (d1[1] - c1[1]) - (d0[1] - c0[1])
    inter_ssim = (d1[0] - c1[0]) - (d0[0] - c0[0])
    inter_lpips = (d1[2] - c1[2]) - (d0[2] - c0[2])
    new_rows.append({
        "recorded_date": today, "row_type": "delta", "scene": "mip360/kitchen",
        "pipeline": "lineB_gaussianobject", "stage": "init_drop_interaction",
        "ckpt_tag": "vggt_effect_Drop1_minus_Drop0", "split": split,
        "init_pcd": vggt_init, "use_drop": "", "drop_rate": "0.2",
        "sparse_id": str(sparse_id), "sparse_view_num": str(sparse_view_num),
        "resolution": "4", "sh_degree": "2", "iterations_coarse": str(iterations),
        "lora_rank": "32", "sd_base": "", "controlnet": "",
        "SSIM": "", "PSNR": "", "LPIPS": "",
        "dSSIM": f"{inter_ssim:.10f}", "dPSNR": f"{inter_psnr:.10f}", "dLPIPS": f"{inter_lpips:.10f}",
        "vs_stage": "vggt_vs_classic_*", "gs_dir": f"output/gs_init/{vggt_d}",
        "lora_exp": "", "final_ply": "",
        "source": "derived (VGGT-classic)|Drop1 - (VGGT-classic)|Drop0",
        "notes": "interaction: does VGGT gap change with Drop",
    })

rows.extend(new_rows)
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=HEADER, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in HEADER})

# summary test
c0 = load_split(classic_nd, "test")
c1 = load_split(classic_d, "test")
v0 = load_split(vggt_nd, "test")
v1 = load_split(vggt_d, "test")
print("--- SUMMARY test 2x2 ---")
print(f"classic+noDrop PSNR={c0[1]:.4f}")
print(f"classic+Drop   PSNR={c1[1]:.4f}")
print(f"VGGT+noDrop    PSNR={v0[1]:.4f}")
print(f"VGGT+Drop      PSNR={v1[1]:.4f}")
print(f"VGGT-classic @Drop0 dPSNR={v0[1]-c0[1]:+.4f}")
print(f"VGGT-classic @Drop1 dPSNR={v1[1]-c1[1]:+.4f}")
print(f"Drop-noDrop @classic dPSNR={c1[1]-c0[1]:+.4f}")
print(f"Drop-noDrop @VGGT    dPSNR={v1[1]-v0[1]:+.4f}")
print(f"wrote {path}")
PY

  echo "[stage 5] done"
}

# ---------- main ----------
echo "=========================================="
echo "2x2 fill: classic+noDrop + ply + CSV"
echo "REPO_ROOT=${REPO_ROOT}"
require_branch
lineb_activate >/dev/null
echo "START_STAGE=${START_STAGE} FORCE_TRAIN=${FORCE_TRAIN} FORCE_HULL=${FORCE_HULL}"
echo "NEW: ${CLASSIC_NODROP_NAME} / ${CLASSIC_INIT} / USE_DROP=0"
echo "=========================================="

[[ "${START_STAGE}" =~ ^[0-5]$ ]] || die "stage must be 0..5, got ${START_STAGE}"

if (( START_STAGE <= 0 )); then stage0_cleanup; fi
if (( START_STAGE <= 1 )); then stage1_hull; fi
if (( START_STAGE <= 2 )); then stage2_classic_nodrop; fi
if (( START_STAGE <= 3 )); then stage3_ensure_others; fi
if (( START_STAGE <= 4 )); then stage4_ply_stats; fi
if (( START_STAGE <= 5 )); then stage5_write_csv; fi

echo "=========================================="
echo "Done."
echo "  classic noDrop: $(results_json "${CLASSIC_NODROP_NAME}")"
echo "  ply CSV: ${PLY_CSV}"
echo "  metrics: ${EVAL_CSV}"
echo "=========================================="
