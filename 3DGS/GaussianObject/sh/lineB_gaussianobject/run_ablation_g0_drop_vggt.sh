#!/bin/bash
# G0 classic+Drop / Drop 消融(noDrop) / 仅有 VGGT(delta 复用 kitchen_drop)
# 计划：docs/PLAN_g0_drop_vggt_ablation.md
#
#   bash run_ablation_g0_drop_vggt.sh        # stage 0..4
#   bash run_ablation_g0_drop_vggt.sh 2      # 从 stage 2 跑到 4
#   FORCE_TRAIN=1 / FORCE_HULL=1 / ALLOW_OTHER_BRANCH=1
#
# Agent 不代跑；请在 feat/sd15-stronger-base-lora + .venv-ddgs-vggt 下执行。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

EXPECTED_BRANCH="feat/sd15-stronger-base-lora"
ALLOW_OTHER_BRANCH="${ALLOW_OTHER_BRANCH:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_HULL="${FORCE_HULL:-0}"
START_STAGE="${1:-0}"

CLASSIC_GS_NAME="kitchen_drop_classic"
CLASSIC_INIT="visual_hull_${SPARSE_ID}"
VGGT_INIT="visual_hull_vggt_enhanced_${SPARSE_ID}"
BASELINE_GS_NAME="kitchen_drop"
NODROP_GS_NAME="kitchen"

EVAL_CSV="${REPO_ROOT}/3DGS/eval_results/lineB_kitchen_mip360.csv"
EVAL_README="${REPO_ROOT}/3DGS/eval_results/README.md"

die() { echo "[ERR] $*" >&2; exit 1; }

require_branch() {
  local br
  br="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "git_branch=${br}"
  if [[ "${br}" != "${EXPECTED_BRANCH}" && "${ALLOW_OTHER_BRANCH}" != "1" ]]; then
    die "expect branch ${EXPECTED_BRANCH} (now: ${br}). Set ALLOW_OTHER_BRANCH=1 to override."
  fi
}

# 新训路径禁止踩 kitchen_drop
assert_safe_train_name() {
  local name="$1"
  [[ "${name}" != "${BASELINE_GS_NAME}" ]] || die "refusing to train into protected GS_NAME=${BASELINE_GS_NAME}"
}

results_json() {
  local gs_name="$1"
  echo "${GO}/output/gs_init/${gs_name}/test/ours_10000/results.json"
}

skip_if_trained() {
  local gs_name="$1"
  local rj
  rj="$(results_json "${gs_name}")"
  if [[ -f "${rj}" && "${FORCE_TRAIN}" != "1" ]]; then
    echo "[skip] ${gs_name} already has ${rj}"
    return 0
  fi
  return 1
}

# ---------- stage 0: D1 cleanup ----------
stage0_cleanup() {
  echo "=========================================="
  echo "[stage 0] D1 cleanup repair logs (keep last.ply / configs / gs_init metrics)"
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
    # 保留 save/last.ply 与 configs
    [[ -f "${root}/save/last.ply" ]] && echo "  keep ${root}/save/last.ply" || true
  done
  # 永不碰 kitchen_drop 粗训基线
  [[ -d "${GO}/output/gs_init/${BASELINE_GS_NAME}" ]] || die "missing baseline gs_init/${BASELINE_GS_NAME}"
  echo "[stage 0] done"
}

# ---------- stage 1: classic hull ----------
stage1_classic_hull() {
  echo "=========================================="
  echo "[stage 1] classic hull → ${CLASSIC_INIT}.ply"
  echo "=========================================="
  FORCE_HULL="${FORCE_HULL}" bash "${SCRIPT_DIR}/1b_classic_hull.sh"
}

# ---------- stage 2: G0 classic+Drop ----------
stage2_g0_classic_drop() {
  echo "=========================================="
  echo "[stage 2] G0 classic+Drop → ${CLASSIC_GS_NAME}"
  echo "=========================================="
  assert_safe_train_name "${CLASSIC_GS_NAME}"
  local ply="${DATA}/${CLASSIC_INIT}.ply"
  [[ -f "${ply}" ]] || die "missing ${ply}; run stage 1 first"

  if skip_if_trained "${CLASSIC_GS_NAME}"; then
    return 0
  fi

  # 清掉可能污染的 override，再显式导出
  unset GS_NAME GS_DIR LOO_DIR INIT_PCD MODEL_DIR || true
  export USE_DROP=1
  export GS_NAME="${CLASSIC_GS_NAME}"
  export GS_DIR="output/gs_init/${CLASSIC_GS_NAME}"
  export LOO_DIR="output/gs_init/${CLASSIC_GS_NAME}_loo"
  export INIT_PCD="${CLASSIC_INIT}"
  export SKIP_PATH="${SKIP_PATH:-1}"

  lineb_print_paths
  bash "${SCRIPT_DIR}/2_train_gs.sh"
  bash "${SCRIPT_DIR}/3_render.sh"
  # 省盘：训练脚本已删 tfevents/input.ply；再清可能的中间图
  rm -rf "${GO}/${GS_DIR}/train" "${GO}/${GS_DIR}/test/ours_10000/renders" 2>/dev/null || true
  [[ -f "$(results_json "${CLASSIC_GS_NAME}")" ]] || die "missing results after G0"
  echo "[stage 2] done → $(results_json "${CLASSIC_GS_NAME}")"
}

# ---------- stage 3: VGGT noDrop ----------
stage3_vggt_nodrop() {
  echo "=========================================="
  echo "[stage 3] Drop ablation noDrop → ${NODROP_GS_NAME}"
  echo "=========================================="
  assert_safe_train_name "${NODROP_GS_NAME}"
  local ply="${DATA}/${VGGT_INIT}.ply"
  [[ -f "${ply}" ]] || die "missing VGGT baseline ply ${ply}"
  [[ -f "$(results_json "${BASELINE_GS_NAME}")" ]] || die "missing baseline $(results_json "${BASELINE_GS_NAME}")"

  if skip_if_trained "${NODROP_GS_NAME}"; then
    return 0
  fi

  unset GS_NAME GS_DIR LOO_DIR INIT_PCD MODEL_DIR || true
  export USE_DROP=0
  export GS_NAME="${NODROP_GS_NAME}"
  export GS_DIR="output/gs_init/${NODROP_GS_NAME}"
  export LOO_DIR="output/gs_init/${NODROP_GS_NAME}_loo"
  export INIT_PCD="${VGGT_INIT}"
  export SKIP_PATH="${SKIP_PATH:-1}"

  lineb_print_paths
  bash "${SCRIPT_DIR}/2_train_gs.sh"
  bash "${SCRIPT_DIR}/3_render.sh"
  rm -rf "${GO}/${GS_DIR}/train" "${GO}/${GS_DIR}/test/ours_10000/renders" 2>/dev/null || true
  [[ -f "$(results_json "${NODROP_GS_NAME}")" ]] || die "missing results after noDrop"
  echo "[stage 3] done → $(results_json "${NODROP_GS_NAME}")"
}

# ---------- stage 4: CSV ----------
stage4_write_csv() {
  echo "=========================================="
  echo "[stage 4] persist metrics → ${EVAL_CSV}"
  echo "=========================================="
  local classic_test kitchen_test drop_test
  classic_test="$(results_json "${CLASSIC_GS_NAME}")"
  kitchen_test="$(results_json "${NODROP_GS_NAME}")"
  drop_test="$(results_json "${BASELINE_GS_NAME}")"
  [[ -f "${classic_test}" ]] || die "missing ${classic_test}"
  [[ -f "${kitchen_test}" ]] || die "missing ${kitchen_test}"
  [[ -f "${drop_test}" ]] || die "missing ${drop_test}"

  python3 - "${EVAL_CSV}" "${GO}" "${CLASSIC_GS_NAME}" "${NODROP_GS_NAME}" "${BASELINE_GS_NAME}" \
    "${CLASSIC_INIT}" "${VGGT_INIT}" "${SPARSE_ID}" "${SPARSE_VIEW_NUM}" "${ITERATIONS}" <<'PY'
import csv, json, sys
from datetime import date
from pathlib import Path

csv_path, go, classic_name, nodrop_name, drop_name, classic_init, vggt_init, sparse_id, sparse_view_num, iterations = sys.argv[1:]
go = Path(go)
sparse_id = int(sparse_id)
sparse_view_num = int(sparse_view_num)
iterations = int(iterations)
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

DROP_STAGES = {
    "coarse_gs_classic_drop",
    "coarse_gs_vggt_nodrop",
    "vggt_vs_classic_hull",
    "drop_vs_nodrop",
}

path = Path(csv_path)
rows = []
if path.exists():
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("stage") in DROP_STAGES:
                continue
            rows.append(r)

def abs_row(gs_name, init_pcd, use_drop, stage, split, note):
    ssim, psnr, lpips, src = load_split(gs_name, split)
    return {
        "recorded_date": today,
        "row_type": "absolute",
        "scene": "mip360/kitchen",
        "pipeline": "lineB_gaussianobject",
        "stage": stage,
        "ckpt_tag": "ours_10000",
        "split": split,
        "init_pcd": init_pcd,
        "use_drop": str(use_drop),
        "drop_rate": "0.2" if use_drop else "",
        "sparse_id": str(sparse_id),
        "sparse_view_num": str(sparse_view_num),
        "resolution": "4",
        "sh_degree": "2",
        "iterations_coarse": str(iterations),
        "lora_rank": "32",
        "sd_base": "",
        "controlnet": "",
        "SSIM": f"{ssim:.10f}",
        "PSNR": f"{psnr:.10f}",
        "LPIPS": f"{lpips:.10f}",
        "dSSIM": "",
        "dPSNR": "",
        "dLPIPS": "",
        "vs_stage": "",
        "gs_dir": f"output/gs_init/{gs_name}",
        "lora_exp": "",
        "final_ply": "",
        "source": src,
        "notes": note,
    }

def delta_row(stage, split, a_name, b_name, a_init, use_drop_field, vs_stage, note):
    # delta = A - B  (A is "treatment", B is reference)
    a = load_split(a_name, split)
    b = load_split(b_name, split)
    dssim = a[0] - b[0]
    dpsnr = a[1] - b[1]
    dlpips = a[2] - b[2]
    return {
        "recorded_date": today,
        "row_type": "delta",
        "scene": "mip360/kitchen",
        "pipeline": "lineB_gaussianobject",
        "stage": stage,
        "ckpt_tag": f"{a_name}_vs_{b_name}",
        "split": split,
        "init_pcd": a_init,
        "use_drop": use_drop_field,
        "drop_rate": "0.2",
        "sparse_id": str(sparse_id),
        "sparse_view_num": str(sparse_view_num),
        "resolution": "4",
        "sh_degree": "2",
        "iterations_coarse": str(iterations),
        "lora_rank": "32",
        "sd_base": "",
        "controlnet": "",
        "SSIM": "",
        "PSNR": "",
        "LPIPS": "",
        "dSSIM": f"{dssim:.10f}",
        "dPSNR": f"{dpsnr:.10f}",
        "dLPIPS": f"{dlpips:.10f}",
        "vs_stage": vs_stage,
        "gs_dir": f"output/gs_init/{a_name}",
        "lora_exp": "",
        "final_ply": "",
        "source": f"derived {a_name} - {b_name} ({split}/ours_10000)",
        "notes": note,
    }

new_rows = []
for split in ("test", "all"):
    new_rows.append(abs_row(
        classic_name, classic_init, 1, "coarse_gs_classic_drop", split,
        "G0 classic hull + Drop (R1); INIT=visual_hull_4",
    ))
    new_rows.append(abs_row(
        nodrop_name, vggt_init, 0, "coarse_gs_vggt_nodrop", split,
        "VGGT hull + no Drop; Drop ablation arm",
    ))
    new_rows.append(delta_row(
        "vggt_vs_classic_hull", split, drop_name, classic_name, vggt_init, "1",
        "coarse_gs_classic_drop",
        "VGGT-only gain (R1): kitchen_drop - kitchen_drop_classic; Drop fixed=1",
    ))
    new_rows.append(delta_row(
        "drop_vs_nodrop", split, drop_name, nodrop_name, vggt_init, "",
        "coarse_gs_vggt_nodrop",
        "Drop ablation: kitchen_drop - kitchen; VGGT init fixed",
    ))

rows.extend(new_rows)
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=HEADER, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        out = {k: r.get(k, "") for k in HEADER}
        w.writerow(out)

# print summary for README
for split in ("test",):
    vggt = delta_row("vggt_vs_classic_hull", split, drop_name, classic_name, vggt_init, "1", "", "")
    drop = delta_row("drop_vs_nodrop", split, drop_name, nodrop_name, vggt_init, "", "", "")
    c = load_split(classic_name, split)
    n = load_split(nodrop_name, split)
    d = load_split(drop_name, split)
    print("--- SUMMARY test ---")
    print(f"classic+Drop  PSNR={c[1]:.4f} SSIM={c[0]:.6f} LPIPS={c[2]:.6f}")
    print(f"VGGT+noDrop   PSNR={n[1]:.4f} SSIM={n[0]:.6f} LPIPS={n[2]:.6f}")
    print(f"VGGT+Drop     PSNR={d[1]:.4f} SSIM={d[0]:.6f} LPIPS={d[2]:.6f}")
    print(f"VGGT vs classic dPSNR={float(vggt['dPSNR']):+.4f} dSSIM={float(vggt['dSSIM']):+.6f} dLPIPS={float(vggt['dLPIPS']):+.6f}")
    print(f"Drop vs noDrop dPSNR={float(drop['dPSNR']):+.4f} dSSIM={float(drop['dSSIM']):+.6f} dLPIPS={float(drop['dLPIPS']):+.6f}")
print(f"wrote {path}")
PY

  # 更新 README 待补表：标记 VGGT/Drop 已由本脚本归档
  if [[ -f "${EVAL_README}" ]]; then
    python3 - "${EVAL_README}" <<'PY'
from pathlib import Path
import sys
import re
p = Path(sys.argv[1])
text = p.read_text()
new_block = """| VGGT vs classic hull | **已归档** → CSV `vggt_vs_classic_hull`（[`run_ablation_g0_drop_vggt.sh`](../GaussianObject/sh/lineB_gaussianobject/run_ablation_g0_drop_vggt.sh)，[`PLAN_g0_drop_vggt_ablation.md`](../../docs/PLAN_g0_drop_vggt_ablation.md)） |
| Drop 消融 | **已归档** → CSV `drop_vs_nodrop`（`kitchen` vs `kitchen_drop`） |"""
pat = re.compile(
    r"\| VGGT vs classic hull \|.*?\|\n\| Drop 消融 \|.*?\|",
    re.M,
)
if pat.search(text):
    p.write_text(pat.sub(new_block.strip(), text, count=1))
    print(f"updated {p}")
else:
    print(f"[WARN] README pending rows not found or already updated: {p}")
PY
  fi
  echo "[stage 4] done"
}

# ---------- main ----------
echo "=========================================="
echo "Ablation G0 / Drop / VGGT (R1+D1)"
echo "REPO_ROOT=${REPO_ROOT}"
require_branch
lineb_activate >/dev/null
echo "START_STAGE=${START_STAGE}  FORCE_HULL=${FORCE_HULL}  FORCE_TRAIN=${FORCE_TRAIN}"
echo "classic: GS=${CLASSIC_GS_NAME} INIT=${CLASSIC_INIT}"
echo "nodrop:  GS=${NODROP_GS_NAME} INIT=${VGGT_INIT}"
echo "baseline(reuse): GS=${BASELINE_GS_NAME}"
echo "=========================================="

[[ "${START_STAGE}" =~ ^[0-4]$ ]] || die "stage must be 0..4, got ${START_STAGE}"

if (( START_STAGE <= 0 )); then stage0_cleanup; fi
if (( START_STAGE <= 1 )); then stage1_classic_hull; fi
if (( START_STAGE <= 2 )); then stage2_g0_classic_drop; fi
if (( START_STAGE <= 3 )); then stage3_vggt_nodrop; fi
if (( START_STAGE <= 4 )); then stage4_write_csv; fi

echo "=========================================="
echo "All requested stages finished."
echo "  G0:      ${GO}/output/gs_init/${CLASSIC_GS_NAME}/"
echo "  noDrop:  ${GO}/output/gs_init/${NODROP_GS_NAME}/"
echo "  CSV:     ${EVAL_CSV}"
echo "=========================================="
