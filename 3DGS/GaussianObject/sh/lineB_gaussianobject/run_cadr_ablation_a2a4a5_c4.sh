#!/bin/bash
# CADR A-full（A2/A4/A5）+ abs16 无 CADR（C4）；结果-only 省盘
# 计划：docs/PLAN_cadr_ablation_c4.md
#
#   bash run_cadr_ablation_a2a4a5_c4.sh      # stage 0..5
#   bash run_cadr_ablation_a2a4a5_c4.sh 2    # 从 A4
#   FORCE_REPAIR=1 / ALLOW_OTHER_BRANCH=1 / SKIP_CLEANUP=1
#
# 复用：RV51 LoRA + kitchen_drop init；A3(rv51_conf) 不重训
# 新 tag：rv51_trust / rv51_cov / rv51_aden / abs16（无 conf）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

EXPECTED_BRANCH="feat/sd15-stronger-base-lora"
ALLOW_OTHER_BRANCH="${ALLOW_OTHER_BRANCH:-0}"
FORCE_REPAIR="${FORCE_REPAIR:-0}"
SKIP_CLEANUP="${SKIP_CLEANUP:-0}"
START_STAGE="${1:-0}"

EVAL_CSV="${REPO_ROOT}/3DGS/eval_results/lineB_kitchen_mip360.csv"
GS_NAME_FIXED="kitchen_drop"
GS_DIR_FIXED="output/gs_init/${GS_NAME_FIXED}"
INIT_FIXED="visual_hull_vggt_enhanced_${SPARSE_ID}"

RV51_CKPT="./models/Realistic_Vision_V5.1.safetensors"
RV51_LORA="controlnet_finetune/kitchen_drop_rv51"
ABS16_CKPT="./models/absolutereality_v16.safetensors"
ABS16_LORA="controlnet_finetune/kitchen_drop_abs16"

die() { echo "[ERR] $*" >&2; exit 1; }

require_branch() {
  local br
  br="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "git_branch=${br}"
  if [[ "${br}" != "${EXPECTED_BRANCH}" && "${ALLOW_OTHER_BRANCH}" != "1" ]]; then
    die "expect ${EXPECTED_BRANCH} (now ${br}); ALLOW_OTHER_BRANCH=1 to override"
  fi
}

results_for_tag() {
  echo "${GO}/${GS_DIR_FIXED}/test/ours_None__$1/results.json"
}

skip_if_done() {
  local tag="$1" rj
  rj="$(results_for_tag "${tag}")"
  if [[ -f "${rj}" && "${FORCE_REPAIR}" != "1" ]]; then
    echo "[skip] ${tag} → ${rj}"
    return 0
  fi
  return 1
}

# 只留 last.ply / configs；删日志与预览图
strip_repair_logs() {
  local tag="$1"
  local root="${GO}/output/gaussian_object/${tag}"
  [[ -d "${root}" ]] || return 0
  local sub
  for sub in tb_logs csv_logs save/controlnet_out; do
    if [[ -e "${root}/${sub}" ]]; then
      echo "  rm -rf ${root}/${sub}"
      rm -rf "${root}/${sub}"
    fi
  done
  # lightning 偶发 events
  rm -f "${root}"/events.out.tfevents.* 2>/dev/null || true
  [[ -f "${root}/save/last.ply" ]] && echo "  keep ${root}/save/last.ply" || true
}

run_repair_render() {
  local tag="$1"
  shift
  # remaining: env assignments as KEY=VAL ...
  echo "=========================================="
  echo "[repair] tag=${tag}"
  echo "=========================================="
  if skip_if_done "${tag}"; then
    strip_repair_logs "${tag}"
    return 0
  fi

  unset REPAIR_TAG BASE_TAG LORA_EXP SD_CKPT USE_CONF_WEIGHT \
        CADR_TRUST_ONLY USE_COV_IN_NEED USE_ADAPTIVE_DENOISE SAVE_CONTROLNET_PREVIEW \
        GS_NAME GS_DIR INIT_PCD FINAL_PLY || true

  export GS_NAME="${GS_NAME_FIXED}"
  export GS_DIR="${GS_DIR_FIXED}"
  export INIT_PCD="${INIT_FIXED}"
  export REPAIR_TAG="${tag}"
  export FINAL_PLY="output/gaussian_object/${tag}/save/last.ply"
  export SAVE_CONTROLNET_PREVIEW="${SAVE_CONTROLNET_PREVIEW:-0}"
  export SKIP_PATH="${SKIP_PATH:-1}"

  local kv
  for kv in "$@"; do
    # shellcheck disable=SC2163
    export "${kv}"
  done

  lineb_print_paths
  echo "  CADR_TRUST_ONLY=${CADR_TRUST_ONLY:-0} USE_COV_IN_NEED=${USE_COV_IN_NEED:-0} USE_ADAPTIVE_DENOISE=${USE_ADAPTIVE_DENOISE:-0}"

  bash "${SCRIPT_DIR}/8_train_repair.sh"
  [[ -f "${GO}/${FINAL_PLY}" ]] || die "missing ${GO}/${FINAL_PLY}"
  bash "${SCRIPT_DIR}/9_render_last.sh"
  [[ -f "$(results_for_tag "${tag}")" ]] || die "missing $(results_for_tag "${tag}")"
  strip_repair_logs "${tag}"
  echo "[repair] done → $(results_for_tag "${tag}")"
}

stage0_cleanup() {
  if [[ "${SKIP_CLEANUP}" == "1" ]]; then
    echo "[stage 0] skip (SKIP_CLEANUP=1)"
    return 0
  fi
  echo "=========================================="
  echo "[stage 0] D1 strip all known repair logs"
  echo "=========================================="
  local tag
  for tag in \
    kitchen_drop kitchen_drop_rv51 kitchen_drop_rv51_conf \
    kitchen_drop_rv51_trust kitchen_drop_rv51_cov kitchen_drop_rv51_aden \
    kitchen_drop_abs16 kitchen_drop_abs16_conf
  do
    strip_repair_logs "${tag}"
  done
  [[ -d "${GO}/${GS_DIR_FIXED}" ]] || die "missing ${GO}/${GS_DIR_FIXED}"
  [[ -d "${GO}/output/${RV51_LORA}" ]] || die "missing LoRA ${GO}/output/${RV51_LORA}"
  [[ -d "${GO}/output/${ABS16_LORA}" ]] || die "missing LoRA ${GO}/output/${ABS16_LORA}"
  echo "[stage 0] done"
}

stage1_a2() {
  run_repair_render "kitchen_drop_rv51_trust" \
    "SD_CKPT=${RV51_CKPT}" \
    "LORA_EXP=${RV51_LORA}" \
    "BASE_TAG=kitchen_drop_rv51" \
    "USE_CONF_WEIGHT=1" \
    "CADR_TRUST_ONLY=1" \
    "USE_COV_IN_NEED=0" \
    "USE_ADAPTIVE_DENOISE=0"
}

stage2_a4() {
  run_repair_render "kitchen_drop_rv51_cov" \
    "SD_CKPT=${RV51_CKPT}" \
    "LORA_EXP=${RV51_LORA}" \
    "BASE_TAG=kitchen_drop_rv51" \
    "USE_CONF_WEIGHT=1" \
    "CADR_TRUST_ONLY=0" \
    "USE_COV_IN_NEED=1" \
    "USE_ADAPTIVE_DENOISE=0"
}

stage3_a5() {
  run_repair_render "kitchen_drop_rv51_aden" \
    "SD_CKPT=${RV51_CKPT}" \
    "LORA_EXP=${RV51_LORA}" \
    "BASE_TAG=kitchen_drop_rv51" \
    "USE_CONF_WEIGHT=1" \
    "CADR_TRUST_ONLY=0" \
    "USE_COV_IN_NEED=0" \
    "USE_ADAPTIVE_DENOISE=1"
}

stage4_c4() {
  run_repair_render "kitchen_drop_abs16" \
    "SD_CKPT=${ABS16_CKPT}" \
    "LORA_EXP=${ABS16_LORA}" \
    "BASE_TAG=kitchen_drop_abs16" \
    "USE_CONF_WEIGHT=0" \
    "CADR_TRUST_ONLY=0" \
    "USE_COV_IN_NEED=0" \
    "USE_ADAPTIVE_DENOISE=0"
}

stage5_csv() {
  echo "=========================================="
  echo "[stage 5] persist CADR/C4 metrics → ${EVAL_CSV}"
  echo "=========================================="
  local tags=(kitchen_drop_rv51_trust kitchen_drop_rv51_cov kitchen_drop_rv51_aden kitchen_drop_abs16)
  local t
  for t in "${tags[@]}"; do
    [[ -f "$(results_for_tag "${t}")" ]] || die "missing $(results_for_tag "${t}")"
  done
  # baselines
  [[ -f "$(results_for_tag kitchen_drop_rv51)" ]] || die "missing A0 $(results_for_tag kitchen_drop_rv51)"
  [[ -f "$(results_for_tag kitchen_drop_rv51_conf)" ]] || die "missing A3 $(results_for_tag kitchen_drop_rv51_conf)"
  [[ -f "$(results_for_tag kitchen_drop_abs16_conf)" ]] || echo "[WARN] missing abs16_conf (appendix CADR); C4 vs conf delta skipped if absent"

  python3 - "${EVAL_CSV}" "${GO}" "${GS_DIR_FIXED}" "${INIT_FIXED}" \
    "${SPARSE_ID}" "${SPARSE_VIEW_NUM}" "${ITERATIONS}" <<'PY'
import csv, json, sys
from datetime import date
from pathlib import Path

csv_path, go, gs_dir, init_pcd, sparse_id, sparse_view_num, iterations = sys.argv[1:]
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
    "lora_repair_rv51_trust", "lora_repair_rv51_cov", "lora_repair_rv51_aden",
    "lora_repair_abs16", "cadr_a2_vs_a0", "cadr_a4_vs_a3", "cadr_a5_vs_a3",
    "cadr_a2_vs_a3", "abs16_nocadr_vs_rv51", "abs16_nocadr_vs_abs16_cadr",
}

def load_tag(tag, split):
    p = go / gs_dir / split / f"ours_None__{tag}" / "results.json"
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

def abs_row(tag, stage, sd_base, lora_exp, split, note, use_conf="1"):
    ssim, psnr, lpips, src = load_tag(tag, split)
    return {
        "recorded_date": today, "row_type": "absolute", "scene": "mip360/kitchen",
        "pipeline": "lineB_gaussianobject", "stage": stage, "ckpt_tag": f"ours_None__{tag}",
        "split": split, "init_pcd": init_pcd, "use_drop": "1", "drop_rate": "0.2",
        "sparse_id": str(sparse_id), "sparse_view_num": str(sparse_view_num),
        "resolution": "4", "sh_degree": "2", "iterations_coarse": str(iterations),
        "lora_rank": "32", "sd_base": sd_base, "controlnet": "control_v11f1e_sd15_tile",
        "SSIM": f"{ssim:.10f}", "PSNR": f"{psnr:.10f}", "LPIPS": f"{lpips:.10f}",
        "dSSIM": "", "dPSNR": "", "dLPIPS": "", "vs_stage": "",
        "gs_dir": gs_dir, "lora_exp": f"output/{lora_exp}",
        "final_ply": f"output/gaussian_object/{tag}/save/last.ply",
        "source": src, "notes": note,
    }

def delta_row(stage, split, a_tag, b_tag, note, vs_stage):
    a, b = load_tag(a_tag, split), load_tag(b_tag, split)
    return {
        "recorded_date": today, "row_type": "delta", "scene": "mip360/kitchen",
        "pipeline": "lineB_gaussianobject", "stage": stage,
        "ckpt_tag": f"{a_tag}_vs_{b_tag}", "split": split, "init_pcd": init_pcd,
        "use_drop": "1", "drop_rate": "0.2",
        "sparse_id": str(sparse_id), "sparse_view_num": str(sparse_view_num),
        "resolution": "4", "sh_degree": "2", "iterations_coarse": str(iterations),
        "lora_rank": "32", "sd_base": "", "controlnet": "",
        "SSIM": "", "PSNR": "", "LPIPS": "",
        "dSSIM": f"{a[0]-b[0]:.10f}", "dPSNR": f"{a[1]-b[1]:.10f}", "dLPIPS": f"{a[2]-b[2]:.10f}",
        "vs_stage": vs_stage, "gs_dir": gs_dir, "lora_exp": "",
        "final_ply": f"output/gaussian_object/{a_tag}/save/last.ply",
        "source": f"derived {a_tag} - {b_tag}", "notes": note,
    }

specs = [
    ("kitchen_drop_rv51_trust", "lora_repair_rv51_trust", "Realistic_Vision_V5.1.safetensors",
     "controlnet_finetune/kitchen_drop_rv51", "A2 CADR trust-only (need=1)"),
    ("kitchen_drop_rv51_cov", "lora_repair_rv51_cov", "Realistic_Vision_V5.1.safetensors",
     "controlnet_finetune/kitchen_drop_rv51", "A4 CADR + use_cov_in_need"),
    ("kitchen_drop_rv51_aden", "lora_repair_rv51_aden", "Realistic_Vision_V5.1.safetensors",
     "controlnet_finetune/kitchen_drop_rv51", "A5 CADR + use_adaptive_denoise"),
    ("kitchen_drop_abs16", "lora_repair_abs16", "absolutereality_v16.safetensors",
     "controlnet_finetune/kitchen_drop_abs16", "C4 abs16 repair without CADR"),
]

new_rows = []
for split in ("test", "all"):
    for tag, stage, sd, lora, note in specs:
        new_rows.append(abs_row(tag, stage, sd, lora, split, note))
    new_rows.append(delta_row(
        "cadr_a2_vs_a0", split, "kitchen_drop_rv51_trust", "kitchen_drop_rv51",
        "A2 trust-only vs A0 no-CADR RV51", "lora_repair_rv51"))
    new_rows.append(delta_row(
        "cadr_a2_vs_a3", split, "kitchen_drop_rv51_trust", "kitchen_drop_rv51_conf",
        "A2 trust-only vs A3 full CADR", "lora_repair_cadr"))
    new_rows.append(delta_row(
        "cadr_a4_vs_a3", split, "kitchen_drop_rv51_cov", "kitchen_drop_rv51_conf",
        "A4 cov-need vs A3 full CADR", "lora_repair_cadr"))
    new_rows.append(delta_row(
        "cadr_a5_vs_a3", split, "kitchen_drop_rv51_aden", "kitchen_drop_rv51_conf",
        "A5 adaptive denoise vs A3 full CADR", "lora_repair_cadr"))
    new_rows.append(delta_row(
        "abs16_nocadr_vs_rv51", split, "kitchen_drop_abs16", "kitchen_drop_rv51",
        "C4 abs16 no-CADR vs RV51 no-CADR (base swap)", "lora_repair_rv51"))
    conf_p = go / gs_dir / split / "ours_None__kitchen_drop_abs16_conf" / "results.json"
    if conf_p.exists():
        new_rows.append(delta_row(
            "abs16_nocadr_vs_abs16_cadr", split, "kitchen_drop_abs16", "kitchen_drop_abs16_conf",
            "C4 isolate CADR on abs16 base", "lora_repair_abs16_cadr"))

rows.extend(new_rows)
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=HEADER, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in HEADER})

print("--- SUMMARY test ---")
for tag, *_ in specs:
    s, p, l, _ = load_tag(tag, "test")
    print(f"{tag}: PSNR={p:.4f} SSIM={s:.6f} LPIPS={l:.6f}")
a0 = load_tag("kitchen_drop_rv51", "test")
a3 = load_tag("kitchen_drop_rv51_conf", "test")
print(f"A0 rv51:        PSNR={a0[1]:.4f}")
print(f"A3 rv51_conf:   PSNR={a3[1]:.4f}")
print(f"wrote {path}")
PY
  echo "[stage 5] done"
}

echo "=========================================="
echo "CADR A-full + C4 (results-only)"
require_branch
lineb_activate >/dev/null
echo "START_STAGE=${START_STAGE} FORCE_REPAIR=${FORCE_REPAIR}"
echo "=========================================="

[[ "${START_STAGE}" =~ ^[0-5]$ ]] || die "stage must be 0..5"

if (( START_STAGE <= 0 )); then stage0_cleanup; fi
if (( START_STAGE <= 1 )); then stage1_a2; fi
if (( START_STAGE <= 2 )); then stage2_a4; fi
if (( START_STAGE <= 3 )); then stage3_a5; fi
if (( START_STAGE <= 4 )); then stage4_c4; fi
if (( START_STAGE <= 5 )); then stage5_csv; fi

echo "=========================================="
echo "Done. CSV=${EVAL_CSV}"
echo "  A2 $(results_for_tag kitchen_drop_rv51_trust)"
echo "  A4 $(results_for_tag kitchen_drop_rv51_cov)"
echo "  A5 $(results_for_tag kitchen_drop_rv51_aden)"
echo "  C4 $(results_for_tag kitchen_drop_abs16)"
echo "=========================================="
