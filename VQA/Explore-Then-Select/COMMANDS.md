# Mage Query-IP — 本地命令（Agent 不代跑 Acc）

工作目录：`/root/autodl-tmp/Sparse-3DGS-heritage-VQA` · 分支 `Mage_LoRA`  
计划：[`docs/PLAN_mage_query_IP_select.md`](../../docs/PLAN_mage_query_IP_select.md)  
产物：[`VQA/eval_results/README.md`](../eval_results/README.md) · 总览：[`VQA/README.md`](../README.md)

**阶段 1 主推：`queryip`（proxy）。** CLIP 语义选区已停。  
**当前优先：OCRBench v1 Mage baseline（§11）+ 图像 QueryIP（§12）。**  
v2 已暂停保全 ~1180（§9.1 以后静默续跑）。§7 视频 300k = 负协议（已停）。

---

## 0. 准备

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
export PATH="/root/autodl-tmp/VLMEvalKit/vlmeval_env/bin:$PATH"
```

CLIP 权重：`VQA/models/CLIP-ViT-Large-patch14`（本地）。

---

## 1. 已完成（勿重跑除非必要）

| 实验 | 结果位置 |
|------|----------|
| 行 A Mage 全量 | `eval_results/baseline_mage/` |
| 行 B QueryIP 全量（非同 K）62.3% | `eval_results/queryip_rowB_full/` |
| Soccer SAMEK4_OK | `eval_results/queryip_samek/soccer/` |
| short-60 / **n=200 主表** | `eval_results/queryip_samek/n60/` · `n200/` |
| proxy K 扫描 | `eval_results/queryip_samek/ksweep_proxy/` |
| CLIP 帧预算 E0/E1/E4 | `eval_results/queryip_samek/clip_frame_n200/`（E4 失败；保留） |

---

## 2. 无卡：CLIP 帧预算 CPU 冒烟

不跑 Mage、不算 Video-MME Acc。

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
# 默认：allocate_budgets（轻量；2Gi cgroup 可过）
python VQA/Explore-Then-Select/scripts/smoke_clip_frame_budget.py

# 接线：select_patches_samek(queryip_clip)（多占一点内存）
# python VQA/Explore-Then-Select/scripts/smoke_clip_frame_budget.py --with-wire

# 真 CLIP（需 GPU 或更大内存；cgroup≈2Gi 会 OOM）:
# python VQA/Explore-Then-Select/scripts/smoke_clip_frame_budget.py --with-clip --device cuda
```

验收：默认过 `allocate_budgets`；有内存再 `--with-wire` / `--with-clip`。

---

## 3. 已完成：真 CLIP + n=200 同 K（E0 / E1 / E4）

产物：`eval_results/queryip_samek/clip_frame_n200/`（保留；E4 失败消融）。复现命令见计划 §4.4.1。

---

## 4. 已完成：α/β 三角对照（E2 · n=200）

结果：queryip=queryip_dyn=63.5%；queryip_sta=64.0% → **题型不作贡献**。产物 `ab_fixed_n200/`（保留）。复现：
```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
git checkout Mage_LoRA
source /root/autodl-tmp/VLMEvalKit/vlmeval_env/bin/activate
export PYTHONPATH="VQA/mage_vl/models:VQA/Explore-Then-Select:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p VQA/eval_results/queryip_samek/ab_fixed_n200 VQA/eval_results/logs
pkill -f 'eval_short60_samek.py.*n200_kr075_ab' 2>/dev/null || true
nohup python VQA/Explore-Then-Select/mage_query_ip/eval_short60_samek.py \
  --n 200 --keep-ratio 0.75 --seed 0 \
  --modes queryip,queryip_dyn,queryip_sta \
  --outdir VQA/eval_results/queryip_samek/ab_fixed_n200 \
  --tag n200_kr075_ab \
  --resume \
  > VQA/eval_results/logs/ab_fixed_n200_nohup.out 2>&1 &
tail -f VQA/eval_results/logs/ab_fixed_n200_nohup.out
```

产物：`eval_results/queryip_samek/ab_fixed_n200/`。判定见计划 §4.4.3。

---

## 5. 预案：满 token 效率曲线

见计划 **§5**。复用 ksweep + 补满预算 Mage 一点 → `efficiency_curve/`。

---

## 6. 区域语义（§9）

### 6.0 已完成：v1 n=200 · **失败归档（勿重跑）**

产物：[`eval_results/queryip_samek/region_sem_n200/`](../eval_results/queryip_samek/region_sem_n200/)  
Acc：queryip **63.5%** / region **58.5%** / shuf **60.5%**（121/200；B−C=−2.0pp）→ **不作贡献**。  
协议：λ=0.4 · 2×2 · 宽池融合。**禁止完整 v2。**

### 6.1–6.2 v1 冒烟 / n=60（历史；可跳过）

见计划附录；冒烟：`scripts/smoke_region_clip_v1.py`。

### 6.3 已完成：边界 edge n=200 · **失败归档（勿重跑）**

产物：[`eval_results/queryip_samek/region_edge_n200/`](../eval_results/queryip_samek/region_edge_n200/)  
Acc：queryip **63.5%** / edge **60.0%** / shuf **62.0%**（B−C=−2.0pp）→ **停 QueryIP 语义选区**。  
口径：否定 CLIP 相似度重排区域；**非**否定一切语义；基线仍 **queryip 63.5%**。不做 v2。

---

## 7. 视频中宽池 textseat · **负协议归档（已停 · 勿按 300k 重跑）**

产物：[`wide_textseat_n200/`](../eval_results/queryip_samek/wide_textseat_n200/)（含部分 jsonl）。  
**失败原因：** `max_pixels=300000` → `n_wide~3–4万`，等价高 keep；proxy 无优势；VMME 上 `need_text≈0`，textseat≡queryip。  
**若再跑视频同 K：** 必须 `--max-pixels 150000`（与历史主表一致）。CPU 冒烟仍可用：`scripts/smoke_text_seat.py`。

---

## 8. 不做

- 同 K 全量 714  
- 主推官方 bitcost / 真 MV  
- CLIP 帧 / 区域 v1 / **edge** 语义选区再试  
- 视频「完整 v2」ρ 分席 / `max_pixels=300000` 中宽池再试  
- Agent 代跑 Acc  

---

## 9. OCRBench_v2 · Mage-VL baseline（已暂停 · ~1180 题已保全）

产物：[`baseline_mage_ocrbench_v2/`](../eval_results/baseline_mage_ocrbench_v2/)  
备份：`_backup_pkl/01_OCRBench_v2_n1180_final.pkl` · 活目录 `Mage-VL/T20261027-112350/`  
**今天优先跑 v1（§11–12）。** 以后续 v2 用 §9.1。

### 9.1 以后静默续跑 v2（不要 `--verbose`）

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
source /root/autodl-tmp/VLMEvalKit/vlmeval_env/bin/activate
REPO=/root/autodl-tmp/Sparse-3DGS-heritage-VQA
export LMUData="$REPO/VQA/datas/lmu"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO/VQA/VLMEvalKit:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SKIP_ERR=1

nohup python "$REPO/VQA/VLMEvalKit/run.py" \
  --model Mage-VL --data OCRBench_v2 \
  --work-dir "$REPO/VQA/eval_results/baseline_mage_ocrbench_v2" \
  --reuse --reuse-aux all \
  > "$REPO/VQA/eval_results/logs/mage_ocrbench_v2_quiet_resume.out" 2>&1 &

# 只看进度条行
tail -f "$REPO/VQA/eval_results/logs/mage_ocrbench_v2_quiet_resume.out" | grep --line-buffered 'Infer Mage-VL'
```

必须日志出现 `reuse … 01_OCRBench_v2.pkl`。

---

## 10. 图像 QueryIP · OCRBench_v2（后置；新文件）

见 [`ocrbench_v2_image_textseat/README.md`](../eval_results/queryip_samek/ocrbench_v2_image_textseat/README.md)。  
CPU 冒烟：`python VQA/Explore-Then-Select/scripts/smoke_image_textseat.py`

---

## 11. OCRBench v1 · Mage-VL baseline（今天优先 · Agent 不代跑 Acc）

TSV 已转好：`VQA/datas/lmu/OCRBench.tsv`（1000 行；源 parquet）。  
重转：`python VQA/datas/ORCBench/convert_parquet_to_ocrbench_tsv.py`

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
git checkout Mage_LoRA
source /root/autodl-tmp/VLMEvalKit/vlmeval_env/bin/activate
REPO=/root/autodl-tmp/Sparse-3DGS-heritage-VQA
export LMUData="$REPO/VQA/datas/lmu"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO/VQA/VLMEvalKit:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SKIP_ERR=1
mkdir -p "$REPO/VQA/eval_results/baseline_mage_ocrbench_v1" "$REPO/VQA/eval_results/logs"
nvidia-smi   # 确认无占卡

# 无 --verbose
nohup python "$REPO/VQA/VLMEvalKit/run.py" \
  --model Mage-VL \
  --data OCRBench \
  --work-dir "$REPO/VQA/eval_results/baseline_mage_ocrbench_v1" \
  > "$REPO/VQA/eval_results/logs/mage_ocrbench_v1_baseline_nohup.out" 2>&1 &

tail -f "$REPO/VQA/eval_results/logs/mage_ocrbench_v1_baseline_nohup.out" | grep --line-buffered 'Infer Mage-VL'
```

中断后续跑（加 `--reuse`）：

```bash
nohup python "$REPO/VQA/VLMEvalKit/run.py" \
  --model Mage-VL --data OCRBench \
  --work-dir "$REPO/VQA/eval_results/baseline_mage_ocrbench_v1" \
  --reuse --reuse-aux all \
  > "$REPO/VQA/eval_results/logs/mage_ocrbench_v1_baseline_resume.out" 2>&1 &
```

产物 / 口径：[`baseline_mage_ocrbench_v1/README.md`](../eval_results/baseline_mage_ocrbench_v1/README.md) · 看 Final Score Norm vs 官方 81.80。

---

## 12. OCRBench v1 · 图像 QueryIP 同 B（baseline 后 · Agent 不代跑）

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
source /root/autodl-tmp/VLMEvalKit/vlmeval_env/bin/activate
REPO=/root/autodl-tmp/Sparse-3DGS-heritage-VQA
export PYTHONPATH="VQA/mage_vl/models:VQA/Explore-Then-Select:${PYTHONPATH:-}"
export LMUData="$REPO/VQA/datas/lmu"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p VQA/eval_results/queryip_samek/ocrbench_v1_image_textseat VQA/eval_results/logs

python VQA/Explore-Then-Select/scripts/smoke_image_textseat.py

# 当前环境无 easyocr → 先不带 --use-ocr（装好后再加）
nohup python VQA/Explore-Then-Select/mage_query_ip/eval_ocrbench_v2_samek.py \
  --tsv "$REPO/VQA/datas/lmu/OCRBench.tsv" \
  --n 0 --keep-ratio 0.75 --seed 0 --patch 32 \
  --modes full_image,random_tiles,queryip_image_textseat,queryip_image_textseat_shuf \
  --outdir VQA/eval_results/queryip_samek/ocrbench_v1_image_textseat \
  --tag n1000_kr075_ocrbench_v1 \
  --resume \
  > VQA/eval_results/logs/ocrbench_v1_image_textseat_nohup.out 2>&1 &

tail -f VQA/eval_results/logs/ocrbench_v1_image_textseat_nohup.out
```

闸门：[`ocrbench_v1_image_textseat/README.md`](../eval_results/queryip_samek/ocrbench_v1_image_textseat/README.md)。  

