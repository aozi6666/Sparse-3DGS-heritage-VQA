# Mage Query-IP — 本地命令（Agent 不代跑 Acc）

工作目录：`/root/autodl-tmp/Sparse-3DGS-heritage-VQA` · 分支 `Mage_LoRA`  
计划：[`docs/PLAN_mage_query_IP_select.md`](../../docs/PLAN_mage_query_IP_select.md)  
产物：[`VQA/eval_results/README.md`](../eval_results/README.md) · 总览：[`VQA/README.md`](../README.md)

**阶段 1 主推：`queryip`（proxy）。** 阶段 2 脚手架：`queryip_clip`（CLIP 帧预算 + 帧内 codec）。

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

## 7. 不做

- 同 K 全量 714  
- 主推官方 bitcost / 真 MV  
- CLIP 帧 / 区域 v1 / **edge** 语义选区再试  
- §9 **完整 v2** ρ 分席  
- Agent 代跑 Acc  

