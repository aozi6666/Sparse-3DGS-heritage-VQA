# VQA 评测产物目录说明

整理日期：2026-10-23。计划：[`docs/PLAN_mage_query_IP_select.md`](../../docs/PLAN_mage_query_IP_select.md)。

## 目录树

```text
eval_results/
  README.md
  baseline_mage/            ← 行 A 满预算 Mage（Video-MME）
  baseline_mage_ocrbench_v1/← 原版 Mage-VL × OCRBench v1（今天优先）
  baseline_mage_ocrbench_v2/← OCRBench_v2（已暂停 · ~1180 已备份）
  queryip_rowB_full/        ← 行 B 全量（非同 K，≈0.75× token）
  queryip_samek/            ← 同 K 消融（主实验）
    soccer/
    n60/
    n200/                   ← 主表（含 bitcost 失败）
    clip_frame_n200/        ← E0/E1/E4：random/queryip/clip（勿删）
    ab_fixed_n200/          ← E2：queryip/dyn/sta（勿删）
    region_sem_n200/        ← §9 v1 宽池+区域CLIP：失败消融（勿删）
    region_edge_n200/       ← §9.12 边界 edge：失败消融（勿删）
    wide_textseat_n200/     ← 视频 300k 中宽池：负协议归档（勿按 300k 重跑）
    ocrbench_v2_image_textseat/ ← 图像 QueryIP + OCR 座位（待跑）
    region_sem_pilot/       ← 早期 n60 断点（可忽略）
    ksweep/                 ← 旧四路扫描（中断，可忽略）
    ksweep_proxy/           ← proxy K 扫描【做图 CSV】
    efficiency_curve/       ← 预案
    stats_mcnemar_n200/     ← 待补：配对检验产物
  tools/
  vlmeval_outputs/
  # logs/  · nohup 临时目录，默认不入库；需要时 mkdir 即可
```

## 主结论（2026-10-23）

**保留 `queryip`（proxy）。** 同设定复现 [`clip_frame_n200/`](queryip_samek/clip_frame_n200/)：

| 条件 | Acc |
|------|-----|
| random | 59.5% |
| **queryip · 保留** | **63.5%**（+4.0 pp · **待 McNemar**） |
| queryip_clip | 58.5% · 失败 |
| queryip_dyn / sta | 63.5% / 64.0% · 题型 **不作贡献**（[`ab_fixed_n200/`](queryip_samek/ab_fixed_n200/)） |
| queryip_region / shuf | 58.5% / **60.5%** · **区域 v1 失败**（[`region_sem_n200/`](queryip_samek/region_sem_n200/)） |
| queryip_edge / shuf | **60.0%** / **62.0%** · **边界 edge 失败**（[`region_edge_n200/`](queryip_samek/region_edge_n200/)；B−C=−2.0pp） |

主表 [`n200/`](queryip_samek/n200/) 另报 queryip 64.0% / bitcost 失败。→ 贡献 = **同 K proxy > 乱砍**。

**QueryIP 语义选区路线停：** CLIP 相似度重排区域（帧/宽池 v1/边界 edge）均无收益；**保留 proxy 63.5%**。否定的是这套具体方法，不是一切语义引导。

旁证：行 A 63.7%（满 token）· 行 B 62.3%（≈0.75× token，非同 K）。

## K 扫描结论（Acc–K 曲线）

[`queryip_samek/ksweep_proxy/`](queryip_samek/ksweep_proxy/) · **做图 CSV：`plot_acc_vs_keep_ratio.csv` / `plot_acc_long.csv`（勿删）**

| keep_ratio | prefix | random | queryip | Δ(q−rand) |
|------------|--------|--------|---------|-----------|
| 0.30 | 53.0% | 51.5% | **55.5%** | +4.0 |
| 0.50 | 53.0% | 57.0% | **61.0%** | +4.0 |
| 0.75 | 57.5% | 59.5% | **64.0%** | +4.5 |
| 0.90 | 58.0% | **61.5%** | 60.5% | −1.0 |

→ **0.3～0.75 有用；0.9 优势消失。** 详见该目录 [`README.md`](queryip_samek/ksweep_proxy/README.md)。

## 还要不要跑全量 714 同 K？

**不建议。** 下一步若补图：满 token Mage 单点拼效率曲线（计划 §5），不是 714 四路。

## 子目录

| 目录 | 内容 |
|------|------|
| `baseline_mage/` | Video-MME overall 63.7% |
| `queryip_rowB_full/` | QueryIP 62.3%；tokens jsonl |
| `queryip_samek/soccer/` | SAMEK4_OK |
| `queryip_samek/n60/` | short 前 60 |
| `queryip_samek/n200/` | 主表 pure + hybrid（bitcost 失败） |
| `queryip_samek/clip_frame_n200/` | E0/E1/E4；queryip +4.0pp vs random；CLIP 失败 |
| `queryip_samek/ab_fixed_n200/` | E2 α/β；题型不作贡献 |
| `queryip_samek/region_sem_n200/` | §9 v1；58.5/60.5 vs queryip 63.5；**失败勿删** |
| `queryip_samek/region_edge_n200/` | §9.12 edge；60.0/62.0 vs 63.5；**失败勿删 · 停语义选区** |
| `queryip_samek/ksweep/` | 旧扫描中断 |
| `queryip_samek/ksweep_proxy/` | **已完成**；`plot_acc_*.csv` 做图 |
| `queryip_samek/efficiency_curve/` | 预案占位 |
| `queryip_samek/stats_mcnemar_n200/` | 待补配对检验 |
| `logs/`（可选） | nohup 临时输出；**不入库**，复现时 `mkdir -p` |
