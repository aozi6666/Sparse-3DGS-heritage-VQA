# VQA 评测产物目录说明

整理日期：2026-10-21。计划：[`docs/PLAN_mage_query_IP_select.md`](../../docs/PLAN_mage_query_IP_select.md)。

## 目录树

```text
eval_results/
  README.md
  baseline_mage/            ← 行 A 满预算 Mage
  queryip_rowB_full/        ← 行 B 全量（非同 K，≈0.75× token）
  queryip_samek/            ← 同 K 消融（主实验）
    soccer/
    n60/
    n200/                   ← 主表定稿
    ksweep/                 ← 旧四路扫描（中断，可忽略）
    ksweep_proxy/           ← proxy K 扫描【已完成 · 做图数据在此】
    efficiency_curve/       ← 预案：满 token 效率曲线（尚未跑）
  logs/
  tools/
  vlmeval_outputs/
```

## 主结论

[`queryip_samek/n200/`](queryip_samek/n200/) · `keep_ratio=0.75`

| 条件 | Acc |
|------|-----|
| prefix | 57.5% |
| random | 59.5% |
| **queryip（proxy）· 主推** | **64.0%** |
| bitcost 纯 / hybrid | 58.5% / 58.0% |

→ **主贡献 = 同 K 下 queryip > 乱砍**；bitcost 仅失败消融。

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
| `queryip_samek/n200/` | 主表 pure + hybrid |
| `queryip_samek/ksweep/` | 旧扫描中断 |
| `queryip_samek/ksweep_proxy/` | **已完成**；`plot_acc_*.csv` 做图 |
| `queryip_samek/efficiency_curve/` | 预案占位 |
| `logs/` | nohup 日志 |
