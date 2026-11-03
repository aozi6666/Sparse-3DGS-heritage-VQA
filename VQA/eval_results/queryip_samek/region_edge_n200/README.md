# 区域边界 edge · n=200（失败消融 · 勿删）

**日期：** 2026-10-23  
**协议：** 冻结 proxy top-\((K-m)\)；band \([K-m,K+m)\) 内按 2×2 区域 CLIP 填 \(m=\lfloor 0.05K\rfloor\) 席；批内循环 shuffle；`keep_ratio=0.75` · seed=0  
**判定：** **失败** — 伤底且 shuffle 反向；**停 QueryIP 语义选区路线**。

## Acc（点估计）

| mode | correct | Acc | 相对 A |
|------|---------|-----|--------|
| `queryip` (A) | 127/200 | **63.5%** | — |
| `queryip_edge` (B) | 120/200 | **60.0%** | **−3.5 pp** |
| `queryip_edge_shuf` (C) | 124/200 | **62.0%** | B−C = **−2.0 pp** |

闸门（PLAN §9.9）未过：需 B≥A−1pp 且 B−C≥+2pp。比 v1（−5.0pp）伤得轻，仍无效。

## 结论口径（写进论文/计划）

- **当前方案不值得继续：** CLIP 区域相似度不能可靠代表答题所需证据；实验无收益。
- **QueryIP 基线保留：** **63.5%**（目前更好）。
- **暂时不要做 v2 / 扩大语义选区**；除非有新机制假设，否则重复试错。
- **否定范围：** 否定的是「用 CLIP 相似度重排局部区域」这套 **QueryIP 语义选区** 具体方法，**不是**所有语义引导机制。

对照：帧 CLIP E4 与区域 v1 / edge 同源失败（图文相似 ≠ 答题证据）。

## 文件

| 文件 | 内容 |
|------|------|
| `Mage-VL-QueryIP_short60_samek4_n200_kr075_region_edge.jsonl` | 逐题三臂 |
| `Mage-VL-QueryIP_short60_samek4_summary_n200_kr075_region_edge.json` | Acc 汇总 |

v1 宽池失败见 [`../region_sem_n200/`](../region_sem_n200/)。
