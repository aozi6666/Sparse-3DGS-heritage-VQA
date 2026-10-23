# 区域语义 v1 · n=200（失败消融 · 勿删）

**日期：** 2026-10-23  
**协议：** 宽池 C + 2×2 区域 CLIP + \(S=\mathrm{norm}(proxy)+0.4\cdot\mathrm{norm}(q)\) top-K；批内循环 shuffle；`keep_ratio=0.75` · seed=0  
**判定：** **失败** — 伤底且 shuffle 反向；**不作贡献**。主推仍只报 `queryip`。

## Acc（点估计）

| mode | correct | Acc | 相对 A |
|------|---------|-----|--------|
| `queryip` (A) | 127/200 | **63.5%** | — |
| `queryip_region` (B) | 117/200 | **58.5%** | **−5.0 pp** |
| `queryip_region_shuf` (C) | 121/200 | **60.5%** | B−C = **−2.0 pp** |

闸门（PLAN §9.9）未过：需 B≥A−1pp 且 B−C≥+2pp。

## 根因（一句话）

区域 CLIP 量的是**图文相似度**，不是「该 patch 对答题是否有用」；语义重排挤掉 proxy 看好的运动/时序证据 → Acc 掉到与 E4 帧 CLIP 同量级（58.5%）。

## 文件

| 文件 | 内容 |
|------|------|
| `Mage-VL-QueryIP_short60_samek4_n200_kr075_region_v1.jsonl` | 逐题三臂预测 |
| `Mage-VL-QueryIP_short60_samek4_summary_n200_kr075_region_v1.json` | Acc 汇总（`queryip_region_shuf` Acc 字段为 **0.605**） |

**后续：** 边界 edge 亦失败（[`../region_edge_n200/`](../region_edge_n200/)：60.0/62.0）→ **停 QueryIP 语义选区**（PLAN §9.12）。否定 CLIP 重排，非否定一切语义。 **禁止** v2。
