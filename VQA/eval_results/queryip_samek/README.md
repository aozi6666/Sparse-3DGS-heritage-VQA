# 同 K 消融产物

`keep_ratio=0.75`（除非文件名带 `krXXX`）。四路：`prefix` / `random` / `queryip` / `queryip_bitcost`；阶段 2：`queryip_clip`；E2：`queryip_dyn` / `queryip_sta`。

## n200（主表）

| 文件 | 含义 |
|------|------|
| `n200/..._n200_purebitcost.*` | miss→0；bitcost **58.5%**，proxy **64.0%** |
| `n200/..._n200_hybrid.*` | miss→proxy；bitcost **58.0%**，proxy **64.0%** |

## clip_frame_n200（阶段 2 · E0/E1/E4 · 2026-10-23）

| 文件 | 含义 |
|------|------|
| `..._n200_kr075_clip.jsonl` | 逐题；200 行 |
| `..._summary_n200_kr075_clip.json` | Acc 汇总 |

| mode | Acc |
|------|-----|
| random | 59.5%（119/200） |
| queryip | **63.5%**（127/200） |
| queryip_clip | **58.5%**（117/200）· **失败消融** |

## ab_fixed_n200（E2 · α/β 三角 · 已归档）

| mode | Acc |
|------|-----|
| queryip | 63.5%（127/200） |
| queryip_dyn | 63.5%（127/200） |
| queryip_sta | 64.0%（128/200） |

→ 题型 α/β **不作贡献**。文件：`*_n200_kr075_ab.*`。不覆盖 `clip_frame_n200/`。

## region_sem_n200（§9 v1 · 宽池融合 · **失败消融 · 勿删**）

| mode | Acc |
|------|-----|
| queryip | **63.5%**（127/200） |
| queryip_region | **58.5%**（117/200）· −5.0 pp |
| queryip_region_shuf | **60.5%**（121/200）· B−C=−2.0 pp |

→ 伤底 + shuffle 反向；**不作贡献**。详见 [`region_sem_n200/README.md`](region_sem_n200/README.md)。文件：`*_n200_kr075_region_v1.*`。

## region_edge_n200（§9.12 边界 · **失败消融 · 勿删**）

| mode | Acc |
|------|-----|
| queryip | **63.5%**（127/200） |
| queryip_edge | **60.0%**（120/200）· −3.5 pp |
| queryip_edge_shuf | **62.0%**（124/200）· B−C=−2.0 pp |

→ 伤底 + shuffle 反向；**停 QueryIP 语义选区**（非否定一切语义）。详见 [`region_edge_n200/README.md`](region_edge_n200/README.md)。文件：`*_n200_kr075_region_edge.*`。

## n60

| 文件 | 含义 |
|------|------|
| `n60/..._short60_samek.jsonl` | 早期三路 proxy |
| `n60/..._kr075.*` | 四路 n=60（bitcost=proxy=61.7） |

## ksweep

仅完成 0.50 / 0.60；0.75 在 n60；0.90 未跑。`ksweep.csv` 可能不完整。
