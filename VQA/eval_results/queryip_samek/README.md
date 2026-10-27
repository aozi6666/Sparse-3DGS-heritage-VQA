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

## wide_textseat_n200（视频 300k 中宽池 · **负协议归档**）

| 项 | 说明 |
|----|------|
| 状态 | **已停**；`n_wide` 过大 + VMME 不触发 textseat |
| 目录 | [`wide_textseat_n200/`](wide_textseat_n200/) |
| 命令 | [`COMMANDS.md` §7](../../Explore-Then-Select/COMMANDS.md) |

## ocrbench_v1_image_textseat（图像硬切 · **失败归档 · 不作主表**）

| 项 | 说明 |
|----|------|
| 结果 | full **84.1%** / random 28.3% / textseat=shuf **37.3%**（n=1000 · 无 OCR） |
| 闸门 | ≥random ✅；−shuf 0 ❌；vs full −46.8 ❌ |
| 目录 | [`ocrbench_v1_image_textseat/`](ocrbench_v1_image_textseat/) |
| 定案 | [`PLAN_mage_query_IP_select.md`](../../../docs/PLAN_mage_query_IP_select.md) **主线 A**；本实验 §2.7 |
| Mage baseline | Norm **83.4**（[`../baseline_mage_ocrbench_v1/`](../baseline_mage_ocrbench_v1/)） |

## ocrbench_v2_image_textseat（图像 + OCR 座位 · 后置）

| 项 | 说明 |
|----|------|
| 目录 | [`ocrbench_v2_image_textseat/`](ocrbench_v2_image_textseat/) |
| Mage baseline | [`../baseline_mage_ocrbench_v2/`](../baseline_mage_ocrbench_v2/) · §9.1 静默续跑 |

## n60

| 文件 | 含义 |
|------|------|
| `n60/..._short60_samek.jsonl` | 早期三路 proxy |
| `n60/..._kr075.*` | 四路 n=60（bitcost=proxy=61.7） |

## ksweep

仅完成 0.50 / 0.60；0.75 在 n60；0.90 未跑。`ksweep.csv` 可能不完整。
