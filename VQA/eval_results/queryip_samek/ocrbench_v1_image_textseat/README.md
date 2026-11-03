# ocrbench_v1_image_textseat · 图像 QueryIP 同 B（OCRBench v1）

> 状态：**已跑完 · 未过闸 · 失败消融归档** · 2026-10-27 · 无 `--use-ocr`  
> 定案：见 [`PLAN_mage_query_IP_select.md`](../../../../docs/PLAN_mage_query_IP_select.md) **主线 A / 加戏 B**；本目录 **不作主表**（§2.7）  
> Mage v1 baseline：Norm **83.4**（[`baseline_mage_ocrbench_v1`](../../baseline_mage_ocrbench_v1/)）

## 协议

| 项 | 值 |
|----|-----|
| 数据 | `VQA/datas/lmu/OCRBench.tsv` + `images/OCRBench/`（1000） |
| 预算 | `keep_ratio=0.75` 同 B（`full_image` 为满图参考） |
| modes | `full_image` · `random_tiles` · `queryip_image_textseat` · `queryip_image_textseat_shuf` |
| 指标 | substring/exact proxy（≠官方 Final Score Norm） |

## 结果

| mode | Acc | n |
|------|-----|---|
| full_image | **0.841** | 1000 |
| random_tiles | 0.283 | 1000 |
| queryip_image_textseat | 0.373 | 1000 |
| queryip_image_textseat_shuf | 0.373 | 1000 |

vs Mage Norm 83.4：`full_image`≈同量级；**textseat 远低于 Mage/full（−46.8 pp）**。  
闸门：≥ random ✅；− shuf = 0 ❌；vs full ❌。

产物：`Mage-VL-ImageQueryIP_ocrbench_v2_n1000_kr075_ocrbench_v1.jsonl` + `_summary_*.json`
