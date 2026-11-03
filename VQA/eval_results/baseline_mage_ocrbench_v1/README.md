# baseline_mage_ocrbench_v1 · 原版 Mage-VL × OCRBench（v1）

> 状态：**已完成** · 2026-10-27 12:10 · `eval_id=T20261027-120048`  
> 命令：[`COMMANDS.md` §11](../../Explore-Then-Select/COMMANDS.md)

## 协议

| 项 | 值 |
|----|-----|
| 模型 | `Mage-VL`（图像） |
| 数据 | OCRBench v1 · 1000 题 |
| TSV | `VQA/datas/lmu/OCRBench.tsv`（由本地 parquet 转换；MD5≠官方下载） |
| 源 parquet | `VQA/datas/ORCBench/test-00000-of-00001.parquet` |
| work-dir | 本目录 |
| 静默 | **不要** `--verbose`；续跑加 `--reuse` |

## 对照

| 对照 | 说明 |
|------|------|
| 官方 Mage **81.80**（Final Score Norm） | 旁证；注明本地自转 TSV |
| 本目录 `*_score.json` Final Score Norm | **主 baseline** |
| QueryIP | [`../queryip_samek/ocrbench_v1_image_textseat/`](../queryip_samek/ocrbench_v1_image_textseat/) |

## 结果（T20261027-120048）

| 指标 | 分数 |
|------|------|
| Text Recognition | 271 |
| Scene Text-centric VQA | 178 |
| Doc-oriented VQA | 172 |
| Key Information Extraction | 162 |
| HMER | 51 |
| **Final Score** | **834** |
| **Final Score Norm** | **83.4** |

相对论文 Mage 81.80：本地自转 TSV，**+1.6**（旁证；不作跨仓硬比）。  
infer/judge fail：0/1000。耗时约 ~9.5 min。

产物：

- `Mage-VL/T20261027-120048/Mage-VL_OCRBench_score.json`
- `Mage-VL/T20261027-120048/Mage-VL_OCRBench.xlsx`
- `Mage-VL/T20261027-120048/status.json`
