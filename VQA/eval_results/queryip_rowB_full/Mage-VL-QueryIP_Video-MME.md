# Mage-VL-QueryIP × Video-MME 评测结果（行 B · 全量）

> 生成时间: 2026-10-21 13:15:31  
> 数据来源: `/root/autodl-tmp/VLMEvalKit/outputs/Mage-VL-QueryIP/T20261021-100938/`  
> 本地副本: [`Mage-VL-QueryIP_Video-MME_score.json`](Mage-VL-QueryIP_Video-MME_score.json)  
> Token 日志: [`Mage-VL-QueryIP_videomme_tokens.jsonl`](Mage-VL-QueryIP_videomme_tokens.jsonl)（714 行）  
> 数据集: Video-MME (32 帧子集, 238 视频 / 714 题)  
> 模型: `Mage-VL-QueryIP`（canvas 级问题重加权，`keep_ratio=0.75`）  
> 推理失败率 **0%** / 打分失败率 **0%**  
> 墙钟: 约 **3.1 h**（`714it [3:05:44]`）

## 一句话结论

**管线可行**（加载、推理、打分全通）；相对默认 Mage（行 A overall **63.7%**）Acc **未提升**（**62.3%**，Δ≈−1.4）。属「故意只留 75% token」的预期代价，**不能**当作「问题条件化选区已在 Video-MME 验证涨点」。

## 总览（vs 行 A）

| 视频时长 | 行 B QueryIP | 行 A 默认 Mage | Δ (B−A) |
| --- | --- | --- | --- |
| Short | 75.3% | 75.8% | −0.5 |
| Medium | 61.6% | 63.6% | −2.0 |
| Long | 51.1% | 52.7% | −1.6 |
| **Overall** | **62.3%** | **63.7%** | **−1.4** |

## 效率（token jsonl）

| 指标 | 值 |
| --- | --- |
| Avg #vis-token（选后） | ≈ **3703** |
| Avg #vis-token（同视频 default N） | ≈ **4937** |
| 相对 token 比 | **0.75×**（与 `keep_ratio` 一致） |
| qtype 计数 | dynamic **592** / static **122** |

## 领域 (Domain) · overall

| 领域 | Acc |
| --- | --- |
| Knowledge | 61.6% |
| Film & Television | 62.8% |
| Sports Competition | 57.3% |
| Artistic Performance | 61.8% |
| Life Record | 65.2% |
| Multilingual | 74.1% |

## 读数边界（归档备忘）

1. 行 A 满预算 vs 行 B 0.75K，**不是同预算消融**。  
2. 当时实现为 **整 canvas 丢弃** + 解码图帧差/方差 proxy，**不是** cv-preinfer 真 MV/残差分。  
3. 后续改正见计划「路线 B」（patch 级 proxy + 同 K 三路）与延期「路线 A」（导出真分）。
