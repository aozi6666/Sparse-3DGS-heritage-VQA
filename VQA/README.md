# 工作 B · 视频问答与评测

系统仓 **VQA / 评测** 工作区（GitHub：`Sparse-3DGS-heritage-VQA`）。

## Mage Query-IP（结论 · 2026-10-23）

**主推：** 固定预算 K 下 **proxy `queryip`**（**63.5%**）> 同 K random（约 +4 pp；**McNemar 待补**）。不重训；不声称超过满预算 Mage。

**已停 / 不作贡献：** CLIP 语义选区（帧 / 区域 v1 / 边界 edge）、题型 α/β、官方 bitcost。  
否定的是「CLIP 相似度重排区域」，**不是**一切语义引导。

| 同 K · short 前 200 · `keep_ratio=0.75` | Acc | 备注 |
|------|-----|------|
| **queryip · 保留** | **63.5%** | [`clip_frame_n200/`](eval_results/queryip_samek/clip_frame_n200/)（主表曾 64.0%） |
| random | 59.5% | 同上 |
| queryip_clip | 58.5% | 失败 |
| queryip_region / shuf | 58.5% / 60.5% | [`region_sem_n200/`](eval_results/queryip_samek/region_sem_n200/) |
| queryip_edge / shuf | 60.0% / 62.0% | [`region_edge_n200/`](eval_results/queryip_samek/region_edge_n200/) |
| queryip_dyn / sta | 63.5% / 64.0% | 题型不作贡献 |
| bitcost 纯 / hybrid | 58.5% / 58.0% | 失败 |
| prefix | 57.5% | `n200/` |

旁证（非同 K）：行 A Mage **63.7%** · 行 B QueryIP@0.75 **62.3%**（≈0.75× token）。  
K 扫描：0.3–0.75 赢 random；0.9 略负 · [`ksweep_proxy/`](eval_results/queryip_samek/ksweep_proxy/)。

| 文档 | 用途 |
|------|------|
| [`docs/PLAN_mage_query_IP_select.md`](../docs/PLAN_mage_query_IP_select.md) | 结论 · 证据 · **后续待办** |
| [`Explore-Then-Select/COMMANDS.md`](Explore-Then-Select/COMMANDS.md) | 复现命令（Agent 不代跑 Acc） |
| [`eval_results/README.md`](eval_results/README.md) | 产物索引 |

**后续优先：** McNemar（§8.1）· 失败消融表 · Acc–K 图 ·（可选）效率曲线。**不做** v2 / 扩大 CLIP 选区 / 真 MV / 714。

---

## 目录

| 路径 | 内容 |
|------|------|
| `mage_vl/` | Mage-VL + streaming；**无** `data/` / `models/` |
| `mage_vl/cultural_lora/` | Cultural LoRA（后置） |
| `Explore-Then-Select/` | Query-IP（`mage_query_ip/` · `COMMANDS.md`） |
| `VLMEvalKit/` | vendored 评测树 |
| `patches/vlmevalkit/` | 相对上游 diff |
| `eval_results/` | 指标归档（请保留） |
| `scripts/` · `envs/` | 复现辅助 / freeze |

## 权重与环境（不进 Git）

- Mage 权重：`mage_vl/models/`（gitignore）  
- 评测环境：仓外 `../VLMEvalKit/vlmeval_env`，editable → **`VQA/VLMEvalKit`**

```bash
ls eval_results/ && cat eval_results/README.md
ls eval_results/queryip_samek/{n200,clip_frame_n200,region_sem_n200,region_edge_n200}/
```
