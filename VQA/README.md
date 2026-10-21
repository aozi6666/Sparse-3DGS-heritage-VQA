# 工作 B · 视频问答与评测

本目录为系统仓中的 **VQA / 评测** 工作区（GitHub：`Sparse-3DGS-heritage-VQA`）。

## 今日定稿：Mage Query-IP（2026-10-21）

**贡献：** 固定减 token 预算 K 下，按问题重加权选 patch，比同预算乱砍更准（不重训；不声称超过满预算 Mage）。

| 同 K · short 前 200 · `keep_ratio=0.75` | Acc |
|------|-----|
| prefix | 57.5% |
| random | 59.5% |
| **queryip（proxy）· 主推** | **64.0%** |
| bitcost 纯 / hybrid | 58.5% / 58.0%（失败消融） |

| 旁证（非同 K） | overall | token |
|----------------|---------|-------|
| 行 A 默认 Mage | 63.7% | 1.00× |
| 行 B QueryIP @0.75 | 62.3% | ≈0.75× |

- **计划（定稿）：** [`docs/PLAN_mage_query_IP_select.md`](../docs/PLAN_mage_query_IP_select.md)  
- **命令（Agent 不代跑）：** [`Explore-Then-Select/COMMANDS.md`](Explore-Then-Select/COMMANDS.md)  
- **产物索引：** [`eval_results/README.md`](eval_results/README.md)  
- **实现：** `Explore-Then-Select/mage_query_ip/`  
- **路线 A 归档：** [`docs/PLAN_routeA_true_mv.md`](../docs/PLAN_routeA_true_mv.md)

**K 扫描（已完成）：** 0.3–0.75 queryip>random；0.9 略负。做图：[`eval_results/queryip_samek/ksweep_proxy/`](eval_results/queryip_samek/ksweep_proxy/)（`plot_acc_*.csv`）。  
**后续预案：** 满 token Mage 效率曲线（计划 §5；复用 ksweep + 补满预算一点）

---

## 目录

| 路径 | 内容 |
|------|------|
| `mage_vl/` | 基于 [microsoft/Mage](https://github.com/microsoft/Mage) 的 `mage_vl`；另含 streaming 相关增量 |
| `mage_vl/cultural_lora/` | Cultural LoRA：子集硬抽、LLM PEFT、冒烟训练 |
| `Explore-Then-Select/` | Query-IP 选区包装 + 评测脚本（`mage_query_ip/`、`COMMANDS.md`） |
| `VLMEvalKit/` | 基于 [open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit) 的可运行树（含本仓改动） |
| `patches/vlmevalkit/` | 相对上游的 diff |
| `eval_results/` | **实验指标与同 K 消融**（请保留） |
| `scripts/` | 复现辅助脚本 |
| `envs/requirements-*.txt` | 评测 / Cultural LoRA 依赖 |

**不包含**：`mage_flow`（仓外）、`mage_vl/data`、`mage_vl/models`、venv、ffmpeg 大包。

## 上游与许可

- Mage-VL：遵循 Mage 仓库中 Mage-VL 许可  
- VLMEvalKit：见 `VLMEvalKit/LICENSE`；来源说明见 `VLMEvalKit/UPSTREAM.md`

## 权重与环境（不进 Git）

- Mage-VL 权重：`mage_vl/models/`（gitignore）或本地路径  
- 评测环境：仓外 `../VLMEvalKit/vlmeval_env`，或按 `envs/requirements-vlmevalkit.txt` 重建，editable 指到 **`VQA/VLMEvalKit`**

## 查看已有结果

```bash
ls eval_results/
cat eval_results/README.md
# 主表
ls eval_results/queryip_samek/n200/
# 行 A / 行 B
ls eval_results/baseline_mage/ eval_results/queryip_rowB_full/
```
