# Proxy keep_ratio 扫描（已完成 · 2026-10-21）

**设定：** Video-MME short 前 **N=200**；modes=`prefix,random,queryip`；seed=0；32 frame / HEVC。  
**0.75** 来自 n200 hybrid summary seed（与主表一致）；`0.3/0.5/0.9` 新跑。  
**逐题 jsonl 已按 `KEEP_JSONL=0` 删除**；Acc 以本目录 summary + CSV 为准（足够画图）。

## 做图用数据（勿删）

| 文件 | 用途 |
|------|------|
| [`Mage-VL-QueryIP_proxy_ksweep.csv`](Mage-VL-QueryIP_proxy_ksweep.csv) | 宽表主表（脚本原生产物） |
| [`plot_acc_vs_keep_ratio.csv`](plot_acc_vs_keep_ratio.csv) | **推荐画图**：含 Acc%、Δ vs random |
| [`plot_acc_long.csv`](plot_acc_long.csv) | 长表（seaborn/plotly 友好） |
| [`Mage-VL-QueryIP_proxy_ksweep.jsonl`](Mage-VL-QueryIP_proxy_ksweep.jsonl) | 每档一行完整 summary |
| `Mage-VL-QueryIP_short60_samek4_summary_n200_kr*.json` | 分档原始 summary |

建议图：横轴 `keep_ratio`，纵轴 Acc%；三条线 prefix / random / queryip。

## 结果表

| keep_ratio | prefix | random | queryip | Δ(queryip−random) |
|------------|--------|--------|---------|-------------------|
| 0.30 | 53.0% | 51.5% | **55.5%** | **+4.0 pp** |
| 0.50 | 53.0% | 57.0% | **61.0%** | **+4.0 pp** |
| 0.75 | 57.5% | 59.5% | **64.0%** | **+4.5 pp**（主表） |
| 0.90 | 58.0% | **61.5%** | 60.5% | **−1.0 pp** |

## 结论（写论文 / 画图口径）

1. **预算偏紧～中等（0.3～0.75）：** queryip 稳定优于同 K random → 支撑「少 token 下按问题选区更准」。  
2. **接近满预算（0.9）：** 优势消失，且略低于 random → **不要写「全程都赢」**；方法价值在受限 K。  
3. **主卖点仍是 0.75 主表**；本扫描是 Acc–K 曲线旁证，不是替换主表。  
4. 本曲线 **相对 random/prefix**，不是相对满预算 Mage（后者见计划 §5 效率曲线预案）。

复现：`VQA/Explore-Then-Select/scripts/eval_proxy_ksweep.sh`（已完成档会 SKIP）。
