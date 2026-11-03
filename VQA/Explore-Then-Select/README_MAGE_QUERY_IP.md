# Mage Query-IP 选区（本目录）

对齐 [`docs/PLAN_mage_query_IP_select.md`](../../docs/PLAN_mage_query_IP_select.md) **结论与后续**。

## 主路径

```text
Mage codec → proxy queryip（运动/残差）同 K top-K → 稠密打包 → Mage-ViT + Qwen3
对照：同 K prefix / random
```

- **主推** `queryip`（约 +4 pp vs random @ n=200）。**不重训**；不改 `mage_vl/` 源码。
- **已停：** CLIP 帧/区域/edge 语义选区；bitcost 主推；真 MV；完整 v2。
- 失败模块文件头有 **【主路径不用】** 中文注释，代码保留供复现。

## 目录

| 路径 | 说明 |
|------|------|
| `mage_query_ip/codec_reweight.py` | **主推** proxy 同 K + pack |
| `mage_query_ip/mage_forward.py` / `eval_short60_samek.py` | 推理与评测 |
| `mage_query_ip/question_type.py` | 静/动 α/β（不作贡献，默认权重仍用） |
| `mage_query_ip/{clip_frame,frame_budget,region_clip,mvres}*.py` | **归档** 失败消融 |
| `COMMANDS.md` | 终端命令（Agent 不代跑 Acc） |

评测适配（VLMEvalKit）：模型名 `Mage-VL-QueryIP`，见 `VQA/VLMEvalKit/vlmeval/vlm/mage_vl_query_ip.py`（全量行 B 仍为 canvas proxy）。

## 同 K 成功标准（soccer 四路）

1. 四路 `n_vis_tokens` / `n_patches` 相同。
2. `queryip_bitcost` kept 与 `queryip` Jaccard **&lt; 1**（有差异）。
3. 答案非空 → `SAMEK4_OK`；完全重合 → `SAMEK4_REDUNDANT`。

**状态：** 代码已就绪；GPU 结果待你跑 §COMMANDS §2。

## 诚实边界

- `queryip`：解码 canvas 上的帧差/方差 **proxy**，不是 codec 内部 MV。
- `queryip_bitcost`：官方 block **bitcost** 单通道（无残差图）；论文可写「官方 codec bitcost + 问题重加权」，**不能**写「真 MV+残差」。
