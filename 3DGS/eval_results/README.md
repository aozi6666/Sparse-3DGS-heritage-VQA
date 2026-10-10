# 工作 A · 稀疏 3DGS 指标归档

与 [`VQA/eval_results/`](../../VQA/eval_results/) 对称：**小体积 CSV 进 Git**，大产物仍在 ignore 的 `**/output/`。

## 文件

| 文件 | 说明 |
|------|------|
| [`lineB_kitchen_mip360.csv`](lineB_kitchen_mip360.csv) | mip360/kitchen · lineB 主线（VGGT hull → Drop 粗训 → LOO → LoRA → repair） |

## 协议（本表）

- 场景：`mip360/kitchen`，`resolution=4`，`sparse_view_num=9`，`sh_degree=2`
- 初始化：`visual_hull_vggt_enhanced_4`（VGGT 增强 hull）
- 粗训：`USE_DROP=1`，`iterations=10000` → `ckpt_tag=ours_10000`
- 修复：SD `v1-5-pruned.ckpt` + ControlNet-Tile + LoRA `rank=32` → `load_ply=last.ply` → `ckpt_tag=ours_None`
- `split=test`：稀疏训练视角以外的测试视角；`split=all`：全部视角

## 已记录摘要（2026-10-10）

### LoRA repair 相对粗训（`ours_10000` → `ours_None`）

| split | dPSNR | dSSIM | dLPIPS |
|-------|------:|------:|-------:|
| **test** | **+0.795** | **+0.00255** | **−0.00627** |
| all | +1.296 | +0.00332 | −0.00711 |

绝对指标见 CSV 中 `row_type=absolute` 行。

### 待补对照（CSV 中 `row_type=delta_pending`）

| 对照 | 怎么补 |
|------|--------|
| **VGGT 补多少点** | 同设定下分别用 `visual_hull_4` 与 `visual_hull_vggt_enhanced_4` 粗训+render，填 `vggt_vs_classic_hull` 的 dPSNR/dSSIM/dLPIPS |
| **Drop 消融** | `bash 2b_train_gs_drop_ablation.sh`，对比 `output/gs_init/kitchen` vs `kitchen_drop` |

当前机器上只有 `kitchen_drop` 产物，**尚无 classic hull / no-Drop 的 `results.json`**，故 VGGT / Drop 增益行先留空占位。

## 追加新结果

1. 跑完对应 `render.py` 后，把新 `results.json` 数值追加进 CSV（或新开 `lineB_<scene>.csv`）。
2. 增益行：`row_type=delta`，填 `dPSNR/dSSIM/dLPIPS` 与 `vs_stage`。
3. 保持小数与源路径，便于复现核对。
