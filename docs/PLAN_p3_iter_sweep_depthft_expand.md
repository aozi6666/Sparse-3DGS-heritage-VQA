# P3：步数 sweep + depthft_mip360 → repair/CADR/garden

> 权重：现成 `depth_head_ft_mip360.pt`（任务内微调再用于 VGGT 场景；非 hold-out 泛化叙事）。  
> Agent 不代跑；结果-only。

## 任务 1：PSNR vs iter

```bash
bash run_iter_sweep_classic_vs_vggt.sh
# 断点：bash run_iter_sweep_classic_vs_vggt.sh 2
```

| GS | init | 点位 |
|----|------|------|
| `kitchen_drop_classic_sweep` | `visual_hull_4` | 3k/5k/7k/10k |
| `kitchen_drop_sweep` | `visual_hull_vggt_enhanced_4` | 同上 |

CSV：[`lineB_kitchen_iter_sweep.csv`](../3DGS/eval_results/lineB_kitchen_iter_sweep.csv)  
不覆盖 `kitchen_drop` / `kitchen_drop_classic`。

## 任务 2a：depthft → RV51 ± CADR

粗训已有：`kitchen_drop_depthft_mip360` ≈ **25.978**（不重训）。

```bash
bash run_depthft_mip360_repair_cadr.sh
```

| tag | CADR |
|-----|------|
| `kitchen_drop_depthft_mip360_rv51` | 关 |
| `kitchen_drop_depthft_mip360_rv51_conf` | 开 |

复用 LoRA `controlnet_finetune/kitchen_drop_rv51`。写入 kitchen 主 CSV。

## 任务 2b：garden classic vs depthft

```bash
bash run_garden_classic_vs_depthft_coarse.sh
# 默认 VGGT_RESO=4
```

- classic：复用 `garden_drop_classic`
- 新：`garden_drop_depthft_mip360`（不覆盖 `garden_drop`）

## 明确不做

hold-out 重训；A2/A4/A5；bonsai；V2 对齐；替换主线 `model.pt`。
