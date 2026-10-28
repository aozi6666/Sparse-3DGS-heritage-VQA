# 摘要 · 粗训 init×Drop / ply / V2（kitchen）

源：[`lineB_kitchen_mip360.csv`](lineB_kitchen_mip360.csv) · [`ply_init_stats_kitchen.csv`](ply_init_stats_kitchen.csv)

## 2×2 test PSNR

| | 无 Drop | Drop |
|--|--------:|-----:|
| classic | **27.11** | 27.10 |
| VGGT-enhanced | 26.05 | 26.14 |

| 效应 | dPSNR |
|------|------:|
| VGGT−classic @Drop0/1 | −1.07 / −0.96 |
| Drop−noDrop @classic/VGGT | −0.01 / +0.09 |

**结论：** init 主导（classic ≫ VGGT）；Drop 弱正/条件性。

## ply

| init | n_points | bbox |
|------|---------:|------|
| visual_hull_4 | 8616 | ~1.5–1.9 |
| visual_hull_vggt_enhanced_4 | 2874 | ~3.7–3.9 |

## V2 纯 VGGT+depthft（负对照）

test PSNR **11.78**（vs kitchen_drop ≈−14 dB）→ 缺度量对齐；勿主打。

## P2 garden（2026-10-28 已归档）

源：[`lineB_garden_mip360.csv`](lineB_garden_mip360.csv) · Drop=1 · 粗训 10k

| 设定 | test PSNR | n_init |
|------|----------:|-------:|
| classic+Drop | **33.486** | 10275 |
| VGGT+Drop | 33.374 | 2455 |
| VGGT−classic | **−0.112** | — |

对照 kitchen（同设定 Drop=1）：VGGT−classic ≈ **−0.96**。

**结论：** 方向一致（classic ≥ VGGT），但 garden 差距仅 **~0.1 dB**，kitchen 的 −1 dB **是场景特异、非普适**。绝对 PSNR garden≫kitchen（~33 vs ~27）也说明难度/尺度不同；论文应写「init 优势场景依赖」，勿把 kitchen −1 dB 写成全局。VGGT hull 用 `VGGT_RESO=4`（garden OOM 兜底）。

## P3 PSNR–iter（已归档，独立 `*_sweep` 目录）

源：[`lineB_kitchen_iter_sweep.csv`](lineB_kitchen_iter_sweep.csv)

| iter | classic+Drop | VGGT+Drop | dPSNR |
|-----:|-------------:|----------:|------:|
| 3000 | 27.40 | 25.36 | −2.04 |
| 5000 | 27.18 | 25.44 | −1.75 |
| 7000 | 27.32 | 25.61 | −1.71 |
| 10000 | 27.36 | 25.83 | −1.53 |

**结论：** 全程 classic ≫ VGGT；差距随步数略收窄但仍 ≥1.5 dB → **不是欠训假象**。sweep 末轮与主表基线有小幅 run 方差（重训独立目录），叙事用相对曲线即可。
