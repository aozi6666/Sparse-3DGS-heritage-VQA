# 摘要 · depthft_mip360 扩展（P3，2026-10-28）

权重：`3DGS/vggt/models/depth_head_ft_mip360.pt`（任务内微调；非 hold-out）。

## Kitchen 粗训

| 设定 | test PSNR |
|------|----------:|
| orig VGGT-enhanced + Drop | 26.141 |
| depthft_mip360 + Drop | **25.978**（vs orig **−0.163**） |

## Kitchen repair / CADR（复用 RV51 LoRA）

| 设定 | PSNR | 对照 |
|------|-----:|------|
| depthft → RV51 无 CADR | **27.022** | vs A0 **+0.055**；vs 自身粗训 **+1.04** |
| depthft → RV51 + CADR | **27.074** | vs A3 **+0.110**；vs depthft-A0 **+0.052** |
| （对照）orig → A0 / A3 | 26.967 / 26.964 | CADR≈持平 |

**读法：** 粗训 depthft 略伤；repair 把弱 init **拉回并略超** 原版 VGGT repair。CADR 在 depthft 上有微小正增益（≈+0.05），仍远低于设计目标 +0.2。

## Garden 粗训

| 设定 | test PSNR |
|------|----------:|
| classic+Drop | 33.486 |
| orig VGGT+Drop | 33.374 |
| depthft+Drop | **33.530** |

| delta | dPSNR |
|-------|------:|
| depthft − classic | **+0.044** |
| depthft − orig VGGT | **+0.156** |

**读法：** garden 上 depthft 略优于 classic / 原版 VGGT（噪声级～小正）；与 kitchen 粗训微负形成**场景依赖**，符合「任务微调」而非跨场景泛化叙事。

## 总判

depth-adapt 可写成：**粗训不稳定/场景依赖；接上既有 LoRA repair 后 kitchen 可恢复到 ≥ 原版 VGGT repair；garden 粗训微正。** 不宜宣称大幅涨点或普适 SOTA。
