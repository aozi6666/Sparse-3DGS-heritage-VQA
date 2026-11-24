# 摘要 · repair / CADR / 底模（kitchen / test）

源 CSV：[`lineB_kitchen_mip360.csv`](lineB_kitchen_mip360.csv) · 计划 [`docs/PLAN_cadr_ablation_c4.md`](../../docs/PLAN_cadr_ablation_c4.md)

## 主表 C0–C4

| ID | 含义 | test PSNR |
|----|------|----------:|
| C0 | classic 粗训 | ≈**27.11** |
| C1 | SD1.5 repair | ≈26.94（vs 旧 VGGT 粗训 +0.80） |
| C2 / A0 | RV51 无 CADR | **26.967** |
| C3 / A3 | RV51 + 完整 CADR | **26.964**（vs A0 −0.003） |
| C4 | abs16 无 CADR | **26.928** |

## CADR 分量（A-full）

| ID | tag | PSNR | vs A0 | vs A3 |
|----|-----|-----:|------:|------:|
| A2 | rv51_trust | 26.997 | +0.031 | +0.034 |
| A4 | rv51_cov | 26.996 | +0.029 | +0.032 |
| A5 | rv51_aden | 26.852 | −0.115 | −0.111 |
| abs16+CADR | abs16_conf | 26.868 | — | C4 相对它 +0.060 |

**结论：** CADR 族≈持平（目标 +0.2 未达）；A5 略伤默认关；换底模无实质涨点。主涨点在 classic 粗训与 LoRA repair vs 粗训，不在 CADR/底模。

## P2 classic init repair（2026-10-28，已归档 CSV）

| 设定 | test PSNR | vs A0 (VGGT→RV51) | vs A3 (CADR) |
|------|----------:|------------------:|-------------:|
| classic+Drop → RV51 无 CADR | **27.085** | **+0.118** | +0.121 |
| （对照）VGGT→RV51 A0 | 26.967 | — | — |
| （对照）classic 粗训 Drop | 27.103 | — | — |

**结论：** classic init + 复用 RV51 LoRA **略优于** VGGT 线 repair（+0.12 dB），但相对自身粗训几乎持平（27.085 vs 27.103，Δ≈−0.02）。消融合理：粗训阶段 classic≫VGGT（≈−1 dB）的优势传到 repair，但 repair **没有翻盘式涨点**——瓶颈仍在 init，不在 CADR/底模。论文可写：下游 repair 无法抹平 init 差距；classic 仍是更强基线。
