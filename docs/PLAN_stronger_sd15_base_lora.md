# 改进设想：更强 SD1.5 底模 + 重训 LoRA（方案 A）

> 状态：**主线已切到 RV5.1**（默认 `Realistic_Vision_V5.1.safetensors` + `*_rv51` LoRA/repair）；本地不再依赖 `v1-5-pruned.ckpt`。  
> 范围：只动扩散修复（步骤 7–8）；**不改 VGGT / 粗训 GS / LOO**。  
> 环境：继续 `.venv-ddgs-vggt`，不引入 Diffusers 全家桶。  
> 分支：`feat/sd15-stronger-base-lora`（可合并 main）

## 1. 背景与目标

线 B 流程：

1. VGGT hull → 2. 粗 GS(±Drop) → 3. 评测 → 5–6. LOO → **7. LoRA** → **8. repair** → 9. 最终评测

当前主线修复底座：

- SD：`models/Realistic_Vision_V5.1.safetensors`
- ControlNet-Tile：`models/control_v11f1e_sd15_tile.pth` + 同名 yaml

本方案：在 **架构不变（仍 SD1.5 + ControlNet-Tile + cldm + minLoRA）** 下使用真实向底模并重训 LoRA；历史 v1-5 A/B 指标见 `3DGS/eval_results/`。

**与 VGGT 的关系：** 换底座只影响步骤 7–8；`1_vggt_hull` / 粗训 / LOO 输出可复用，零改动。

## 2. 实际采用的新底模

| 项 | 值 |
| --- | --- |
| 模型 | Realistic Vision V5.1（SD1.5） |
| 文件 | `Realistic_Vision_V5.1.safetensors`（约 4.27GB） |
| 来源 | https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE |
| 直链 | https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1.safetensors |
| 放置目录 | `3DGS/GaussianObject/models/Realistic_Vision_V5.1.safetensors` |

**不要：**

- 换 ControlNet（继续 `control_v11f1e_sd15_tile`）
- 使用 inpainting / SDXL / Diffusers 多文件夹权重
- 动 `vggt_visual_hull_enhanced.py`、`1_vggt_hull.sh`、`2_train_gs.sh`、LOO 脚本

备选（后续 A/B，非本阶段）：

| 模型 | 文件 | 地址 |
| --- | --- | --- |
| AbsoluteReality v1.6 | `absolutereality_v16.safetensors` | https://huggingface.co/Yntec/AbsoluteReality/resolve/main/absolutereality_v16.safetensors |
| epiCRealism | Civitai **单文件** SD1.5 checkpoint（不要用 Diffusers 版） | https://civitai.com/models/25694/epicrealism |

下载示例：

```bash
cd 3DGS/GaussianObject/models
wget -c -O Realistic_Vision_V5.1.safetensors \
  "https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1.safetensors"
```

`cldm/model.py` 已支持 `.safetensors`（见 `load_state_dict`），一般无需再转 ckpt。

## 3. 代码改动清单（已实施）

### 3.1 `train_lora.py`：`--sd_ckpt`

- 默认 `./models/Realistic_Vision_V5.1.safetensors`
- ControlNet 仍：`./models/{model_name}.pth`，默认 `control_v11f1e_sd15_tile`

### 3.2 repair：`system.sd_ckpt`

- [`gaussian_object_system.py`](../3DGS/GaussianObject/threestudio/systems/gaussian_object_system.py) `Config.sd_ckpt`（默认同上 RV5.1）
- `8_train_repair.sh` 传入 `system.sd_ckpt="${SD_CKPT}"`（必须与训 LoRA 同一底模）

### 3.3 脚本：`7` / `8` / `9` + `_common.sh`

```bash
SD_CKPT="${SD_CKPT:-./models/Realistic_Vision_V5.1.safetensors}"
LORA_EXP="${LORA_EXP:-controlnet_finetune/${GS_NAME}_rv51}"
REPAIR_TAG="${REPAIR_TAG:-${GS_NAME}_rv51}"
```

- `7_train_lora.sh`：`--sd_ckpt "${SD_CKPT}"`
- `8_train_repair.sh`：`tag` / `exp_name` 跟 `REPAIR_TAG` / `LORA_EXP`；`init_dreamer` 仍为 `GS_DIR`
- `9_render_last.sh`：`REPAIR_TAG != GS_NAME` 时备份/归档 `ours_None`

### 3.4 明确不改

- `vggt_visual_hull_enhanced.py`
- `1_vggt_hull.sh` / `2_train_gs.sh` / `5_loo_*.sh` / `6_loo_*.sh`
- ControlNet 权重与 yaml
- 全局另装 Diffusers 环境

## 4. 实验协议（A/B）

| 组 | 底模 | LoRA 输出 | repair tag | 说明 |
| --- | --- | --- | --- | --- |
| **主线 RV51** | `Realistic_Vision_V5.1.safetensors` | `controlnet_finetune/kitchen_drop_rv51` | `kitchen_drop_rv51` | 默认 |
| 历史对照（已归档指标） | 曾用官方 SD1.5 ckpt | `controlnet_finetune/kitchen_drop` | `kitchen_drop` | 见 eval_results CSV |

默认跑法（权重就绪后）：

```bash
cd 3DGS/GaussianObject/sh/lineB_gaussianobject
USE_DROP=1 SKIP_PATH=1 bash 0_run_all.sh 7
```

（从步骤 7 起；1–6 复用已有结果。默认即 RV51。）

## 5. 论文级依据：涨点仍主要靠 LOO + Tile LoRA

来源：[GaussianObject](https://arxiv.org/abs/2402.10259)（Yang et al.）

要点：

1. **修复模型不是裸用 SD 文生图**，而是 ControlNet-Tile（SD1.5）+ 场景 LoRA，输入为「损坏渲染」，输出为高保真图，再回流优化 3DGS。
2. **训练对靠自生成**：leave-one-out（LOO）3DGS +（可选）高斯属性 3D noise，得到 corrupted / clean 对；没有这对数据，换再强底模也学不会「修稀视图伪影」。
3. **实现细节（论文）**：ControlNet-Tile + SD1.5；LoRA（minLoRA）打在 text-encoder 与 transformer；约 1800 step、rank 64、lr 1e-3。
4. **消融含义**：去掉 repair setup / repair process 会明显掉 LPIPS/PSNR/SSIM；说明论文增益核心在 **「LOO（及噪声）造对 → Tile 条件修复 → LoRA 适配场景 → repair 优化 GS」**，而不是单纯换一代生成模型。
5. **对本方案的推论**：换 Realistic Vision 等更强 SD1.5 底模，预期是改善纹理/真实感先验的 **边际增益**；若跳过重训 LoRA 或削弱 LOO，不应期待「论文级」涨点。后续若还要冲指标，优先保证 LOO 质量与 Tile LoRA 训练充分，再考虑换底模或更强 Tile 变体。

相关后续工作（非本阶段）：RI3D 仍沿用 ControlNet + LOO 修复思路；WaveletGaussian 在小波域做扩散，属于方法级改动，不是「换底座」。

## 6. 实施顺序（已完成）

1. 参数化 `--sd_ckpt` / `system.sd_ckpt` / 脚本环境变量。
2. 下载 RV5.1，默认切到 RV51 + `*_rv51` 目录。
3. 本地删除官方 SD1.5 大权重；下载脚本改指 RV5.1。
4. 合并 `feat/sd15-stronger-base-lora` → `main`（见仓库 README 旁操作）。

## 7. 风险

- 社区底模 key 与 SD1.5 略有差异：继续 `strict=False`。
- 磁盘：RV5.1 ≈ 4G；repair 后及时删 `tb_logs`。

## 8. 检查清单

- [x] `models/Realistic_Vision_V5.1.safetensors` 主线默认
- [x] `train_lora.py` / repair / `_common.sh` 默认 RV51 + `*_rv51`
- [x] `models/download_hf_models.py` + `models/README.md` 已更新
- [x] ControlNet 仍为 `control_v11f1e_sd15_tile`
- [x] VGGT / 粗训 / LOO 脚本未改
- [x] A/B 指标见 `3DGS/eval_results/`（RV51 ≈ 持平，无实质涨点）
