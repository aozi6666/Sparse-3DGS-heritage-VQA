# 改进设想：更强 SD1.5 底模 + 重训 LoRA（方案 A）

> 状态：设想 / 待实施。当前主线仍用原版 `v1-5-pruned.ckpt` 跑通基线。  
> 范围：只动扩散修复（步骤 7–8）；**不改 VGGT / 粗训 GS / LOO**。  
> 环境：继续 `.venv-ddgs-vggt`，不引入 Diffusers 全家桶。  
> 分支：`feat/sd15-stronger-base-lora`

## 1. 背景与目标

线 B 流程：

1. VGGT hull → 2. 粗 GS(±Drop) → 3. 评测 → 5–6. LOO → **7. LoRA** → **8. repair** → 9. 最终评测

原链路修复底座：

- SD：`models/v1-5-pruned.ckpt`
- ControlNet-Tile：`models/control_v11f1e_sd15_tile.pth` + 同名 yaml

本方案目标：在 **架构不变（仍 SD1.5 + ControlNet-Tile + cldm + minLoRA）** 的前提下，换更强的真实向 SD1.5 底模，并 **重新训练 LoRA**，与原底座做 A/B。

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

- 覆盖损坏或不完整的 `v1-5-pruned.ckpt`（基线对照用；坏文件可删残片另下完整版）
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

## 3. 代码改动清单（待实施）

### 3.1 `train_lora.py`：底模路径参数化

现状写死：

```python
model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cpu'), strict=False)
model.load_state_dict(load_state_dict(f'./models/{args.model_name}.pth', location='cpu'), strict=False)
```

建议：

- 新增 `--sd_ckpt`（默认仍 `./models/v1-5-pruned.ckpt`，保证原链路零行为变化）
- ControlNet 仍：`./models/{model_name}.pth`，默认 `control_v11f1e_sd15_tile` **先不换**

示例：

```python
parser.add_argument('--sd_ckpt', type=str, default='./models/v1-5-pruned.ckpt')
# ...
model.load_state_dict(load_state_dict(args.sd_ckpt, location='cpu'), strict=False)
model.load_state_dict(load_state_dict(f'./models/{args.model_name}.pth', location='cpu'), strict=False)
```

### 3.2 `7_train_lora.sh`：新实验目录 + 传入新底模

- 新 `LORA_EXP`，例如：`controlnet_finetune/kitchen_drop_rv51`
- 传入：`--sd_ckpt ./models/Realistic_Vision_V5.1.safetensors`
- **必须重跑 LoRA**（旧 LoRA 绑定旧底模，不能直接复用）
- 旧目录 `controlnet_finetune/kitchen_drop` **保留**作对照

### 3.3 `8_train_repair.sh`：指向新 LoRA，粗模不动

- `system.exp_name="output/${LORA_EXP}"` → 新 LoRA 目录
- `system.init_dreamer="${GS_DIR}"` → 仍为现有 VGGT+Drop 粗模（如 `output/gs_init/kitchen_drop`）
- 建议新 `tag`，例如 `kitchen_drop_rv51`，避免覆盖旧 `last.ply`
- **VGGT 链路零改动**

### 3.4 `_common.sh`（可选）

增加可覆盖变量，便于 A/B：

```bash
SD_CKPT="${SD_CKPT:-./models/v1-5-pruned.ckpt}"
LORA_EXP="${LORA_EXP:-controlnet_finetune/kitchen_drop}"   # 新实验改 kitchen_drop_rv51
REPAIR_TAG="${REPAIR_TAG:-kitchen_drop}"                    # 新实验改 kitchen_drop_rv51
```

### 3.5 明确不改

- `vggt_visual_hull_enhanced.py`
- `1_vggt_hull.sh` / `2_train_gs.sh` / `5_loo_*.sh` / `6_loo_*.sh`
- ControlNet 权重与 yaml
- 全局另装 Diffusers 环境

## 4. 实验协议（A/B）

| 组 | 底模 | LoRA 输出 | repair tag | 说明 |
| --- | --- | --- | --- | --- |
| Baseline | `v1-5-pruned.ckpt` | `controlnet_finetune/kitchen_drop` | `kitchen_drop` | 当前原链路 |
| RV51 | `Realistic_Vision_V5.1.safetensors` | `controlnet_finetune/kitchen_drop_rv51` | `kitchen_drop_rv51` | 本方案 |

对比：`results.json`（PSNR/SSIM/LPIPS）+ 主观图；粗模 / LOO 数据两侧共用。

建议跑法（新实验，参数化完成后）：

```bash
cd 3DGS/GaussianObject/sh/lineB_gaussianobject
SD_CKPT=./models/Realistic_Vision_V5.1.safetensors \
LORA_EXP=controlnet_finetune/kitchen_drop_rv51 \
REPAIR_TAG=kitchen_drop_rv51 \
USE_DROP=1 SKIP_PATH=1 bash 0_run_all.sh 7
```

（从步骤 7 起；1–6 复用已有结果。）

## 5. 论文级依据：涨点仍主要靠 LOO + Tile LoRA

来源：[GaussianObject](https://arxiv.org/abs/2402.10259)（Yang et al.）

要点：

1. **修复模型不是裸用 SD 文生图**，而是 ControlNet-Tile（SD1.5）+ 场景 LoRA，输入为「损坏渲染」，输出为高保真图，再回流优化 3DGS。
2. **训练对靠自生成**：leave-one-out（LOO）3DGS +（可选）高斯属性 3D noise，得到 corrupted / clean 对；没有这对数据，换再强底模也学不会「修稀视图伪影」。
3. **实现细节（论文）**：ControlNet-Tile + SD1.5；LoRA（minLoRA）打在 text-encoder 与 transformer；约 1800 step、rank 64、lr 1e-3。
4. **消融含义**：去掉 repair setup / repair process 会明显掉 LPIPS/PSNR/SSIM；说明论文增益核心在 **「LOO（及噪声）造对 → Tile 条件修复 → LoRA 适配场景 → repair 优化 GS」**，而不是单纯换一代生成模型。
5. **对本方案的推论**：换 Realistic Vision 等更强 SD1.5 底模，预期是改善纹理/真实感先验的 **边际增益**；若跳过重训 LoRA 或削弱 LOO，不应期待「论文级」涨点。后续若还要冲指标，优先保证 LOO 质量与 Tile LoRA 训练充分，再考虑换底模或更强 Tile 变体。

相关后续工作（非本阶段）：RI3D 仍沿用 ControlNet + LOO 修复思路；WaveletGaussian 在小波域做扩散，属于方法级改动，不是「换底座」。

## 6. 实施顺序（建议）

1. **现在（原链路）**：完整 `v1-5-pruned.ckpt` + 原 `LORA_EXP`，跑通 7→8，记下基线指标。
2. **本分支**：按 §3 做路径参数化（默认值保持原行为）。
3. 下载 `Realistic_Vision_V5.1.safetensors` 到 `models/`（独立文件名）。
4. 新 `LORA_EXP` / `REPAIR_TAG` 从步骤 7 重跑，与基线对比。
5. 文档与代码就绪后再决定是否合并回 `main`。

## 7. 风险与回滚

- 社区底模 key 与 SD1.5 略有差异：继续 `strict=False`；加载后看日志缺失 key 是否异常多。
- 磁盘：RV5.1 ≈ 4.3G；数据盘紧时先处理损坏的 2G `v1-5-pruned.ckpt` 残片。
- 回滚：不传 `--sd_ckpt` / 不改 `LORA_EXP` 即回到原链路；旧 LoRA 目录勿删。

## 8. 检查清单

- [ ] 基线：`v1-5-pruned.ckpt` 可完整加载 / 原 7→8 出数
- [ ] `models/Realistic_Vision_V5.1.safetensors` 已就位且完整
- [ ] `train_lora.py` 增加 `--sd_ckpt`，默认仍指向 v1-5
- [ ] `7_train_lora.sh` / `_common.sh` 支持新 `LORA_EXP` + `SD_CKPT`
- [ ] `8_train_repair.sh` 指向新 LoRA，`init_dreamer` 仍为原 `GS_DIR`
- [ ] ControlNet 仍为 `control_v11f1e_sd15_tile`
- [ ] VGGT / 粗训 / LOO 脚本未改
- [ ] A/B 指标与主观对比记录
