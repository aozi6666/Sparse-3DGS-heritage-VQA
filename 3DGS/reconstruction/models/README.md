# GaussianObject 权重（本地，不进 Git）

| 文件 | 用途 | 约大小 |
|------|------|--------|
| `absolutereality_v16.safetensors` | **当前主线** SD 底模（LoRA / repair） | ~2.0G |
| `Realistic_Vision_V5.1.safetensors` | 对照底模（可用环境变量切回） | ~4.0G |
| `control_v11f1e_sd15_tile.pth` | ControlNet-Tile | ~1.4G |
| `control_v11f1e_sd15_tile.yaml` | ControlNet 配置（仓内已有） | 小 |

下载 AbsoluteReality（在本目录执行）：

```bash
cd 3DGS/GaussianObject/models
wget -c -O absolutereality_v16.safetensors \
  "https://huggingface.co/Yntec/AbsoluteReality/resolve/main/absolutereality_v16.safetensors"
```

线 B 默认：`SD_CKPT=./models/absolutereality_v16.safetensors`，  
`LORA_EXP=controlnet_finetune/<scene>_abs16`，`REPAIR_TAG=<scene>_abs16_conf`（CADR 开）。

回退 RV51：

```bash
SD_CKPT=./models/Realistic_Vision_V5.1.safetensors \
LORA_EXP=controlnet_finetune/${GS_NAME}_rv51 \
BASE_TAG=${GS_NAME}_rv51 \
...
```
