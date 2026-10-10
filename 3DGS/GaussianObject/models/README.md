# GaussianObject 权重（本地，不进 Git）

| 文件 | 用途 | 约大小 |
|------|------|--------|
| `Realistic_Vision_V5.1.safetensors` | **主线** SD 底模（LoRA / repair） | ~4.0G |
| `control_v11f1e_sd15_tile.pth` | ControlNet-Tile | ~1.4G |
| `control_v11f1e_sd15_tile.yaml` | ControlNet 配置（仓内已有） | 小 |

下载（在本目录执行）：

```bash
cd 3DGS/GaussianObject/models
python download_hf_models.py
# 或
wget -c -O Realistic_Vision_V5.1.safetensors \
  "https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1.safetensors"
```

线 B 默认：`SD_CKPT=./models/Realistic_Vision_V5.1.safetensors`，  
`LORA_EXP=controlnet_finetune/<scene>_rv51`，`REPAIR_TAG=<scene>_rv51`。
