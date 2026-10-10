> 本文档由仓库根目录 `README.md` 迁入，全文保留（文中历史路径多为当时的 `/root/autodl-tmp/GaussianObject`）。  
> **当前布局**：工作 A 在 `3DGS/GaussianObject/`、`3DGS/vggt/`；工作 B 在 `VQA/`。系统级总览见仓库根 [`README.md`](../../README.md)。

<div align="center">

# GaussianObject复现记录

2025.5.28

![截屏2025-05-29 11.26.14](/Users/zhangao/Desktop/gasussianObject/复现记录/截屏2025-05-29 11.26.14.png)

GitHub：https://github.com/chensjtu/GaussianObject

论文：《GaussianObject: High-Quality 3D Object Reconstruction from Four Views with Gaussian Splatting》

https://arxiv.org/abs/2402.10259

# 〇、本机环境与依赖复现（实际运行环境）

> 说明：下方第一、二节的「CUDA 环境 / conda 环境配置」是早期复现记录（Python 3.11 / CUDA 11.8 / torch 2.0.0），
> 与当前真实环境不一致。**当前实际使用的环境如下**，统一用 [`uv`](https://github.com/astral-sh/uv) 管理（不是 conda）。

## 系统与硬件

| 项目 | 值 |
| --- | --- |
| 操作系统 | Ubuntu 22.04.5 LTS |
| GPU | NVIDIA GeForce RTX 4090 D（24GB） |
| 显卡驱动 | 595.71.05 |
| Python 管理器 | uv 0.12.20 |

## 虚拟环境一览

| 环境目录 | Python | PyTorch | 用途 | requirements 快照 |
| --- | --- | --- | --- | --- |
| `3DGS/.venv-ddgs-vggt` | 3.10.21 | 2.5.1+cu124（CUDA 12.4） | 主线：GaussianObject + VGGT 点云 + DropGaussian 正则化 | `envs/requirements-venv-ddgs-vggt.txt` |
| `3DGS/ddgs` | 3.10.21 | 2.11.0+cu128（CUDA 12.8） | DDGS 结构正则化训练（旧 lineA） | `envs/requirements-ddgs.txt` |
| `VLMEvalKit/vlmeval_env` | 3.11.16 | 2.11.0+cu128（CUDA 12.8） | VLMEvalKit 视觉语言模型评测 | `envs/requirements-vlmevalkit.txt` |
| `Mage_ViT/mage` | 3.11.16 | 2.11.0+cu128（CUDA 12.8） | Mage_ViT 图像 / 视频生成 | `envs/requirements-mage-vit.txt` |

## 如何复现

每个环境的完整依赖已用 `uv pip freeze` 导出到 `envs/` 目录，可直接重建：

```bash
# 主线环境（GaussianObject + VGGT + DropGaussian）
uv venv .venv-ddgs-vggt --python 3.10
uv pip install -r envs/requirements-venv-ddgs-vggt.txt --python .venv-ddgs-vggt/bin/python

# DDGS 环境
uv venv ddgs --python 3.10
uv pip install -r envs/requirements-ddgs.txt --python ddgs/bin/python

# VLMEvalKit 环境（需先在仓库同级目录准备好 VLMEvalKit）
uv venv vlmeval_env --python 3.11
uv pip install -r envs/requirements-vlmevalkit.txt --python vlmeval_env/bin/python

# Mage_ViT 环境
uv venv mage --python 3.11
uv pip install -r envs/requirements-mage-vit.txt --python mage/bin/python
```

注意事项：

- 主线环境和 `ddgs` 环境依赖若干本地子模块（`pytorch3d`、`diff-gaussian-rasterization`、`simple-knn` 等），
  requirements 文件中用相对路径引用，需先递归拉取子模块：
  `git submodule update --init --recursive`。
- `ddgs` 环境引用的是 `DDGS/` 子模块（已归档），重建时需单独恢复该子模块。
- `+cu124` / `+cu128` 的 PyTorch 版本需要从 PyTorch 官方 index 安装；`uv pip freeze` 已尽量保留版本标记，
  若装成 CPU 版请参考 PyTorch 官网对应 CUDA 的安装命令覆盖 torch/torchvision/torchaudio。

#  一、 **CUDA** 环境

AutoDL 算力市场：https://www.autodl.com

![截屏2025-05-29 10.29.53](/Users/zhangao/Desktop/gasussianObject/复现记录/截屏2025-05-29 10.29.53.png)

**GPU** RTX 4090 24GB显存

CUDA 11.8

PyTorch 2.0.0

Python 3.8(ubuntu20.04)



#  二、部署GaussianObject项目

## 1.克隆仓库

   —可以本地（网络环境允许克隆后，上传autoDL服务器 /root/autodl-tmp/）—

（1）由于此仓库包含子模块，需要递归克隆整个项目。

```python
git clone https://github.com/GaussianObject/GaussianObject.git --recursive
```

（2）更新子模块

```python
cd GaussianObject

git submodule update --init --recursive
```



## 2.下载数据集

GaussianObject 支持 Mip-NeRF360 和 OmniObject3D 数据集。

![image-20250817184029154](/Users/zhangao/Library/Application Support/typora-user-images/image-20250817184029154.png)

下载网站: https://drive.google.com/drive/folders/1DUOxFybdsSYJHI5p79O_QH87TIODiJ8h

下载到本地上传到/root/autodl-tmp/GaussianObject/data目录

## 3.**conda环境配置**

（1）创建并激活新的conda环境

```python
conda create --name GaussianObject python=3.11

conda activate GaussianObject
```

（2）虚拟环境下安装PyTorch(支持CUDA)

```python
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

（3）**检查 CUDA 和 PyTorch 版本匹配：**

确保 PyTorch 版本和 CUDA 版本匹配。

```python
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

（4）安装项目所需的 Python 包

```python
pip install -r requirements.txt
```

问题：

文件 `/submodules/diff_gaussian_rasterization` 和 `/submodules/diff-gaussian-rasterization-w-pose` 的安装名称相同，导致它们发生冲突。

解决方法：

- **修改 setup.py 文件**
  文件路径：/submodules/diff-gaussian-rasterization-w-pose/setup.py

  修改内容：

  将setup.py 文件中的所有 `"diff_gaussian_rasterization"` 替换为 `"diff_gaussian_rasterization_w_pose"`

- **修改diff_gaussian_rasterization文件名**

  文件路径：/submodules/diff-gaussian-rasterization-w-pose/diff_gaussian_rasterization

  文件名 `"diff_gaussian_rasterization"` 更改为 `"diff_gaussian_rasterization_w_pose"`

终端重新运行
pip install -r requirements.txt

pip install xformers

## 4.加载预训练ControlNet 模型

GaussianObject 依赖 **SD1.5 兼容底模**（主线：Realistic Vision V5.1）和 ControlNet Tile。

详见 [`3DGS/GaussianObject/models/README.md`](../../3DGS/GaussianObject/models/README.md)。在 `models/` 下执行 `python download_hf_models.py` 即可拉：

- `Realistic_Vision_V5.1.safetensors`（`SG161222/Realistic_Vision_V5.1_noVAE`）
- `control_v11f1e_sd15_tile.pth`（`lllyasviel/ControlNet-v1-1`）

保存到 `3DGS/GaussianObject/models/`（权重不进 Git）。

# 三、**运行 GaussianObject 项目**

## 1.生成视觉轮廓（Visual Hull）

此步骤通过稀疏视角生成场景的视觉轮廓，轮廓为一个粗略的3D模型。

```python
conda activate GaussianObject

cd autodl-tmp/GaussianObject  #进入代码路径
```

**终端运行命令**：

```python
python visual_hull.py \
    --sparse_id 4 \
    --data_dir /root/autodl-tmp/GaussianObject/data/mip360/kitchen \
    --reso 2 --not_vis
```

注意：--data_dir （自己本地的数据目录路径）

这里是：/root/autodl-tmp/GaussianObject/data/mip360/kitchen

参数解释：

- `--sparse_id 4`：选择稀疏视角ID为4
- `--data_dir`：数据目录路径，指定数据集的位置
- `--reso 2`：设置分辨率为2
- `--not_vis`：不进行可视化操作

**![截屏2025-05-30 14.10.37](/Users/zhangao/Desktop/gasussianObject/复现记录/截屏2025-05-30 14.10.37.png)输出**：生成的视觉轮廓所保存的文件路径：

`/root/autodl-tmp/GaussianObject/data/mip360/kitchen/visual_hull_4.ply` 

## 2.训练粗略的 3DGS 模型

此步骤训练一个粗略的3D高斯点云表示 (3DGS)。

**终端运行** 命令：

```python
python train_gs.py \
-s /root/autodl-tmp/GaussianObject/data/mip360/kitchen \
    -m output/gs_init/kitchen \
    -r 4 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --random_background
```

注意：--data_dir （自己本地的数据目录路径）

这里是：/root/autodl-tmp/GaussianObject/data/mip360/kitchen

​           -m (模型输出路径)

我这里是：output/gs_init/kitchen

参数解释：

- `-s`：数据目录。
- `-m`：模型输出路径。
- `--init_pcd_name`：初始化点云名称（即视觉轮廓文件）。
- `--white_background`：使用白色背景。
- `--random_background`：随机背景。

![截屏2025-05-30 14.16.37](/Users/zhangao/Desktop/gasussianObject/复现记录/截屏2025-05-30 14.16.37.png)

**渲染粗略模型**：

渲染测试集：

    python render.py \
        -m output/gs_init/kitchen \
        --sparse_view_num 4 --sh_degree 2 \
        --init_pcd_name visual_hull_4 \
        --white_background --skip_all --skip_train

问题：

​	渲染时没有，试图删除 `renders.mp4`和 `gt.mp4`的文件，但该文件 **不存在**

![截屏2025-05-30 14.22.52](/Users/zhangao/Desktop/gasussianObject/复现记录/截屏2025-05-30 14.22.52.png)

解决方法：

- 修改 render.py 文件

  - 文件路径：/root/autodl-tmp/GaussianObject/render.py

  原内容：

  ```python
  os.remove(renders_path) # 224行
  os.remove(gt_path)
  ```

  修改为：

  ```python
  if os.path.exists(renders_path):
  	os.remove(renders_path)
  if os.path.exists(gt_path):
  	os.remove(gt_path)
  ```


重新运行render.py

![截屏2025-05-30 14.38.45](/Users/zhangao/Desktop/gasussianObject/复现记录/截屏2025-05-30 14.38.45.png)

渲染路径：

```python
python render.py \
    -m output/gs_init/kitchen \
    --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --render_path
```

![截屏2025-05-30 14.40.57](/Users/zhangao/Desktop/gasussianObject/复现记录/截屏2025-05-30 14.40.57.png)

## 3.留一法分析 (Leave-One-Out Analysis)

(两个阶段)该步骤通过逐个排除视角分析模型性能，进一步优化3DGS模型。

（1）第一阶段：

**终端运行**命令： 

    python leave_one_out_stage1.py -s /root/autodl-tmp/GaussianObject/data/mip360/kitchen \
        -m output/gs_init/kitchen_loo \
        -r 4 --sparse_view_num 4 --sh_degree 2 \
        --init_pcd_name visual_hull_4 \
        --white_background --random_background

![截屏2025-05-30 14.51.33](/Users/zhangao/Desktop/gasussianObject/复现记录/截屏2025-05-30 14.51.33.png)

![截屏2025-05-30 14.52.58](/Users/zhangao/Desktop/gasussianObject/复现记录/截屏2025-05-30 14.52.58.png)

（2）第二阶段：

**终端运行**命令：

```python
python leave_one_out_stage2.py -s /root/autodl-tmp/GaussianObject/data/mip360/kitchen \
    -m output/gs_init/kitchen_loo \
    -r 4 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --random_background
```

问题：

`torch.load()` 加载 checkpoint 文件失败了，原因是 PyTorch 在 2.6 版本开始默认开启了 `weights_only=True` 模式

![截屏2025-05-29 10.19.33](/Users/zhangao/Desktop/gasussianObject/复现记录/截屏2025-05-29 10.19.33.png)

解决方法：

- leave_one_out_stage2.py

  - 文件路径：/root/autodl-tmp/GaussianObject/leave_one_out_stage2.py
  
  原内容：
  
  ```python
  (model_params, first_iter) = torch.load(checkpoint)
  ```

​       更改为：

	(model_params, first_iter) = torch.load(checkpoint, weights_only=False)
**重新运行** leave_one_out_stage2.py

![截屏2025-05-30 15.00.02](/Users/zhangao/Desktop/gasussianObject/复现记录/截屏2025-05-30 15.00.02.png)

![截屏2025-05-30 15.01.01](/Users/zhangao/Desktop/gasussianObject/复现记录/截屏2025-05-30 15.01.01.png)

## 4. 使用LoRA进行微调

使用LoRA进行微调，通过控制提示符和其他适应性调整增强模型细节。

该步骤通过LoRA优化模型，使用提示符 `xxy5syt00` 进行微调。

需要 `Realistic_Vision_V5.1.safetensors` 和 `control_v11f1e_sd15_tile.pth` 放在指定路径：

/root/autodl-tmp/GaussianObject/models/文件夹下

**终端运行**命令：

```python
python train_lora.py --exp_name controlnet_finetune/kitchen \
    --prompt xxy5syt00 --sh_degree 2 --resolution 4 --sparse_num 4 \
    --data_dir /root/autodl-tmp/GaussianObject/data/mip360/kitchen \
    --gs_dir output/gs_init/kitchen \
    --loo_dir output/gs_init/kitchen_loo \
    --bg_white --sd_locked --train_lora --use_prompt_list \
    --add_diffusion_lora --add_control_lora --add_clip_lora
```

问题1:

- autoDL无法连接huggingface下载 CLIP 模型

解决方案：

- 本地（网络环境允许）下载好--> 上传到

- 本地新建个 load_CLIP.py，内容如下：

```python
from transformers import CLIPTokenizer, CLIPTextModel

model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

model.save_pretrained("./clip-vit-large-patch14")
tokenizer.save_pretrained("./clip-vit-large-patch14")
```

- 本地（网络环境允许）运行 load_CLIP.py

会在load_CLIP.py目录下得到 **clip-vit-large-patch14 **文件夹

复制到服务器/root/autodl-tmp/GaussianObject/pretrained/文件夹下

（在/root/autodl-tmp/GaussianObject/目录下新建pretrained文件下放入clip-vit-large-patch14整个文件夹）



问题2：

- `torch.load()` 加载 `.ckpt` 模型时，里面保存了一个 **`pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` 对象**， PyTorch**默认不再允许反序列化（加载）这类 Python 类**。

![截屏2025-05-29 10.43.25](/Users/zhangao/Desktop/gasussianObject/复现记录/截屏2025-05-29 10.43.25.png)

解决方法：

- 修改model.py文件

  文件路径：/root/autodl-tmp/GaussianObject/cldm/model.py

  ```python
  state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location))
  ```

  修改为

  ```python
  state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location), weights_only=False))
  ```

重新运行 train_lora.py

![截屏2025-05-30 15.21.34](/Users/zhangao/Desktop/gasussianObject/复现记录/截屏2025-05-30 15.21.34.png)

![截屏2025-05-30 15.21.51](/Users/zhangao/Desktop/gasussianObject/复现记录/截屏2025-05-30 15.21.51.png)

## 5. **高斯修复 (Gaussian Repair)**

该步骤应用高斯修复模型，进一步修复和优化3D表示。

**终端运行** 命令：

    python train_repair.py \
        --config configs/gaussian-object.yaml \
        --train --gpu 0 \
        tag="kitchen" \
        system.init_dreamer="output/gs_init/kitchen" \
        system.exp_name="output/controlnet_finetune/kitchen" \
        system.refresh_size=8 \
        data.data_dir="/root/autodl-tmp/GaussianObject/data/mip360/kitchen" \
        data.resolution=4 \
        data.sparse_num=4 \
        data.prompt="a photo of a xxy5syt00" \
        data.refresh_size=8 \
        system.sh_degree=2

![截屏2025-05-30 15.34.49](/Users/zhangao/Desktop/gasussianObject/复现记录/截屏2025-05-30 15.34.49.png)

![截屏2025-05-30 15.35.35](/Users/zhangao/Desktop/gasussianObject/复现记录/截屏2025-05-30 15.35.35.png)

![截屏2025-05-30 15.36.04](/Users/zhangao/Desktop/gasussianObject/复现记录/截屏2025-05-30 15.36.04.png)

![截屏2025-05-30 15.36.42](/Users/zhangao/Desktop/gasussianObject/复现记录/截屏2025-05-30 15.36.42.png)

最终的 3DGS 表示保存在 `output/gaussian_object/kitchen/save/last.ply`

## 6. **渲染最终的3DGS表示**

渲染最终修复后的3D模型：

**渲染测试集**：

```python
python render.py \
    -m output/gs_init/kitchen \
    --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --skip_all --skip_train \
    --load_ply output/gaussian_object/kitchen/save/last.ply
```

![截屏2025-05-30 15.40.01](/Users/zhangao/Desktop/gasussianObject/复现记录/截屏2025-05-30 15.40.01.png)

**渲染路径**：

    python render.py \
        -m output/gs_init/kitchen \
        --sparse_view_num 4 --sh_degree 2 \
        --init_pcd_name visual_hull_4 \
        --white_background --render_path \
        --load_ply output/gaussian_object/kitchen/save/last.ply

![截屏2025-05-30 15.43.26](/Users/zhangao/Desktop/gasussianObject/复现记录/截屏2025-05-30 15.43.26.png)

渲染结果保存在`output/gs_init/kitchen/test/ours_None`和 `output/gs_init/kitchen/render/ours_None`中

# 四、last.ply成像效果

在线可视化工具：https://superspl.at/editor

![截屏2025-05-30 15.53.48](/Users/zhangao/Desktop/gasussianObject/复现记录/截屏2025-05-30 15.53.48.png)

![截屏2025-05-30 17.14.16](/Users/zhangao/Desktop/gasussianObject/复现记录/截屏2025-05-30 17.14.16.png)

![截屏2025-05-30 17.12.26](/Users/zhangao/Desktop/gasussianObject/复现记录/截屏2025-05-30 17.12.26.png)

# 五、 COLMAP-Free GaussianObject (CF-GaussianObject)

## 1.下载[SAM](https://github.com/facebookresearch/segment-anything)和[DUSt3R](https://github.com/naver/dust3r)或[MASt3R](https://github.com/naver/mast3r)检查点

```sh
cd models
sh download_preprocess_models.sh
cd ..
```

整理目录结构

`./data`下 准备 4 张图片的数据集

```text
GaussianObject
├── data
│   ├── <your dataset name>
│   │   ├── images
│   │   │   ├── 0001.png
│   │   │   ├── 0002.png
│   │   │   ├── 0003.png
│   │   │   └── 0004.png
│   │   ├── sparse_4.txt
│   │   └── sparse_test.txt
│   └── ...
└── ...
```

其中`sparse_4.txt`和`sparse_test.txt`包含与输入图像相同的序列号，从 0 开始。如果所有图像都用于训练，则文件应该是

```text
0
1
2
3
```

执行 downsample.py 文件，进行 图像下采样

```sh
python preprocess/downsample.py -s data/realcap/cuc
```

## 2.生成掩码

执行 `segment_anything.ipynb`使用 SAM 生成掩码

## 3. 生成粗略姿势

[DUSt3R](https://github.com/naver/dust3r)用于估计输入图像的粗略姿态。您可以使用以下方法获取姿态：

```
python pred_poses.py -s data/realcap/cuc --sparse_num 4
```

![image-20250628172216013](/Users/zhangao/Library/Application Support/typora-user-images/image-20250628172216013.png)

中提供了替代的[MASt3R](https://github.com/naver/mast3r)`pred_poses_mast3r.py`脚本。



问题1: 

![img](https://cdn.nlark.com/yuque/0/2025/png/49848623/1750385291023-6d7fbd15-a54b-4998-97d9-353732a0524a.png)

解决方法：

转到 dust3r/inference.py  文件

```python
ckpt = torch.load(model_path, map_location='cpu')
```

改为

```python
ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
```



问题2：

![img](https://cdn.nlark.com/yuque/0/2025/png/49848623/1750385239768-39ac69aa-92b1-4bb1-8f04-2ce5e38424c2.png)

解决方法：

找到pred_poses.py

```python
original_masks = [np.array(Image.open(mask).resize(image.size))[:, :, 0] / 255.0 for mask, image in zip(masks, original_images)]
```

改为：

```python
# ✅ 安全兼容灰度图和RGB图的写法
original_masks = []
for mask_path, image in zip(masks, original_images):
    mask_img = Image.open(mask_path).resize(image.size)
    mask_np = np.array(mask_img)
    if mask_np.ndim == 3:  # RGB 图像
        mask_np = mask_np[:, :, 0]
    original_masks.append(mask_np / 255.0)
```

再次运行：python pred_poses.py -s data/realcap/cuc --sparse_num 4

![image-20250628172216013](/Users/zhangao/Library/Application Support/typora-user-images/image-20250628172216013.png)

## 4.高斯修复

数据准备好后，后续步骤与标准 GaussianObject 类似,一步步执行即可。(修改文件路径/文件名)

```sh
python train_gs.py -s data/realcap/cuc \
    -m output/gs_init/cuc \
    -r 8 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name dust3r_4 \
    --white_background --random_background --use_dust3r

python render.py \
    -m output/gs_init/cuc \
    --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name dust3r_4 \
    --dust3r_json output/gs_init/cuc/refined_cams.json \
    --white_background --render_path --use_dust3r

python leave_one_out_stage1.py -s data/realcap/cuc \
    -m output/gs_init/cuc_loo \
    -r 8 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name dust3r_4 \
    --dust3r_json output/gs_init/cuc/refined_cams.json \
    --white_background --random_background --use_dust3r

python leave_one_out_stage2.py -s data/realcap/cuc \
    -m output/gs_init/cuc_loo \
    -r 8 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name dust3r_4 \
    --dust3r_json output/gs_init/cuc/refined_cams.json \
    --white_background --random_background --use_dust3r

python train_lora.py --exp_name controlnet_finetune/cuc \
    --prompt xxy5syt00 --sh_degree 2 --resolution 8 --sparse_num 4 \
    --data_dir data/realcap/cuc \
    --gs_dir output/gs_init/cuc \
    --loo_dir output/gs_init/cuc_loo \
    --bg_white --sd_locked --train_lora --use_prompt_list \
    --add_diffusion_lora --add_control_lora --add_clip_lora --use_dust3r

python train_repair.py \
    --config configs/gaussian-object-colmap-free.yaml \
    --train --gpu 0 \
    tag="cuc" \
    system.init_dreamer="output/gs_init/cuc" \
    system.exp_name="output/controlnet_finetune/cuc" \
    system.refresh_size=8 \
    data.data_dir="data/realcap/cuc" \
    data.resolution=8 \
    data.sparse_num=4 \
    data.prompt="a photo of a xxy5syt00" \
    data.json_path="output/gs_init/cuc/refined_cams.json" \
    data.refresh_size=8 \
    system.sh_degree=2

python render.py \
    -m output/gs_init/cuc \
    --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name dust3r_4 \
    --white_background --render_path --use_dust3r \
    --load_ply output/gaussian_object/cuc/save/last.ply
```

![image-20250701101843483](/Users/zhangao/Library/Application Support/typora-user-images/image-20250701101843483.png)

![image-20250701102027420](/Users/zhangao/Library/Application Support/typora-user-images/image-20250701102027420.png)

# 9.10接入 VGGT 增强点云初始化
## model: VGGT 1B 版本，预训练模型
https://huggingface.co/facebook/VGGT-1B

# 9.16接入 DropGaussian 结构正则化-高斯训练
