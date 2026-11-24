import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

data_path = 'data/realcap/cuc'
sparse_num = 13
image_path = os.path.join(data_path, 'images')

# 加载稀疏ID
sparse_ids = np.loadtxt(os.path.join(data_path, f'sparse_{sparse_num}.txt'), dtype=np.int32)
print(f"稀疏ID: {sparse_ids}")

# 获取所有图像文件名
all_image_names = sorted(os.listdir(image_path))
print(f"总图像数量: {len(all_image_names)}")

# 根据稀疏ID选择图像
image_names = []
for idx in sparse_ids:
    if idx < len(all_image_names):
        image_names.append(all_image_names[idx])
    else:
        print(f"警告: 索引 {idx} 超出范围")

print(f"选择的图像: {image_names}")

# 逐个加载图像（避免列表推导式的问题）
images = []
for image_name in image_names:
    image_file_path = os.path.join(image_path, image_name)
    print(f"正在加载: {image_name}")
    
    # 读取图像
    img = cv2.imread(image_file_path)
    if img is None:
        print(f"错误: 无法读取 {image_name}")
        continue
    
    # 转换颜色
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)
        print(f"成功: {image_name}")
    except Exception as e:
        print(f"错误: 颜色转换失败 {image_name} - {e}")
        continue

print(f"成功加载了 {len(images)} 个图像")

# 显示图像
for i, image in enumerate(images):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.title(f"图像 {i+1}: {image_names[i]}")
    plt.axis('on')
    plt.show() 