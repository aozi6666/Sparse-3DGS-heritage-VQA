import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source-path', type=str, default='data/realcap/rabbit')
    parser.add_argument('--sparse_num', type=int, default=4)
    args = parser.parse_args()

    images_path = os.path.join(args.source_path, 'images')

    factors = (2, 4, 8)
    for factor in factors:
        images_path_resize = f'{images_path}_{factor}'
        if not os.path.exists(images_path_resize):
            os.mkdir(images_path_resize)
        # 只处理图像文件，过滤掉系统文件如 .DS_Store
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        image_files = [f for f in sorted(os.listdir(images_path)) 
                      if os.path.splitext(f)[1].lower() in image_extensions]
        
        for image_name in tqdm(image_files):
            image = Image.open(os.path.join(images_path, image_name))
            orig_w, orig_h = image.size[0], image.size[1]
            resolution = round(orig_w / factor), round(orig_h / factor)
            image = image.resize(resolution)
            image.save(os.path.join(images_path_resize, image_name))
