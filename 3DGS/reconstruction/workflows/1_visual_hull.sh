# 可视化hull


# 输出： 在 data/mip360/kitchen/ 下生成 visual_hull_4.ply 文件 视觉外壳点云文件，用于初始化3D高斯

CUDA_VISIBLE_DEVICES=2  python visual_hull_test.py \
    --sparse_id 9 \
    --data_dir /data/zhangao_data/3DGS/reconstruction/data/mip360/kitchen \
    --reso 2 --not_vis