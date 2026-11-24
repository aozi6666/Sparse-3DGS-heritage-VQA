# 本地数据接入约定

## OmniObject3D

复制到 data/raw/omniobject3d，可以直接保留官方blender_renders外层。程序递归寻找transforms.json；每个文件定义一个对象的render目录，内含frames。

支持常见NeRF Blender格式：frame.file_path、frame.transform_matrix（OpenGL camera-to-world），meta.camera_angle_x 或 fl_x/fl_y/cx/cy。图片可以通过file_path精确定位或唯一扩展名补齐。深度使用frame.depth_file_path，或depths下与RGB同stem前缀的唯一EXR/NPY；多个候选会停止。

深度必须明确：metadata.depth_type 或命令 --depth-type。camera_z_depth直接读取；ray_distance根据K反投影射线长度换成Z。整数灰度、未知单位或未知深度类型拒绝。RGBA前景alpha与浮点深度有效性共同生成mask；背景必须是无效深度，不能已被裁成任意伪深度。

只有raw_scans/视频RGB并不等于RGB-D，当前入口不自动渲染模型或生成伪GT。

## CO3Dv2

保留 <category>/frame_annotations.jgz、<sequence>/{images,depths,depth_masks,masks}，以及已有的官方本地序列划分资料。

程序支持官方uint16 PNG内float16位模式和scale_adjustment，不能把PNG像素直接除以1000。相机R/T按PyTorch3D右乘约定转换，focal_length/principal_point支持ndc_norm_image_bounds与ndc_isotropic。

先创建 data/splits/co3d_sequences.json，明确每条序列唯一的集合：

```json
{
  "cup/SEQUENCE_ID_A": "train",
  "cup/SEQUENCE_ID_B": "val",
  "cup/SEQUENCE_ID_C": "test"
}
```

映射应来自你下载的官方划分；不要把同一序列中的known/train帧和unknown/test帧当成不同对象集合。若官方文件只有同序列的帧划分，需要先整理独立序列协议，再评估。工具不会猜测官方标注或随机把官方测试序列加入训练。

首次转换不加--co3d-depth-reviewed。先查看reports/data_quality的RGB/GT/mask和GT跨视角投影覆盖，再确认深度质量、坐标及尺度；确认后用独立processed目录重新转换并加该标记。Stage B拒绝未经review的CO3D深度。当前未实现“不可靠深度仅作无监督MV训练”，不会拿它冒充已完成分支。

## 转换后的统一格式

```text
data/processed/<dataset>/<object>/
  images/00000.jpg
  depths/00000.npy        # float32二维相机Z-depth，RGB同尺寸
  masks/00000.png         # 0或255
  cameras.json
```

cameras.json包括schema_version、id、group_id、dataset、depth_type=camera_z_depth、coordinate_convention=opencv_camera_from_world、depth_reviewed、provenance和frames。

每个frame：image/depth/mask（相对路径）、K（3×3像素内参）、E（4×4世界到相机矩阵）。depth与E的平移必须同单位，K无畸变。RGB缩放或letterbox后K同步更新，mask/depth最近邻。不要单独归一化每张深度。

同一个实物多个扫描应合并成一个group_id，不得跨train/val/test。manifest会拒绝重复组；不要仅改object文件名绕过检查。
