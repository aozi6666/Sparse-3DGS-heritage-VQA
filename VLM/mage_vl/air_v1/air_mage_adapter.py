"""显式帧选择的 Mage 适配层；不改模型文件，导入时不加载 Torch 或模型。

Mage 的预解码列表入口会把时间轴重置为 0..T-1。本模块保留原生像素处理，
只在输出处补回来源视频的真实帧号和秒数，最终仍由原生 MageProcessor 展开提示词。
"""
from contextlib import contextmanager
import math
import sys


def validate_selection(frames, indices, fps, frame_count):
    """只检查身份／顺序；禁止重复帧、越界和伪造稠密时间轴。"""
    indices = tuple(indices)
    if type(frame_count) is not int or frame_count <= 0:
        raise ValueError('来源视频frame_count必须为正整数')
    try:
        fps_value = float(fps)
    except (TypeError, ValueError) as exc:
        raise ValueError('来源视频fps必须为数值') from exc
    if isinstance(fps, bool) or not math.isfinite(fps_value) or fps_value <= 0:
        raise ValueError('来源视频fps必须有限且大于0，不猜测30fps')
    if not indices or len(frames) != len(indices):
        raise ValueError('帧内容与原始帧号必须一一对应且非空')
    if any(type(i) is not int or not 0 <= i < frame_count for i in indices):
        raise ValueError('原始帧号非法或超出来源视频范围')
    if any(a >= b for a, b in zip(indices, indices[1:])):
        raise ValueError('最终帧号必须严格递增，不能重复或按相关度顺序输入')
    return indices


def _native_symbols(base):
    """仅从已加载的原生模块取函数；不执行第三方源码或导入权重。"""
    current = base
    for _ in range(8):
        module = sys.modules.get(type(current).__module__)
        names = ('smart_resize', 'build_patch_positions', 'format_timestamp', 'time_str_to_seconds')
        if module is not None and all(callable(getattr(module, n, None)) for n in names):
            return tuple(getattr(module, n) for n in names)
        # 兼容已有透明 CaptureVideoProcessor；它的 base 才是原生处理器。
        if not hasattr(current, 'base'):
            break
        current = current.base
    raise TypeError('未找到Mage原生视频处理函数；拒绝以其他处理器替代')


def _prepare_frames(base, frames, smart_resize):
    """复现原生decord路径的预缩放，然后才进入原生image processor。

    只把原始RGB图片直接交给image processor会少一次原生视频预缩放，
    即使最终网格相同也可能产生不同像素，必须完整保留两阶段流程。
    """
    import numpy as np
    import torch
    from PIL import Image

    arrays = []
    for frame in frames:
        if isinstance(frame, Image.Image):
            if frame.mode != 'RGB':
                raise ValueError('来源视频帧必须为RGB，不能静默改变色彩模式')
            arr = np.asarray(frame)
        else:
            arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
            raise ValueError('来源视频帧必须为HxWx3 uint8 RGB')
        if min(arr.shape[:2]) <= 0:
            raise ValueError('视频帧不能为空')
        arrays.append(arr)
    if len({a.shape for a in arrays}) != 1:
        raise ValueError('同一视频选择的帧尺寸不一致')
    arr = np.stack(arrays, axis=0)
    height, width = arr.shape[1:3]
    if base.resize_frames and (base.min_pixels or base.max_pixels):
        resized_h, resized_w = smart_resize(
            height, width, base.patch_size,
            min_pixels=base.min_pixels, max_pixels=base.max_pixels,
            align_patch_size=base.patch_size * 2,
        )
        if min(resized_h, resized_w) <= 0:
            raise ValueError('原生预缩放产生非法尺寸')
        if (resized_h, resized_w) != (height, width):
            from torchvision import transforms as transforms
            from torchvision.transforms import InterpolationMode
            tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
            tensor = transforms.functional.resize(
                tensor, [resized_h, resized_w],
                interpolation=InterpolationMode.BICUBIC, antialias=True,
            )
            arr = tensor.permute(0, 2, 3, 1).contiguous().numpy()
    return [Image.fromarray(frame) for frame in arr]


class SelectedFramesVideoProcessor:
    """一次显式帧选择的透明包装器，保留原生处理器的属性覆盖／恢复约定。"""
    _OWN = {'base', 'frames', 'indices', 'seconds', 'fps', 'frame_count', 'video_path',
            'calls', '_smart_resize', '_build_positions'}

    def __init__(self, base, frames, indices, fps, frame_count, video_path=None):
        indices = validate_selection(frames, indices, fps, frame_count)
        smart_resize, build_positions, format_timestamp, parse_timestamp = _native_symbols(base)
        object.__setattr__(self, 'base', base)
        object.__setattr__(self, 'frames', tuple(frames))
        object.__setattr__(self, 'indices', indices)
        object.__setattr__(self, 'fps', float(fps))
        object.__setattr__(self, 'frame_count', frame_count)
        object.__setattr__(self, 'video_path', None if video_path is None else str(video_path))
        # 与原生路径一样经六位小数时间戳往返，确保uniform回归逐token一致。
        seconds = tuple(parse_timestamp(format_timestamp(i / float(fps))) for i in indices)
        if any(not math.isfinite(s) or s < 0 for s in seconds):
            raise ValueError('原生时间戳转换非法')
        if any(a >= b for a, b in zip(seconds, seconds[1:])):
            raise ValueError('秒数在原生时间精度下不递增')
        object.__setattr__(self, 'seconds', seconds)
        object.__setattr__(self, '_smart_resize', smart_resize)
        object.__setattr__(self, '_build_positions', build_positions)
        object.__setattr__(self, 'calls', 0)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __setattr__(self, name, value):
        if name in self._OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(self.base, name, value)

    def __call__(self, videos, return_tensors='pt', **kwargs):
        import torch
        if return_tensors not in (None, 'pt'):
            raise ValueError('仅支持原生pt输出')
        if not isinstance(videos, (list, tuple)) or len(videos) != 1:
            raise ValueError('显式帧适配仅支持一次一条视频')
        if self.video_path is not None and videos[0] != self.video_path:
            raise ValueError('显式选择的来源视频路径与本次提示词不符')
        prepared = _prepare_frames(self.base, self.frames, self._smart_resize)
        values = dict(self.base(videos=[prepared], return_tensors=return_tensors, **kwargs))
        grid = values['video_grid_thw']
        if grid.ndim != 2 or grid.shape[0] != 1 or grid.shape[1] != 3:
            raise ValueError('原生视频网格不是单视频[T,H,W]')
        if int(grid[0, 0]) != len(self.indices):
            raise ValueError('原生处理后的帧数与选择帧数不符')
        expected_patches = int(grid.prod(dim=1).sum())
        if values['pixel_values_videos'].shape[0] != expected_patches:
            raise ValueError('原生像素patch数量与网格不符')
        values['patch_positions'] = self._build_positions(
            grid, spatial_merge_size=self.base.spatial_merge_size,
            frame_indices=[torch.tensor(self.indices, dtype=torch.int64)],
        )
        if values['patch_positions'].shape != (expected_patches, 3):
            raise ValueError('原始帧号patch位置形状不符')
        values['frame_timestamps'] = [list(self.seconds)]
        self.calls += 1
        return values


@contextmanager
def selected_video_inputs(processor, frames, indices, fps, frame_count, video_path=None):
    """临时接管采样入口；异常也恢复原对象，不污染下一题／原生基线。"""
    original = processor.video_processor
    if isinstance(original, SelectedFramesVideoProcessor):
        raise RuntimeError('不能嵌套显式帧选择，也不能并发共享同一processor')
    wrapper = SelectedFramesVideoProcessor(original, frames, indices, fps, frame_count, video_path)
    sampling = (original.fixed_num_frames, original.max_frames, original.target_fps)
    processor.video_processor = wrapper
    try:
        yield wrapper
    finally:
        original.fixed_num_frames, original.max_frames, original.target_fps = sampling
        processor.video_processor = original


def image_inputs(processor, image, prompt):
    """VLM逐帧分析直接用原生单图路径；不把时间序列伪装成图片批次。"""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('逐帧分析提示词不能为空')
    chat = processor.apply_chat_template(
        [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}],
        tokenize=False, add_generation_prompt=True,
    )
    return processor(text=[chat], images=[image], return_tensors='pt', padding=True)


def assert_inputs_equal(left, right):
    """原生uniform回归的硬性门槛：比较所有字段，包含像素和时间位置。"""
    import torch
    if set(left) != set(right):
        raise ValueError(f'原生输入字段不一致: {sorted(set(left) ^ set(right))}')
    for name in left:
        a, b = left[name], right[name]
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise TypeError(f'待比较的最终输入字段不是Tensor: {name}')
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            raise ValueError(f'原生输入逐tensor回归失败: {name}')
    return True
